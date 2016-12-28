import tensorflow as tf
import numpy as np


class BeamSearchDecoderResult(object):

    def __init__(self, context_indices, starts, ends, probs):

        assert len(context_indices) == len(starts) == len(ends) == len(probs)

        self.context_indices = context_indices
        self.starts = starts
        self.ends = ends
        self.probs = probs

    def __iter__(self):

        return zip(self.context_indices, self.starts, self.ends, self.probs)


class BeamSearchDecoder(object):
    """From a QASetting batch, computes most likely (start, end) pairs via beam search."""


    def __init__(self, sess, model, beam_size):

        self._sess = sess
        self._model = model
        self._beam_size = beam_size


    def decode(self, qa_settings):
        """For each QASetting, finds most likely (start, end) pairs via beam search.

        :param qa_settings: List fo QASetting objects
        :return: List of BeamSearchDecoderResult objects
        """

        self._model.set_eval(self._sess)

        # Get start probs, context partition and all relevant intermediate results
        # to predict end pointer later.
        context_partition, matched_output, question_representation, start_probs = self._sess.run(
                [self._model.context_partition,
                 self._model.matched_output,
                 self._model.question_representation,
                 self._model.start_probs],
                self._model.get_feed_dict(qa_settings))

        num_questions = context_partition[-1] + 1
        assert num_questions == len(qa_settings)

        contexts, starts, top_start_probs = self._compute_top_starts(start_probs,
                                                                     context_partition)

        predicted_starts = starts.flatten()
        batch_index_contexts = self._context_index_to_batch_index(contexts,
                                                                  qa_settings)
        predicted_contexts = batch_index_contexts.flatten()

        # Compute end probs by feeding all necessary itermediate results & start pointers
        feed_dict = self._model.get_feed_dict(qa_settings)
        feed_dict.update({
            # TODO: Find out why I need to feed this:
            self._model.correct_start_pointer: [],
            # Starts
            self._model.predicted_answer_starts: predicted_starts,
            self._model.answer_context_indices: predicted_contexts,
            # Intermediate Results (so no recomputation needed)
            self._model.matched_output: matched_output,
            self._model.question_representation: question_representation,
        })
        [end_probs] = self._sess.run([self._model.end_probs], feed_dict)

        contexts, starts, ends, probs = self._compute_top_spans(
                contexts, starts, top_start_probs, end_probs)

        assert len(contexts) == len(starts) == len(ends) == len(probs) == num_questions

        return [BeamSearchDecoderResult(context_indices=contexts[i],
                                        starts=starts[i],
                                        ends=ends[i],
                                        probs=probs[i])
                for i in range(num_questions)]

    def _compute_top_starts(self, start_probs, context_partition):
        """
        Computes the top starts from a start_probs matrix.
        :param start_probs: Shape [num_contexts, max_token_length] np array
                with probabilities.
        :param context_partition: Shape [num_contexts] np array that maps each
                context index to its question index.
        :return: (contexts, starts, probs) np arrays, each of shape
                [num_questions, beam_size]:
                    - context[q, k]: the index (within the question's contexts)
                            of the kth most probable context for question q.
                    - starts[q, k]: the kth most probable answer pointer for
                            question q.
                    - probs[q, k]: the probability of the kth most
                            probable answer for question q.
        """

        contexts, starts, probs = self._top_k_2d(start_probs, context_partition)

        return contexts, starts, probs


    def _compute_top_spans(self, contexts, top_starts, top_start_probs, end_probs):
        """
        Performs one beam search step to find the top (start, end) pairs.
        :param contexts: Shape [num_questions, beam_size] np array of context indices.
        :param top_starts: Shape [num_questions, beam_size] np array of starts.
        :param top_start_probs: Shape [num_questions, beam_size] np array of start probs.
        :param end_probs: Shape [num_questions * beam_size, num_tokens] np array
                of end probabilities, one for each start token.
        :return: contexts, top_starts, top_ends, top_probs, all
                [num_questions, beam_size] arrays.
        """

        # Compute top ends for each start
        _, top_ends, top_end_probs = self._top_k_2d(end_probs)

        # Convert everything to [num_questions, beam_size * beam_size] arrays
        beam_size_squared = self._beam_size * self._beam_size
        top_ends = np.reshape(top_ends, [-1, beam_size_squared])
        top_end_probs = np.reshape(top_end_probs, [-1, beam_size_squared])
        top_starts = np.repeat(top_starts, self._beam_size, axis=1)
        top_start_probs = np.repeat(top_start_probs, self._beam_size, axis=1)
        contexts = np.repeat(contexts, self._beam_size, axis=1)

        # Compute top spans
        total_probs = top_start_probs * top_end_probs
        _, top_indices, top_probs = self._top_k_2d(total_probs)

        # Gather [num_questions, beam_size] arrays
        top_ends = self._gather_rowwise(top_ends, top_indices)
        top_starts = self._gather_rowwise(top_starts, top_indices)
        contexts = self._gather_rowwise(contexts, top_indices)

        return contexts, top_starts, top_ends, top_probs


    def _top_k_2d(self, values, partition=None):
        """
        Computes top row & column indices and values for each partition. If
        partition is not defined, each row is a partition, otherwise partitions
        are collections of consecutive rows.
        :param values: 2D Array of values.
        :param partition: Optional. Shape [len(values] int array.
        :return: rows, cols, values, all shape [num_partitions, beam_size] arrays.
        """

        if partition is None:
            partition = np.arange(len(values))

        num_partitions = partition[-1] + 1

        rows = np.zeros([num_partitions, self._beam_size], dtype=np.int64)
        cols = np.zeros([num_partitions, self._beam_size], dtype=np.int64)
        top_values = np.zeros([num_partitions, self._beam_size], dtype=np.float32)

        for p in range(num_partitions):

            # Collect values
            current_values = values[partition == p]
            n_rows, n_cols = current_values.shape
            values_indices = [(values[row, col], row, col)
                              for row in range(n_rows)
                              for col in range(n_cols)]

            # Sort by values descending
            values_indices = sorted(values_indices,
                                    key=lambda v: -v[0])[:self._beam_size]

            # Unpack
            rows[p], cols[p], top_values[p] = zip(*values_indices)

        return rows, cols, top_values


    def _gather_rowwise(self, values, indices):
        """
        For each row in indices, select the values from the corresponding row
        in values.
        :param values: 2D Array
        :param indices: Shape [n_rows, k] int array of indices.s
        :return: Shape [n_rows, k] result array where
                result[r, c] = values[r, indices[r, c]].
        """

        result = np.zeros(indices.shape, dtype=values.dtype)

        for row in range(len(indices)):
            result[row,:] = values[row, indices[row]]

        return result


    def _context_index_to_batch_index(self, contexts, qa_settings):
        """
        Convertes per-question context indices to context batch indices.
        :param contexts: Shape [num_questions, k] np array of context indices.
        :param qa_settings: List of all num_questions QASetting objects.
        :return: Shape [num_questions, k] np array of transformed indices
        """

        offsets = np.cumsum([len(s.contexts) for s in qa_settings])
        offsets = np.roll(offsets, 1)
        offsets[0] = 0

        return contexts + offsets.reshape([-1, 1])
