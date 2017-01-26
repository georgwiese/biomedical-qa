import tensorflow as tf
import numpy as np


class BeamSearchDecoderResult(object):

    def __init__(self, context_indices, starts, ends, probs,
                 start_probs, end_probs):

        assert len(context_indices) == len(starts) == len(ends) == len(probs)

        self.context_indices = context_indices
        self.starts = starts
        self.ends = ends
        self.probs = probs
        self.start_probs = start_probs
        self.end_probs = end_probs

    def __iter__(self):

        return zip(self.context_indices, self.starts, self.ends, self.probs)


class ModelEnsemble(object):


    def __init__(self, sess, models):

        self._sess = sess
        self._models = models
        self._intermediate_results = {}

        assert self._models[0].start_output_unit == "sigmoid"


    def set_eval(self):

        for model in self._models:
            model.set_eval(self._sess)


    def get_start_probs(self, qa_settings):

        # Should be the same across models
        context_partition = None

        # Added across models
        start_scores = None

        for model in self._models:

            context_partition, matched_output, question_representation, model_start_scores = \
                self._sess.run([model.context_partition,
                                model.matched_output,
                                model.question_representation,
                                model.start_scores], model.get_feed_dict(qa_settings))

            self._intermediate_results[model] = {
                model.matched_output: matched_output,
                model.question_representation: question_representation,
            }

            if start_scores is None:
                start_scores = model_start_scores
            else:
                start_scores += model_start_scores

        # Average start scores
        start_scores /= len(self._models)
        start_probs = 1 / (1 + np.exp(-start_scores))

        return context_partition, start_probs


    def get_end_probs(self, qa_settings, predicted_starts, predicted_contexts):

        end_scores = None

        for model in self._models:

            feed_dict = model.get_feed_dict(qa_settings)
            feed_dict.update(self._intermediate_results[model])
            feed_dict.update({
                # TODO: Find out why I need to feed this:
                model.correct_start_pointer: [],
                # Starts
                model.predicted_answer_starts: predicted_starts,
                model.answer_context_indices: predicted_contexts,
            })
            [model_end_scores] = self._sess.run([model.end_scores], feed_dict)

            if end_scores is None:
                end_scores = model_end_scores
            else:
                end_scores += model_end_scores

        # Average end scores
        end_scores /= len(self._models)

        # Compute softmax
        end_scores -= end_scores.max(axis=1).reshape([-1, 1])
        end_scores_exp = np.exp(end_scores)
        end_probs = end_scores_exp / end_scores_exp.sum(axis=1).reshape([-1, 1])

        return end_probs



class BeamSearchDecoder(object):
    """From a QASetting batch, computes most likely (start, end) pairs via beam search."""


    def __init__(self, sess, model, beam_size):

        self._sess = sess
        self._model_ensemble = ModelEnsemble(sess, [model])
        self._beam_size = beam_size


    def decode(self, qa_settings):
        """For each QASetting, finds most likely (start, end) pairs via beam search.

        :param qa_settings: List fo QASetting objects
        :return: List of BeamSearchDecoderResult objects
        """

        self._model_ensemble.set_eval()

        context_partition, start_probs = self._model_ensemble.get_start_probs(qa_settings)

        num_questions = context_partition[-1] + 1
        assert num_questions == len(qa_settings)

        contexts, starts, top_start_probs = self._compute_top_starts(start_probs,
                                                                     context_partition)

        predicted_starts = starts.flatten()
        batch_index_contexts = self._context_index_to_batch_index(contexts,
                                                                  qa_settings)
        predicted_contexts = batch_index_contexts.flatten()

        # Compute end probs by feeding all necessary itermediate results & start pointers
        end_probs = self._model_ensemble.get_end_probs(qa_settings, predicted_starts, predicted_contexts)

        contexts, starts, ends, probs = self._compute_top_spans(
                contexts, starts, top_start_probs, end_probs)

        assert len(contexts) == len(starts) == len(ends) == len(probs) == num_questions

        return [BeamSearchDecoderResult(context_indices=contexts[i],
                                        starts=starts[i],
                                        ends=ends[i],
                                        probs=probs[i],
                                        start_probs=start_probs[i],
                                        end_probs=end_probs[i])
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
            values_indices = [(current_values[row, col], row, col)
                              for row in range(n_rows)
                              for col in range(n_cols)]

            # Sort by values descending
            values_indices = sorted(values_indices,
                                    key=lambda v: -v[0])[:self._beam_size]

            # Unpack
            top_values[p], rows[p], cols[p] = zip(*values_indices)

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
