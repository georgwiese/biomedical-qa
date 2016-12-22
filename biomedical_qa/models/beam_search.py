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
    """From a QASetting batch, computes most likely (start, end) pairs vie beam search."""


    def __init__(self, sess, model, beam_size):

        self._sess = sess
        self._model = model
        self._beam_size = beam_size

        assert self._beam_size == 1, "Current Implementation supports only beam size 1"


    def decode(self, qa_settings):
        """For each QASetting, finds most likely (start, end) pairs via beam search.

        :param qa_settings: List fo QASetting objects
        :return: List of BeamSearchDecoderResult objects
        """
        # TODO: Implement for beam_size > 1

        self._model.set_eval(self._sess)

        # Get start probs, context partition and all relevant intermediate results
        # to predict end pointer later.
        context_partition, matched_output, question_representation, start_probs = self._sess.run(
                [self._model.context_partition,
                 self._model.matched_output,
                 self._model.question_representation,
                 self._model.start_probs],
                self._model.get_feed_dict(qa_settings))

        # Compute top starts their row indices
        num_partitions = context_partition[-1] + 1
        assert num_partitions == len(qa_settings)
        top_start_indices_per_context = np.argmax(start_probs, axis=1)

        top_start_probs_per_partition = - np.ones([num_partitions], dtype=np.float32)
        top_start_row_indices_per_partition = np.zeros([num_partitions], dtype=np.int64)
        top_start_col_indices_per_partition = np.zeros([num_partitions], dtype=np.int64)

        for row, col in enumerate(top_start_indices_per_context):
            partition = context_partition[row]
            if start_probs[row, col] > top_start_probs_per_partition[partition]:
                top_start_probs_per_partition = start_probs[row, col]
                top_start_row_indices_per_partition = row
                top_start_col_indices_per_partition = col

        # Compute end probs by feeding all necessary itermediate results & start pointers
        end_probs = self._sess.run(
                [self._model.end_probs],
                {
                    # Starts
                    self._model.correct_start_pointer: top_start_col_indices_per_partition,
                    self._model.answer_context_indices: top_start_row_indices_per_partition,
                    # Intermediate Results (so no recomputation needed)
                    self._model.matched_output: matched_output,
                    self._model.question_representation: question_representation,
                })

        row_indices_per_question = [0] + np.cumsum([len(s.contexts) for s in qa_settings])[:-1]
        context_indices = [row_index - start_row_index
                           for row_index, start_row_index
                           in zip(top_start_row_indices_per_partition,
                                  row_indices_per_question)]
        ends = np.argmax(end_probs, axis=1)
        starts = top_start_col_indices_per_partition
        assert len(ends) == num_partitions

        return [BeamSearchDecoderResult(context_indices=[context_indices[i]],
                                        starts=[starts[i]],
                                        ends=[ends[i]],
                                        probs=[top_start_probs_per_partition[i] * end_probs[i]])
                for i in range(num_partitions)]
