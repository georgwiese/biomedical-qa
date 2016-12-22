import tensorflow as tf
import numpy as np


class BeamSearchDecoderResult(object):

    def __init__(self, starts, ends, probs):

        self.starts = starts
        self.ends = ends
        self.probs = probs


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
        num_partitions = context_partition[-1]
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

        ends = np.argmax(end_probs, axis=1)
        starts = zip(top_start_row_indices_per_partition,
                     top_start_col_indices_per_partition)
        assert len(ends) == num_partitions

        return [BeamSearchDecoderResult(starts=[starts[i]],
                                        ends=[ends[i]],
                                        probs=[top_start_probs_per_partition[i] * end_probs[i]])
                for i in range(num_partitions)]



class TfBeamSearchDecoder(object):


    def __init__(self, beam_size, answer_context_indices):

        self._beam_size = beam_size
        self._answer_context_indices = answer_context_indices


    def receive_start_scores(self, start_scores):

        self._start_scores = tf.gather(start_scores, self._answer_context_indices)

        start_probs = tf.nn.softmax(start_scores)
        start_probs = tf.gather(start_probs, self._answer_context_indices)
        top_start_probs, top_starts = tf.nn.top_k(start_probs, self._beam_size)

        self._top_start_probs = tf.reshape(top_start_probs, [-1])
        self._top_starts = tf.cast(tf.reshape(top_starts, [-1]), tf.int64)

        return self._top_starts


    def receive_end_scores(self, end_scores):

        # Get [*, beam_size] Tensor of top ends for each start
        end_probs = tf.nn.softmax(end_scores)
        top_end_probs, top_ends = tf.nn.top_k(end_probs, self._beam_size)

        # Get [*, beam_size * beam_size] Tensor of total probs for all start/end combinations
        squared_beam_size = self._beam_size * self._beam_size
        top_end_probs = tf.reshape(top_end_probs, [-1, squared_beam_size])
        top_ends = tf.cast(tf.reshape(top_ends, [-1, squared_beam_size]), tf.int64)
        top_start_probs = tf.reshape(self.expand_batch(self._top_start_probs),
                                     [-1, squared_beam_size])
        top_starts = tf.reshape(self.expand_batch(self._top_starts),
                                [-1, squared_beam_size])
        top_total_probs = top_start_probs * top_end_probs

        # Reduce all Tensors to [*, beam_size] tensors according to top_total_probs
        top_total_probs, top_indices = tf.nn.top_k(top_total_probs, self._beam_size, sorted=True)
        _top_indices = self.gather_rowwise_indices(top_indices)

        end_scores = tf.reshape(self.expand_batch(end_scores),
                                [-1, squared_beam_size, tf.shape(end_scores)[1]])

        self._top_starts = tf.gather_nd(top_starts, _top_indices)
        self._top_ends = tf.gather_nd(top_ends, _top_indices)
        self._end_scores = tf.gather_nd(end_scores, _top_indices)
        self._top_total_probs = top_total_probs


    def get_top_spans(self):
        """Returns the top <beam size> starts with their most likely end."""

        return self._top_starts, self._top_ends, self._top_total_probs


    def get_final_prediction(self):

        return self._start_scores, self._end_scores[:, 0, :], \
                self._top_starts[:, 0], self._top_ends[:, 0]


    def get_beam_search_partition(self, partitions):
        """Compute partition indices, which each index repeated <beam_size> times."""

        indices = tf.range(partitions)
        indices = tf.reshape(indices, [-1, 1])
        indices = tf.tile(indices, tf.pack([1, self._beam_size]))
        indices = tf.reshape(indices, [-1])

        return indices


    def expand_batch(self, tensor):

        partitions = tf.shape(tensor)[0]
        indices = self.get_beam_search_partition(partitions)
        return tf.gather(tensor, indices)


    def gather_rowwise_indices(self, indices):
        """Transforms 2D indices tensor to _indices such that:
            tf.gather_nd(some_2d_tensor, _indices)
            is equivalent to:
                rows = [tf.gather(some_2d_tensor[i], indices[i]) for i in range(n_rows)]
                result = tf.pack(rows)
        """

        rows = tf.shape(indices)[0]
        cols = tf.shape(indices)[1]

        # Compute [rows, cols, 2] indices tensor of [row_index, col_index] entries
        row_index = tf.reshape(tf.range(rows), tf.pack([rows, 1]))
        row_index_tiled = tf.tile(row_index, tf.pack([1, cols]))
        _indices = tf.pack([row_index_tiled, indices])
        _indices = tf.transpose(_indices, [1, 2, 0])

        return _indices

