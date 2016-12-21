import tensorflow as tf

class BeamSearchDecoder(object):


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

