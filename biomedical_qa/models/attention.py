import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from biomedical_qa import tfutil


def bilinear_attention(att_states, att_lengths, queries, query_lengths, size, batch_size=None):
    attention_key = tf.contrib.layers.fully_connected(att_states, size,
                                                             activation_fn=None,
                                                             weights_initializer=None)
    # [B, Q, L] --  Q is length of query
    attention_scores = tf.matmul(queries, attention_key, adj_y=True)
    max_length = tf.cast(tf.reduce_max(query_lengths), tf.int32)
    max_query_length = tf.cast(tf.reduce_max(att_lengths), tf.int32)
    mask = tfutil.mask_for_lengths(att_lengths, batch_size, max_length=max_query_length)
    mask = tf.tile(tf.expand_dims(mask, 1), tf.stack([1, max_length, 1]))
    attention_scores = attention_scores + mask
    attention_scores_reshaped = tf.reshape(attention_scores, tf.stack([-1, max_query_length]))
    attention_weights = tf.reshape(tf.nn.softmax(attention_scores_reshaped), tf.shape(attention_scores))
    # [B, Q, L] x [B, L, S] --> [B, L, S]
    ctxt_aligned_att_states = tf.matmul(attention_weights, att_states)
    return ctxt_aligned_att_states


def dot_co_attention(states1, lengths1, states2, lengths2, batch_size=None):
    # [B, L1, L2]
    attention_scores = tf.matmul(states1, states2, adj_y=True)
    return extract_co_attention_states(attention_scores, states1, lengths1, states2, lengths2, batch_size)


def extract_co_attention_states(affinity_scores, states1, lengths1, states2, lengths2, batch_size=None):
    max_length2 = tf.cast(tf.reduce_max(lengths2), tf.int32)
    max_length1 = tf.cast(tf.reduce_max(lengths1), tf.int32)

    # [B, L1]
    mask1 = tfutil.mask_for_lengths(lengths1, batch_size, max_length=max_length1)
    # [B, L2, L1]
    mask1 = tf.tile(tf.expand_dims(mask1, 1), tf.stack([1, max_length2, 1]))

    # [B, L2]
    mask2 = tfutil.mask_for_lengths(lengths2, batch_size, max_length=max_length2)
    # [B, L1, L2]
    mask2 = tf.tile(tf.expand_dims(mask2, 1), tf.stack([1, max_length1, 1]))
    # [B, L1, L2]
    attention_scores1 = affinity_scores + mask2
    # [B, L2, L1]
    attention_scores2 = tf.transpose(affinity_scores, [0,2,1]) + mask1

    # [B, L1, L2]
    attention_weights1 = _my_softmax(attention_scores1)
    # [B, L2, L1]
    attention_weights2 = _my_softmax(attention_scores2)

    # [B, L2, L1] x [B, L1, S] --> [B, L2, S]
    att_states2 = tf.matmul(attention_weights2, states1)

    # [B, L2, 2*S]
    new_states2 = tf.concat(axis=2, values=[att_states2, states2])

    # [B, L1, 2*S]
    att_states1 = tf.matmul(attention_weights1, new_states2)

    # [B, L1, 3*S]
    new_states1 = tf.concat(axis=2, values=[att_states1, states1])

    return new_states1

def _my_softmax(inputs):
    #softmax in last dimension
    dims = len(inputs.get_shape())-1
    max_v = tf.reduce_max(inputs, [dims], keep_dims=True)
    inputs = inputs - max_v
    exp_inputs = tf.exp(inputs)
    z = tf.reduce_sum(exp_inputs, dims, keep_dims=True)
    return exp_inputs / z


def attention(att_states, att_lengths, queries, query_lengths, size, batch_size=None):
    # [B, L, S]
    inter_states = tf.contrib.layers.fully_connected(att_states, size,
                                                     activation_fn=None,
                                                     weights_initializer=None,
                                                     scope="inter_states")
    # [B, Q, S]
    inter_queries = tf.contrib.layers.fully_connected(queries, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      scope="inter_queries")

    # [B, L, Q, S] --  Inter
    inter = tf.tanh(tf.expand_dims(inter_states, 2) + tf.expand_dims(inter_queries, 1))

    # [B, L, Q, 1]
    attention_scores = tf.contrib.layers.fully_connected(inter, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         scope="attention_scores")

    attention_scores = tf.squeeze(attention_scores, [3])

    max_length = tf.cast(tf.reduce_max(query_lengths), tf.int32)
    max_question_length = tf.cast(tf.reduce_max(att_lengths), tf.int32)
    mask = tfutil.mask_for_lengths(att_lengths, batch_size, max_length=max_question_length)
    mask = tf.tile(tf.expand_dims(mask, 1), tf.stack([1, max_length, 1]))
    attention_scores = attention_scores + mask
    attention_scores_reshaped = tf.reshape(attention_scores, tf.stack([-1, max_question_length]))
    attention_weights = tf.reshape(tf.nn.softmax(attention_scores_reshaped), tf.shape(attention_scores))
    # [B, L, Q] x [B, Q, S] --> [B, L, S]
    ctxt_aligned_att_states = tf.matmul(attention_weights, att_states)
    return ctxt_aligned_att_states


def conditional_attention(att_states, att_lengths, queries, query_lengths, ctr_cell, bidirectional=True):
    with tf.variable_scope("conditional_attention"):
        if bidirectional:
            cell_fw = ControllerWrapper(ctr_cell, AttentionCell(att_states, att_lengths, num_units=ctr_cell.output_size))
            cell_bw = ControllerWrapper(ctr_cell, AttentionCell(att_states, att_lengths, num_units=ctr_cell.output_size))
            #ctxt_aligned_fw_att_states = tf.nn.dynamic_rnn(cell_fw, queries, query_lengths, dtype=tf.float32)[0]
            #tf.get_variable_scope().reuse_variables()
            #ctxt_aligned_bw_att_states = tf.nn.dynamic_rnn(cell_bw, queries, query_lengths, dtype=tf.float32)[0]
            #ctxt_aligned_att_states = tf.concat(2, [ctxt_aligned_fw_att_states, ctxt_aligned_bw_att_states])
            ctxt_aligned_att_states = tf.concat(axis=2, values=tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, queries, query_lengths, dtype=tf.float32)[0])
        else:
            cell = ControllerWrapper(ctr_cell, BilinearAttentionCell(att_states, att_lengths, num_units=ctr_cell.output_size))
            ctxt_aligned_att_states = tf.nn.dynamic_rnn(cell, queries, query_lengths, dtype=tf.float32)[0]

    return ctxt_aligned_att_states


class ControllerWrapper(RNNCell):
    """
    A ControllerWrapper wraps a controller RNNCell and controlled RNNCell. The controller cell pre-processes
    the current input and latest output of the controlled cell and feeds its output to the controlled cell.
    Like this Attention can be realized using the AttentionCell as controlled cell and another RNNCell as
    controller cell. The output of this wrapper is the concatenation of the controller cell output and the
    controlled cell output. This output can also be projected via a output projection function.
    """

    def __init__(self, controller_cell, cell, controller_scope=None, controller_first=False):
        """
        :param controller_cell: controller that controls the input to the cell
        :param cell: controlled cell, typically some form of memory
        :return:
        """
        self._cell = cell
        self._controller_cell = controller_cell
        self._controller_scope = controller_scope
        self._controller_first = controller_first

    @property
    def output_size(self):
        if self._controller_first:
            return self._cell.output_size + self._controller_cell.output_size
        else:
            return self._controller_cell.output_size

    def __call__(self, inputs, state, scope=None):
        prev_ctr_state, prev_crtld_state, prev_out = None, None, None
        if self._cell.state_size > 0:
            prev_ctr_state, prev_crtld_state, prev_out = state
        else:
            prev_ctr_state, prev_out = state

        if self._controller_first:
            prev_ctrld_out = tf.slice(prev_out, [0, self._controller_cell.output_size], [-1, -1])
            ctr_out, ctr_state = self._controller_cell(tf.concat(axis=1, values=[inputs, prev_ctrld_out]), prev_ctr_state,
                                                       self._controller_scope)
            out, crtld_state = self._cell(tf.concat(axis=1, values=[ctr_out, inputs]), prev_crtld_state)

            out = tf.concat(axis=1, values=[ctr_out, out])
        else:
            ctrld_out, crtld_state = self._cell(tf.concat(axis=1, values=[prev_out, inputs]), prev_crtld_state)
            out, ctr_state = self._controller_cell(tf.concat(axis=1, values=[inputs, ctrld_out]), prev_ctr_state, self._controller_scope)

        if self._cell.state_size > 0:
            return out, (ctr_state, crtld_state, out)
        else:
            return out, (ctr_state, out)

    @property
    def state_size(self):
        if self._cell.state_size > 0:
            return (self._controller_cell.state_size, self._cell.state_size, self.output_size)
        else:
            return (self._controller_cell.state_size, self.output_size)


class NoInputControllerWrapper(RNNCell):
    """
    A ControllerWrapper wraps a controller RNNCell and controlled RNNCell. The controller cell pre-processes
    the current input and latest output of the controlled cell and feeds its output to the controlled cell.
    Like this Attention can be realized using the AttentionCell as controlled cell and another RNNCell as
    controller cell. The output of this wrapper is the concatenation of the controller cell output and the
    controlled cell output. This output can also be projected via a output projection function.
    """

    def __init__(self, controller_cell, cell):
        """
        :param controller_cell: controller that controls the input to the cell
        :param cell: controlled cell, typically some form of memory
        :return:
        """
        self._cell = cell
        self._controller_cell = controller_cell

    @property
    def output_size(self):
        return self._controller_cell.output_size

    def __call__(self, inputs, state, scope=None):
        prev_ctr_state, prev_crtld_state, prev_ctr_out = None, None, None
        if self._cell.state_size > 0:
            prev_ctr_state, prev_crtld_state, prev_ctr_out = state
        else:
            prev_ctr_state, prev_ctr_out = state

        ctrld_out, crtld_state = self._cell(prev_ctr_out, prev_crtld_state)
        if isinstance(prev_ctr_state, tuple):
            to_tile = tf.div(tf.shape(ctrld_out)[0], tf.shape(prev_ctr_state[0])[0])
            prev_ctr_state = (tf.reshape(tf.tile(s, tf.stack([1, to_tile])),
                                        [-1, s.get_shape()[1].value]) for s in prev_ctr_state)
        else:
            to_tile = tf.div(tf.shape(ctrld_out)[0], tf.shape(prev_ctr_state[0]))
            prev_ctr_state = tf.reshape(tf.tile(prev_ctr_state, tf.stack([1, to_tile])),
                                        [-1, prev_ctr_state.get_shape()[1].value])
        out, ctr_state = self._controller_cell(ctrld_out, prev_ctr_state)

        if self._cell.state_size > 0:
            return out, (ctr_state, crtld_state, out)
        else:
            return out, (ctr_state, out)

    @property
    def state_size(self):
        if self._cell.state_size > 0:
            return (self._controller_cell.state_size, self._cell.state_size, self._controller_cell.output_size)
        else:
            return (self._controller_cell.state_size, self._controller_cell.output_size)


class AttentionCell(RNNCell):
    """
    This RNNCell only makes sense in conjunction with ControllerWrapper
    """

    def __init__(self, attention_states, attention_length, num_heads=1,
                 num_units=None, k=1, reuse=False):
        """
        :param attention_states: [B, L, S]-tensor, B-batch_size, L-attention-length, S-attention-size
        :param attention_length: [B]-tensor, with batch-specific sequence-lengths to attend over
        :param num_heads: number of attention read heads
        :return:
        """
        self._attention_states = attention_states
        self._attention_length = attention_length
        self._num_heads = num_heads
        self._hidden_features = None
        self._num_units = self._attention_states.get_shape()[2].value if num_units is None else num_units
        self.attention_scores = [list() for _ in range(self._num_heads)]
        self.attention_weights = [list() for _ in range(self._num_heads)]
        self._reuse = reuse

    @property
    def output_size(self):
        return self._attention_states.get_shape()[2].value * self._num_heads

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "attention_cell"):
            if self._reuse:
                tf.get_variable_scope().reuse_variables()
            if self._hidden_features is None:
                # [B, L, S]
                attention_states = self._attention_states
                self._hidden_features = []

                for a in range(self._num_heads):
                    # [B, L, S]
                    self._hidden_features.append(tf.contrib.layers.fully_connected(attention_states, self._num_units,
                                                                                   activation_fn=None,
                                                                                   weights_initializer=None))

            ds = []  # Results of attention reads will be stored here.

            batch_size = tf.shape(inputs)[0]
            mask = tfutil.mask_for_lengths(self._attention_length, batch_size)

            # some parts are copied from tensorflow attention code-base
            for a in range(self._num_heads):
                with tf.variable_scope("Attention_%d" % a):
                    with tf.variable_scope("features%d" % a):
                        # [B, S]
                        y = tf.contrib.layers.fully_connected(inputs, self._num_units, activation_fn=None,
                                                              weights_initializer=None)
                        y = tf.tanh(self._hidden_features[a] + tf.expand_dims(y, 1))
                    with tf.variable_scope("scores%d" % a):
                        # [B, L, 1]
                        s = tf.contrib.layers.fully_connected(y, 1, activation_fn=None,
                                                              weights_initializer=None)
                    s = tf.squeeze(s, [2])
                    self.attention_scores[a].append(s)
                    # [B, L]
                    weights = tf.nn.softmax(s + mask)
                    # Now calculate the attention-weighted vector d.
                    self.attention_weights[a].append(weights)

                    d = tf.reduce_sum(tf.expand_dims(weights, 2) * self._attention_states, [1])
                    ds.append(d)

            if len(ds) > 1:
                return tf.concat(axis=1, values=ds), None
            else:
                return ds[0], None

    @property
    def state_size(self):
        return 0


class BilinearAttentionCell(RNNCell):
    """
    This RNNCell only makes sense in conjunction with ControllerWrapper
    """

    def __init__(self, attention_states, attention_length, num_heads=1, k=1):
        """
        :param attention_states: [B, L, S]-tensor, B-batch_size, L-attention-length, S-attention-size
        :param attention_length: [B]-tensor, with batch-specific sequence-lengths to attend over
        :param num_heads: number of attention read heads
        :return:
        """
        self._attention_states = attention_states
        self._attention_length = attention_length
        self._num_heads = num_heads
        self._hidden_features = None
        self.attention_scores = [list() for _ in range(self._num_heads)]
        self.attention_weights = [list() for _ in range(self._num_heads)]
        self._k = k

    @property
    def output_size(self):
        return self._attention_states.get_shape()[2].value * self._num_heads

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "bilinear_attention_cell"):
            if self._hidden_features is None:
                # [B, L, S]
                attention_states = self._attention_states
                self._hidden_features = []

                for a in range(self._num_heads):
                    # [B, L, S]
                    self._hidden_features.append(tf.contrib.layers.fully_connected(attention_states, inputs.get_shape()[-1].value,
                                                                                   activation_fn=None,
                                                                                   weights_initializer=None,
                                                                                   biases_initializer=None))

                if attention_states.get_shape()[-1].value == inputs.get_shape()[-1].value:
                    self._hidden_features[0] = self._hidden_features[0] + attention_states

                self.eval = tf.get_variable("attention_is_eval", dtype=tf.bool, initializer=False, trainable=False)
                self.set_eval = tf.assign(self.eval, True)

            ds = []  # Results of attention reads will be stored here.

            batch_size = tf.shape(inputs)[0]
            mask = tfutil.mask_for_lengths(self._attention_length, batch_size)

            # some parts are copied from tensorflow attention code-base
            for a in range(self._num_heads):
                with tf.variable_scope("Attention_%d" % a):
                    # [B, S]
                    query = inputs
                    # [B, L, 1]
                    s = tf.matmul(self._hidden_features[a], tf.expand_dims(query, 2))
                    s = tf.squeeze(s, [2])
                    self.attention_scores[a].append(s)
                    # [B, L]
                    weights = tf.nn.softmax(s + mask)
                    # Now calculate the attention-weighted vector d.
                    self.attention_weights[a].append(weights)

                    d = tf.reduce_sum(tf.expand_dims(weights, 2) * self._attention_states, [1])
                    ds.append(d)

            if len(ds) > 1:
                return tf.concat(axis=1, values=ds), None
            else:
                return ds[0], None

    @property
    def state_size(self):
        return 0
