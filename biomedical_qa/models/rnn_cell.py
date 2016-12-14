from tensorflow.python.ops.rnn_cell import RNNCell, GRUCell
import tensorflow as tf
import numpy as np

from biomedical_qa import tfutil


class ParamAssociativeMemory(RNNCell):

    def __init__(self, num_slots, size, input_size):
        """
        :param num_slots: number of slots
        :param size: size of slots
        :return:
        """
        self._num_slots = num_slots
        self._size = size
        self._keys = tf.get_variable("keys", shape=[self._num_slots * 5, self._num_slots+1],
                                     initializer=tf.random_normal_initializer(0.0, 0.1))
        self._proj = tf.get_variable("key_proj", shape=[input_size, self._num_slots * 5],
                                     initializer=tf.random_normal_initializer(0.0, 0.1))

    @property
    def state_size(self):
        if self._num_slots > 1:
            return (self._num_slots * self._size, self._num_slots+1)
        else:
            return self._num_slots * self._size

    @property
    def output_size(self):
        return self._size * self._num_slots + self._num_slots

    def __call__(self, inputs, prev_state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # [B, L, S]
            state = prev_state
            if self._num_slots > 1:
                state = prev_state[0]
            state = tf.reshape(state, [-1, self._num_slots, self._size])

            with tf.variable_scope("address"):
                lru_scores = prev_state[1]
                lru_gamma, beta = tf.split(1, 2,
                                           tf.contrib.layers.fully_connected(inputs, 2,
                                                              activation_fn=None,
                                                              weights_initializer=None,
                                                              scope="gamma_beta"))

                beta = tf.log(1+tf.exp(beta)) + 1

                address = tf.matmul(inputs, self._proj)
                scores = beta * tf.matmul(tf.nn.l2_normalize(address, 1), tf.nn.l2_normalize(self._keys, 1))

                weights = tf.nn.softmax(scores - lru_scores * tf.sigmoid(lru_gamma))
                weights = tf.slice(weights, [0, 0], [-1, self._num_slots])
                scaling = min(0.5, 0.05*self._num_slots)
                new_lru_scores = scaling * lru_scores + (1.0 - scaling) * scores

                weights_exp = tf.expand_dims(weights, 2)

            with tf.variable_scope("read"):
                read = tf.reduce_sum(state * weights_exp, reduction_indices=[1])

            with tf.variable_scope("write"):
                to_write = tf.contrib.layers.fully_connected(inputs, self._size,
                                                             activation_fn=None,
                                                             weights_initializer=None,
                                                             scope="to_write")
                to_write_exp = tf.expand_dims(to_write, 1)
                to_write_tiled = tf.tile(to_write_exp, [1, self._num_slots, 1])

                # [B, 1]
                e = tf.contrib.layers.fully_connected(tf.concat(1, [inputs, read]), 1,
                                                      activation_fn=tf.sigmoid,
                                                      weights_initializer=None,
                                                      scope="erase")
                # [B, N, S]
                e = tf.expand_dims(e, 1)

                new_state = tf.reshape(weights_exp * to_write_tiled + (1-weights_exp * e) * state,
                                       [-1, self._num_slots * self._size])

            return tf.concat(1, [new_state, weights]), (new_state, new_lru_scores)


class MultiConcatRNNCell(RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells, state_is_tuple=False):
        """Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  By default (False), the states are all
            concatenated along the column axis.

        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        self._cells = cells
        self._state_is_tuple = state_is_tuple
        if not state_is_tuple:
            if any(tf.nn.nest.is_sequence(c.state_size) for c in self._cells):
                raise ValueError("Some cells return tuples of states, but the flag "
                                 "state_is_tuple is not set.  State sizes are: %s"
                                 % str([c.state_size for c in self._cells]))

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return sum(c.output_size for c in self._cells)

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            outs = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("Cell%d" % i):
                    if self._state_is_tuple:
                        if not tf.nn.nest.is_sequence(state):
                            raise ValueError(
                                "Expected state to be a tuple of length %d, but received: %s"
                                % (len(self.state_size), state))
                        cur_state = state[i]
                    else:
                        cur_state = tf.slice(
                            state, [0, cur_state_pos], [-1, cell.state_size])
                        cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
                    outs.append(cur_inp)
        new_states = (tuple(new_states) if self._state_is_tuple
                      else tf.concat(1, new_states))
        return tf.concat(1, outs), new_states


class MultiMemoryRNN(RNNCell):

    def __init__(self, memories, size):
        self._rnn_memories = memories
        self._cell = GRUCell(size)
        self._size = size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            mem_states = []
            cell_state = tf.slice(state, [0, 0], [-1, self._cell.state_size])
            offset = self._cell.state_size
            for m in self._rnn_memories:
                mem_states.append(tf.slice(state, [0, offset], [-1, m.state_size]))
                offset += m.state_size

            cell_output, _ = self._cell(inputs, cell_state)

            # read from memories
            mem_input = tf.concat(1, [cell_output, inputs])
            mem_out_states = [m(mem_input, s, "memory"+str(i)) for i, m, s in zip(range(len(self._rnn_memories)), self._rnn_memories, mem_states)]
            # [B, N+1, S]
            output= tf.concat(1, [tf.expand_dims(m[0], 1) for m in mem_out_states]+ [tf.expand_dims(cell_output, 1)])
            # [B, N+1]
            gates = tf.contrib.layers.fully_connected(tf.reshape(output, [-1,
                                                                          (len(self._rnn_memories)+1) * self._size]),
                                                      len(self._rnn_memories)+1, activation_fn=tf.sigmoid,
                                                      weights_initializer=None,
                                                      biases_initializer=tf.constant_initializer(0.0))
            # [B, N+1, S]
            output = output * tf.expand_dims(gates, 2)
            output = tf.reduce_sum(output, [1])

            #new_input = tf.contrib.layers.fully_connected(read, self._size, activation_fn=tf.tanh, weights_initializer=None)

            new_mem_states = [out_state[1] for out_state in mem_out_states]
            new_mem_states = tf.concat(1, [cell_output] + new_mem_states)
            return output, new_mem_states #tf.concat(1, [output, new_mem_states])

    def zero_state(self, batch_size, dtype):
        return tf.concat(1, [self._cell.zero_state(batch_size, dtype)] +
                         [m.zero_state(batch_size, dtype) for m in self._rnn_memories])

    @property
    def state_size(self):
        return self._cell.state_size + sum(m.state_size for m in self._rnn_memories)

    @property
    def output_size(self):
        return self._size


class ParamNTM(RNNCell):

    def __init__(self, num_slots, size, input_size, ctr_cell, weights_given=False):
        """
        :param num_slots: number of slots
        :param size: size of slots
        :return:
        """
        self._num_slots = num_slots
        self._size = size
        if not weights_given:
            self._keys = tf.get_variable("keys", shape=[self._num_slots * 5, self._num_slots+1],
                                         initializer=tf.random_normal_initializer(0.0, 0.1))
            self._proj = tf.get_variable("key_proj", shape=[input_size+self._size, self._num_slots * 5],
                                         initializer=tf.random_normal_initializer(0.0, 0.1))
        self._weights_given = weights_given
        self._ctr_cell = ctr_cell

    @property
    def state_size(self):
        if not self._weights_given:
            return (self._ctr_cell.state_size, self._num_slots * self._size,
                    self._ctr_cell.output_size, self._num_slots+1)
        else:
            return (self._ctr_cell.state_size, self._num_slots * self._size)

    @property
    def output_size(self):
        if not self._weights_given:
            return self._size * self._num_slots + self._size + self._num_slots
        else:
            return self._size * self._num_slots + self._size

    def __call__(self, inputs, prev_state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # [B, L, S]
            if not self._weights_given:
                prev_ctr_state, memory, prev_ctr_out, lru_scores = prev_state
            else:
                prev_ctr_state, memory = prev_state
            memory = tf.reshape(memory, [-1, self._num_slots, self._size])

            if not self._weights_given:
                conc_input = tf.concat(1, [inputs, prev_ctr_out])

                with tf.variable_scope("address"):
                    lru_gamma, beta = tf.split(1, 2,
                                               tf.contrib.layers.fully_connected(conc_input, 2,
                                                                  activation_fn=None,
                                                                  weights_initializer=None,
                                                                  scope="gamma_beta"))

                    beta = tf.log(1+tf.exp(beta)) + 1

                    address = tf.matmul(conc_input, self._proj)
                    scores = beta * tf.matmul(tf.nn.l2_normalize(address, 1), tf.nn.l2_normalize(self._keys, 0))

                    weights = tf.nn.softmax(scores - lru_scores * tf.sigmoid(lru_gamma))
                    weights = tf.slice(weights, [0, 0], [-1, self._num_slots])
                    scaling = min(0.5, 0.05*self._num_slots)
                    new_lru_scores = scaling * lru_scores + (1.0 - scaling) * scores
            else:
                weights = tf.slice(inputs, [0, 0], [-1, self._num_slots])
                inputs = tf.slice(inputs, [0, self._num_slots], [-1, -1])

            weights_exp = tf.expand_dims(weights, 2)

            with tf.variable_scope("read"):
                read = tf.reduce_sum(memory * weights_exp, reduction_indices=[1])

            new_out, ctr_state = self._ctr_cell(tf.concat(1, [read, inputs]), prev_ctr_state)

            with tf.variable_scope("write"):
                to_write = tf.contrib.layers.fully_connected(tf.concat(1, [inputs, new_out]),
                                                             self._size, activation_fn=None,
                                                             weights_initializer=None,
                                                             scope="to_write")
                to_write_exp = tf.expand_dims(to_write, 1)
                to_write_tiled = tf.tile(to_write_exp, [1, self._num_slots, 1])

                # [B, 1]
                e = tf.contrib.layers.fully_connected(tf.concat(1, [to_write, read]), 1,
                                                      activation_fn=tf.sigmoid,
                                                      weights_initializer=None,
                                                      scope="erase")
                # [B, N, S]
                e = tf.expand_dims(e, 1)

                new_memory = tf.reshape(weights_exp * to_write_tiled + (1-weights_exp * e) * memory,
                                       [-1, self._num_slots * self._size])
            if not self._weights_given:
                return tf.concat(1, [new_out, new_memory, weights]), (ctr_state, new_memory, new_out, new_lru_scores)
            else:
                return tf.concat(1, [new_out, new_memory]), (ctr_state, new_memory)


class BackwardNTM(RNNCell):

    def __init__(self, num_slots, size):
        """
        :param num_slots: number of slots
        :param size: size of slots
        :return:
        """
        self._num_slots = num_slots
        self._size = size

    @property
    def state_size(self):
        return [self._size] * self._num_slots

    @property
    def output_size(self):
        return self._size * self._num_slots + self._size

    def __call__(self, inputs, prev_state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # [B, L, S]
            next_states = prev_state

            current_state = inputs
            current_states = tf.split(1, self._num_slots + 1, current_state)

            # interpolate between current next_states and nex_state
            ctr_out = current_states[0]

            s = [ctr_out]
            for s_c, s_n in zip(current_states[1:], next_states):
                g = tf.contrib.layers.fully_connected(tf.concat(1, [s_c, s_n, ctr_out]),
                                                      self._size,
                                                      activation_fn=tf.sigmoid,
                                                      weights_initializer=None)
                s.append(g * s_c + (1 - g) * s_n)

            return tf.concat(1, s), s[1:]


class DynamicPointerRNN(RNNCell):
    """
    """

    def __init__(self, size, pool_size, ctr_cell, input_states, lengths,
                 num_layers):
        self._ctr_cell = ctr_cell
        self._size = size
        self._input_states = input_states
        self._max_length = tf.reduce_max(lengths)
        self._max_length_32 = tf.cast(self._max_length, tf.int32)
        self._lengths = lengths
        self._pool_size = pool_size
        self._num_layers = num_layers

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return (None, None)

    def __call__(self, u, prev_ctr_state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            ctr_out, ctr_state = self._ctr_cell(u, prev_ctr_state)

            with tf.variable_scope("start"):
                next_start_scores = _highway_maxout_network(
                    self._num_layers, self._pool_size, tf.concat(1, [u, ctr_out]),
                    self._input_states, self._lengths, self._max_length_32,
                    self._size)

            with tf.variable_scope("end"):
                next_end_scores = _highway_maxout_network(
                    self._num_layers, self._pool_size, tf.concat(1, [u, ctr_out]),
                    self._input_states, self._lengths, self._max_length_32,
                    self._size)

        return (next_start_scores, next_end_scores), ctr_state


def _highway_maxout_network(num_layers, pool_size, inputs, states, lengths, max_length, size):
    r = tf.contrib.layers.fully_connected(inputs, size, activation_fn=tf.tanh, weights_initializer=None, scope="r")

    r_tiled = tf.tile(tf.expand_dims(r, 1), tf.pack([1, max_length, 1]))

    ms = []
    hm_inputs = tf.concat(2, [states, r_tiled])
    hm_inputs.set_shape([None, None, size + states.get_shape()[-1].value])
    for i in range(num_layers):
        m = tf.contrib.layers.fully_connected(hm_inputs,
                                              size * pool_size,
                                              activation_fn=None,
                                              weights_initializer=None,
                                              scope="m_%d" % i)

        m = tf.reshape(m, tf.pack([-1, max_length, size, pool_size]))
        m = tf.reduce_max(m, [3])
        hm_inputs = m
        ms.append(m)

    if num_layers <= 0:
        out = tf.contrib.layers.fully_connected(hm_inputs, pool_size,
                                                activation_fn=None,
                                                weights_initializer=None,
                                                scope="out")
    else:
        out = tf.contrib.layers.fully_connected(tf.concat(2, ms), pool_size,
                                                activation_fn=None,
                                                weights_initializer=None,
                                                scope="out")
    # [B, L]
    out = tf.reduce_max(out, [2])
    out = out + tfutil.mask_for_lengths(lengths, max_length=tf.shape(states)[1])

    return out
