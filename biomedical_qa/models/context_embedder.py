from tensorflow.python.ops.rnn_cell import RNNCell
from biomedical_qa.models.embedder import *


class ContextEmbedder(StackedEmbedder):

    def __init__(self, underlying_embedder, size, dropout=0.0,
                 devices=None, name="ContextEmbedder", reuse=False):
        self.unk_mask = None
        self.size = size
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._dropout = dropout

        StackedEmbedder.__init__(self, underlying_embedder, size, dropout, name, reuse)

    def _init(self):
        StackedEmbedder._init(self)
        with tf.device(self._device0):
            with tf.variable_scope("context_encoding"):
                self._embedded_context, self._output = self._context_encoder()

        self._train_variables = [p for p in tf.trainable_variables() if self.name in p.name]
        self._save_vars = list(self._train_variables)
        for v in self.underlying.save_variables:
            if v not in self._save_vars:
                self._save_vars.append(v)

    def _context_encoder(self):
        pass

    @property
    def output(self):
        return self._output

    @property
    def embedded_context(self):
        return self._embedded_context

    @property
    def train_variables(self):
        return self._train_variables

    @property
    def save_variables(self):
        return self._save_vars

    def run(self, sess, goal, batch):
        feed_dict = {
            self.inputs: batch[0],
            self.seq_lengths: batch[1]
        }

        return sess.run(goal, feed_dict=feed_dict)

    def get_config(self):
        config = super().get_config()
        config["size"] = self.size
        config["name"] = self.name
        return config


class RNNContextEmbedder(ContextEmbedder):

    def __init__(self, underlying_embedder, size, dropout=0.0, devices=None,
                 name="RNNContextEmbedder", composition="GRU", forward_only=False, reuse=False):
        self.composition = composition
        if composition == "GRU":
            self._cell = GRUCell(size)
        else:
            self._cell = BasicLSTMCell(size)

        self._forward_only = forward_only

        if not forward_only:
            with tf.variable_scope(name + "_util", reuse=reuse):
                self.forward_only = tf.get_variable("forward_only", dtype=tf.bool,
                                                    initializer=self._forward_only,
                                                    trainable=False)
                self.set_forward_only = tf.assign(self.forward_only, True)

        ContextEmbedder.__init__(self, underlying_embedder, size, dropout, devices, name, reuse)

    def _context_encoder(self):
        """
        Encodes all embedded inputs with bi-rnn, up to max(self._seq_lengths)
        :return: [B * T, S] context_encoded input
        """
        cell = self._cell
        seq_lengths = self.seq_lengths
        embedded = self.underlying.output
        if self._dropout > 0.0:
            embedded = tf.nn.dropout(embedded, self.keep_prob)
        with tf.device(self._device0):
            with tf.variable_scope("forward"):
                init_state_fw = cell.zero_state(self.batch_size, dtype=tf.float32)

                outs_fw_tmp = tf.nn.dynamic_rnn(cell, embedded, seq_lengths,
                                                initial_state=init_state_fw, time_major=False)[0]

                outs_fw = tf.slice(tf.concat(1, [tf.expand_dims(init_state_fw, 1), outs_fw_tmp]),
                                   [0, 0, 0], tf.shape(outs_fw_tmp))
                out_fw = tf.reshape(outs_fw, [-1, self.size])

                if self._forward_only:
                    context_encoded = tf.contrib.layers.fully_connected(out_fw, self.size, weights_initializer=None, activation_fn=None)

                    context_encoded = tf.reshape(context_encoded, tf.pack([-1, self.max_length, self.size])) + outs_fw
                    context_encoded.set_shape((None, None, self.size))

                    return context_encoded

        with tf.device(self._device1):
            # use other device for backward rnn
            with tf.variable_scope("backward"):
                init_state_bw = cell.zero_state(self.batch_size, dtype=tf.float32)

                rev_embedded = tf.reverse_sequence(embedded, seq_lengths, 1, 0)
                outs_bw_tmp = tf.nn.dynamic_rnn(cell, rev_embedded, seq_lengths, initial_state=init_state_bw, time_major=False)[0]

                outs_bw = tf.slice(tf.concat(1, [tf.expand_dims(init_state_bw, 1), outs_bw_tmp]),
                                   [0, 0, 0], tf.shape(outs_bw_tmp))

                outs_bw = tf.reverse_sequence(outs_bw, seq_lengths, 1, 0)
                out_bw = tf.reshape(outs_bw, [-1, self.size])

                out_bw = tf.cond(self.forward_only,
                                 lambda: tf.zeros_like(out_bw),
                                 lambda: out_bw)

            context_encoded = tf.contrib.layers.fully_connected(
                tf.concat(1, [out_fw, out_bw]), self.size,
                weights_initializer=None, activation_fn=None
            )

            context_encoded = tf.add_n([context_encoded, out_fw, out_bw])
            context_encoded = tf.reshape(context_encoded, tf.pack([-1, self.max_length, self.size]))
            context_encoded.set_shape((None, None, self.size))
        return context_encoded, tf.concat(2, [outs_fw_tmp, outs_bw_tmp])

    def clone(self, **kwargs):
        return RNNContextEmbedder(self.underlying.clone(),
                                  self.size,
                                  self._dropout,
                                  [self._device0, self._device1],
                                  self.name,
                                  self.composition,
                                  self._forward_only,
                                  reuse=True)

    def get_config(self):
        config = ContextEmbedder.get_config(self)
        config["composition"] = self.composition
        config["forward_only"] = self._forward_only
        config["type"] = "rnn_context"
        return config

    @staticmethod
    def create_from_config(config, devices=None, dropout=0.0,
                           underlying_embedder=None, reuse=False, **kwargs):
        # todo: load parameters of the model
        # todo: dump config dictionary as json
        from quebap.projects.autoread import model_from_config
        autoreader = RNNContextEmbedder(
            underlying_embedder if underlying_embedder else model_from_config(config["underlying"], devices, reuse=reuse),
            config["size"],
            dropout=dropout,
            composition=config.get("composition", "GRU"),
            devices=devices,
            name=config.get("name", "AutoReader"),
            forward_only=config.get("forward_only", False),
            reuse=reuse
        )

        return autoreader


class _AttentionTapeRNNCell(RNNCell):
    # Attention using dot-product over input-key and recurrent key to retrieve value

    def __init__(self, tape_length, size):
        self._tape_length = tape_length
        self._size = size

    def __call__(self, inputs, state, scope=None):
        # [B, L* S_k], [B, L* S_v], [B, S_k]
        (context_tape, word_tape, last_key) = state
        context_tape_reshaped = tf.reshape(context_tape, [-1, self._tape_length, self._size])
        word_tape_reshaped = tf.reshape(word_tape, [-1, self._tape_length, self._size])
        # [B, S]
        current_context = tf.slice(inputs, [0, 0], [-1, self._size])
        current_word = tf.slice(inputs, [0, self._size], [-1, -1])

        with tf.variable_scope("key_gate"):
            g = tf.contrib.layers.fully_connected(
                    tf.concat(1, [current_context, last_key]), self._size,
                    weights_initializer=None, activation_fn=tf.sigmoid,
                    biases_initializer=tf.constant_initializer(1.0, tf.float32)
                )
            new_key = g * current_context + (1-g) * last_key

        # attention - [B, L, 1]
        with tf.variable_scope("interaction"):
            #with tf.variable_scope("current_context"):
            #    inter_key = tf.contrib.layers.fully_connected(
            #                    new_key, self._key_size,
            #                    weights_initializer=None, activation_fn=None
            #                )
            #with tf.variable_scope("tape"):
            #    inter_tape = tf.contrib.layers.fully_connected(
            #                    context_tape, self._key_size,
            #                    weights_initializer=None, activation_fn=None,
            #                    biases_initializer=None
            #                )
            # [B, L, S_k]
            new_key_exp = tf.expand_dims(new_key, 1)
            inter = tf.concat(2, [context_tape_reshaped * new_key_exp,
                                  word_tape_reshaped * new_key_exp])

        with tf.variable_scope("attention"):
            # [B, L, 1]
            scores = tf.contrib.layers.fully_connected(inter, 1,
                            weights_initializer=tf.constant_initializer(1.0), activation_fn=None,
                            biases_initializer=None
                        )
            sharpen = tf.get_variable("sharpen", dtype=tf.float32, initializer=0.0)
            # [B, L]
            scores = tf.squeeze(scores, [2]) * tf.exp(sharpen)
            #tf.squeeze(tf.batch_matmul(context_tape_reshaped, tf.expand_dims(new_key, 2)), [2]) * tf.exp(sharpen)
            weights = tf.nn.softmax(scores)

        matched_word = tf.reduce_sum(tf.expand_dims(weights, 2) * word_tape_reshaped, [1])
        matched_context = tf.reduce_sum(tf.expand_dims(weights, 2) * context_tape_reshaped, [1])
        
        with tf.variable_scope("selection"):
            inter = tf.concat(1, [matched_context * new_key, matched_word * new_key])
            selection_gate = tf.contrib.layers.fully_connected(inter, self._size,
                                                     weights_initializer=None, activation_fn=tf.sigmoid)

            matched = selection_gate * matched_word + (1-selection_gate) * matched_context
            matched = tf.contrib.layers.fully_connected(new_key - matched, self._size,
                                                        weights_initializer=None,
                                                        activation_fn=None) + matched

        update_gate = tf.contrib.layers.fully_connected(matched * new_key, self._size, weights_initializer=None,
                                                        activation_fn=tf.sigmoid,
                                                        biases_initializer=tf.constant_initializer(-1.0))

        new_context = update_gate * matched + (1-update_gate) * current_context

        new_context_tape = tf.concat(1, [tf.slice(context_tape, [0, self._size], [-1, -1]), new_context])
        new_word_tape = tf.concat(1, [tf.slice(word_tape, [0, self._size], [-1, -1]), current_word])

        weighted_attention_weights = tf.reduce_mean(update_gate, [1], keep_dims=True) * weights
        return tf.concat(1, [new_context, weighted_attention_weights]), (new_context_tape, new_word_tape, new_key)

    @property
    def output_size(self):
        return self._size + self._tape_length

    @property
    def state_size(self):
        return (self._tape_length * self._size,  # key_tape
                self._tape_length * self._size,  # word_tape
                self._size) # last key

    def zero_state(self, batch_size, dtype):
        return super().zero_state(batch_size, dtype)


class AttentionMemoryContextEmbedder(ContextEmbedder):

    def __init__(self, underlying_embedder, tape_length, dropout=0.0, devices=None,
                 name="AttentionContextEmbedder", forward_only=False, reuse=False):
        self.tape_length = tape_length
        assert isinstance(underlying_embedder, ContextEmbedder), \
            "AttentionContextEmbedder needs context embedder as underlying embedder"
        #self._forward_only = forward_only

        #if not forward_only:
        #    with tf.variable_scope(name + "_util", reuse=reuse):
        #        self.forward_only = tf.get_variable("forward_only", dtype=tf.bool,
        #                                            initializer=self._forward_only,
        #                                            trainable=False)
        #        self.set_forward_only = tf.assign(self.forward_only, True)

        ContextEmbedder.__init__(self, underlying_embedder, underlying_embedder.size, dropout, devices, name, reuse)

    def _context_encoder(self):
        """
        Encodes all embedded inputs with bi-rnn, up to max(self._seq_lengths)
        :return: [B * T, S] encoded input
        """
        embedded_context = tf.nn.dropout(self.underlying.embedded_context, self.keep_prob)
        embedded_words = tf.nn.dropout(self.embedded_words, self.keep_prob)

        with tf.device(self._device1):
            with tf.variable_scope("attention"):
                inputs = tf.concat(2, [embedded_context, embedded_words])
                self.attention_cell = _AttentionTapeRNNCell(self.tape_length, self.size)
                retrieved = tf.nn.dynamic_rnn(self.attention_cell, inputs, self.seq_lengths, dtype=tf.float32)[0]
                output = tf.slice(retrieved, [0, 0, 0], [-1, -1, self.size])
                self.attention_weights = tf.slice(retrieved, [0, 0, self.size], [-1, -1, -1])

        return output

    def clone(self, **kwargs):
        return AttentionMemoryContextEmbedder(self.size,
                                              self.underlying.clone(),
                                              self._dropout,
                                              [self._device0, self._device1],
                                              self.name,
                                              reuse=True)

    def get_config(self):
        config = ContextEmbedder.get_config(self)
        config["tape_length"] = self.tape_length
        config["type"] = "attention_context"
        return config

    @staticmethod
    def create_from_config(config, devices=None, dropout=0.0,
                           underlying_embedder=None, reuse=False):
        # todo: load parameters of the model
        # todo: dump config dictionary as json
        from quebap.projects.autoread import model_from_config
        autoreader = AttentionMemoryContextEmbedder(
            underlying_embedder if underlying_embedder else model_from_config(config["underlying"], devices, reuse=reuse),
            config["tape_length"],
            devices=devices,
            name=config.get("name", "AutoReader"),
            forward_only=config.get("forward_only", False),
            reuse=reuse
        )

        return autoreader