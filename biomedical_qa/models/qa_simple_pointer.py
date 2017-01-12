import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUBlockCell, LSTMBlockCell, LSTMBlockFusedCell, FusedRNNCellAdaptor
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import BasicRNNCell

from biomedical_qa import tfutil
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.models.rnn_cell import LayerNormGRUCell, LayerNormLSTMCell, GatedAggregationRNNCell


class QASimplePointerModel(ExtractionQAModel):

    def __init__(self, size, embedder, keep_prob=1.0, composition="LSTM", devices=None, name="QASimplePointerModel",
                 with_features=True, num_intrafusion_layers=1, with_inter_fusion=True, layer_norm=False,
                 start_output_unit="softmax"):
        self._composition = composition
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1%len(devices)] if devices is not None else "/cpu:0"
        self._num_intrafusion_layers = num_intrafusion_layers
        self._with_features = with_features
        self._with_inter_fusion = with_inter_fusion
        self._layer_norm = layer_norm
        self.start_output_unit = start_output_unit
        assert start_output_unit in ["softmax", "sigmoid"]
        ExtractionQAModel.__init__(self, size, embedder, keep_prob, name)

    def _init(self):
        ExtractionQAModel._init(self)
        if self._composition == "GRU":
            if self._layer_norm:
                rnn_constructor = lambda size: FusedRNNCellAdaptor(LayerNormGRUCell(size), use_dynamic_rnn=True)
            else:
                rnn_constructor = lambda size: FusedRNNCellAdaptor(GRUBlockCell(size), use_dynamic_rnn=True)
        elif self._composition == "RNN":
            rnn_constructor = lambda size: FusedRNNCellAdaptor(BasicRNNCell(size), use_dynamic_rnn=True)
        else:
            if self._layer_norm:
                rnn_constructor = lambda size: FusedRNNCellAdaptor(LayerNormLSTMCell(size), use_dynamic_rnn=True)
            else:
                rnn_constructor = lambda size: LSTMBlockFusedCell(size)

        with tf.device(self._device0):
            self._eval = tf.get_variable("is_eval", initializer=False, trainable=False)
            self._set_train = self._eval.initializer
            self._set_eval = self._eval.assign(True)

            self.context_mask = tfutil.mask_for_lengths(self.context_length, self._batch_size, self.embedder.max_length)

            question_binary_mask = tfutil.mask_for_lengths(self.question_length,
                                                           self.question_embedder.batch_size,
                                                           self.question_embedder.max_length,
                                                           value=1.0,
                                                           mask_right=False)

            with tf.variable_scope("preprocessing_layer"):
                if self._with_features:
                    in_question_feature = tf.ones(tf.pack([self.question_embedder.batch_size,
                                                           self.question_embedder.max_length, 2]))
                    embedded_question = tf.concat(2, [self.embedded_question, in_question_feature])
                else:
                    embedded_question = self.embedded_question

                self.encoded_question = self._preprocessing_layer(rnn_constructor, embedded_question,
                                                                  self.question_length, projection_scope="question_proj")

                # single time attention over question
                attention_scores = tf.contrib.layers.fully_connected(self.encoded_question, 1,
                                                                     activation_fn=None,
                                                                     weights_initializer=None,
                                                                     biases_initializer=None,
                                                                     scope="attention")
                attention_scores = attention_scores + tf.expand_dims(
                    tfutil.mask_for_lengths(self.question_length, self.question_embedder.batch_size,
                                            self.question_embedder.max_length), 2)
                attention_weights = tf.nn.softmax(attention_scores, 1)
                self.question_attention_weights = attention_weights
                self.question_representation = tf.reduce_sum(attention_weights * self.encoded_question, [1])

                # Multiply question features for each paragraph
                self.encoded_question = tf.gather(self.encoded_question, self.context_partition)
                self._embedded_question_not_dropped = tf.gather(self._embedded_question_not_dropped, self.context_partition)
                self.question_representation = tf.gather(self.question_representation, self.context_partition)
                self.question_length = tf.gather(self.question_length, self.context_partition)
                question_binary_mask = tf.gather(question_binary_mask, self.context_partition)

                # context
                if self._with_features:
                    mask = tf.get_variable("attention_mask", [1, 1, self._embedded_question_not_dropped.get_shape()[-1].value],
                                           initializer=tf.constant_initializer(1.0))
                    # compute word wise features
                    #masked_question = self.question_embedder.output * mask
                    # [B, Q, L]
                    q2c_scores = tf.batch_matmul(self._embedded_question_not_dropped * mask,
                                                 self._embedded_context_not_dropped, adj_y=True)
                    q2c_scores = q2c_scores + tf.expand_dims(self.context_mask, 1)
                    #c2q_weights = tf.reduce_max(q2c_scores / (tf.reduce_max(q2c_scores, [2], keep_dims=True) + 1e-5), [1])

                    q2c_weights = tf.reduce_sum(tf.nn.softmax(q2c_scores) * \
                                                tf.expand_dims(question_binary_mask, 2), [1])

                    # [B, L , 1]
                    self.context_features = tf.concat(2, [tf.expand_dims(self._word_in_question, 2),
                                                          #tf.expand_dims(c2q_weights, 2),
                                                          tf.expand_dims(q2c_weights,  2)])

                    embedded_ctxt = tf.concat(2, [self.embedded_context, self.context_features])
                else:
                    embedded_ctxt = self.embedded_context

                self.encoded_ctxt = self._preprocessing_layer(rnn_constructor, embedded_ctxt, self.context_length,
                                                              share_rnn=True, projection_scope="context_proj",
                                                              num_fusion_layers=self._num_intrafusion_layers)

            if self._with_inter_fusion:
                with tf.variable_scope("inter_fusion"):
                    with tf.variable_scope("associative") as vs:
                        mask = tf.get_variable("attention_mask", [1, 1, self.size], initializer=tf.constant_initializer(1.0))
                        mask = tf.nn.relu(mask)
                        for i in range(1):
                            # [B, Q, L]
                            inter_scores = tf.batch_matmul(self.encoded_question * mask, self.encoded_ctxt, adj_y=True)
                            inter_scores = inter_scores + tf.expand_dims(self.context_mask, 1)

                            inter_weights = tf.nn.softmax(inter_scores)
                            inter_weights = inter_weights * tf.expand_dims(question_binary_mask, 2)
                            # [B, L, Q] x [B, Q, S] -> [B, L, S]
                            co_states = tf.batch_matmul(inter_weights, self.encoded_question, adj_x=True)

                            u = tf.contrib.layers.fully_connected(tf.concat(2, [self.encoded_ctxt, co_states]), self.size,
                                                                  activation_fn=tf.sigmoid,
                                                                  biases_initializer=tf.constant_initializer(1.0),
                                                                  scope="update_gate")
                            self.encoded_ctxt = u * self.encoded_ctxt + (1.0 - u) * co_states
                            vs.reuse_variables()

                    with tf.variable_scope("recurrent") as vs:
                        self.encoded_ctxt.set_shape([None, None, self.size])
                        self.encoded_ctxt = dynamic_rnn(GatedAggregationRNNCell(self.size),
                                                        tf.reverse_sequence(self.encoded_ctxt, self.context_length, 1),
                                                        self.context_length,
                                                        dtype=tf.float32, time_major=False, scope="backward")[0]

                        self.encoded_ctxt = dynamic_rnn(GatedAggregationRNNCell(self.size),
                                                        tf.reverse_sequence(self.encoded_ctxt, self.context_length, 1),
                                                        self.context_length,
                                                        dtype=tf.float32, time_major=False, scope="forward")[0]

            # No matching layer, so set matched_output to encoded_ctxt (for compatibility)
            self.matched_output = self.encoded_ctxt

            with tf.variable_scope("pointer_layer"):
                self.predicted_context_indices, \
                self._start_scores, self._start_pointer, self.start_probs, \
                self._end_scores, self._end_pointer, self.end_probs = \
                    self._spn_answer_layer(self.question_representation, self.encoded_ctxt)

            self._train_variables = [p for p in tf.trainable_variables() if self.name in p.name]

    def _preprocessing_layer(self, rnn_constructor, inputs, length, share_rnn=False,
                             projection_scope=None, num_fusion_layers=0):
        projection_initializer = tf.constant_initializer(np.concatenate([np.eye(self.size), np.eye(self.size)]))
        fused_rnn = rnn_constructor(self.size)
        with tf.variable_scope("RNN") as vs:
            if share_rnn:
                vs.reuse_variables()
            encoded = tfutil.fused_birnn(fused_rnn, inputs, sequence_length=length, dtype=tf.float32, time_major=False,
                                         backward_device=self._device1)[0]
            encoded = tf.concat(2, encoded)

        projected = tf.contrib.layers.fully_connected(encoded, self.size,
                                                      activation_fn=tf.tanh,
                                                      weights_initializer=projection_initializer,
                                                      scope=projection_scope)
        if num_fusion_layers > 0:
            with tf.variable_scope("intra_fusion") as vs:
                diag = tf.zeros(tf.unpack(tf.shape(inputs))[:2], dtype=tf.float32)
                mask = tf.get_variable("attention_mask", [1, 1, self.size], initializer=tf.constant_initializer(1.0))
                mask = tf.nn.relu(mask)
                for i in range(num_fusion_layers):
                    if i > 1:
                        vs.reuse_variables()
                    # [B, L, L]
                    intra_scores = tf.batch_matmul(mask * projected, projected, adj_y=True) \
                                   + tf.expand_dims(self.context_mask, 1)
                    intra_scores = tf.matrix_set_diag(intra_scores, diag)
                    intra_weights = tf.nn.softmax(intra_scores)
                    # [B, L, L] x [B, L, S] -> [B, L, S]
                    co_states = tf.batch_matmul(intra_weights, projected)

                    u = tf.contrib.layers.fully_connected(tf.concat(2, [projected, co_states]), self.size,
                                                          activation_fn=tf.sigmoid,
                                                          biases_initializer=tf.constant_initializer(1.0),
                                                          scope="update_gate")
                    projected = u*projected + (1.0-u) * co_states

                    projected = dynamic_rnn(GatedAggregationRNNCell(self.size),
                                            tf.reverse_sequence(projected, length, 1), length,
                                            dtype=tf.float32, time_major=False, scope="backward")[0]

                    projected = dynamic_rnn(GatedAggregationRNNCell(self.size),
                                            tf.reverse_sequence(projected, length, 1), length,
                                            dtype=tf.float32, time_major=False, scope="forward")[0]
        return projected

    def _spn_answer_layer(self, question_state, context_states):

        input_size = context_states.get_shape()[-1].value
        context_states_flat = tf.reshape(context_states, [-1, input_size])
        offsets = tf.cast(tf.range(0, self._batch_size), dtype=tf.int64) * (tf.reduce_max(self.context_length))

        #START
        start_input = tf.concat(2, [tf.expand_dims(question_state, 1) * context_states,
                                    context_states])

        q_start_inter = tf.contrib.layers.fully_connected(question_state, self.size,
                                                          activation_fn=None,
                                                          weights_initializer=None,
                                                          scope="q_start_inter")

        q_start_state = tf.contrib.layers.fully_connected(start_input, self.size,
                                                          activation_fn=None,
                                                          weights_initializer=None,
                                                          scope="q_start") + tf.expand_dims(q_start_inter, 1)

        start_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_start_state), 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="start_scores")
        start_scores = tf.squeeze(start_scores, [2])
        start_scores = start_scores + self.context_mask

        contexts, starts = tfutil.segment_argmax(start_scores, self.context_partition)
        if self.start_output_unit == "softmax":
            start_probs = tfutil.segment_softmax(start_scores, self.context_partition)
        else:
            start_probs = tf.sigmoid(start_scores)

        # From now on, answer_context_indices need to be fed.
        # There will be an end pointer prediction for each start pointer.
        starts = tf.gather(starts, self.context_partition)
        starts = tf.gather(starts, self.answer_context_indices)
        question_state = tf.gather(question_state, self.answer_context_indices)
        context_states = tf.gather(context_states, self.answer_context_indices)
        start_input = tf.gather(start_input, self.answer_context_indices)
        offsets = tf.gather(offsets, self.answer_context_indices)
        context_mask = tf.gather(self.context_mask, self.answer_context_indices)

        start_pointer = tf.cond(self._eval,
                                lambda: starts,
                                lambda: self.correct_start_pointer)

        u_s = tf.gather(context_states_flat, start_pointer + offsets)

        #END
        end_input = tf.concat(2, [tf.expand_dims(u_s, 1) * context_states, start_input])

        q_end_inter = tf.contrib.layers.fully_connected(tf.concat(1, [question_state, u_s]), self.size,
                                                        activation_fn=None,
                                                        weights_initializer=None,
                                                        scope="q_end_inter")

        q_end_state = tf.contrib.layers.fully_connected(end_input, self.size,
                                                        activation_fn=None,
                                                        weights_initializer=None,
                                                        scope="q_end") + tf.expand_dims(q_end_inter, 1)

        end_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_end_state), 1,
                                                       activation_fn=None,
                                                       weights_initializer=None,
                                                       biases_initializer=None,
                                                       scope="end_scores")
        end_scores = tf.squeeze(end_scores, [2])
        end_scores = end_scores + context_mask
        ends = tf.argmax(end_scores, axis=1)
        end_probs = tf.nn.softmax(end_scores)

        return contexts, start_scores, starts, start_probs, end_scores, ends, end_probs

    def set_eval(self, sess):
        super().set_eval(sess)
        sess.run(self._set_eval)

    def set_train(self, sess):
        super().set_train(sess)
        sess.run(self._set_train)

    @property
    def end_scores(self):
        return self._end_scores

    @property
    def start_scores(self):
        return self._start_scores

    @property
    def predicted_answer_starts(self):
        # for answer extraction models
        return self._start_pointer

    @property
    def predicted_answer_ends(self):
        # for answer extraction models
        return self._end_pointer

    @property
    def train_variables(self):
        return self._train_variables

    def get_config(self):
        config = super().get_config()
        config["type"] = "simple_pointer"
        config["composition"] = self._composition
        config["with_features"] = self._with_features
        config["with_inter_fusion"] = self._with_inter_fusion
        config["num_intrafusion_layers"] = self._num_intrafusion_layers
        config["layer_norm"] = self._layer_norm
        config["composition"] = self._composition
        config["start_output_unit"] = self.start_output_unit
        return config


    @staticmethod
    def create_from_config(config, devices, dropout=0.0, reuse=False):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """

        if "start_output_unit" not in config:
            config["start_output_unit"] = "softmax"

        from biomedical_qa.models import model_from_config
        embedder = model_from_config(config["transfer_model"], devices)
        qa_model = QASimplePointerModel(
            config["size"],
            embedder=embedder,
            name=config["name"],
            composition=config["composition"],
            keep_prob=1.0 - dropout,
            devices=devices,
            with_features=config["with_features"],
            with_inter_fusion=config["with_inter_fusion"],
            num_intrafusion_layers=config["num_intrafusion_layers"],
            layer_norm=config.get("layer_norm", False),
            start_output_unit=config["start_output_unit"])

        return qa_model


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

    return out
