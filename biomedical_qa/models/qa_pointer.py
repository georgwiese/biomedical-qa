from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn import *

from biomedical_qa.models.attention import *
from biomedical_qa.models.context_embedder import ContextEmbedder
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.models.rnn_cell import DynamicPointerRNN
from biomedical_qa.models.transfer import gated_transfer
from biomedical_qa import tfutil
import numpy as np

class QAPointerModel(ExtractionQAModel):

    def __init__(self, size, transfer_model, keep_prob=1.0, transfer_layer_size=None,
                 composition="GRU", devices=None, name="QAPointerModel", depends_on=[],
                 answer_layer_depth=1, answer_layer_poolsize=8):
        self._composition = composition
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"
        self._depends_on = depends_on
        self._transfer_layer_size = size if transfer_layer_size is None else transfer_layer_size
        self._answer_layer_depth = answer_layer_depth
        self._answer_layer_poolsize = answer_layer_poolsize

        ExtractionQAModel.__init__(self, size, transfer_model, keep_prob, name)

    def _init(self):
        ExtractionQAModel._init(self)
        if self._composition == "GRU":
            cell_constructor = lambda size: GRUCell(size)
        elif self._composition == "RNN":
            cell_constructor = lambda size: BasicRNNCell(size)
        else:
            cell_constructor = lambda size: BasicLSTMCell(size, state_is_tuple=True)

        with tf.device(self._device0):
            self._eval = tf.get_variable("is_eval", initializer=False, trainable=False)
            self._set_train = self._eval.initializer
            self._set_eval = self._eval.assign(True)

            with tf.control_dependencies(self._depends_on):
                with tf.variable_scope("preprocessing_layer"):
                    null_word = tf.get_variable("NULL_WORD", shape=[self.embedded_question.get_shape()[2]],
                                                initializer=tf.constant_initializer(0.0))
                    tiled_null_word = tf.tile(null_word, [self._batch_size])
                    reshaped_null_word = tf.reshape(tiled_null_word, [-1, 1, null_word.get_shape()[0].value])

                    # question
                    rev_embedded_question = tf.reverse_sequence(self.embedded_question, self.question_length, 1)
                    rev_embedded_question = tf.concat(1, [reshaped_null_word, rev_embedded_question])
                    embedded_question = tf.reverse_sequence(rev_embedded_question, self.question_length, 1)

                    self.encoded_question = self._preprocessing_layer(
                        cell_constructor, embedded_question,
                        self.question_length + 1, projection_scope="question_proj")

                    # single time attention over question
                    enc_question = tf.slice(self.encoded_question, [0, 0, 0], [-1, -1, self.size])
                    attention_scores = tf.contrib.layers.fully_connected(enc_question, 1,
                                                                         activation_fn=None,
                                                                         weights_initializer=None,
                                                                         biases_initializer=None,
                                                                         scope="attention")
                    attention_scores = tf.squeeze(attention_scores, [2])
                    attention_weights = tf.nn.softmax(attention_scores)
                    attention_weights = tf.expand_dims(attention_weights, 2)
                    self.question_representation = tf.reduce_sum(attention_weights * self.encoded_question, [1])

                    # context
                    rev_embedded_context = tf.reverse_sequence(self.embedded_context, self.context_length, 1)
                    rev_embedded_context = tf.concat(1, [reshaped_null_word, rev_embedded_context])
                    embedded_context = tf.reverse_sequence(rev_embedded_context, self.context_length, 1)

                    self.encoded_ctxt = self._preprocessing_layer(
                        cell_constructor, embedded_context, self.context_length + 1,
                        share_rnn=True, projection_scope="context_proj")

                with tf.variable_scope("match_layer"):
                    self.matched_output = self._match_layer(
                        self.encoded_question, self.encoded_ctxt,
                        cell_constructor)

                with tf.variable_scope("pointer_layer"):
                    self._start_scores, self._end_scores, self._start_pointer, self._end_pointer = \
                        self._answer_layer(self.question_representation, self.matched_output)

                self._train_variables = [p for p in tf.trainable_variables() if self.name in p.name]

    def _preprocessing_layer(self, cell_constructor, inputs, length, share_rnn=False,
                             projection_scope=None):

        projection_initializer = tf.constant_initializer(np.concatenate([np.eye(self.size), np.eye(self.size)]))
        cell = cell_constructor(self.size)
        with tf.variable_scope("RNN") as vs:
            if share_rnn:
                vs.reuse_variables()
            # Does this do use the same weights for forward & backward? Because
            # same cell instance is passed
            encoded = bidirectional_dynamic_rnn(cell, cell, inputs, length,
                                                dtype=tf.float32, time_major=False)[0]
        encoded = tf.concat(2, encoded)
        projected = tf.contrib.layers.fully_connected(encoded, self.size,
                                                      activation_fn=tf.tanh,
                                                      weights_initializer=projection_initializer,
                                                      scope=projection_scope)

        return projected

    def _match_layer(self, encoded_question, encoded_ctxt, cell_constructor):
        size = self.size

        matched_output = dot_co_attention(encoded_ctxt, self.context_length + 1,
                                          encoded_question, self.question_length + 1)
        matched_output = tf.nn.bidirectional_dynamic_rnn(cell_constructor(size),
                                                         cell_constructor(size),
                                                         matched_output, sequence_length=self.context_length,
                                                         dtype=tf.float32)[0]
        matched_output = tf.concat(2, matched_output)
        matched_output.set_shape([None, None, 2 * size])

        return matched_output

    def _answer_layer(self, question_state, context_states):
        context_states = tf.nn.dropout(context_states, self.keep_prob)
        # dynamic pointing decoder
        controller_cell = GRUCell(question_state.get_shape()[1].value)
        input_size = context_states.get_shape()[-1].value
        context_states_flat = tf.reshape(context_states, [-1, context_states.get_shape()[-1].value])
        offsets = tf.cast(tf.range(0, self._batch_size), dtype=tf.int64) * (tf.reduce_max(self.context_length) + 1)

        pointer_rnn = DynamicPointerRNN(self.size, self._answer_layer_poolsize,
                                        controller_cell, context_states,
                                        self.context_length + 1,
                                        self._answer_layer_depth)

        cur_state = question_state
        u = tf.zeros(tf.pack([self._batch_size, 2 * input_size]))
        is_stable = tf.constant(False, tf.bool, [1])
        is_stable = tf.tile(is_stable, tf.pack([tf.cast(self._batch_size, tf.int32)]))
        current_start, current_end = None, None
        start_scores, end_scores = [], []

        self.answer_partition = tf.cast(tf.range(0, self._batch_size), dtype=tf.int64)

        for i in range(4):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            (next_start_scores, next_end_scores), cur_state = \
                pointer_rnn(u, cur_state)

            next_start = tf.arg_max(next_start_scores, 1)
            next_end_scores_heuristic = next_end_scores + tfutil.mask_for_lengths(next_start,
                                                                                  max_length=self.embedder.max_length + 1,
                                                                                  mask_right=False)
            next_end = tf.arg_max(next_end_scores_heuristic, 1)

            u_s = tf.gather(context_states_flat, next_start + offsets)
            u_e = tf.gather(context_states_flat, next_end + offsets)
            u = tf.concat(1, [u_s, u_e])

            if i > 0:
                # Once is_stable is true, it'll stay stable
                is_stable = tf.logical_or(is_stable, tf.logical_and(tf.equal(next_start, current_start),
                                                                    tf.equal(next_end, current_end)))
                is_stable_int = tf.cast(is_stable, tf.int64)
                current_start = current_start * is_stable_int + (1 - is_stable_int) * next_start
                current_end = current_end * is_stable_int + (1 - is_stable_int) * next_end
            else:
                current_start = next_start
                current_end = next_end

            start_scores.append(tf.gather(next_start_scores, self.answer_partition))
            end_scores.append(tf.gather(next_end_scores, self.answer_partition))

        end_pointer = tf.gather(current_end, self.answer_partition)
        start_pointer = tf.gather(current_start, self.answer_partition)

        return start_scores, end_scores, start_pointer, end_pointer

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
        config["type"] = "pointer"
        config["composition"] = self._composition
        config["answer_layer_depth"] = self._answer_layer_depth
        config["answer_layer_poolsize"] = self._answer_layer_poolsize
        return config

    @staticmethod
    def create_from_config(config, devices, dropout=0.0, reuse=False):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """
        # size, max_answer_length, embedder, keep_prob, name="QAModel", reuse=False

        # Set defaults for backword compatibility
        if "answer_layer_depth" not in config:
            config["answer_layer_depth"] = 1
        if "answer_layer_poolsize" not in config:
            config["answer_layer_poolsize"] = 8

        from biomedical_qa.models import model_from_config
        transfer_model = model_from_config(config["transfer_model"], devices)
        if transfer_model is None:
            transfer_model = model_from_config(config["transfer_model"], devices)
        qa_model = QAPointerModel(
            config["size"],
            transfer_model=transfer_model,
            name=config["name"],
            composition=config["composition"],
            keep_prob=1.0 - dropout,
            devices=devices,
            answer_layer_depth=config["answer_layer_depth"],
            answer_layer_poolsize=config["answer_layer_poolsize"])

        return qa_model
