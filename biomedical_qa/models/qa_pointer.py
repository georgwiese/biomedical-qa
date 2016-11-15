import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn import *

from biomedical_qa.models.attention import *
from biomedical_qa.models.embedder import WordEmbedder
from biomedical_qa.models.context_embedder import ContextEmbedder
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.models.rnn_cell import ParamAssociativeMemory, MultiConcatRNNCell, ParamNTM, BackwardNTM, \
    DynamicPointerRNN
from biomedical_qa.models.transfer import gated_transfer
from biomedical_qa.training.qa_trainer import ExtractionQATrainer
from biomedical_qa import tfutil
import numpy as np

class QAPointerModel(ExtractionQAModel):

    def __init__(self, size, transfer_model, keep_prob=1.0, transfer_layer_size=None, num_slots=0,
                 composition="GRU", devices=None, name="QAPointerModel", depends_on=[]):
        self._composition = composition
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"
        self._depends_on = depends_on
        self._transfer_layer_size = size if transfer_layer_size is None else transfer_layer_size
        self._num_slots = num_slots

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

                    transfer_input = None
                    if isinstance(self.transfer_model, QAPointerModel):
                        transfer_input = self.transfer_model.encoded_question
                    elif isinstance(self.question_embedder, ContextEmbedder):
                        zero_padding = tf.zeros(shape=[1, 1, self.question_embedder.output.get_shape()[2]])
                        tiled_zero_padding = tf.tile(zero_padding, tf.pack([self._batch_size, 1, 1]))
                        transfer_input = tf.concat(1, [self.question_embedder.output, tiled_zero_padding])

                    self.encoded_question, self.question_fw_weights, self.question_bw_weights = \
                        self._preprocessing_layer(cell_constructor, embedded_question, self.question_length + 1,
                                                  transfer_input, projection_scope="question_proj")

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

                    transfer_input = None
                    if isinstance(self.transfer_model, QAPointerModel):
                        transfer_input = self.transfer_model.encoded_ctxt
                    elif isinstance(self.embedder, ContextEmbedder):
                        zero_padding = tf.zeros(shape=[1, 1, self.embedder.output.get_shape()[2]])
                        tiled_zero_padding = tf.tile(zero_padding, tf.pack([self._batch_size, 1, 1]))
                        transfer_input = tf.concat(1, [self.embedder.output, tiled_zero_padding])

                    self.embedded_context = tf.concat(1, [self.embedded_context, reshaped_null_word])
                    self.encoded_ctxt, self.ctxt_fw_weights, self.ctxt_bw_weights = \
                        self._preprocessing_layer(cell_constructor, embedded_context, self.context_length + 1,
                                                  transfer_input, share_rnn=True, projection_scope="context_proj")

                with tf.variable_scope("match_layer"):
                    self.matched_output = self._match_layer(self.encoded_question, self.encoded_ctxt, cell_constructor)

                with tf.variable_scope("pointer_layer"):
                    self._start_scores, self._end_scores, self._start_pointer, self._end_pointer = \
                        self._answer_layer(self.question_representation, self.matched_output)

                self._train_variables = [p for p in tf.trainable_variables() if self.name in p.name]

    def _preprocessing_layer(self, cell_constructor, inputs, length, transfer_input=None, share_rnn=False,
                             projection_scope=None):
        if self._num_slots > 0:
            with tf.variable_scope("memory") as vs:
                if share_rnn:
                    vs.reuse_variables()

                bw_outputs = dynamic_rnn(cell_constructor(self.size), tf.reverse_sequence(inputs, length, 1),
                                         length, dtype=tf.float32, time_major=False)[0]

                fw_inputs = tf.concat(2, [tf.reverse_sequence(bw_outputs, length, 1), inputs])
                cell_fw = ParamNTM(self._num_slots, self.size, fw_inputs.get_shape()[2].value, cell_constructor(self.size))
                with tf.variable_scope("forward"):
                    fw_memory, last_state = dynamic_rnn(cell_fw, fw_inputs, length, dtype=tf.float32, time_major=False)
                fw_weights = bw_weights = tf.slice(fw_memory, [0, 0, cell_fw.output_size - self._num_slots],
                                                   [-1, -1, -1])
                fw_memory = tf.slice(fw_memory, [0, 0, 0], [-1, -1, cell_fw.output_size - self._num_slots])

                last_ctr_state, last_memory, last_ctr_out, _ = last_state

                inputs_bw = tf.reverse_sequence(fw_memory, length - 1, 1)
                start_state = tf.split(1, self._num_slots, last_memory)
                cell_bw = BackwardNTM(self._num_slots, self.size)
                with tf.variable_scope("backward"):
                    partial_memory_bw = \
                        dynamic_rnn(cell_bw, inputs_bw, length - 1, initial_state=start_state, time_major=False)[0]
                    last_memory = tf.expand_dims(tf.concat(1, [last_state[2], last_state[1]]), 1)
                    #last memory comes from dummy word and should be set to zero
                    last_memory = tf.concat(2, [tf.slice(last_memory, [0, 0, 0], [-1, -1, self.size]),
                                                tf.zeros_like(tf.slice(last_memory, [0, 0, self.size], [-1, -1, -1]))])
                    memory_bw = tf.concat(1, [last_memory, partial_memory_bw])
                    memory_bw = tf.slice(memory_bw, [0, 0, 0], tf.shape(partial_memory_bw))
                    memory = tf.reverse_sequence(memory_bw, length, 1)
                    memory.set_shape([None, None, (self._num_slots + 1) * self.size])

            return memory, fw_weights, bw_weights
        else:
            if transfer_input is not None:
                if self._transfer_layer_size <= 0:
                    encoded_dropped = tf.nn.dropout(transfer_input, self.keep_prob)
                    projected = tf.contrib.layers.fully_connected(encoded_dropped, self.size,
                                                                  activation_fn=tf.tanh,
                                                                  weights_initializer=None,
                                                                  scope=projection_scope)
                else:
                    cell = cell_constructor(self._transfer_layer_size)
                    with tf.variable_scope("RNN") as vs:
                        if share_rnn:
                            vs.reuse_variables()
                        encoded = bidirectional_dynamic_rnn(cell, cell, inputs, length,
                                                            dtype=tf.float32, time_major=False)[0]
                    with tf.variable_scope(projection_scope or "projection"):
                        projected = \
                            gated_transfer(2, list(encoded), transfer_input, self.keep_prob,
                                           self.size, 2 * self._transfer_layer_size, activation_fn=tf.tanh)

            else:
                projection_initializer = tf.constant_initializer(np.concatenate([np.eye(self.size), np.eye(self.size)]))
                cell = cell_constructor(self.size)
                with tf.variable_scope("RNN") as vs:
                    if share_rnn:
                        vs.reuse_variables()
                    encoded = bidirectional_dynamic_rnn(cell, cell, inputs, length,
                                                        dtype=tf.float32, time_major=False)[0]
                encoded = tf.concat(2, encoded)
                projected = tf.contrib.layers.fully_connected(encoded, self.size,
                                                              activation_fn=tf.tanh,
                                                              weights_initializer=projection_initializer,
                                                              scope=projection_scope)

            return projected, None, None

    def _match_layer(self, encoded_question, encoded_ctxt, cell_constructor):
        size = self.size
        if isinstance(self.transfer_model, QAPointerModel):
            if self._transfer_layer_size <= 0:
                return tf.nn.dropout(self.transfer_model.matched_output, self.keep_prob)
            else:
                size = self._transfer_layer_size

        if self._num_slots > 0:
            eye_initializer = tf.constant_initializer(np.eye(self.size))
            split_question = tf.split(2, self._num_slots + 1, encoded_question)
            context_state_rep = tf.slice(encoded_ctxt, [0, 0, 0], [-1, -1, self.size])
            inter_question_memories = []
            for i, q in enumerate(split_question):
                inter_question_memories.append(tf.contrib.layers.fully_connected(q, self.size,
                                                                                 activation_fn=None,
                                                                                 weights_initializer=eye_initializer,
                                                                                 biases_initializer=None,
                                                                                 scope="inter_states_%d" % i))
            inter_question_memories = tf.concat(2, inter_question_memories)

            affinity_scores = tf.batch_matmul(encoded_ctxt, inter_question_memories, adj_y=True)
            matched_output = extract_co_attention_states(affinity_scores, context_state_rep, self.context_length + 1,
                                                         split_question[0], self.question_length + 1)
        else:
            matched_output = dot_co_attention(encoded_ctxt, self.context_length + 1,
                                              encoded_question, self.question_length + 1)
        matched_output = tf.nn.bidirectional_dynamic_rnn(cell_constructor(size),
                                                         cell_constructor(size),
                                                         matched_output, sequence_length=self.context_length,
                                                         dtype=tf.float32)[0]
        matched_output = tf.concat(2, matched_output)
        matched_output.set_shape([None, None, 2 * size])

        if isinstance(self.transfer_model, QAPointerModel):
            matched_output = gated_transfer(2, matched_output,
                                            self.transfer_model.matched_output,
                                            self.keep_prob,
                                            2 * self.size)

        return matched_output

    def _answer_layer(self, question_state, context_states):
        context_states = tf.nn.dropout(context_states, self.keep_prob)
        # dynamic pointing decoder
        controller_cell = GRUCell(question_state.get_shape()[1].value)
        input_size = context_states.get_shape()[-1].value
        context_states_flat = tf.reshape(context_states, [-1, context_states.get_shape()[-1].value])
        offsets = tf.cast(tf.range(0, self._batch_size), dtype=tf.int64) * (tf.reduce_max(self.context_length) + 1)

        pointer_rnn = DynamicPointerRNN(self.size, 8, controller_cell, context_states, self.context_length + 1)

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
        config["num_slots"] = self._num_slots
        config["composition"] = self._composition
        return config

    @staticmethod
    def create_from_config(config, devices, dropout=0.0, reuse=False):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """
        # size, max_answer_length, embedder, keep_prob, name="QAModel", reuse=False
        from quebap.projects.autoread import model_from_config, model_from_config
        transfer_model = model_from_config(config["transfer_model"], devices)
        if transfer_model is None:
            transfer_model = model_from_config(config["transfer_model"], devices)
        qa_model = QAPointerModel(
            config["size"],
            transfer_model=transfer_model,
            name=config["name"],
            composition=config["composition"],
            num_slots=config.get("num_slots", 0),
            keep_prob=1.0 - dropout,
            devices=devices)

        return qa_model
