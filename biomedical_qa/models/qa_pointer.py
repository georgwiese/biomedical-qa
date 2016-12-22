import tensorflow as tf

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import LSTMBlockCell, GRUBlockCell
from tensorflow.python.ops.rnn_cell import BasicRNNCell

from biomedical_qa.models.attention import dot_co_attention
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.models.rnn_cell import _highway_maxout_network
from biomedical_qa import tfutil
import numpy as np

MAX_ANSWER_LENGTH_HEURISTIC = 10

class QAPointerModel(ExtractionQAModel):

    def __init__(self, size, transfer_model, keep_prob=1.0, transfer_layer_size=None,
                 composition="GRU", devices=None, name="QAPointerModel", depends_on=[],
                 answer_layer_depth=1, answer_layer_poolsize=8,
                 answer_layer_type="dpn"):
        self._composition = composition
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"
        self._depends_on = depends_on
        self._transfer_layer_size = size if transfer_layer_size is None else transfer_layer_size
        self._answer_layer_depth = answer_layer_depth
        self._answer_layer_poolsize = answer_layer_poolsize
        self._answer_layer_type = answer_layer_type

        ExtractionQAModel.__init__(self, size, transfer_model, keep_prob, name)

    def _init(self):
        ExtractionQAModel._init(self)
        if self._composition == "GRU":
            cell_constructor = lambda size: GRUBlockCell(size)
        elif self._composition == "RNN":
            cell_constructor = lambda size: BasicRNNCell(size)
        else:
            cell_constructor = lambda size: LSTMBlockCell(size)

        with tf.device(self._device0):
            self._eval = tf.get_variable("is_eval", initializer=False, trainable=False)
            self._set_train = self._eval.initializer
            self._set_eval = self._eval.assign(True)

            self.paragraph2question = tf.placeholder(tf.int64, [None], "paragraph2question")

            # Fed during Training & end pointer prediction
            self.correct_start_pointer = tf.placeholder(tf.int64, [None])
            self.answer_context_indices = tf.placeholder(tf.int64, [None])

            with tf.control_dependencies(self._depends_on):
                with tf.variable_scope("preprocessing_layer"):

                    self.encoded_question = self._preprocessing_layer(
                        cell_constructor, self.embedded_question,
                        self.question_length, projection_scope="question_proj")

                    # single time attention over question
                    attention_scores = tf.contrib.layers.fully_connected(self.encoded_question, 1,
                                                                         activation_fn=None,
                                                                         weights_initializer=None,
                                                                         biases_initializer=None,
                                                                         scope="attention")
                    attention_scores = tf.squeeze(attention_scores, [2])
                    attention_scores += tfutil.mask_for_lengths(self.question_length)
                    attention_weights = tf.nn.softmax(attention_scores)
                    attention_weights = tf.expand_dims(attention_weights, 2)
                    self.question_representation = tf.reduce_sum(attention_weights * self.encoded_question, [1])

                    # Multiply question features for each paragraph
                    self.encoded_question = tf.gather(self.encoded_question, self.paragraph2question)
                    self.question_representation = tf.gather(self.question_representation, self.paragraph2question)

                    self.encoded_ctxt = self._preprocessing_layer(
                        cell_constructor, self.embedded_context, self.context_length,
                        share_rnn=True, projection_scope="context_proj")

                    # Append NULL word
                    null_word = tf.get_variable(
                        "NULL_WORD", shape=[self.encoded_ctxt.get_shape()[2]],
                        initializer=tf.constant_initializer(0.0))
                    self.encoded_question, self.question_length = self.append_null_word(
                        self.encoded_question, self.question_length, null_word)
                    self.encoded_ctxt, self.context_length = self.append_null_word(
                        self.encoded_ctxt, self.context_length, null_word)

                with tf.variable_scope("match_layer"):
                    self.matched_output = self._match_layer(
                        self.encoded_question, self.encoded_ctxt,
                        cell_constructor)

                with tf.variable_scope("pointer_layer"):
                    if self._answer_layer_type == "spn":
                        self.predicted_context_indices, \
                        self._start_scores, self._start_pointer, self.start_probs, \
                        self._end_scores, self._end_pointer, self.end_probs = \
                            self._spn_answer_layer(self.question_representation, self.matched_output)
                    else:
                        raise ValueError("Unknown answer layer type: %s" % self._answer_layer_type)

                self._train_variables = [p for p in tf.trainable_variables() if self.name in p.name]

    def append_null_word(self, tensor, lengths, null_word):

        tiled_null_word = tf.tile(null_word, [self._batch_size])
        reshaped_null_word = tf.reshape(tiled_null_word,
                                        [-1, 1, null_word.get_shape()[0].value])

        rev_tensor = tf.reverse_sequence(tensor, lengths, 1)
        rev_tensor = tf.concat(1, [reshaped_null_word, rev_tensor])
        new_tensor = tf.reverse_sequence(rev_tensor, lengths + 1, 1)

        return new_tensor, lengths + 1

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

        matched_output = dot_co_attention(encoded_ctxt, self.context_length,
                                          encoded_question, self.question_length)
        # TODO: Append feature if token is in question
        matched_output = tf.nn.bidirectional_dynamic_rnn(cell_constructor(size),
                                                         cell_constructor(size),
                                                         matched_output, sequence_length=self.context_length,
                                                         dtype=tf.float32)[0]
        matched_output = tf.concat(2, matched_output)
        matched_output.set_shape([None, None, 2 * size])

        return matched_output

    def _spn_answer_layer(self, question_state, context_states):

        context_states = tf.nn.dropout(context_states, self.keep_prob)
        context_shape = tf.shape(context_states)
        input_size = context_states.get_shape()[-1].value
        context_states_flat = tf.reshape(context_states, [-1, input_size])
        offsets = tf.cast(tf.range(0, self._batch_size), dtype=tf.int64) \
                  * (tf.reduce_max(self.context_length))

        def hmn(input, states, context_lengths):
            # Use context_length - 1 so that the null word is never selected.
            return _highway_maxout_network(self._answer_layer_depth,
                                           self._answer_layer_poolsize,
                                           input,
                                           states,
                                           context_lengths - 1,
                                           context_shape[1],
                                           self.size)

        with tf.variable_scope("start"):
            start_scores = hmn(question_state, context_states,
                               self.context_length)
            # TODO: Handle new format
            contexts, starts = tfutil.segment_argmax(start_scores, self.paragraph2question)
            start_probs = tfutil.segment_softmax(start_scores, self.paragraph2question)

        # From now on, answer_context_indices and correct_start_pointer need to be fed.
        # There will be an end pointer prediction for each start pointer.
        question_state = tf.gather(question_state, self.answer_context_indices)
        context_states = tf.gather(context_states, self.answer_context_indices)
        offsets = tf.gather(offsets, self.answer_context_indices)
        context_lengths = tf.gather(self.context_length, self.answer_context_indices)

        start_pointer = self.correct_start_pointer
        u_s = tf.gather(context_states_flat, start_pointer + offsets)

        with tf.variable_scope("end"):
            end_input = tf.concat(1, [u_s, question_state])
            end_scores = hmn(end_input, context_states, context_lengths)
            ends = tf.argmax(end_scores, axis=1)
            end_probs = tf.nn.softmax(end_scores)

        # Mask end scores for evaluation
        masked_end_scores = end_scores + tfutil.mask_for_lengths(
            start_pointer, mask_right=False, max_length=self.embedder.max_length + 1)
        masked_end_scores = masked_end_scores + tfutil.mask_for_lengths(
            start_pointer + MAX_ANSWER_LENGTH_HEURISTIC + 1,
            max_length=self.embedder.max_length + 1)
        end_scores = tf.cond(self._eval,
                             lambda: masked_end_scores,
                             lambda: end_scores)

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
        config["type"] = "pointer"
        config["composition"] = self._composition
        config["answer_layer_depth"] = self._answer_layer_depth
        config["answer_layer_poolsize"] = self._answer_layer_poolsize
        config["answer_layer_type"] = self._answer_layer_type
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
        if "answer_layer_type" not in config:
            config["answer_layer_type"] = "dpn"

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
            answer_layer_poolsize=config["answer_layer_poolsize"],
            answer_layer_type=config["answer_layer_type"])

        return qa_model
