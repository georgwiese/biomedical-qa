import tensorflow as tf

from biomedical_qa.models.embedder import Embedder
from biomedical_qa.models.model import ConfigurableModel


class QAModel(ConfigurableModel):

    def __init__(self, size, transfer_model, keep_prob, name="QAModel", reuse=False):
        self.size = size
        self.vocab_size = transfer_model.vocab_size
        self.name = name

        self.transfer_model = transfer_model

        if isinstance(transfer_model, Embedder):
            self.embedder = transfer_model
            self.question_embedder = self.embedder.clone()
        else:
            self.embedder = transfer_model.embedder
            self.question_embedder = transfer_model.question_embedder

        with tf.variable_scope(name, initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse):
            self.keep_prob = tf.get_variable("keep_prob", [], initializer=tf.constant_initializer(keep_prob),
                                             trainable=False)
            self._init()
            self._save_vars = list(self.train_variables)
            for v in self.transfer_model.save_variables:
                if v not in self._save_vars:
                    self._save_vars.append(v)

    def set_train(self, sess):
        sess.run(self.keep_prob.initializer)

    def set_eval(self, sess):
        sess.run(self.keep_prob.assign(1.0))

    def _embed(self, input, length, e):
        max_length = tf.cast(tf.reduce_max(length), tf.int32)
        input = tf.slice(input, [0, 0], tf.pack([-1, max_length]))
        embedded = tf.nn.embedding_lookup(e, input)
        embedded = tf.nn.dropout(embedded, self.keep_prob)
        return embedded, input

    def _init(self):
        pass

    @property
    def predicted_answers(self):
        raise NotImplementedError()

    @property
    def predicted_lengths(self):
        raise NotImplementedError()

    @property
    def decoder_outputs(self):
        raise NotImplementedError()

    @property
    def train_variables(self):
        raise NotImplementedError()

    @property
    def save_variables(self):
        return self._save_vars

    def get_config(self):
        config = dict()
        config["transfer_model"] = self.transfer_model.get_config()
        config["size"] = self.size
        config["name"] = self.name
        return config

    @property
    def run(self, sess, goal, qa_settings):
        raise NotImplementedError()


class ExtractionQAModel(QAModel):

    def _init(self):
        with tf.device("/cpu:0"):
            self.question = self.question_embedder.inputs
            self.question_length = self.question_embedder.seq_lengths

            self.context = self.embedder.inputs
            self.context_length = self.embedder.seq_lengths
            self.top_k = tf.get_variable("k_answers", trainable=False, dtype=tf.int32, initializer=1)
            self._top_k_placeholder = tf.placeholder(tf.int32, tuple(), "top_k_placeholder")
            self._set_top_k = self.top_k.assign(self._top_k_placeholder)

            # Maps context index to question index
            self.context_partition = tf.placeholder(tf.int64, [None], "context_partition")

            with tf.variable_scope("embeddings"):
                # embeddings
                self._batch_size = self.embedder.batch_size
                # [B, MAX_T, S']
                embedded_question = self.question_embedder.embedded_words
                embedded_context = self.embedder.embedded_words
                self.embedded_question = tf.nn.dropout(embedded_question, self.keep_prob)
                self.embedded_context = tf.nn.dropout(embedded_context, self.keep_prob)

    def set_top_k(self, sess, k):
        return sess.run(self._set_top_k, feed_dict={self._top_k_placeholder:k})

    @property
    def predicted_answer_starts(self):
        # for answer extraction models
        raise NotImplementedError()

    @property
    def predicted_answer_ends(self):
        # for answer extraction models
        raise NotImplementedError()

    @property
    def start_scores(self):
        # for answer extraction models
        raise NotImplementedError()

    @property
    def end_scores(self):
        # for answer extraction models
        raise NotImplementedError()


    @property
    def predicted_lengths(self):
        return self.predicted_answer_ends - self.predicted_answer_starts + 1

    def get_feed_dict(self, qa_settings):
        question = []
        question_length = []

        context = []
        context_length = []

        context_partition = []

        max_q_length = max([len(s.question) for s in qa_settings])
        max_c_length = max([len(c) for s in qa_settings for c in s.contexts])
        for i, qa_setting in enumerate(qa_settings):
            question.append(qa_setting.question + [0] * (max_q_length - len(qa_setting.question)))
            question_length.append(len(qa_setting.question))
            assert len(qa_setting.contexts) > 0
            for c in qa_setting.contexts:
                context.append(c + [0] * (max_c_length - len(c)))
                context_length.append(len(c))
                context_partition.append(i)

        feed_dict = dict()
        feed_dict[self.context_partition] = context_partition
        feed_dict.update(self.embedder.get_feed_dict(context, context_length))
        feed_dict.update(self.question_embedder.get_feed_dict(question, question_length))

        return feed_dict

    def run(self, sess, goal, qa_settings):
        return sess.run(goal, feed_dict=self.get_feed_dict(qa_settings))
