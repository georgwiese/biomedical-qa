"""
            __                       __
 ___ ___ __/ /____  _______ ___ ____/ /
/ _ `/ // / __/ _ \/ __/ -_) _ `/ _  /
\_,_/\_,_/\__/\___/_/  \__/\_,_/\_,_/ v0.1

Learning to read, unsupervised
"""
import tensorflow as tf
import numpy as np
import math
import pickle

from biomedical_qa import tfutil
from biomedical_qa.models.model import ConfigurableModel


class Embedder(ConfigurableModel):
    def __init__(self, size, vocab_size, vocab, name="Embedder", reuse=False):
        self.vocab = vocab
        if vocab_size > 0:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(vocab)
        self.size = size
        self.name = name
        with tf.variable_scope(name, initializer=tf.contrib.layers.xavier_initializer(), reuse=True):
            # temporary HACK so that we can reuse this scope multiple times with same name, needed for loading older model
            tf.get_variable_scope()._reuse = reuse
            self._init()

    def _init(self):
        pass

    @property
    def output(self):
        """
        :return: output embedding of this layer
        """
        raise NotImplementedError()

    def get_feed_dict(self, inputs, seq_lengths):
        return None

    @property
    def embedded_words(self):
        raise NotImplementedError()

    @property
    def word_embeddings(self):
        raise NotImplementedError()

    @property
    def word_embedder(self):
        raise NotImplementedError()

    @property
    def max_length(self):
        raise NotImplementedError()

    @property
    def inputs(self):
        raise NotImplementedError()

    @property
    def sliced_inputs(self):
        raise NotImplementedError()

    @property
    def seq_lengths(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        raise NotImplementedError()

    @property
    def train_variables(self):
        raise NotImplementedError()

    @property
    def save_variables(self):
        return self.train_variables

    def clone(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        raise NotImplementedError()


class WordEmbedder(Embedder):

    def __init__(self, size, vocab_size, vocab, unk_id, name="Embedder", reuse=False, inputs=None, seq_lengths=None):
        self._unk_id = unk_id
        self._inputs = inputs
        with tf.device("/cpu:0"):
            if inputs is None:
                self._inputs = tf.placeholder(tf.int64, shape=[None, None], name="inputs")
            else:
                self._inputs = inputs
            if seq_lengths is None:
                self._seq_lengths = tf.placeholder(tf.int64, shape=[None], name="input_lengths")
            else:
                self._seq_lengths = seq_lengths
        super().__init__(size, vocab_size, vocab, name, reuse)

    def _init(self):
        with tf.device("/cpu:0"):
            inputs = self._inputs
            if self.vocab_size > 0:
                # set idxs > vocab_size to unk_id
                inputs = inputs + tf.cast(tf.greater_equal(inputs, self.vocab_size), tf.int64) * (self._unk_id - inputs)

            with tf.variable_scope("embeddings"):
                self._embedding_matrix = \
                    tf.get_variable("embedding_matrix", shape=(self.vocab_size, self.size),
                                    initializer=tf.random_normal_initializer(0.0, 0.1), trainable=True)

                self._max_length = tf.cast(tf.reduce_max(self.seq_lengths), tf.int32)
                self._batch_size = tf.shape(self.seq_lengths)[0]
                self._sliced_inputs = tf.slice(inputs, (0, 0), tf.pack((-1, self.max_length)))
                self._embedded_words = tf.nn.embedding_lookup(self._embedding_matrix, self.sliced_inputs)

        self._train_variables = [self._embedding_matrix]

    @property
    def max_length(self):
        return self._max_length

    @property
    def output(self):
        return self._embedded_words

    @property
    def embedded_words(self):
        return self._embedded_words

    @property
    def word_embeddings(self):
        return self._embedding_matrix

    def get_feed_dict(self, inputs, seq_lengths):
        feed_dict = dict()
        feed_dict[self._inputs] = inputs
        feed_dict[self._seq_lengths] = seq_lengths
        return feed_dict

    @property
    def word_embedder(self):
        return self

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def inputs(self):
        return self._inputs

    @property
    def sliced_inputs(self):
        return self._sliced_inputs

    @property
    def train_variables(self):
        return self._train_variables

    def clone(self, inputs=None, seq_lengths=None, **kwargs):
        return WordEmbedder(self.size, self.vocab_size, self.vocab, self._unk_id, self.name, inputs=inputs,
                            seq_lengths=seq_lengths, reuse=True)

    def get_config(self):
        config = dict()
        config["size"] = self.size
        config["vocab_size"] = self.vocab_size
        config["vocab"] = self.vocab
        config["unk_id"] = self._unk_id
        config["name"] = self.name
        config["type"] = "word"
        return config

    @staticmethod
    def create_from_config(config, reuse=False, inputs=None, seq_lengths=None):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """
        # todo: load parameters of the model
        # todo: dump config dictionary as json
        autoreader = WordEmbedder(
            config["size"],
            config["vocab_size"],
            config["vocab"],
            config["unk_id"],
            name=config.get("name", "AutoReader"),
            reuse=reuse,
            inputs=inputs,
            seq_lengths=seq_lengths
        )

        return autoreader


class ConstantWordEmbedder(WordEmbedder):

    def __init__(self, size, vocab, unk_id, embeddings, name="Embedder",
                 reuse=False, inputs=None, seq_lengths=None, embeddings_config=None):
        self._embeddings = embeddings
        self.embeddings_config = embeddings_config
        super().__init__(size, len(vocab), vocab, unk_id, name, reuse, inputs, seq_lengths)

    def _init(self):
        with tf.device("/cpu:0"):
            inputs = self._inputs
            if self.vocab_size > 0:
                # set idxs > vocab_size to unk_id
                inputs = inputs + tf.cast(tf.greater_equal(inputs, self.vocab_size), tf.int64) * (self._unk_id - inputs)

            with tf.variable_scope("embeddings"):
                self._max_length = tf.cast(tf.reduce_max(self.seq_lengths), tf.int32)
                self._batch_size = tf.shape(self.seq_lengths)[0]
                self._sliced_inputs = tf.slice(inputs, (0, 0), tf.pack((-1, self.max_length)))
                self._embedded_words = tf.placeholder(tf.float32, [None, None, self.size], "embedded_words")
                dummy_variable = tf.get_variable("dummy", dtype=tf.float32, initializer=0.0)
        self._train_variables = [dummy_variable]

    def get_feed_dict(self, inputs, seq_lengths):
        feed_dict = super().get_feed_dict(inputs, seq_lengths)

        if isinstance(inputs, list):
            batch_size = len(inputs)
            max_l = max(seq_lengths)
        else:
            batch_size = inputs.shape[0]
            max_l = np.max(seq_lengths)

        embedded_inputs = np.zeros([batch_size, max_l, self.size])
        for i in range(len(inputs)):
            for j in range(seq_lengths[i]):
                embedded_inputs[i, j] = self._embeddings[inputs[i][j]]
        feed_dict[self._embedded_words] = embedded_inputs
        return feed_dict

    def clone(self, inputs=None, seq_lengths=None, **kwargs):
        return ConstantWordEmbedder(self.size, self.vocab, self._unk_id, self._embeddings,
                                    self.name, inputs=inputs, seq_lengths=seq_lengths, reuse=True)

    def get_config(self):
        config = dict()
        config["size"] = self.size
        if self.embeddings_config is not None:
            config["embeddings_config"] = self.embeddings_config
        else:
            config["embeddings"] = self._embeddings
        config["vocab"] = self.vocab
        config["unk_id"] = self._unk_id
        config["name"] = self.name
        config["type"] = "constant_word"
        return config

    @property
    def word_embeddings(self):
        return self._embeddings

    @staticmethod
    def create_from_config(config, reuse=False, inputs=None, seq_lengths=None):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """
        # todo: load parameters of the model
        # todo: dump config dictionary as json

        if "embeddings_config" in config:
            with open(config["embeddings_config"], "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = config["embeddings"]

        autoreader = ConstantWordEmbedder(
            config["size"],
            config["vocab"],
            config["unk_id"],
            embeddings,
            name=config.get("name", "AutoReader"),
            reuse=reuse,
            inputs=inputs,
            seq_lengths=seq_lengths,
            embeddings_config=config.get("embeddings_config", None)
        )

        return autoreader


class CharWordEmbedder(WordEmbedder):

    def __init__(self, size, vocab, device="/cpu:0", name="CharWordEmbedder", reuse=False, inputs=None, seq_lengths=None):
        self._device = device
        super().__init__(size, -1, vocab, -1, name, reuse, inputs, seq_lengths)

    def _init(self):
        # build char_vocab
        # reset vocab_size to size of actual vocabulary
        conv_width = 5
        pad_right = math.ceil(conv_width / 2) # "fixed PAD o right side"
        self.vocab_size = max(self.vocab.values())+ 1
        max_l = max(len(w) for w in self.vocab) + pad_right
        self.char_vocab = {"PAD": 0}
        self._word_to_chars_arr = np.zeros((self.vocab_size, max_l), np.int16)
        self._word_lengths_arr = np.zeros([self.vocab_size], np.int8)
        for w, i in sorted(self.vocab.items()):
            for k, c in enumerate(w):
                j = self.char_vocab.get(c)
                if j is None:
                    j = len(self.char_vocab)
                    self.char_vocab[c] = j
                self._word_to_chars_arr[i, k] = j
            self._word_lengths_arr[i] = len(w) + conv_width - 1

        with tf.device("/cpu:0"):
            with tf.variable_scope("embeddings"):
                self._word_to_chars = tf.placeholder(tf.int64, [None, None], "word_to_chars")
                self._word_lengths = tf.placeholder(tf.int64, [None], "word_lengths")
                self.char_embedding_matrix = \
                    tf.get_variable("char_embedding_matrix", shape=(len(self.char_vocab), self.size),
                                    initializer=tf.random_normal_initializer(0.0, 0.1), trainable=True)

                self._max_length = tf.cast(tf.reduce_max(self.seq_lengths), tf.int32)
                self._batch_size = tf.shape(self.seq_lengths)[0]
                self._sliced_inputs = tf.slice(self.inputs, (0, 0), tf.pack((-1, self.max_length)))

                self.unique_words = tf.placeholder(tf.int64, [None], "unique_words") #tf.unique(tf.reshape(self._sliced_inputs, [-1]))
                self._word_idx = tf.placeholder(tf.int64, [None], "word_idx")
                self._new_inputs = tf.reshape(self._word_idx, tf.shape(self._sliced_inputs))

                chars = tf.nn.embedding_lookup(self._word_to_chars, self.unique_words)
                wl = tf.nn.embedding_lookup(self._word_lengths, self.unique_words)
                max_word_length = tf.cast(tf.reduce_max(wl), tf.int32)
                chars = tf.slice(chars, [0, 0], tf.pack([-1, max_word_length]))

                embedded_chars = tf.nn.embedding_lookup(self.char_embedding_matrix, chars)
                #embedded_chars_reshaped = tf.reshape(embedded_chars, tf.pack([-1, max_word_length, 4 *  self.size]))
                with tf.device(self._device):
                    with tf.variable_scope("conv"):
                        # [B, T, S]
                        filter = tf.get_variable("filter", [conv_width*self.size, self.size])
                        filter_reshaped = tf.reshape(filter, [conv_width, self.size, self.size])
                        # [B, T, S]
                        conv_out = tf.nn.conv1d(embedded_chars, filter_reshaped, 1, "SAME")
                        conv_mask = tf.expand_dims(tfutil.mask_for_lengths(self._word_lengths - pad_right,
                                                                           max_length=max_word_length), 2)
                        conv_out = conv_out + conv_mask

                    self.unique_embedded_words = tf.reduce_max(conv_out, [1])

                    embedded_words = tf.gather(self.unique_embedded_words, self._word_idx)
                    self._embedded_words = tf.reshape(embedded_words, tf.pack([-1, self.max_length, self.size]))

        self._train_variables = [p for p in tf.trainable_variables() if self.name+"/embeddings" in p.name]

    @property
    def max_length(self):
        return self._max_length

    @property
    def output(self):
        """
        :return: output embedding of this layer
        """
        return self._embedded_words

    def get_feed_dict(self, inputs, seq_lengths):
        unique_ids = dict()
        new_inputs = list()
        unique_word_idxs = []
        max_l = max(seq_lengths)
        for i in range(len(inputs)):
            for j in range(max_l):
                k = inputs[i][j]
                if k not in unique_ids:
                    unique_ids[k] = len(unique_ids)
                    unique_word_idxs.append(k)
                new_inputs.append(unique_ids[k])

        feed_dict = super().get_feed_dict(inputs, seq_lengths)

        feed_dict[self._word_to_chars] = self._word_to_chars_arr[unique_word_idxs]
        feed_dict[self._word_lengths] = self._word_lengths_arr[unique_word_idxs]
        feed_dict[self.unique_words] = list(range(len(unique_ids)))
        feed_dict[self._word_idx] = new_inputs
        return feed_dict

    @property
    def embedded_words(self):
        return self._embedded_words

    @property
    def word_embedder(self):
        return self

    @property
    def word_embeddings(self):
        return self.unique_embedded_words

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def inputs(self):
        return self._inputs

    @property
    def sliced_inputs(self):
        return self._new_inputs

    @property
    def train_variables(self):
        return self._train_variables

    def clone(self, inputs=None, seq_lengths=None, **kwargs):
        return CharWordEmbedder(self.size, self.vocab, self._device, self.name, inputs=inputs,
                                seq_lengths=seq_lengths, reuse=True)

    def get_config(self):
        config = dict()
        config["size"] = self.size
        config["vocab_size"] = self.vocab_size
        config["vocab"] = self.vocab
        config["name"] = self.name
        config["type"] = "charword"

        return config

    @staticmethod
    def create_from_config(config, device, reuse=False, inputs=None, seq_lengths=None):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """
        # todo: load parameters of the model
        # todo: dump config dictionary as json
        autoreader = CharWordEmbedder(
            config["size"],
            config["vocab"],
            device=device,
            name=config.get("name", "AutoReader"),
            reuse=reuse,
            inputs=inputs,
            seq_lengths=seq_lengths
        )

        return autoreader


class StackedEmbedder(Embedder):

    def __init__(self, underlying_embedder, size, dropout=0.0, name="Embedder", reuse=False):
        self.underlying = underlying_embedder
        self._dropout = dropout
        Embedder.__init__(self, size, underlying_embedder.vocab_size, underlying_embedder.vocab, name, reuse)

    def _init(self):
        if self._dropout > 0.0:
            self.keep_prob = tf.get_variable("keep_prob", [], initializer=tf.constant_initializer(1.0-self._dropout),
                                             trainable=False)
            self._keep_prob_placeholder = tf.placeholder(tf.float32, name="keep_prob_placeholder")
            self._keep_prob_assign = tf.assign(self.keep_prob, self._keep_prob_placeholder)

    def set_train(self, sess):
        if hasattr(self, "keep_prob"):
            sess.run(self.keep_prob.initializer)

    def set_eval(self, sess):
        if hasattr(self, "keep_prob"):
            sess.run(self._keep_prob_assign, feed_dict={self._keep_prob_placeholder: 1.0})

    def get_feed_dict(self, inputs, seq_lengths):
        return self.underlying.get_feed_dict(inputs, seq_lengths)

    @property
    def max_length(self):
        return self.underlying.max_length

    @property
    def embedded_words(self):
        return self.underlying.embedded_words

    @property
    def embedded_words_norm(self):
        return self.underlying.embedded_words_norm

    @property
    def word_embeddings(self):
        return self.underlying.word_embeddings

    @property
    def word_embedder(self):
        return self.underlying.word_embedder

    @property
    def seq_lengths(self):
        return self.underlying.seq_lengths

    @property
    def inputs(self):
        return self.underlying.inputs

    @property
    def sliced_inputs(self):
        return self.underlying.sliced_inputs

    @property
    def batch_size(self):
        return self.underlying.batch_size

    def get_config(self):
        config = dict()
        config["underlying"] = self.underlying.get_config()
        return config


class ConcatEmbedder(Embedder):

    def __init__(self, embedders):
        self.embedders = embedders
        self.all_word_embedders = all(isinstance(e, WordEmbedder) for e in embedders)
        if not self.all_word_embedders:
            assert all(e.word_embedder is embedders[0].word_embedder for e in embedders), "embedders must share word embedders"
            self._word_embedder = embedders[0].word_embedder
        else:
            self._word_embedder = self

        super().__init__(sum(e.size for e in embedders), -1, embedders[0].vocab, "_".join(e.name for e in embedders))

    def _init(self):

        self._train_variables = list()
        for e in self.embedders:
            self._train_variables.extend(e.train_variables)

        self._embedded_words = tf.concat(2, [e.embedded_words for e in self.embedders])
        if self.all_word_embedders:
            self._output = self._embedded_words
        else:
            self._output = tf.concat(2, [e.output for e in self.embedders])

    @property
    def max_length(self):
        return self.embedders[0].max_length

    @property
    def output(self):
        """
        :return: output embedding of this layer
        """
        return self._embedded_words

    def get_feed_dict(self, inputs, seq_lengths):
        feed_dict = dict()
        for e in self.embedders:
            feed_dict.update(e.get_feed_dict(inputs, seq_lengths))
        return feed_dict

    @property
    def embedded_words(self):
        return self._embedded_words

    @property
    def word_embedder(self):
        return self._word_embedder

    @property
    def word_embeddings(self):
        return self.embedders[0].word_embeddings

    @property
    def seq_lengths(self):
        return self.embedders[0].seq_lengths

    @property
    def batch_size(self):
        return self.embedders[0].batch_size

    @property
    def inputs(self):
        return self.embedders[0].inputs

    @property
    def sliced_inputs(self):
        return self.embedders[0].sliced_inputs

    @property
    def train_variables(self):
        return self._train_variables

    def clone(self, **kwargs):
        first_clone = self.embedders[0].clone()

        return ConcatEmbedder([first_clone] + [e.clone(inputs=first_clone.inputs, seq_lengths=first_clone.seq_lengths)
                                               for e in self.embedders[1:]])

    def get_config(self):
        config = dict()
        config["embedders"] = [e.get_config() for e in self.embedders]
        return config

    @staticmethod
    def create_from_config(config, devices, dropout=0.0, inputs=None, seq_lengths=None, reuse=False, **kwargs):
        """
        :param config: dictionary of parameters for creating an embedder
        :return:
        """
        from biomedical_qa.models import model_from_config
        first_embedder = model_from_config(config["embedders"][0], devices, dropout, inputs, seq_lengths, reuse)
        inputs = first_embedder.inputs
        seq_lengths = first_embedder.seq_lengths
        embedder = ConcatEmbedder([first_embedder] +
                                  [model_from_config(ce, devices, dropout, inputs, seq_lengths, reuse)
                                   for ce in config["embedders"][1:]])
        return embedder
