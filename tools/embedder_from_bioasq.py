import tensorflow as tf
import os
import numpy as np
import pickle

from quebap.projects.autoread import WordEmbedder
from quebap.projects.autoread.models.embedder import ConstantWordEmbedder
from quebap.projects.autoread.util import *

tf.app.flags.DEFINE_string('vocab_file', None, 'list file of words in vocab.')
tf.app.flags.DEFINE_string('embedding_file', None, 'path to embeddings')
tf.app.flags.DEFINE_string('out_dir', None, 'path to saved model and config')
tf.app.flags.DEFINE_string('name', 'WordEmbedder', 'Name of the model')

FLAGS = tf.app.flags.FLAGS


vocab = {"<S>": 0, "</S>": 1, "<UNK>": 2}
duplicate = set()

with open(FLAGS.vocab_file, "rb") as f:
    i = 0
    for l in f:
        word = l.decode("utf-8").strip()
        if word in vocab:
            duplicate.add(i)
        else:
            vocab[word] = len(vocab)
        i += 1
    print("Vocab. size: %d" % len(vocab))

with open(FLAGS.embedding_file, "rb") as f:
    print("Filling embeddings...")
    embeddings, embedding_size = None, None
    i = 0
    k = 3
    for l in f:
        if i not in duplicate:
            split = l.decode("utf-8").rstrip().split(" ")
            if embedding_size is None:
                embedding_size = len(split)
                embeddings = np.zeros([len(vocab), embedding_size], np.float32)
            embeddings[k] = [float(v) for v in split]
            k += 1
        i += 1

with tf.Session() as sess:
    embedder = ConstantWordEmbedder(embedding_size, vocab, 2, embeddings, name=FLAGS.name)
    sess.run(tf.initialize_all_variables())
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    config = embedder.get_config()
    with open(os.path.join(FLAGS.out_dir, "config.pickle"), 'wb') as f:
        pickle.dump(config, f)

    embedder.model_saver.save(sess, os.path.join(FLAGS.out_dir, "model.tf"), write_meta_graph=False)
    print("Done")
