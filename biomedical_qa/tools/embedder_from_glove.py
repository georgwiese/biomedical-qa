import tensorflow as tf
import os
import numpy as np
import pickle

from biomedical_qa.models.embedder import WordEmbedder, ConstantWordEmbedder
from biomedical_qa.util import load_vocab

tf.app.flags.DEFINE_integer('max_vocab', -1, 'maximum size of vocabulary used')
tf.app.flags.DEFINE_string('vocab_file', None, 'tsv file in format: id\tword\tcount.')
tf.app.flags.DEFINE_string('embedding_file', None,
                           'path to embeddings')
tf.app.flags.DEFINE_string('out_dir', None, 'path to saved model and config')
tf.app.flags.DEFINE_string('name', 'WordEmbedder', 'Name of the model')

FLAGS = tf.app.flags.FLAGS


if FLAGS.vocab_file is None:
    vocab = {"<S>": 0, "</S>": 1, "<UNK>": 2}
else:
    vocab, _, _ = load_vocab(FLAGS.vocab_file)

with open(FLAGS.embedding_file, "rb") as f:
    print("Collecting vocab...")
    for l in f:
        split = l.decode("utf-8").rstrip().split(" ", 1)
        i = vocab.get(split[0], len(vocab))
        if i >= FLAGS.max_vocab and FLAGS.max_vocab > 0:
            break
        if i == len(vocab):
            vocab[split[0]] = i

    print("Vocab. size: %d" % len(vocab))

with open(FLAGS.embedding_file, "rb") as f:
    print("Filling embeddings...")
    embeddings, embedding_size = None, None
    for l in f:
        split = l.decode("utf-8").rstrip().split(" ")
        i = vocab.get(split[0], len(vocab))
        if embedding_size is None:
            embedding_size = len(split)-1
            embeddings = np.zeros([len(vocab), embedding_size], np.float32)
        if i < len(vocab):
            embeddings[i] = [float(v) for v in split[1:]]

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
