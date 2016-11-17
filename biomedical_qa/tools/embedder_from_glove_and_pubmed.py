import tensorflow as tf
import os
import numpy as np
import pickle

from biomedical_qa.models.embedder import ConstantWordEmbedder
from biomedical_qa.util import load_vocab

tf.app.flags.DEFINE_string('glove_embedding_file', None, 'path to embeddings')
tf.app.flags.DEFINE_string('pubmed_vocab_file', None, 'path to vocab file')
tf.app.flags.DEFINE_string('pubmed_embedding_file', None, 'path to embeddings')
tf.app.flags.DEFINE_string('out_dir', None, 'path to saved model and config')
tf.app.flags.DEFINE_string('name', 'GlovePubmedEmbedder', 'Name of the model')

FLAGS = tf.app.flags.FLAGS


vocab = {"<S>": 0, "</S>": 1, "<UNK>": 2}

print("Collecting vocab...")
with open(FLAGS.glove_embedding_file, "rb") as f:
    for l in f:
        split = l.decode("utf-8").rstrip().split(" ", 1)
        i = vocab.get(split[0], len(vocab))
        if i == len(vocab):
            vocab[split[0]] = i

print("Vocab. size after glove: %d" % len(vocab))

with open(FLAGS.pubmed_vocab_file, "rb") as f:
    pubmed_word_mapping = []
    for l in f:
        word = l.decode("utf-8").strip()
        pubmed_word_mapping.append(word)
        if not word in vocab:
            vocab[word] = len(vocab)

print("Final Vocab. size: %d" % len(vocab))

with open(FLAGS.glove_embedding_file, "rb") as f:
    print("Filling Glove embeddings...")
    glove_embeddings, glove_embedding_size = None, None
    for l in f:
        split = l.decode("utf-8").rstrip().split(" ")
        i = vocab.get(split[0], len(vocab))
        if glove_embedding_size is None:
            glove_embedding_size = len(split) - 1
            glove_embeddings = np.zeros([len(vocab), glove_embedding_size], np.float32)
        if i < len(vocab):
            glove_embeddings[i] = [float(v) for v in split[1:]]

with open(FLAGS.pubmed_embedding_file, "rb") as f:
    print("Filling Pubmed embeddings...")
    pubmed_embeddings, pubmed_embedding_size = None, None
    for i, l in enumerate(f):
        word = pubmed_word_mapping[i]
        split = l.decode("utf-8").rstrip().split(" ")
        if pubmed_embedding_size is None:
            pubmed_embedding_size = len(split)
            pubmed_embeddings = np.zeros([len(vocab), pubmed_embedding_size], np.float32)
        pubmed_embeddings[vocab.get(word)] = [float(v) for v in split]

print("Concatenating both embeddings...")
embeddings = np.concatenate((glove_embeddings, pubmed_embeddings), axis=1)
print("Final embeddings shape:", embeddings.shape)

with tf.Session() as sess:
    embedder = ConstantWordEmbedder(embeddings.shape[1], vocab, 2, embeddings,
                                    name=FLAGS.name)
    sess.run(tf.initialize_all_variables())
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    config = embedder.get_config()
    with open(os.path.join(FLAGS.out_dir, "config.pickle"), 'wb') as f:
        pickle.dump(config, f, protocol=2)

    embedder.model_saver.save(sess, os.path.join(FLAGS.out_dir, "model.tf"), write_meta_graph=False)
    print("Done")
