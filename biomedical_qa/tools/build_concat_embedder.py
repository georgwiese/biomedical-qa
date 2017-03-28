"""Concatenates two existing embedders."""

import tensorflow as tf
import os
import pickle

from biomedical_qa.models import model_from_config

from biomedical_qa.models.embedder import ConcatEmbedder

tf.app.flags.DEFINE_string('embedder1_config', None, 'Config file fo the first embedder')
tf.app.flags.DEFINE_string('embedder1_model', None, 'TF Model file of the first embedder')
tf.app.flags.DEFINE_string('embedder2_config', None, 'Config file fo the second embedder')
tf.app.flags.DEFINE_string('embedder2_model', None, 'TF Model file of the second embedder')
tf.app.flags.DEFINE_string('out_dir', None, 'path to saved model and config')
tf.app.flags.DEFINE_string('name', 'ConcatEmbedder', 'Name of the model')

FLAGS = tf.app.flags.FLAGS

def load_embedding_model(sess, config_file, model_file):
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    embedder = model_from_config(config)
    embedder.model_saver.restore(sess, model_file)
    return embedder

with tf.Session() as sess:

    print("Loading Embedders")
    embedder1 = load_embedding_model(sess, FLAGS.embedder1_config,
                                     FLAGS.embedder1_model)
    embedder2 = load_embedding_model(sess, FLAGS.embedder2_config,
                                     FLAGS.embedder2_model)

    print("Building Concat Embedder")
    embedder = ConcatEmbedder([embedder1, embedder2])

    sess.run(tf.global_variables_initializer())
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    config = embedder.get_config()

    print("Write Config")
    with open(os.path.join(FLAGS.out_dir, "config.pickle"), 'wb') as f:
        pickle.dump(config, f)

    print("Write Model")
    embedder.model_saver.save(sess, os.path.join(FLAGS.out_dir, "model.tf"), write_meta_graph=False)
    print("Done")
