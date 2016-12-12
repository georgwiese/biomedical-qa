import pickle
import os

import tensorflow as tf

from biomedical_qa.models import model_from_config


def initialize_model(model_config_file, model_weights_file, devices, beam_size):

    print("Loading Model...")
    with open(model_config_file, 'rb') as f:
        model_config = pickle.load(f)
    model = model_from_config(model_config, devices)

    print("Restoring Weights...")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if model_weights_file is None:
        train_dir = os.path.dirname(model_config_file)
        model_weights_file = tf.train.latest_checkpoint(train_dir)
        print("Using weights: %s" % model_weights_file)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    model.model_saver.restore(sess, model_weights_file)
    model.set_eval(sess)
    model.set_beam_size(sess, beam_size)

    return model, sess
