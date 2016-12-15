import pickle
import os
import logging

import tensorflow as tf

from biomedical_qa.sampling.squad import trfm
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


def extract_answer(context, answer_span):

    token_start, token_end = answer_span
    _, char_offsets = trfm(context)

    if token_start == len(char_offsets):
        logging.warning("Null word selected! Using first token instead.")
        token_start = 0

    char_start = char_offsets[token_start]

    if token_end == len(char_offsets):
        logging.warning("Null word selected! Using last token instead.")
        token_end = len(char_offsets) - 1

    if token_end == len(char_offsets) - 1:
        # Span continues until the very end
        return context[char_start:]
    else:
        # token_end is inclusive
        char_end = char_offsets[token_end + 1]
        return context[char_start:char_end].strip()


def extract_answers(context, starts, ends):

    answers_set = set()
    answers = []

    assert len(starts) == len(ends)

    for i in range(len(starts)):
        answer = extract_answer(context, (starts[i], ends[i]))

        # Deduplicate
        if answer not in answers_set:
            answers_set.add(answer)
            answers.append(answer)

    return answers
