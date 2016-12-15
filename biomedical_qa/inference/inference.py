import pickle
import os
import logging
import sys

import tensorflow as tf

from biomedical_qa.sampling.squad import trfm
from biomedical_qa.models import model_from_config


class InferenceResult(object):

    def __init__(self, starts, ends, probs, answer_strings):

        self.starts = starts
        self.ends = ends
        self.probs = probs
        self.answer_strings = answer_strings


class Inferrer(object):


    def __init__(self, model_config_file, devices, beam_size,
                 model_weights_file=None):

        print("Loading Model...")
        with open(model_config_file, 'rb') as f:
            model_config = pickle.load(f)
        self.model = model_from_config(model_config, devices)

        print("Restoring Weights...")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        if model_weights_file is None:
            train_dir = os.path.dirname(model_config_file)
            model_weights_file = tf.train.latest_checkpoint(train_dir)
            print("Using weights: %s" % model_weights_file)

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.model.model_saver.restore(self.sess, model_weights_file)
        self.model.set_eval(self.sess)
        self.model.set_beam_size(self.sess, beam_size)


    def get_predictions(self, sampler):

        predictions = {}

        sampler.reset()
        epoch = sampler.epoch

        while sampler.epoch == epoch:

            batch = sampler.get_batch()

            starts, ends, probs = self.sess.run(
                    [self.model.top_starts,
                     self.model.top_ends,
                     self.model.top_probs],
                     self.model.get_feed_dict(batch))

            for i, question in enumerate(batch):
                context = question.paragraph_json["context_original_capitalization"]


                predictions[question.id] = InferenceResult(
                    starts[i], ends[i], probs[i],
                    self.extract_answers(context, starts[i], ends[i]))

        return predictions


    def extract_answers(self, context, starts, ends):

        answers_set = set()
        answers = []

        assert len(starts) == len(ends)

        for i in range(len(starts)):
            answer = self.extract_answer(context, (starts[i], ends[i]))

            # Deduplicate
            if answer not in answers_set:
                answers_set.add(answer)
                answers.append(answer)

        return answers


    def extract_answer(self, context, answer_span):

        token_start, token_end = answer_span
        _, char_offsets = trfm(context)

        if token_start >= len(char_offsets):
            logging.warning("Null word selected! Using first token instead.")
            token_start = 0

        char_start = char_offsets[token_start]

        if token_end >= len(char_offsets):
            logging.warning("Null word selected! Using last token instead.")
            token_end = len(char_offsets) - 1

        if token_end == len(char_offsets) - 1:
            # Span continues until the very end
            return context[char_start:]
        else:
            # token_end is inclusive
            char_end = char_offsets[token_end + 1]
            return context[char_start:char_end].strip()
