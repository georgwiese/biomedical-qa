import pickle
import os
import logging

import tensorflow as tf

from biomedical_qa.models import model_from_config
from biomedical_qa.models.beam_search import BeamSearchDecoder


class InferenceResult(object):

    def __init__(self, prediction, answer_strings, answer_probs, question):

        self.prediction = prediction
        self.answer_strings = answer_strings
        self.answer_probs = answer_probs
        self.question = question


    def __iter__(self):

        return zip(self.answer_strings, self.answer_probs)


def get_model(sess, model_config_file, devices, model_weights_file=None):


    print("Loading Model...")
    with open(model_config_file, 'rb') as f:
        model_config = pickle.load(f)
    model = model_from_config(model_config, devices)

    if model_weights_file is None:
        train_dir = os.path.dirname(model_config_file)
        model_weights_file = tf.train.latest_checkpoint(train_dir)
        print("Using weights: %s" % model_weights_file)

    print("Restoring Weights...")
    sess.run(tf.global_variables_initializer())
    model.model_saver.restore(sess, model_weights_file)

    return model


def get_session():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class Inferrer(object):


    def __init__(self, model, sess, beam_size):

        self.model = model
        self.sess = sess
        self.beam_size = beam_size

        # If true, each start has its own probability to allow for multiple starts
        self.unnormalized_probs = self.model.start_output_unit == "sigmoid"

        self.model.set_eval(self.sess)

        self.beam_search_decoder = BeamSearchDecoder(self.sess, self.model,
                                                     beam_size)


    def get_predictions(self, sampler):

        predictions = {}

        sampler.reset()
        epoch = sampler.epoch

        while sampler.epoch == epoch:

            batch = sampler.get_batch()

            network_predictions = self.beam_search_decoder.decode(batch)

            for i, question in enumerate(batch):
                context = question.paragraph_json["context_original_capitalization"]

                answers, answer_probs = self.extract_answers(context, network_predictions[i],
                                                             sampler.char_offsets[question.id])
                predictions[question.id] = InferenceResult(
                    network_predictions[i], answers, answer_probs, question)

        return predictions


    def extract_answers(self, context, prediction, all_char_offsets):

        answer2index = {}
        answers = []
        filtered_probs = []

        for context_index, start, end, prob in prediction:
            char_offsets = {token_index: char_offset
                            for (c_index, token_index), char_offset in all_char_offsets.items()
                            if c_index == context_index}
            answer = self.extract_answer(context, (start, end), char_offsets)

            # Deduplicate
            if answer.lower() not in answer2index:
                answer2index.update({answer.lower() : len(answers)})
                answers.append(answer)
                filtered_probs.append(prob)
            else:
                # Duplicate mentions should add their probs
                index = answer2index[answer.lower()]
                if self.unnormalized_probs:
                    filtered_probs[index] = max(filtered_probs[index], prob)
                else:
                    filtered_probs[index] += prob

        # Sort by new probability
        answers_probs = list(zip(answers, filtered_probs))
        answers_probs.sort(key=lambda x : -x[1])

        return zip(*answers_probs)


    def extract_answer(self, context, answer_span, char_offsets):

        token_start, token_end = answer_span

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
