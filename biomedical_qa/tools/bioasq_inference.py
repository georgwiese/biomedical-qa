import json
import os

import tensorflow as tf

from biomedical_qa.data.bioasq_squad_builder import BioAsqSquadBuilder
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.tools import util

tf.app.flags.DEFINE_string('bioasq_file', None, 'Path to the BioASQ JSON file.')
tf.app.flags.DEFINE_string('out_file', None, 'Path to the output file.')
tf.app.flags.DEFINE_string('model_config', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")

tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")

tf.app.flags.DEFINE_integer("beam_size", 5, "Beam size used for decoding.")

FLAGS = tf.app.flags.FLAGS


def load_dataset(path):

    with open(path) as f:
        bioasq_json = json.load(f)

    squad_json = BioAsqSquadBuilder(bioasq_json, include_answers=False) \
                    .build() \
                    .get_reult_object("BioASQ")

    return bioasq_json, squad_json


def predict_answers(sess, model, sampler):
    """Returns a <question id> -> [(<start token>, <end token>), ...] map."""

    sampler.reset()
    start_epoch = sampler.epoch
    answers = {}

    while sampler.epoch == start_epoch:
        batch = sampler.get_batch()

        top_starts, top_ends = sess.run([model.top_starts,
                                         model.top_ends],
                                        model.get_feed_dict(batch))

        for i in range(len(batch)):
            question = batch[i]


            answers[question.id] = [(top_starts[i, k], top_ends[i, k])
                                    for k in range(5)]

    return answers


def insert_answers(bioasq_json, answers, contexts, sampler):
    """Inserts answers into bioasq_json from a
    <question id> -> [(<start token>, <end token>), ...]."""

    questions = []

    for question in bioasq_json["questions"]:
        q_id = question["id"]
        if q_id in answers:
            question["exact_answer"] = [[extract_answer(contexts[q_id], answer_span, sampler)]
                                        for answer_span in answers[q_id]]
            questions.append(question)

    return {"questions": questions}


def extract_answer(context, answer_span, sampler):

    token_start, token_end = answer_span
    _, char_offsets = sampler.trfm(context)

    char_start = char_offsets[token_start]

    if token_end == len(char_offsets) - 1:
        # Span continues until the very end
        return context[char_start:]
    else:
        # token_end is inclusive
        char_end = char_offsets[token_end + 1]
        return context[char_start:char_end].strip()


if __name__ == "__main__":

    devices = FLAGS.devices.split(",")

    model, sess = util.initialize_model(FLAGS.model_config, FLAGS.model_weights,
                                        devices, FLAGS.beam_size)

    # Build sampler from dataset JSON
    bioasq_json, squad_json = load_dataset(FLAGS.bioasq_file)
    sampler = SQuADSampler(None, None, FLAGS.batch_size, model.embedder.vocab,
                           shuffle=False, dataset_json=squad_json)

    contexts = {p["qas"][0]["id"] : p["context"]
                for p in squad_json["data"][0]["paragraphs"]}
    answers = predict_answers(sess, model, sampler)
    bioasq_json = insert_answers(bioasq_json, answers, contexts, sampler)

    os.makedirs(os.path.dirname(FLAGS.out_file), exist_ok=True)
    with open(FLAGS.out_file, "w") as f:
        json.dump(bioasq_json, f, indent=2)
