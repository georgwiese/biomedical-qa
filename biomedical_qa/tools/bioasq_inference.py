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
    """Returns a <question id> -> <answers array> map."""

    pass


def insert_answers(bioasq_json, answers):
    """Inserts answers into bioasq_json from a <question id> -> <answers array>."""

    return bioasq_json


if __name__ == "__main__":

    devices = FLAGS.devices.split(",")

    model, sess = util.initialize_model(FLAGS.model_config, FLAGS.model_weights,
                                        devices, FLAGS.beam_size)

    # Build sampler from dataset JSON
    bioasq_json, squad_json = load_dataset(FLAGS.bioasq_file)
    sampler = SQuADSampler(None, None, FLAGS.batch_size, model.embedder.vocab,
                           shuffle=False, dataset_json=squad_json)

    answers = predict_answers(sess, model, sampler)
    bioasq_json = insert_answers(bioasq_json, answers)

    os.makedirs(os.path.dirname(FLAGS.out_file), exist_ok=True)
    with open(FLAGS.out_file, "w") as f:
        json.dump(bioasq_json, f, indent=2)
