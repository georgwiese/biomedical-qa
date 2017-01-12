"""Split a BioASQ into dev and train, given a dev ID file."""

import os
import json
import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_string('bioasq_file', None, 'Path to the BioASQ JSON file.')
tf.app.flags.DEFINE_string('out_dir', None, 'Path to the output directory.')
tf.app.flags.DEFINE_string('dev_path_types', "factoid,list", 'Comma-separated list of question types.')
tf.app.flags.DEFINE_string('random_assign_types', "yesno", 'Comma-separated list of question types.')
tf.app.flags.DEFINE_float('random_assign_train_fraction', 0.8, 'Fraction of train data.')
tf.app.flags.DEFINE_string('dev_id_file', None, 'Path to a text file with dev question IDs, one ID per line.')

FLAGS = tf.app.flags.FLAGS

np.random.seed(1234)


dev_path_types = FLAGS.dev_path_types.split(",")
random_assign_types = FLAGS.random_assign_types.split(",")


def split_bioasq(bioasq_file_path, out_dir, dev_id_file):

    with open(dev_id_file) as f:
        dev_ids = set(f.read().split("\n"))

    with open(bioasq_file_path) as f:
        all_questions = json.load(f)["questions"]

    dev_questions = []
    train_questions = []
    for question in all_questions:
        if question["type"] in dev_path_types:

            if question["id"] in dev_ids:
                dev_questions.append(question)
            else:
                train_questions.append(question)

        elif question["type"] in random_assign_types:

            probs = [FLAGS.random_assign_train_fraction, 1 - FLAGS.random_assign_train_fraction]
            if np.random.choice([True, False], p=probs):
                train_questions.append(question)
            else:
                dev_questions.append(question)


    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "dev.json"), "w") as f:
        json.dump({"questions": dev_questions}, f, indent=2)
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump({"questions": train_questions}, f, indent=2)


if __name__ == "__main__":

    split_bioasq(FLAGS.bioasq_file, FLAGS.out_dir, FLAGS.dev_id_file)
