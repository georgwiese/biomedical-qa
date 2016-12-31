"""Split a BioASQ into dev and train, given a dev ID file."""

import os
import json
import tensorflow as tf

tf.app.flags.DEFINE_string('bioasq_file', None, 'Path to the BioASQ JSON file.')
tf.app.flags.DEFINE_string('out_dir', None, 'Path to the output directory.')
tf.app.flags.DEFINE_string('types', "factoid,list", 'Comma-separated list of question types.')
tf.app.flags.DEFINE_string('dev_id_file', None, 'Path to a text file with dev question IDs, one ID per line.')

FLAGS = tf.app.flags.FLAGS


types = FLAGS.types.split(",")


def split_bioasq(bioasq_file_path, out_dir, dev_id_file):

    with open(dev_id_file) as f:
        dev_ids = set(f.read().split("\n"))

    with open(bioasq_file_path) as f:
        all_questions = json.load(f)["questions"]

    dev_questions = []
    train_questions = []
    for question in all_questions:
        if question["type"] in types:

            if question["id"] in dev_ids:
                dev_questions.append(question)
            else:
                train_questions.append(question)

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "dev.json"), "w") as f:
        json.dump({"questions": dev_questions}, f, indent=2)
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump({"questions": train_questions}, f, indent=2)


if __name__ == "__main__":

    split_bioasq(FLAGS.bioasq_file, FLAGS.out_dir, FLAGS.dev_id_file)
