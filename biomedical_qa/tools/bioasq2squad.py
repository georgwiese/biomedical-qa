import os
import sys
import json
import numpy as np
import tensorflow as tf

from biomedical_qa.data.bioasq_squad_builder import BioAsqSquadBuilder


tf.app.flags.DEFINE_string('bioasq_file', None, 'Path to the BioASQ JSON file.')
tf.app.flags.DEFINE_string('out_dir', None, 'Path to the output directory.')
tf.app.flags.DEFINE_string('types', "factoid,list", 'Comma-separated list of question types.')
tf.app.flags.DEFINE_float('train_fraction', 0.8, 'Fraction of training data.')
tf.app.flags.DEFINE_integer('context_token_limit', -1, 'Maximum number of context tokens. Max on BioASQ is ~4300 and on SQuAD ~700.')

FLAGS = tf.app.flags.FLAGS

np.random.seed(1234)


def convert_to_squad(bioasq_file_path, out_dir):

    with open(bioasq_file_path) as json_file:
        data = json.load(json_file)

    squad_builder = BioAsqSquadBuilder(data, FLAGS.context_token_limit,
                                       types=FLAGS.types.split(","))
    squad_builder.build()
    stats = squad_builder.get_stats()

    paragraphs = squad_builder.get_paragraphs()
    train_paragraphs, dev_paragraphs = split_paragraphs(paragraphs)

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train.json"), "w") as out_file:
       name = bioasq_file_path + " - train"
       result = squad_builder.build_result_object(name, train_paragraphs)
       json.dump(result, out_file, indent=2)

    with open(os.path.join(out_dir, "dev.json"), "w") as out_file:
       name = bioasq_file_path + " - dev"
       result = squad_builder.build_result_object(name, dev_paragraphs)
       json.dump(result, out_file, indent=2)

    print("Done. Extracted %d questions (%d train & %d dev)"
          % (len(paragraphs), len(train_paragraphs), len(dev_paragraphs)))

    num_factoid_original = len([q for q in data["questions"] if q["type"] == "factoid"])
    num_list_original = len([q for q in data["questions"] if q["type"] == "list"])
    num_factoid = len([p for p in paragraphs
                     if p["qas"][0]["question_type"] == "factoid"])
    num_list = len([p for p in paragraphs
                    if p["qas"][0]["question_type"] == "list"])

    print("Max Context Length: %d tokens" % stats["max_context_length"])
    print("Contexts truncated: %d" % stats["contexts_truncated"])
    print("Used Factoid questions: %d / %d" % (num_factoid, num_factoid_original))
    print("Used List questions: %d / %d" % (num_list, num_list_original))


def split_paragraphs(paragraphs):

    dataset_size = len(paragraphs)
    train_size = int(FLAGS.train_fraction * dataset_size)

    train_indices = np.random.choice(dataset_size, train_size, replace=False)

    train_paragraphs, dev_paragraphs = [], []
    for i in np.random.permutation(dataset_size):
        if i in train_indices:
            train_paragraphs += [paragraphs[i]]
        else:
            dev_paragraphs += [paragraphs[i]]

    return train_paragraphs, dev_paragraphs


if __name__ == "__main__":

    convert_to_squad(FLAGS.bioasq_file, FLAGS.out_dir)
