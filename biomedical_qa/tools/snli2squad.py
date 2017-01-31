import os
import sys
import json
import tensorflow as tf


tf.app.flags.DEFINE_string('snli_file', None, 'Path to the SNLI JSONL file.')
tf.app.flags.DEFINE_string('out_file', None, 'Path to the output JSON file.')

FLAGS = tf.app.flags.FLAGS


def main():

    with open(FLAGS.snli_file) as f:
        lines = f.readlines()

    sentence_pairs = [json.loads(line) for line in lines]

    paragraphs = []
    for sentence_pair in sentence_pairs:

        if sentence_pair["gold_label"] == "neutral":
            # Skip
            continue

        paragraphs.append({
            "context_original_capitalization": sentence_pair["sentence1"],
            "context": sentence_pair["sentence1"].lower(),
            "qas": [{
                "id": sentence_pair["pairID"],
                "question_type": "yesno",
                "question": sentence_pair["sentence2"].lower(),
                "answer_is_yes": sentence_pair["gold_label"] == "entailment",
            }]
        })

    data = {
        "data": [{
            "paragraphs": paragraphs
        }]
    }

    with open(FLAGS.out_file, "w") as f:
        json.dump(data, f, indent=2)

    print("Done. Used %d pairs." % len(paragraphs))

main()
