"""Given two SQuAD JSON files, save the concatenation."""

import json
import os
import tensorflow as tf

tf.app.flags.DEFINE_string('ds1', None, 'SQuAD JSON file 1')
tf.app.flags.DEFINE_string('ds2', None, 'SQuAD JSON file 2')
tf.app.flags.DEFINE_string('out_file', None, 'Out file path')
tf.app.flags.DEFINE_integer('repeat1', 1, 'Number of repetitions of ds1')
tf.app.flags.DEFINE_integer('repeat2', 1, 'Number of repetitions of ds2')

FLAGS = tf.app.flags.FLAGS


def concat_jsons(file1, file2, out_file, repeat1, repeat2):

    with open(file1) as f:
        ds1 = json.load(f)
    with open(file2) as f:
        ds2 = json.load(f)

    ds_merged = merge(ds1, ds2, repeat1, repeat2)

    print("Length ds1: %d" % count_questions(ds1["data"]))
    print("Length ds2: %d" % count_questions(ds2["data"]))
    print("Length merged: %d" % count_questions(ds_merged["data"]))

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ds_merged, f)


def count_questions(data):
    return len([q
                for d in data
                for p in d["paragraphs"]
                for q in p["qas"]])


def merge(ds1, ds2, repeat1, repeat2):

    return {
        "version": "1.0",
        "data": get_data(ds1, repeat1) + get_data(ds2, repeat2)
    }


def get_data(ds, repeat):

    result = ds["data"].copy()
    for _ in range(repeat - 1):
        result += ds["data"]

    return result


if __name__ == "__main__":
    concat_jsons(FLAGS.ds1, FLAGS.ds2, FLAGS.out_file, FLAGS.repeat1, FLAGS.repeat2)
