import json
import os
import pickle

import sys
import tensorflow as tf

from biomedical_qa.models import model_from_config
from biomedical_qa.sampling.squad import SQuADSampler

tf.app.flags.DEFINE_string('dir', None, 'Directory containing file.')
tf.app.flags.DEFINE_string('file', None, 'Filename of dataset.')
tf.app.flags.DEFINE_string('model_config', None, 'Config of model.')
tf.app.flags.DEFINE_string('model_path', None, 'path of model.')
tf.app.flags.DEFINE_string('device', "/cpu:0", 'Device to use.')
tf.app.flags.DEFINE_string('out', "results.json", 'Result file path.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')

FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

results = {}
with tf.Session(config=config) as sess:
    with open(FLAGS.model_config, "rb") as f:
        config = pickle.load(f)

    model = model_from_config(config, [FLAGS.device])

    sess.run(tf.global_variables_initializer())
    model.model_saver.restore(sess, FLAGS.model_path)
    model.set_eval(sess)

    sampler = SQuADSampler(FLAGS.dir, [FLAGS.file], FLAGS.batch_size, model.embedder.vocab)

    with open(os.path.join(FLAGS.dir, FLAGS.file)) as dataset_file:
        squad = json.load(dataset_file)["data"]

    question2real_context = {}
    for article in squad:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question2real_context[qa["id"]] = context
    counter = 0
    while sampler.epoch == 0:
        batch = sampler.get_batch()
        context_indices, starts, ends = model.run(sess,
                                                  [model.predicted_context_indices,
                                                   model.predicted_answer_starts,
                                                   model.predicted_answer_ends],
                                                  batch)
        for i, qa_setting in enumerate(batch):
            context_index = context_indices[i]
            start = starts[i]
            end = ends[i]

            char_offsets = sampler.char_offsets[qa_setting.id]
            context = question2real_context[qa_setting.id]

            char_start = char_offsets[(context_index, start)]
            char_end = char_offsets[(context_index, end + 1)] \
                       if (context_index, end + 1) in char_offsets \
                       else len(context)

            answer = context[char_start:char_end]
            answer = answer.strip()

            results[qa_setting.id] = answer
        counter += len(batch)
        sys.stdout.write("\r%d" % counter)
        sys.stdout.flush()

with open(FLAGS.out, "w") as out_file:
    json.dump(results, out_file)
