import json
import os
import pickle
import math
import tensorflow as tf
from nltk import RegexpTokenizer

from biomedical_qa.models import model_from_config
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer

tf.app.flags.DEFINE_string('eval_data', None, 'Path to the SQuAD JSON file.')
tf.app.flags.DEFINE_string('model_config', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")

tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")
tf.app.flags.DEFINE_integer("subsample", -1, "Number of samples to do the evaluation on.")

tf.app.flags.DEFINE_boolean("squad_evaluation", False, "If true, measures F1 and exact match acc on answer spans.")
tf.app.flags.DEFINE_boolean("bioasq_evaluation", False, "If true, runs BioASQ evaluation measures.")
tf.app.flags.DEFINE_boolean("verbose", False, "If true, prints correct and given answers.")

FLAGS = tf.app.flags.FLAGS

def bioasq_evaluation(sampler, sess, model):
    with open(FLAGS.eval_data) as f:
        paragraphs = json.load(f)["data"][0]["paragraphs"]

    assert paragraphs[0]["qas"][0]["original_answers"] is not None, \
        "Questions must be augmented with original_answers to perform BioASQ evaluation."
    assert paragraphs[0]["qas"][0]["question_type"] is not None, \
        "Questions must be augmented with question_type to perform BioASQ evaluation."

    # Assuming one question per paragraph
    paragraphs_by_id = {p["qas"][0]["id"] : p for p in paragraphs}
    count = len(paragraphs)
    print("  (Questions: %d, using %d)" %
          (count, FLAGS.subsample if FLAGS.subsample > 0 else count))

    factoid_correct, factoid_total = 0, 0
    list_correct, list_total = 0, 0

    sampler.reset()
    epoch = sampler.epoch

    while sampler.epoch == epoch:

        batch = sampler.get_batch()
        paragraphs_batch = [paragraphs_by_id[q.id] for q in batch]
        assert len(batch) == len(paragraphs_batch)

        correct_answers = [p["qas"][0]["original_answers"] for p in paragraphs_batch]
        question_types = [p["qas"][0]["question_type"] for p in paragraphs_batch]
        contexts = [p["context"] for p in paragraphs_batch]

        starts, ends = sess.run([model.predicted_answer_starts,
                                 model.predicted_answer_ends],
                                model.get_feed_dict(batch))

        assert len(starts) == len(batch)
        assert len(ends) == len(batch)

        for i in range(len(batch)):

            tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
            answer_tokens = tokenizer.tokenize(contexts[i])[starts[i] : ends[i] + 1]
            answer = " ".join(answer_tokens)

            if FLAGS.verbose:
                print("-------------")
                print("  Given: ", answer)
                print("  Correct: ", correct_answers[i])

            if question_types[i] == "factoid":
                # TODO: Implement MRR once ranked answers are implemented
                factoid_total += 1
                for correct_answer in correct_answers[i][0]:
                    if correct_answer.lower() == answer.lower():
                        if FLAGS.verbose:
                            print("  Correct!")
                        factoid_correct += 1
                        break

            if question_types[i] == "list":
                # TODO: Evaluate F1 once multiple answers are implemented
                list_total += 1
                for answer_option in correct_answers[i]:
                    for correct_answer in answer_option:
                        if correct_answer.lower() == answer.lower():
                            if FLAGS.verbose:
                                print("  Correct!")
                            factoid_correct += 1
                            break

    print("Factoid Summary: %d / %d" % (factoid_correct, factoid_total))
    print("List Summary: %d / %d" % (list_correct, list_total))

def main():
    devices = FLAGS.devices.split(",")

    print("Loading Model...")
    with open(FLAGS.model_config, 'rb') as f:
        model_config = pickle.load(f)
    model = model_from_config(model_config, devices)

    vocab = model.embedder.vocab

    print("Restoring Weights...")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.model_saver.restore(sess, FLAGS.model_weights)
        model.set_eval(sess)

        print("Initializing Sampler & Trainer...")
        data_dir = os.path.dirname(FLAGS.eval_data)
        data_filename = os.path.basename(FLAGS.eval_data)
        instances = FLAGS.subsample if FLAGS.subsample > 0 else None
        sampler = SQuADSampler(data_dir, [data_filename], FLAGS.batch_size,
                               vocab, instances_per_epoch=instances)
        trainer = ExtractionQATrainer(0, model, devices[0])

        if FLAGS.squad_evaluation:
            print("Running SQuAD Evaluation...")
            trainer.eval(sess, sampler, verbose=True)

        if FLAGS.bioasq_evaluation:
            print("Running BioASQ Evaluation...")
            bioasq_evaluation(sampler, sess, model)

main()
