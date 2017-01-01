import logging
import json
import os
import sys
import numpy as np
import tensorflow as tf

from biomedical_qa.data.bioasq_squad_builder import BioAsqSquadBuilder
from biomedical_qa.inference.inference import Inferrer
from biomedical_qa.sampling.bioasq import BioAsqSampler
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer

tf.app.flags.DEFINE_string('eval_data', None, 'Path to the SQuAD JSON file.')
tf.app.flags.DEFINE_boolean('split_contexts', False, 'Whether to split contexts on newline.')
tf.app.flags.DEFINE_string('model_config', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")

tf.app.flags.DEFINE_boolean("is_bioasq", False, "Whether the provided dataset is a BioASQ json.")
tf.app.flags.DEFINE_boolean("bioasq_include_synonyms", False, "Whether BioASQ synonyms should be included.")
tf.app.flags.DEFINE_integer("bioasq_context_token_limit", -1, "Token limit for BioASQ contexts.")

tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")
tf.app.flags.DEFINE_integer("subsample", -1, "Number of samples to do the evaluation on.")

tf.app.flags.DEFINE_integer("beam_size", 5, "Beam size used for decoding.")

tf.app.flags.DEFINE_boolean("squad_evaluation", False, "If true, measures F1 and exact match acc on answer spans.")
tf.app.flags.DEFINE_boolean("bioasq_evaluation", False, "If true, runs BioASQ evaluation measures.")
tf.app.flags.DEFINE_boolean("verbose", False, "If true, prints correct and given answers.")

FLAGS = tf.app.flags.FLAGS


def bioasq_evaluation(sampler, inferrer):

    if FLAGS.beam_size < 5:
        logging.warning("Beam size should be at least 5 in order to get 5 ranked answers.")

    questions = sampler.get_questions()

    # Assuming one question per paragraph
    count = len(questions)
    print("  (Questions: %d, using %d)" %
          (count, FLAGS.subsample if FLAGS.subsample > 0 else count))

    factoid_correct, factoid_total = 0, 0
    factoid_reciprocal_rank_sum = 0
    list_f1_sum, list_total = 0, 0

    print("  Doing predictions...")
    predictions = inferrer.get_predictions(sampler)

    for question in questions:

        prediction = predictions[question.id]

        correct_answers = question.question_json["original_answers"]
        question_type = question.q_type

        answers = prediction.answer_strings[:5]

        if FLAGS.verbose:
            print("-------------")
            print("  ID:", question.id)
            print("  Given: ", answers)
            print("  Correct: ", correct_answers)

        if question_type == "factoid":
            factoid_total += 1
            exact_math_found = False
            rank = sys.maxsize
            for correct_answer in correct_answers[0]:
                # Compute exact match
                if not exact_math_found and \
                        answers[0].lower() == correct_answer.lower():
                    if FLAGS.verbose:
                        print("  Correct!")
                    factoid_correct += 1
                    exact_math_found = True
                # Compute rank
                for k in range(min(len(answers), 5)):
                    if answers[k].lower() == correct_answer.lower():
                        rank = min(rank, k + 1)

            if FLAGS.verbose:
                print("  Rank: %d" % (rank if rank <= 5 else -1))
            factoid_reciprocal_rank_sum += 1 / rank if rank <= 5 else 0


        if question_type == "list":
            list_total += 1
            answer_correct = np.zeros([len(answers)], dtype=np.bool)

            for answer_option in correct_answers:
                for correct_answer in answer_option:
                    for k in range(len(answers)):

                        # Count answer if it hasn't yet been counted as correct.
                        if not answer_correct[k] and \
                                answers[k].lower() == correct_answer.lower():
                            answer_correct[k] = True
                            # Only count one synonym.
                            break

            tp = np.count_nonzero(answer_correct)
            precision = tp / len(answers)
            recall = tp / len(correct_answers)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            if FLAGS.verbose:
                print("F1: %f" % f1)
            list_f1_sum += f1

    print("Factoid correct: %d / %d" % (factoid_correct, factoid_total))
    print("Factoid MRR: %f" % (factoid_reciprocal_rank_sum / factoid_total))
    print("List mean F1: %f (%d Questions)" % (list_f1_sum / list_total,
                                               list_total))

def main():
    devices = FLAGS.devices.split(",")

    inferrer = Inferrer(FLAGS.model_config, devices, FLAGS.beam_size,
                        FLAGS.model_weights)

    print("Initializing Sampler & Trainer...")
    data_dir = os.path.dirname(FLAGS.eval_data)
    data_filename = os.path.basename(FLAGS.eval_data)
    instances = FLAGS.subsample if FLAGS.subsample > 0 else None
    if not FLAGS.is_bioasq:
        sampler = SQuADSampler(data_dir, [data_filename], FLAGS.batch_size,
                               inferrer.model.embedder.vocab,
                               instances_per_epoch=instances, shuffle=False,
                               split_contexts_on_newline=FLAGS.split_contexts)
    else:
        sampler = BioAsqSampler(data_dir, [data_filename], FLAGS.batch_size,
                                inferrer.model.embedder.vocab,
                                instances_per_epoch=instances, shuffle=False,
                                split_contexts_on_newline=FLAGS.split_contexts,
                                context_token_limit=FLAGS.bioasq_context_token_limit,
                                include_synonyms=FLAGS.bioasq_include_synonyms)

    if FLAGS.squad_evaluation:
        print("Running SQuAD Evaluation...")
        trainer = ExtractionQATrainer(0, inferrer.model, devices[0])
        trainer.eval(inferrer.sess, sampler, verbose=True)

    if FLAGS.bioasq_evaluation:
        print("Running BioASQ Evaluation...")
        bioasq_evaluation(sampler, inferrer)

main()
