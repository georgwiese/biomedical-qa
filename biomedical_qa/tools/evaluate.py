import json
import os
import pickle
import sys
import numpy as np
import tensorflow as tf
from nltk import RegexpTokenizer

from biomedical_qa.tools import util
from biomedical_qa.models import model_from_config
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer

tf.app.flags.DEFINE_string('eval_data', None, 'Path to the SQuAD JSON file.')
tf.app.flags.DEFINE_string('model_config', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")

tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")
tf.app.flags.DEFINE_integer("subsample", -1, "Number of samples to do the evaluation on.")

tf.app.flags.DEFINE_integer("beam_size", 5, "Beam size used for decoding.")

tf.app.flags.DEFINE_boolean("squad_evaluation", False, "If true, measures F1 and exact match acc on answer spans.")
tf.app.flags.DEFINE_boolean("bioasq_evaluation", False, "If true, runs BioASQ evaluation measures.")
tf.app.flags.DEFINE_boolean("verbose", False, "If true, prints correct and given answers.")

FLAGS = tf.app.flags.FLAGS

TOKENIZER = RegexpTokenizer(r'\w+|[^\w\s]')

def answer_equal(answer_string, predicted_tokens):

    answer_tokens = TOKENIZER.tokenize(answer_string.lower())

    if len(answer_tokens) != len(predicted_tokens):
        return False

    for answer_token, predicted_token in zip(answer_tokens, predicted_tokens):
        if answer_token != predicted_token.lower():
            return False

    return True


def bioasq_evaluation(sampler, sess, model):
    with open(FLAGS.eval_data) as f:
        paragraphs = json.load(f)["data"][0]["paragraphs"]

    assert paragraphs[0]["qas"][0]["original_answers"] is not None, \
        "Questions must be augmented with original_answers to perform BioASQ evaluation."
    assert paragraphs[0]["qas"][0]["question_type"] is not None, \
        "Questions must be augmented with question_type to perform BioASQ evaluation."
    assert FLAGS.beam_size >= 5, "Beamsize must be at least 5 to get 5 ranked answers."

    # Assuming one question per paragraph
    paragraphs_by_id = {p["qas"][0]["id"] : p for p in paragraphs}
    count = len(paragraphs)
    print("  (Questions: %d, using %d)" %
          (count, FLAGS.subsample if FLAGS.subsample > 0 else count))

    factoid_correct, factoid_total = 0, 0
    factoid_reciprocal_rank_sum = 0
    list_f1_sum, list_total = 0, 0

    sampler.reset()
    epoch = sampler.epoch

    while sampler.epoch == epoch:

        batch = sampler.get_batch()
        paragraphs_batch = [paragraphs_by_id[q.id] for q in batch]
        assert len(batch) == len(paragraphs_batch)

        correct_answers = [p["qas"][0]["original_answers"] for p in paragraphs_batch]
        question_types = [p["qas"][0]["question_type"] for p in paragraphs_batch]
        contexts = [p["context"] for p in paragraphs_batch]

        starts, ends, top_starts, top_ends = sess.run([model.predicted_answer_starts,
                                                       model.predicted_answer_ends,
                                                       model.top_starts,
                                                       model.top_ends],
                                                       model.get_feed_dict(batch))

        assert len(starts) == len(batch)
        assert len(ends) == len(batch)
        assert len(top_starts) == len(batch)
        assert len(top_ends) == len(batch)

        for i in range(len(starts)):
            assert starts[i] == top_starts[i, 0]
            assert ends[i] == top_ends[i, 0]

        for i in range(len(batch)):

            current_correct_answers = correct_answers[i]
            if isinstance(current_correct_answers[0], str):
                current_correct_answers = [current_correct_answers]

            context_tokens = TOKENIZER.tokenize(contexts[i])
            answers = [context_tokens[top_starts[i, k] : top_ends[i, k] + 1]
                       for k in range(5)]

            if FLAGS.verbose:
                print("-------------")
                print("  Given: ", [" ".join(answer) for answer in answers])
                print("  Correct: ", current_correct_answers)

            if question_types[i] == "factoid":
                factoid_total += 1
                exact_math_found = False
                rank = sys.maxsize
                for correct_answer in current_correct_answers[0]:
                    # Compute exact match
                    if not exact_math_found and answer_equal(correct_answer, answers[0]):
                        if FLAGS.verbose:
                            print("  Correct!")
                        factoid_correct += 1
                        exact_math_found = True
                    # Compute rank
                    for k in range(5):
                        if answer_equal(correct_answer, answers[k]):
                            rank = min(rank, k + 1)

                if FLAGS.verbose:
                    print("  Rank: %d" % (rank if rank <= 5 else -1))
                factoid_reciprocal_rank_sum += 1 / rank if rank <= 5 else 0


            if question_types[i] == "list":
                list_total += 1
                answer_correct = np.zeros([len(answers)], dtype=np.bool)

                for answer_option in current_correct_answers:
                    for correct_answer in answer_option:
                        for j, answer in enumerate(answers):

                            # Count answer if it hasn't yet been counted as correct.
                            if not answer_correct[j] and \
                                answer_equal(correct_answer, answers[0]):
                                answer_correct[j] = True
                                # Only count one synonym.
                                break

                tp = np.count_nonzero(answer_correct)
                precision = tp / len(answers)
                recall = tp / len(current_correct_answers)
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

    model, sess = util.initialize_model(FLAGS.model_config, FLAGS.model_weights,
                                        devices, FLAGS.beam_size)

    print("Initializing Sampler & Trainer...")
    data_dir = os.path.dirname(FLAGS.eval_data)
    data_filename = os.path.basename(FLAGS.eval_data)
    instances = FLAGS.subsample if FLAGS.subsample > 0 else None
    sampler = SQuADSampler(data_dir, [data_filename], FLAGS.batch_size,
                           model.embedder.vocab, instances_per_epoch=instances)
    trainer = ExtractionQATrainer(0, model, devices[0])

    if FLAGS.squad_evaluation:
        print("Running SQuAD Evaluation...")
        trainer.eval(sess, sampler, verbose=True)

    if FLAGS.bioasq_evaluation:
        print("Running BioASQ Evaluation...")
        bioasq_evaluation(sampler, sess, model)

main()
