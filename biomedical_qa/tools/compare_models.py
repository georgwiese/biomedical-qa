import os
import tensorflow as tf

from biomedical_qa.inference.inference import Inferrer, get_session, get_model
from biomedical_qa.sampling.bioasq import BioAsqSampler
from biomedical_qa.evaluation.bioasq_evaluation import BioAsqEvaluator, \
    element_wise_mean

tf.app.flags.DEFINE_string('eval_data', None, 'Path to the SQuAD JSON file.')
tf.app.flags.DEFINE_boolean('split_contexts', False, 'Whether to split contexts on newline.')
tf.app.flags.DEFINE_string('model_config1', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights1', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string('model_config2', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights2', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")

tf.app.flags.DEFINE_boolean("is_bioasq", False, "Whether the provided dataset is a BioASQ json.")
tf.app.flags.DEFINE_boolean("bioasq_include_synonyms", False, "Whether BioASQ synonyms should be included.")
tf.app.flags.DEFINE_integer("bioasq_context_token_limit", -1, "Token limit for BioASQ contexts.")

tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")
tf.app.flags.DEFINE_integer("subsample", -1, "Number of samples to do the evaluation on.")

tf.app.flags.DEFINE_integer("beam_size", 5, "Beam size used for decoding.")
tf.app.flags.DEFINE_float("list_answer_prob_threshold1", 0.5, "Probability threshold to include answers to list questions. Used start output unit is sigmoid.")
tf.app.flags.DEFINE_integer("list_answer_count1", 5, "Number of answers to list questions. Used start output unit is softmax.")
tf.app.flags.DEFINE_float("list_answer_prob_threshold2", 0.5, "Probability threshold to include answers to list questions. Used start output unit is sigmoid.")
tf.app.flags.DEFINE_integer("list_answer_count2", 5, "Number of answers to list questions. Used start output unit is softmax.")

tf.app.flags.DEFINE_integer("verbosity_level", 0, "Verbosity Level.")

FLAGS = tf.app.flags.FLAGS


def get_evaluation_for_model(config, devices, weights, answer_count, prob_threshold):

    with tf.Graph().as_default() as g:
        sess = get_session()
        model = get_model(sess, config, devices, weights)
        inferrer = Inferrer(model, sess, FLAGS.beam_size)

        data_dir = os.path.dirname(FLAGS.eval_data)
        data_filename = os.path.basename(FLAGS.eval_data)
        instances = FLAGS.subsample if FLAGS.subsample > 0 else None

        sampler = BioAsqSampler(data_dir, [data_filename], FLAGS.batch_size,
                                inferrer.model.embedder.vocab,
                                instances_per_epoch=instances, shuffle=False,
                                split_contexts_on_newline=FLAGS.split_contexts,
                                context_token_limit=FLAGS.bioasq_context_token_limit,
                                include_synonyms=FLAGS.bioasq_include_synonyms)

        evaluator = BioAsqEvaluator(sampler, inferrer)
        results =  evaluator.evaluate_questions(verbosity_level=FLAGS.verbosity_level,
                                                list_answer_count=answer_count,
                                                list_answer_prob_threshold=prob_threshold)

        sess.close()
        return results


def print_performance(factoid_performances, list_performances):

    acc, mrr = element_wise_mean(factoid_performances.values())
    f1, precision, recall = element_wise_mean(list_performances.values())

    print("Factoid:")
    print("  Accuracy: %.1f%%" % (acc * 100.0))
    print("  MRR: %.1f%%" % (mrr * 100.0))
    print("List:")
    print("  F1: %.1f%%" % (f1 * 100.0))
    print("  Precision: %.1f%%" % (precision * 100.0))
    print("  Recall: %.1f%%" % (recall * 100.0))


def main():
    devices = FLAGS.devices.split(",")

    print("Doing model 1 predictions...")
    factiod_performances1, list_performances1 = get_evaluation_for_model(
        FLAGS.model_config1, devices, FLAGS.model_weights1,
        FLAGS.list_answer_count1, FLAGS.list_answer_count2)

    print("Doing model 2 predictions...")
    factiod_performances2, list_performances2 = get_evaluation_for_model(
        FLAGS.model_config2, devices, FLAGS.model_weights2,
        FLAGS.list_answer_count2, FLAGS.list_answer_count2)

    print("Intersecting predictions...")
    best_factoid_perfomances = {}
    best_list_performances = {}

    for q_id, (correct1, reciprocal_rank1) in factiod_performances1.items():
        correct2, reciprocal_rank2 = factiod_performances2[q_id]

        if reciprocal_rank1 > reciprocal_rank2:
            best_factoid_perfomances[q_id] = (correct1, reciprocal_rank1)
        else:
            best_factoid_perfomances[q_id] = (correct2, reciprocal_rank2)

    for q_id, (f11, precision1, recall1) in list_performances1.items():
        f12, precision2, recall2 = list_performances2[q_id]

        if f11 > f12:
            best_list_performances[q_id] = (f11, precision1, recall1)
        else:
            best_list_performances[q_id] = (f12, precision2, recall2)


    print("=== Model 1 stand-alone:")
    print_performance(factiod_performances1, list_performances1)
    print()
    print("=== Model 2 stand-alone:")
    print_performance(factiod_performances2, list_performances2)
    print()
    print("=== Best performance:")
    print_performance(best_factoid_perfomances, best_list_performances)



main()
