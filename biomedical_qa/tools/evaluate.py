import os
import tensorflow as tf
import numpy as np

from biomedical_qa.inference.inference import Inferrer
from biomedical_qa.sampling.bioasq import BioAsqSampler
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer
from biomedical_qa.evaluation.bioasq_evaluation import BioAsqEvaluator

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
tf.app.flags.DEFINE_float("list_answer_prob_threshold", 0.5, "Probability threshold to include answers to list questions. Used start output unit is sigmoid.")
tf.app.flags.DEFINE_integer("list_answer_count", 5, "Number of answers to list questions. Used start output unit is softmax.")

tf.app.flags.DEFINE_boolean("squad_evaluation", False, "If true, measures F1 and exact match acc on answer spans.")
tf.app.flags.DEFINE_boolean("bioasq_evaluation", False, "If true, runs BioASQ evaluation measures.")
tf.app.flags.DEFINE_boolean("find_optimal_threshold", False, "If true, will find the threshold which optimizes list performance.")
tf.app.flags.DEFINE_boolean("verbose", False, "If true, prints correct and given answers.")

tf.app.flags.DEFINE_float("threshold_search_step", 0.01, "Step size to use for threshold search.")

FLAGS = tf.app.flags.FLAGS



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
        evaluator = BioAsqEvaluator(sampler, inferrer)
        evaluator.evaluate(verbosity_level=2 if FLAGS.verbose else 1,
                           list_answer_count=FLAGS.list_answer_count,
                           list_answer_prob_threshold=FLAGS.list_answer_prob_threshold)

    if FLAGS.find_optimal_threshold:
        assert FLAGS.is_bioasq

        sampler = BioAsqSampler(data_dir, [data_filename], FLAGS.batch_size,
                                inferrer.model.embedder.vocab,
                                types=["list"],
                                instances_per_epoch=instances, shuffle=False,
                                split_contexts_on_newline=FLAGS.split_contexts,
                                context_token_limit=FLAGS.bioasq_context_token_limit,
                                include_synonyms=FLAGS.bioasq_include_synonyms)
        evaluator = BioAsqEvaluator(sampler, inferrer)
        evaluator.find_optimal_threshold(FLAGS.threshold_search_step,
                                         verbosity_level=2 if FLAGS.verbose else 1)

main()
