import os
import sys
import functools
import pickle
import random
import time

import tensorflow as tf

from biomedical_qa.data.entity_tagger import EntityTagger
from biomedical_qa.models import model_from_config
from biomedical_qa.models.embedder import CharWordEmbedder, ConcatEmbedder
from biomedical_qa.models.qa_pointer import QAPointerModel
from biomedical_qa.models.qa_simple_pointer import QASimplePointerModel
from biomedical_qa.sampling.bioasq import BioAsqSampler
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionGoalDefiner, BioAsqGoalDefiner
from biomedical_qa.training.trainer import Trainer
from biomedical_qa.training.yesno_trainer import YesNoGoalDefiner

# data loading specifics
tf.app.flags.DEFINE_string('data', None, 'Directory containing dataset files.')
tf.app.flags.DEFINE_string('yesno_data', None, 'Directory containing Yes/No dataset files.')
tf.app.flags.DEFINE_boolean('split_contexts', False, 'Whether to split contexts on newline.')
tf.app.flags.DEFINE_string("trainset_prefix", "train", "Prefix of training files.")
tf.app.flags.DEFINE_string("validset_prefix", "valid", "Prefix of validation files.")
tf.app.flags.DEFINE_string("dataset", "squad", "[wikireading,squad].")
tf.app.flags.DEFINE_string("task", "qa", "qa, multiple_choice, question_generation")

# BioASQ data loading
tf.app.flags.DEFINE_boolean("is_bioasq", False, "Whether the provided dataset is a BioASQ json.")
tf.app.flags.DEFINE_boolean("bioasq_include_synonyms", False, "Whether BioASQ synonyms should be included.")
tf.app.flags.DEFINE_integer("bioasq_context_token_limit", -1, "Token limit for BioASQ contexts.")

# model
tf.app.flags.DEFINE_string("model_config", None, "Config of the model.")
tf.app.flags.DEFINE_integer("size", 150, "hidden size of model")
tf.app.flags.DEFINE_integer("max_length", 30, "max length of answer or question depending on task.")
tf.app.flags.DEFINE_string("composition", 'GRU', "'LSTM', 'GRU'")
tf.app.flags.DEFINE_string("model_type", "qa_pointer", "[pointer, simple_pointer].")

# qa_simple_pointer settings
tf.app.flags.DEFINE_bool("with_fusion", False, "Whether Inter & Intra fusion is activated.")
tf.app.flags.DEFINE_bool("with_question_type_features", False, "Whether Question types are passed to the network.")
tf.app.flags.DEFINE_bool("with_entity_tag_features", False, "Whether entity tags are passed to the network.")

tf.app.flags.DEFINE_string("terms_file", None, "UML Terms file (MRCONSO.RRF).")
tf.app.flags.DEFINE_string("types_file", None, "UMLS Types file (MRSTY.RRF).")

# qa_pointer settings
tf.app.flags.DEFINE_string("answer_layer_type", "dpn", "Type of answer layer ([dpn]).")
tf.app.flags.DEFINE_integer("answer_layer_depth", 1, "Number of layer in the answer layer")
tf.app.flags.DEFINE_integer("answer_layer_poolsize", 8, "Maxout poolsize in answer layer")

#training
tf.app.flags.DEFINE_string("start_output_unit", "softmax", "softmax or sigmoid.")
tf.app.flags.DEFINE_float("dropout", 0.0, "Dropout.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimal learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("batch_size", 30, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", 1000, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("min_epochs", 1, "Minimum number of epochs.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from, or 'latest' if model_config is used.")
tf.app.flags.DEFINE_string("embeddings", None, "Init with word embeddings from given path in w2v binary format.")
tf.app.flags.DEFINE_integer("max_context_length", 300, "Maximum length of context.")
tf.app.flags.DEFINE_integer("max_vocab", -1, "Maximum vocab size if no embedder is given.")
tf.app.flags.DEFINE_integer("max_instances", None, "Maximum number of training instances.")
tf.app.flags.DEFINE_integer("subsample_validation", None, "Maximum number of validation instances.")
tf.app.flags.DEFINE_integer("max_epochs", 40, "Maximum number of epochs.")
tf.app.flags.DEFINE_string("train_variable_prefixes", "", "Comma-seperated list of variable name prefixes that should be trained.")

#embedder
tf.app.flags.DEFINE_boolean("with_chars", False, "Use char word-embedder additionally.")
tf.app.flags.DEFINE_string("transfer_model_config", None, "Path to transfer model config.")
tf.app.flags.DEFINE_string("transfer_model_path", None, "Path to transfer model model.")
tf.app.flags.DEFINE_integer("transfer_layer_size", None, "Learning rate for transfer model.")


FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)
train_dir = FLAGS.save_dir

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def make_sampler(dir, filenames, vocab, types, tagger):

    args = {
        "dir": dir,
        "filenames": filenames,
        "batch_size": FLAGS.batch_size,
        "vocab": vocab,
        "instances_per_epoch": FLAGS.max_instances,
        "split_contexts_on_newline": FLAGS.split_contexts,
        "types": types,
        "tagger": tagger,
    }

    if FLAGS.is_bioasq:
        args.update({
            "context_token_limit": FLAGS.bioasq_context_token_limit,
            "include_synonyms": FLAGS.bioasq_include_synonyms,
        })
        return BioAsqSampler(**args)
    else:
        return SQuADSampler(**args)



with tf.Session(config=config) as sess:
    devices = FLAGS.devices.split(",")
    train_variable_prefixes = FLAGS.train_variable_prefixes.split(",") \
                              if FLAGS.train_variable_prefixes else []

    if FLAGS.model_config is not None:

        with open(FLAGS.model_config, 'rb') as f:
            model_config = pickle.load(f)
        model = model_from_config(model_config, devices, FLAGS.dropout)

    else:
        print("Creating transfer model from config %s" % FLAGS.transfer_model_config)
        with open(FLAGS.transfer_model_config, 'rb') as f:
            transfer_model_config = pickle.load(f)
        transfer_model = model_from_config(transfer_model_config, devices[0:1])

        if FLAGS.with_chars:
            print("Use additional char-based word-embedder")
            char_embedder = CharWordEmbedder(FLAGS.size, transfer_model.vocab, devices[0])
            FLAGS.embedder_lr = FLAGS.learning_rate
            embedder = ConcatEmbedder([transfer_model, char_embedder])

        print("Creating model of type %s..." % FLAGS.model_type)
        if FLAGS.model_type == "pointer":
            model = QAPointerModel(FLAGS.size, transfer_model, devices=devices,
                                   keep_prob=1.0-FLAGS.dropout, composition=FLAGS.composition,
                                   answer_layer_depth=FLAGS.answer_layer_depth,
                                   answer_layer_poolsize=FLAGS.answer_layer_poolsize,
                                   answer_layer_type=FLAGS.answer_layer_type,
                                   start_output_unit=FLAGS.start_output_unit)
        elif FLAGS.model_type == "simple_pointer":
            with_inter_fusion = FLAGS.with_fusion
            num_intrafusion_layers = 1 if FLAGS.with_fusion else 0
            model = QASimplePointerModel(FLAGS.size, transfer_model, devices=devices,
                                         keep_prob=1.0-FLAGS.dropout, composition=FLAGS.composition,
                                         num_intrafusion_layers=num_intrafusion_layers,
                                         with_inter_fusion=with_inter_fusion,
                                         start_output_unit=FLAGS.start_output_unit,
                                         with_question_type_features=FLAGS.with_question_type_features,
                                         with_entity_tag_features=FLAGS.with_entity_tag_features)
        else:
            raise ValueError("Unknown model type: %s" % FLAGS.model_type)

    if FLAGS.yesno_data is not None:
        model.add_yesno()

    print("Preparing Samplers ...")
    train_samplers = []
    valid_samplers = []

    tagger = None
    if FLAGS.terms_file and FLAGS.types_file:
        tagger = EntityTagger(FLAGS.terms_file, FLAGS.types_file, case_sensitive=True)

    for dir, types in [(FLAGS.data, ["factoid", "list"]), (FLAGS.yesno_data, ["yesno"])]:
        if dir is not None:
            train_fns = [fn for fn in os.listdir(dir) if fn.startswith(FLAGS.trainset_prefix)]
            train_samplers.append(make_sampler(dir, train_fns,
                                               model.transfer_model.vocab,
                                               types, tagger))

            valid_fns = [fn for fn in os.listdir(dir) if fn.startswith(FLAGS.validset_prefix)]
            valid_samplers.append(make_sampler(dir, valid_fns,
                                               model.transfer_model.vocab,
                                               types, tagger))

    # Free memory
    tagger = None

    goal_definers = []
    if FLAGS.data is not None:
        if FLAGS.is_bioasq:
            goal_definers.append(BioAsqGoalDefiner(model, devices[0]))
        else:
            goal_definers.append(ExtractionGoalDefiner(model, devices[0]))

    if FLAGS.yesno_data is not None:
        goal_definers.append(YesNoGoalDefiner(model, devices[0]))

    trainer = Trainer(model, FLAGS.learning_rate, goal_definers, devices[0],
                      train_variable_prefixes)

    print("Created %s!" % type(model).__name__)

    print("Setting up summary writer...")
    train_summary_writer = tf.train.SummaryWriter(FLAGS.save_dir + '/train',
                                                  sess.graph)
    dev_summary_writer = tf.train.SummaryWriter(FLAGS.save_dir + '/dev',
                                                sess.graph)

    print("Initializing variables ...")
    sess.run(tf.global_variables_initializer())
    if FLAGS.transfer_model_path is not None:
        print("Loading transfer model from %s" % FLAGS.transfer_model_path)
        transfer_model.model_saver.restore(sess, FLAGS.transfer_model_path)

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if FLAGS.init_model_path:
        init_model_path = FLAGS.init_model_path
        if FLAGS.init_model_path == "latest":
            assert FLAGS.model_config is not None, "Provide model_config to use latest checkpoint."
            init_model_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.model_config))
        print("Loading from path " + init_model_path)
        model.model_saver.restore(sess, init_model_path)
    elif latest_checkpoint is not None:
        print("Loading from checkpoint " + latest_checkpoint)
        trainer.all_saver.restore(sess, latest_checkpoint)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(model.train_variables), 0)
    print("Num params: %d" % num_params)

    print("Initialized model.")
    with open(os.path.join(train_dir, "config.pickle"), 'wb') as f:
        pickle.dump(model.get_config(), f, protocol=4)

    best_path = None
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    previous_performances = list()
    epoch = 0

    def validate(global_step, trainer):

        global best_path

        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        performance, summaries = trainer.eval(sess, valid_samplers, verbose=True)
        print("####################################")
        model.set_train(sess)

        for summary in summaries:
            dev_summary_writer.add_summary(summary, global_step)

        if not best_path or performance > max(previous_performances):
            best_path = trainer.all_saver.save(sess, checkpoint_path, global_step=trainer.global_step,
                                               write_meta_graph=False)

        if previous_performances and performance < previous_performances[-1]:
            print("Decaying learningrate.")
            lr = sess.run(trainer.learning_rate)
            decay = FLAGS.min_learning_rate/lr if lr * FLAGS.learning_rate_decay < FLAGS.min_learning_rate else FLAGS.learning_rate_decay
            if sess.run(trainer.learning_rate) > FLAGS.min_learning_rate:
                trainer.decay_learning_rate(sess, decay)

        previous_performances.append(performance)
        return performance

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    loss = 0.0
    step_time = 0.0
    ckpt_result = 0.0
    i = 0

    model.set_train(sess)

    epochs = 0
    validate(0, trainer)

    while True:
        i += 1
        start_time = time.time()

        batch_loss, summaries = trainer.run_train_steps(sess, train_samplers,
                                                        with_summaries=i % FLAGS.ckpt_its == 0)

        loss += batch_loss
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f, step-time %.3f" % (i*100.0 / FLAGS.ckpt_its, loss / i, step_time / i))
        sys.stdout.flush()

        if i % FLAGS.ckpt_its == 0:
            global_step = trainer.global_step.eval()
            for summary in summaries:
                train_summary_writer.add_summary(summary, global_step)

            epochs += 1
            i = 0
            loss /= FLAGS.ckpt_its
            print("")
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.5f, step-time %.3f, loss %.4f" % (global_step,
                                                                                    trainer.learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            result = validate(global_step, trainer)
            if result < ckpt_result and epochs >= FLAGS.max_epochs:
                print("Stop learning!")
                break
            else:
                ckpt_result = result

    best_valid_performance = max(previous_performances) if previous_performances else 0.0
    print("Restore model to best performance on validation: %.3f" % best_valid_performance)
    trainer.all_saver.restore(sess, best_path)
    model_name = best_path.split("/")[-1]
    trainer.model.model_saver.save(sess, os.path.join(train_dir, "final_model.tf"), write_meta_graph=False)
