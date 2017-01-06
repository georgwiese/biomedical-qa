import os
import sys
import functools
import pickle
import random
import time

import tensorflow as tf

from biomedical_qa.models import model_from_config
from biomedical_qa.models.embedder import CharWordEmbedder
from biomedical_qa.models.qa_pointer import QAPointerModel
from biomedical_qa.models.qa_simple_pointer import QASimplePointerModel
from biomedical_qa.sampling.bioasq import BioAsqSampler
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer

from biomedical_qa.util import load_vocab

# data loading specifics
tf.app.flags.DEFINE_string('data', None, 'Directory containing dataset files.')
tf.app.flags.DEFINE_boolean('split_contexts', False, 'Whether to split contexts on newline.')
tf.app.flags.DEFINE_string("trainset_prefix", "train", "Prefix of training files.")
tf.app.flags.DEFINE_string("validset_prefix", "valid", "Prefix of validation files.")
tf.app.flags.DEFINE_string("testset_prefix", "test", "Prefix of test files.")
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
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from.")
tf.app.flags.DEFINE_string("embeddings", None, "Init with word embeddings from given path in w2v binary format.")
tf.app.flags.DEFINE_integer("max_context_length", 300, "Maximum length of context.")
tf.app.flags.DEFINE_integer("max_vocab", -1, "Maximum vocab size if no embedder is given.")
tf.app.flags.DEFINE_integer("max_instances", None, "Maximum number of training instances.")
tf.app.flags.DEFINE_integer("subsample_validation", None, "Maximum number of validation instances.")
tf.app.flags.DEFINE_integer("max_epochs", 40, "Maximum number of epochs.")
tf.app.flags.DEFINE_string("train_variable_prefixes", "", "Comma-seperated list of variable name prefixes that should be trained.")

#embedder
tf.app.flags.DEFINE_boolean("transfer_qa", False, "Tranfer-model (if given) is a QAModel")
tf.app.flags.DEFINE_string("transfer_model_config", None, "Path to transfer model config.")
tf.app.flags.DEFINE_string("transfer_model_path", None, "Path to transfer model model.")
tf.app.flags.DEFINE_integer("transfer_layer_size", None, "Learning rate for transfer model.")


FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)
train_dir = FLAGS.save_dir

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def make_sampler(filenames, vocab):
    args = {
        "dir": FLAGS.data,
        "filenames": filenames,
        "batch_size": FLAGS.batch_size,
        "vocab": vocab,
        "instances_per_epoch": FLAGS.max_instances,
        "split_contexts_on_newline": FLAGS.split_contexts,
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

        if FLAGS.transfer_model_config is None:
            vocab, _, _ = load_vocab(os.path.join(FLAGS.data, "document.vocab"))
            if FLAGS.max_vocab < 0:
                FLAGS.max_vocab = len(vocab)
            transfer_model = CharWordEmbedder(FLAGS.size, vocab, devices[0])
        else:
            print("Creating transfer model from config %s" % FLAGS.transfer_model_config)
            with open(FLAGS.transfer_model_config, 'rb') as f:
                transfer_model_config = pickle.load(f)

            if FLAGS.transfer_qa:
                transfer_model = model_from_config(transfer_model_config, devices[0:1])
            else:
                transfer_model = model_from_config(transfer_model_config, devices[0:1])

        print("Creating model of type %s..." % FLAGS.model_type)
        if FLAGS.model_type == "qa_pointer":
            model = QAPointerModel(FLAGS.size, transfer_model, devices=devices,
                                         keep_prob=1.0-FLAGS.dropout, composition=FLAGS.composition,
                                         answer_layer_depth=FLAGS.answer_layer_depth,
                                         answer_layer_poolsize=FLAGS.answer_layer_poolsize,
                                         answer_layer_type=FLAGS.answer_layer_type)
        elif FLAGS.model_type == "qa_simple_pointer":
            with_inter_fusion = FLAGS.with_fusion
            num_intrafusion_layers = 1 if FLAGS.with_fusion else 0
            model = QASimplePointerModel(FLAGS.size, transfer_model, devices=devices,
                                         keep_prob=1.0-FLAGS.dropout, composition=FLAGS.composition,
                                         num_intrafusion_layers=num_intrafusion_layers,
                                         with_inter_fusion=with_inter_fusion)
        else:
            raise ValueError("Unknown model type: %s" % FLAGS.model_type)

    print("Preparing Samplers ...")
    train_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.trainset_prefix)]
    random.shuffle(train_fns)
    print("Training sets (first 100): ", train_fns[:100])
    sampler = make_sampler(train_fns, model.transfer_model.vocab)

    valid_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.validset_prefix)]
    print("Valid sets: (first 100)", valid_fns[:100])
    valid_sampler = make_sampler(valid_fns, model.transfer_model.vocab)
    test_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.testset_prefix)]
    if test_fns:
        print("Test sets: (first 100)", test_fns[:100])
        test_sampler = SQuADSampler(FLAGS.data, test_fns, FLAGS.batch_size, model.transfer_model.vocab, FLAGS.max_instances)

    trainer = ExtractionQATrainer(FLAGS.learning_rate, model, devices[0],
                                  train_variable_prefixes=train_variable_prefixes,
                                  start_output_unit=FLAGS.start_output_unit)

    print("Created %s!" % type(model).__name__)

    print("Setting up summary writer...")
    train_summaries = tf.merge_all_summaries()
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
        print("Loading from path " + FLAGS.init_model_path)
        trainer.model.model_saver.restore(sess, FLAGS.init_model_path)
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

    best_path = []
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    previous_loss = list()
    epoch = 0

    def validate(global_step):
        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        f1, exact = trainer.eval(sess, valid_sampler, verbose=True)#, subsample=min(10000, FLAGS.batch_size * 100))
        print("####################################")
        trainer.model.set_train(sess)

        dev_summary = tf.Summary()
        dev_summary.value.add(tag="valid_f1_mean", simple_value=f1)
        dev_summary.value.add(tag="valid_exact_mean", simple_value=exact)
        dev_summary_writer.add_summary(dev_summary, global_step)

        l = -f1
        if not best_path or l < min(previous_loss):
            if best_path:
                best_path[0] = trainer.all_saver.save(sess, checkpoint_path, global_step=trainer.global_step,
                                                      write_meta_graph=False)
            else:
                best_path.append(
                    trainer.all_saver.save(sess, checkpoint_path, global_step=trainer.global_step, write_meta_graph=False))

        if previous_loss and l > previous_loss[-1]:
            print("Decaying learningrate.")
            lr = sess.run(trainer.learning_rate)
            decay = FLAGS.min_learning_rate/lr if lr * FLAGS.learning_rate_decay < FLAGS.min_learning_rate else FLAGS.learning_rate_decay
            if sess.run(trainer.learning_rate) > FLAGS.min_learning_rate:
                trainer.decay_learning_rate(sess, decay)

        previous_loss.append(l)
        return l

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    loss = 0.0
    step_time = 0.0
    ckpt_result = 0.0
    i = 0
    trainer.model.set_train(sess)
    epochs = 0
    validate(0)
    while True:
        i += 1
        start_time = time.time()
        batch = sampler.get_batch()
        # already fetch next batch parallel to running model
        goals = [trainer.update, trainer.loss]
        if i % FLAGS.ckpt_its == 0:
            goals += [train_summaries]
        results = trainer.run(sess, goals, batch)
        loss += results[1]

        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f, step-time %.3f" % (i*100.0 / FLAGS.ckpt_its, loss / i, step_time / i))
        sys.stdout.flush()

        if i % FLAGS.ckpt_its == 0:
            global_step = trainer.global_step.eval()
            batch_summary = results[2]
            train_summary_writer.add_summary(batch_summary, global_step)

            epochs += 1
            i = 0
            loss /= FLAGS.ckpt_its
            print("")
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.5f, step-time %.3f, loss %.4f" % (global_step,
                                                                                    trainer.learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            result = validate(global_step)
            if result > ckpt_result and epochs >= FLAGS.max_epochs:
                print("Stop learning!")
                break
            else:
                ckpt_result = result

    best_valid_loss = min(previous_loss) if previous_loss else 0.0
    print("Restore model to best loss on validation: %.3f" % best_valid_loss)
    trainer.all_saver.restore(sess, best_path[0])
    model_name = best_path[0].split("/")[-1]
    trainer.model.model_saver.save(sess, os.path.join(train_dir, "final_model.tf"), write_meta_graph=False)
    if test_fns:
        print("########## Test ##############")
        trainer.eval(sess, test_sampler, verbose=True)
        print("####################################")
