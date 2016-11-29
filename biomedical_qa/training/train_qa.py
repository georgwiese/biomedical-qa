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
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer

from biomedical_qa.util import load_vocab

# data loading specifics
tf.app.flags.DEFINE_string('data', None, 'Directory containing dataset files.')
tf.app.flags.DEFINE_string("trainset_prefix", "train", "Prefix of training files.")
tf.app.flags.DEFINE_string("validset_prefix", "valid", "Prefix of validation files.")
tf.app.flags.DEFINE_string("testset_prefix", "test", "Prefix of test files.")
tf.app.flags.DEFINE_string("dataset", "squad", "[wikireading,squad].")
tf.app.flags.DEFINE_string("task", "qa", "qa, multiple_choice, question_generation")

# model
tf.app.flags.DEFINE_integer("size", 150, "hidden size of model")
tf.app.flags.DEFINE_integer("max_length", 30, "max length of answer or question depending on task.")
tf.app.flags.DEFINE_string("composition", 'GRU', "'LSTM', 'GRU'")
tf.app.flags.DEFINE_string("name", "QAModel", "Name of the model.")

tf.app.flags.DEFINE_string("answer_layer_type", "dpn", "Type of answer layer ([dpn]).")
tf.app.flags.DEFINE_integer("answer_layer_depth", 1, "Number of layer in the answer layer")
tf.app.flags.DEFINE_integer("answer_layer_poolsize", 8, "Maxout poolsize in answer layer")

#training
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

#embedder
tf.app.flags.DEFINE_boolean("transfer_qa", False, "Tranfer-model (if given) is a QAModel")
tf.app.flags.DEFINE_string("transfer_model_config", None, "Path to transfer model config.")
tf.app.flags.DEFINE_string("transfer_model_path", None, "Path to transfer model model.")
tf.app.flags.DEFINE_float("transfer_model_lr", 0.0, "Learning rate for transfer model.")
tf.app.flags.DEFINE_integer("transfer_layer_size", None, "Learning rate for transfer model.")


FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)
train_dir = FLAGS.save_dir

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    devices = FLAGS.devices.split(",")

    if FLAGS.transfer_model_config is None:
        vocab, _, _ = load_vocab(os.path.join(FLAGS.data, "document.vocab"))
        if FLAGS.max_vocab < 0:
            FLAGS.max_vocab = len(vocab)
        transfer_model = CharWordEmbedder(FLAGS.size, vocab, devices[0], name=FLAGS.name)
        FLAGS.transfer_model_lr = FLAGS.learning_rate
    else:
        print("Creating transfer model from config %s" % FLAGS.transfer_model_config)
        with open(FLAGS.transfer_model_config, 'rb') as f:
            transfer_model_config = pickle.load(f)

        if FLAGS.transfer_qa:
            transfer_model = model_from_config(transfer_model_config, devices[0:1])
            vocab = transfer_model.embedder.vocab
        else:
            transfer_model = model_from_config(transfer_model_config, devices[0:1])
            vocab = transfer_model.vocab

    print("Preparing Samplers ...")
    train_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.trainset_prefix)]
    random.shuffle(train_fns)
    print("Training sets (first 100): ", train_fns[:100])
    sampler = SQuADSampler(FLAGS.data, train_fns, FLAGS.batch_size, vocab, FLAGS.max_instances)

    valid_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.validset_prefix)]
    print("Valid sets: (first 100)", valid_fns[:100])
    valid_sampler = SQuADSampler(FLAGS.data, valid_fns, FLAGS.batch_size, vocab, FLAGS.max_instances)
    test_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.testset_prefix)]
    if test_fns:
        print("Test sets: (first 100)", test_fns[:100])
        test_sampler = SQuADSampler(FLAGS.data, test_fns, FLAGS.batch_size, vocab, FLAGS.max_instances)

    embedder_device = devices[0]
    if len(devices) > 1:
        devices = devices[1:]

    print("Creating model with name %s..." % FLAGS.name)
    model = QAPointerModel(FLAGS.size, transfer_model, devices=devices, name=FLAGS.name,
                           keep_prob=1.0-FLAGS.dropout, composition=FLAGS.composition,
                           answer_layer_depth=FLAGS.answer_layer_depth,
                           answer_layer_poolsize=FLAGS.answer_layer_poolsize,
                           answer_layer_type=FLAGS.answer_layer_type)

    trainer = ExtractionQATrainer(FLAGS.learning_rate, model, devices[0], FLAGS.transfer_model_lr)

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

    if FLAGS.init_model_path:
        print("Loading from path " + FLAGS.init_model_path)
        trainer.model.model_saver.restore(sess, FLAGS.init_model_path)
    elif os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: not x.endswith(".meta") and "ckpt" in x, os.listdir(train_dir))),
                     key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        trainer.all_saver.restore(sess, newest)
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
