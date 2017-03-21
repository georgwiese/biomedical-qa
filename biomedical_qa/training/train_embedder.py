import os
import random
import time
from quebap.projects.autoread import *
import tensorflow as tf
import sys
import functools
import web.embeddings as embeddings
import numpy as np
import pickle

from quebap.projects.autoread.models import *
from quebap.projects.autoread.models.embedder import ConcatEmbedder
from quebap.projects.autoread.util import load_vocab

tf.app.flags.DEFINE_string('data', None, 'Path to extracted TFRecord data. Assuming contains document.vocab')
tf.app.flags.DEFINE_string("trainset_prefix", "train", "Comma separated datasets to train on.")
tf.app.flags.DEFINE_string("validset_prefix", "valid", "Development set")
#tf.app.flags.DEFINE_string("testset_prefix", "test", "Test set.")

# model
tf.app.flags.DEFINE_integer("size", 256, "hidden size of model")
tf.app.flags.DEFINE_integer("tape_length", 50, "length of memory tape if embedder_type is 'attention_context'")
tf.app.flags.DEFINE_string("composition", 'GRU', "'LSTM', 'GRU'")
tf.app.flags.DEFINE_string("name", 'AutoReader', "Name of the model")

#training
tf.app.flags.DEFINE_float("dropout", 0.0, "Dropout.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_float("beta", 0.0, "Word freq smoothing factor for weighted training.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", 1000, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("min_epochs", 1, "Minimum number of epochs.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from.")
tf.app.flags.DEFINE_string("embeddings", None, "Init with word embeddings from given path in w2v binary format.")
tf.app.flags.DEFINE_string("max_context_length", 300, "Maximum length of context.")
tf.app.flags.DEFINE_string("dataset", "wikireading", "wikireading, txt")
tf.app.flags.DEFINE_string("task", "cloze", "task on which to train ['cloze']")


#underlying embedder
tf.app.flags.DEFINE_string("embedder_type", "rnn_context", "Type of Embedder to train. ['word','context'].")
tf.app.flags.DEFINE_string("embedder_config", None, "Path to underlying embedder config.")
tf.app.flags.DEFINE_string("embedder_path", None, "Path to underlying embedder model.")

FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

if os.path.exists(os.path.join(FLAGS.data, "document.vocab")):
    vocab, idxword, word_freq = load_vocab(os.path.join(FLAGS.data, "document.vocab"))
else:
    vocab, word_freq = dict(), dict()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    devices = FLAGS.devices.split(",")
    if FLAGS.embedder_config is not None:
        with open(FLAGS.embedder_config, 'rb') as f:
            emb_config = pickle.load(f)

        vocab = emb_config["vocab"]
        if "<UNK>" not in vocab:
            vocab["<UNK>"] = len(vocab)

        embedder = CharWordEmbedder(FLAGS.size, vocab, devices[0], name=FLAGS.name)
        pretrained_embedder = model_from_config(emb_config, devices, 0.0, embedder.inputs,
                                                embedder.seq_lengths)
        embedder = ConcatEmbedder([embedder, pretrained_embedder])
    else:
        embedder = CharWordEmbedder(FLAGS.size, vocab, devices[0], name=FLAGS.name)

    if FLAGS.embedder_type == "rnn_context":
        model = RNNContextEmbedder(embedder, FLAGS.size, devices=devices,
                                   dropout=FLAGS.dropout,
                                   composition=FLAGS.composition, name=FLAGS.name)
    elif FLAGS.embedder_type == "attention_context":
        model = AttentionMemoryContextEmbedder(embedder, FLAGS.tape_length, devices=devices,
                                               dropout=FLAGS.dropout,
                                               name=FLAGS.name)


    print("Preparing Samplers ...")
    train_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.trainset_prefix)]
    random.shuffle(train_fns)
    print("Training sets: ", train_fns)
    sampler = sampler_for(FLAGS.dataset, FLAGS.task, sess, FLAGS.data, train_fns, FLAGS.batch_size,
                          max_length=FLAGS.max_context_length, vocab=vocab, word_freq=word_freq, beta=FLAGS.beta)

    train_dir = os.path.join(FLAGS.save_dir)
    dev_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.validset_prefix)]
    print("Valid sets: ", dev_fns)
    valid_sampler = sampler_for(FLAGS.dataset, FLAGS.task, sess, FLAGS.data, dev_fns, FLAGS.batch_size,
                                max_length=FLAGS.max_context_length, vocab=vocab,
                                instances_per_epoch=100 * FLAGS.batch_size)
    #test_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.testset_prefix)]
    #print("Test sets: ", test_fns)
    #test_sampler = BatchSampler(sess, FLAGS.data, test_fns, FLAGS.batch_size, max_vocab=FLAGS.max_vocab,
    #                            max_answer_vocab=FLAGS.max_vocab,
    #                            max_length=FLAGS.max_context_length, vocab=word_ids)

    trainer = trainer_for(FLAGS.task, FLAGS.learning_rate, model, devices[0], sampler.unk_id)
    print("Created model!")

    checkpoint_path = os.path.join(train_dir, "model.ckpt")
    model_path = os.path.join(train_dir, "model.tf")

    previous_loss = list()
    epoch = 0

    print("Initializing variables ...")
    sess.run(tf.global_variables_initializer())
    if FLAGS.embedder_path is not None:
        pretrained_embedder.model_saver.restore(sess, FLAGS.embedder_path)

    if FLAGS.init_model_path:
        print("Loading from path " + FLAGS.init_model_path)
        model.model_saver.restore(sess, FLAGS.init_model_path)
    elif os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: not x.endswith(".meta") and "ckpt" in x, os.listdir(train_dir))),
                     key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        trainer.all_saver.restore(sess, newest)
    elif not os.path.exists(train_dir):
            os.makedirs(train_dir)

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)

    print("Initialized model.")
    with open(os.path.join(train_dir, "config.pickle"), 'wb') as f:
        pickle.dump(model.get_config(), f)

    def validate():
        # Run evals as cloze-QA on development set and print(their loss.)
        print("########## Validation ##############")
        l = trainer.eval(sess, valid_sampler, verbose=True)
        print("####################################")
        trainer.model.set_train(sess)
        if not previous_loss or l < min(previous_loss):
            trainer.all_saver.save(sess, checkpoint_path, global_step=trainer.global_step, write_meta_graph=False)
            trainer.model.model_saver.save(sess, model_path, write_meta_graph=False)

        if previous_loss and l > previous_loss[-1]:
            # if no significant improvement decay learningrate
            print("Decaying learningrate.")
            trainer.decay_learning_rate(sess, FLAGS.learning_rate_decay)

        previous_loss.append(l)
        return l

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    loss = 0.0
    step_time = 0.0
    i = 0
    while True:
        i += 1
        start_time = time.time()
        batch = sampler.get_batch()
        # already fetch next batch parallel to running model
        loss += trainer.run(sess, [trainer.update, trainer.loss], batch)[1]
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" % (i*100.0 / FLAGS.ckpt_its, loss / i))
        sys.stdout.flush()

        if i % FLAGS.ckpt_its == 0:
            i = 0
            loss /= FLAGS.ckpt_its
            print("")
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.5f, step-time %.3f, loss %.4f" % (trainer.global_step.eval(),
                                                                                    trainer.learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            validate()
