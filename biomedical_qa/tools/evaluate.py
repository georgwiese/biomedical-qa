import os
import pickle
import tensorflow as tf

from biomedical_qa.models import model_from_config
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.training.qa_trainer import ExtractionQATrainer

tf.app.flags.DEFINE_string('eval_data', None, 'Path to the SQuAD JSON file.')
tf.app.flags.DEFINE_string('transfer_model_config', None, 'Path to the Transfer Model config (needed for vocab).')
tf.app.flags.DEFINE_string('model_config', None, 'Path to the Model config.')
tf.app.flags.DEFINE_string('model_weights', None, 'Path to the Model weights.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")

tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")

FLAGS = tf.app.flags.FLAGS

devices = FLAGS.devices.split(",")

print("Loading Model...")
with open(FLAGS.model_config, 'rb') as f:
    model_config = pickle.load(f)
model = model_from_config(model_config, devices)

print("Loading Vocab...")
with open(FLAGS.transfer_model_config, 'rb') as f:
    transfer_model_config = pickle.load(f)
vocab = transfer_model_config["vocab"]

print("Restoring Weights...")
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    model.model_saver.restore(sess, FLAGS.model_weights)

    print("Initializing Sampler & Trainer...")
    data_dir = os.path.dirname(FLAGS.eval_data)
    data_filename = os.path.basename(FLAGS.eval_data)
    valid_sampler = SQuADSampler(data_dir, [data_filename], FLAGS.batch_size, vocab)
    trainer = ExtractionQATrainer(0, model, devices[0])

    print("Running SQuAD Evaluation...")
    trainer.eval(sess, valid_sampler, verbose=True)


