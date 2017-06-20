import json

from flask import Flask, request
import tensorflow as tf


from biomedical_qa.data.bioasq_squad_builder import BioAsqSquadBuilder
from biomedical_qa.data.entity_tagger import get_entity_tagger
from biomedical_qa.inference.inference import Inferrer, get_session, get_model
from biomedical_qa.sampling.squad import SQuADSampler
from biomedical_qa.inference.bioasq import insert_answers, clean_bioasq_json

tf.app.flags.DEFINE_integer('port', 5000, 'Port for the server.')
tf.app.flags.DEFINE_string('model_config', None, 'Comma-separated list of paths to the model configs.')
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Number of examples in each batch.")
tf.app.flags.DEFINE_integer("beam_size", 5, "Beam size used for decoding.")
tf.app.flags.DEFINE_float("list_answer_prob_threshold", 0.04, "Beam size used for decoding.")
tf.app.flags.DEFINE_boolean("preferred_terms", False, "If true, uses preferred terms when available.")

FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/answer", methods=['POST'])
def predict():

    data = request.data.decode("utf-8")
    bioasq_json = json.loads(data)
    bioasq_json = clean_bioasq_json(bioasq_json)
    squad_json = BioAsqSquadBuilder(bioasq_json, include_answer_spans=False) \
        .build().get_result_object()


    sampler = SQuADSampler(None, None, FLAGS.batch_size,
                           inferrer.models[0].embedder.vocab,
                           shuffle=False, dataset_json=squad_json,
                           tagger=tagger)
    answers = inferrer.get_predictions(sampler)
    bioasq_json = insert_answers(bioasq_json, answers, FLAGS.list_answer_prob_threshold,
                                 FLAGS.preferred_terms, FLAGS.terms_file)

    return json.dumps(bioasq_json, indent=2)

if __name__ == "__main__":


    devices = FLAGS.devices.split(",")

    sess = get_session()
    models = [get_model(sess, config, devices, scope="model_%d" % i)
              for i, config in enumerate(FLAGS.model_config.split(","))]
    inferrer = Inferrer(models, sess, FLAGS.beam_size)

    tagger = get_entity_tagger()

    app.run(host="0.0.0.0", port=FLAGS.port)
