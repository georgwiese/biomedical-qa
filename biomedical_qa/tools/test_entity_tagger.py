import tensorflow as tf
from nltk.tokenize import RegexpTokenizer

from biomedical_qa.data.entity_tagger import DictionaryEntityTagger, OleloEntityTagger, CtakesEntityTagger


tf.app.flags.DEFINE_string('terms_file', None, 'UML Terms file (MRCONSO.RRF).')
tf.app.flags.DEFINE_string('types_file', None, 'UMLS Types file (MRSTY.RRF).')
tf.app.flags.DEFINE_string('blacklist_file', None, 'Entities to black list.')
tf.app.flags.DEFINE_string('olelo_url', 'https://ares.epic.hpi.uni-potsdam.de/CJosfa64Kz46H7M6/rest/api1/analyze', 'Olelo URL.')
tf.app.flags.DEFINE_string('ctakes_url', 'http://localhost:9876/ctakes', 'CTakes URL.')


FLAGS = tf.app.flags.FLAGS


def main():

    # tagger = OleloEntityTagger(FLAGS.types_file, FLAGS.olelo_url)
    tagger = CtakesEntityTagger(FLAGS.types_file, FLAGS.ctakes_url)
    # tagger = DictionaryEntityTagger(FLAGS.terms_file, FLAGS.types_file,
    #                                 case_sensitive=True,
    #                                 blacklist_file=FLAGS.blacklist_file)
    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

    # text = "What are the indications for hydrochlorothiazide?"
    text = "Here, we show that BIBW2992, an anilino-quinazoline designed to irreversibly bind EGFR and HER2, potently suppresses the kinase activity of wild-type and activated EGFR and HER2 mutants, including erlotinib-resistant isoforms"


    tags, tag_ids, entities = tagger.tag(text, tokenizer)
    print(tags)
    print(tag_ids)
    print(entities)

main()
