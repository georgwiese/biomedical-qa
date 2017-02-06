import tensorflow as tf
import os
import tarfile
import json
import nltk.data
import xml.etree.ElementTree as ET
import logging
from html2text import html2text
from multiprocessing import Pool

tf.app.flags.DEFINE_string('data_dir', None, 'Path to directory containing all tar.gz files.')
tf.app.flags.DEFINE_string('out_json', None, 'Path to the output JSON file.')
tf.app.flags.DEFINE_string('out_dir', None, 'Path to the output directory for extracted xml.')
tf.app.flags.DEFINE_string('extract', "question_titles", '(question_titles|all_questions|all).')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'Keep prob.')
tf.app.flags.DEFINE_integer('threads', 4, 'Number of threads.')

FLAGS = tf.app.flags.FLAGS

def iter_xmls(tar_file):

    import time, random

    random.seed(hash(tar_file))

    print("Processing tarfile:", tar_file)

    with tarfile.open(os.path.join(FLAGS.data_dir, tar_file)) as tar:

        members = tar.getmembers()
        start_time = time.time()
        for i, member in enumerate(members):

            if i % 1000 == 0:
                end_time = time.time()
                total_time = end_time - start_time
                start_time = end_time
                print("  [%s] Parsing file %d / %d. (%fs / 1000 files)" % (
                    tar_file, i+1, len(members), total_time))

            if not member.name.endswith(".nxml"):
                continue

            if random.random() > FLAGS.keep_prob:
                continue

            try:
                with tar.extractfile(member) as f:
                    xml_text = f.read()
                    yield xml_text, member.name
            except:
                logging.error("Error Parsing member: %s" % member.name)


def process_tarfile_question_titles(tar):

    data = []

    for xml_text, filename in iter_xmls(tar):

        root = ET.fromstring(xml_text)

        title_node = root.find("front/article-meta/title-group/article-title")

        if title_node is None or title_node.text is None:
            logging.warning("No title: %s" % filename)
            continue

        title = title_node.text

        if title is None or title[-1] != "?":
            continue

        abstract_node = root.find("**/abstract")
        if abstract_node is None:
            logging.warning("No abstract: %s" % filename)
            continue
        abstract_xml = ET.tostring(abstract_node).decode("utf-8")

        data.append({
            "id": filename,
            "title": title,
            "abstract_xml": abstract_xml,
        })

    print("Done processing tarfile %s. %d Questions added." % (tar, len(data)))

    return data


def process_tarfile_all_questions(tar):

    data = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for xml_text, filename in iter_xmls(tar):

        text = html2text(xml_text.decode("utf-8"))
        sentences = tokenizer.tokenize(text)
        questions = [s for s in sentences if s[-1] == "?"]

        for question in questions:
            data.append({
                "filename": filename,
                "question": question,
            })

    print("Done processing tarfile %s. %d Questions added." % (tar, len(data)))

    return data


def process_tarfile_all(tar):

    for xml_text, filename in iter_xmls(tar):

        os.makedirs(FLAGS.out_dir, exist_ok=True)
        full_path = os.path.join(FLAGS.out_dir, filename.replace("/", "__"))

        with open(full_path, "w") as f:
            f.write(xml_text.decode("utf-8"))


def main():

    pool = Pool(FLAGS.threads)

    process_tarfile = None
    if FLAGS.extract == "question_titles":
        process_tarfile = process_tarfile_question_titles
    elif FLAGS.extract == "all_questions":
        process_tarfile = process_tarfile_all_questions
    elif FLAGS.extract == "all":
        process_tarfile = process_tarfile_all

    files = [f for f in os.listdir(FLAGS.data_dir) if f.endswith(".tar.gz")]
    data = pool.map(process_tarfile, files)

    # process_tarfile_all() already writes everything and doesn't return
    if FLAGS.extract != "all":
        data = [d for group in data for d in group]

        with open(FLAGS.out_json, "w") as f:
            json.dump({"data": data}, f, indent=2)


main()
