import tensorflow as tf
import os
import tarfile
import json
import xml.etree.ElementTree as ET

tf.app.flags.DEFINE_string('data_dir', None, 'Path to directory containing all tar.gz files.')
tf.app.flags.DEFINE_string('out_json', None, 'Path to the output JSON file.')

FLAGS = tf.app.flags.FLAGS


def process_tarfile(tar):

    data = []

    members = tar.getmembers()
    for i, member in enumerate(members):

        if i % 10000 == 0:
            print("  Parsing file %d / %d" % (i+1, len(members)))

        if not member.name.endswith(".nxml"):
            continue

        with tar.extractfile(member) as f:
            root = ET.parse(f).getroot()
            title = root.find("front/article-meta/title-group/article-title").text

            if title is None or title[-1] != "?":
                continue

            abstract_xml = ET.tostring(root.find("**/abstract")).decode("utf-8")

            data.append({
                "title": title,
                "abstract_xml": abstract_xml,
            })

    return data


def main():

    data = []

    files = [f for f in os.listdir(FLAGS.data_dir) if f.endswith(".tar.gz")]
    for file in files:
        with tarfile.open(os.path.join(FLAGS.data_dir, file)) as f:
            print("Processing tarfile:", file)
            new_data = process_tarfile(f)
            data += new_data
            print("Done processing tarfile %s. %d Questions added." % (file, len(new_data)))

    with open(FLAGS.out_json, "w") as f:
        json.dump({"data": data}, f, indent=2)


main()
