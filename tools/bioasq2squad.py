import sys
import json
import logging
import re
import numpy as np


TRAIN_FRACTION = 0.8


def convert_to_squad(bioasq_file_path):

  with open(bioasq_file_path) as json_file:
    data = json.load(json_file)

  paragraphs = build_paragraphs(data)
  train_paragraphs, dev_paragraphs = split_paragraphs(paragraphs)

  with open("train.json", "w") as out_file:
    name = bioasq_file_path + " - train"
    result = build_result_object(name, train_paragraphs)
    json.dump(result, out_file, indent=2)

  with open("dev.json", "w") as out_file:
    name = bioasq_file_path + " - dev"
    result = build_result_object(name, dev_paragraphs)
    json.dump(result, out_file, indent=2)

  print("Done. Extracted %d questions (%d train & %d dev)"
        % (len(paragraphs), len(train_paragraphs), len(dev_paragraphs)))


def split_paragraphs(paragraphs):

  dataset_size = len(paragraphs)
  train_size = int(TRAIN_FRACTION * dataset_size)
  dev_size = dataset_size - train_size

  train_indices = np.random.choice(dataset_size, train_size, replace=False)

  train_paragraphs, dev_paragraphs = [], []
  for i in np.random.permutation(dataset_size):
    if i in train_indices:
      train_paragraphs += [paragraphs[i]]
    else:
      dev_paragraphs += [paragraphs[i]]

  return train_paragraphs, dev_paragraphs


def build_result_object(name, paragraphs):

  return {
    "version": "1.0",
    "data": [
      {
        "title": name,
        "paragraphs": paragraphs
      }
    ]
  }


def build_paragraphs(data):

  paragraphs = [build_paragraph(question)
                for question in filter_questions(data["questions"])]
  return [p for p in paragraphs if p is not None]


def filter_questions(questions):

  result = []

  for question in questions:

    if not question["type"] in ["factoid", "list"]:
      continue

    if len(question["snippets"]) == 0:
      logging.warning("Skipping question %s. No snippets." % question["id"])
      continue

    result.append(question)

  return result


def build_paragraph(question):

  context = get_context(question)
  answers = get_answers(question, context)

  if answers is None:
    return None

  return {
    "context": context.lower(),
    "qas": [
      {
        "id": question["id"],
        "question": question["body"].lower(),
        "answers": answers
      }
    ]
  }


def get_context(question):

  snippets = [snippet["text"] for snippet in question["snippets"]]
  return "\n".join(snippets)


def get_answers(question, context):

  context = context.lower()

  def find_best_answer(answers):

    for answer in answers:
      if answer.lower() in context and len(answer):
        return answer

    return answers[0]

  answers = [a if isinstance(a, str) else find_best_answer(a)
             for a in question["exact_answer"]]

  assert len(answers)

  answer_objects = []

  for answer in answers:

    for start_position in find_all_substring_positions(context, answer.lower()):
      answer_objects += [
        {
          "answer_start": start_position,
          "text": answer.lower()
        }
      ]

  if not len(answer_objects):
    # Skip question
    logging.warning("Skipping question %s. No matching answer." %
                    question["id"])
    return None

  return answer_objects


def find_all_substring_positions(string, substring):

  if not len(substring):
    return []

  search_string = "\\W%s\\W" % re.escape(substring)
  return [m.start() + 1 for m in re.finditer(search_string, string)]


if __name__ == "__main__":

  if len(sys.argv) < 2:

    print("Usage: %s <BioASQ json file>" % sys.argv[0])
    exit(1)

  bioasq_file_path = sys.argv[1]
  convert_to_squad(bioasq_file_path)
