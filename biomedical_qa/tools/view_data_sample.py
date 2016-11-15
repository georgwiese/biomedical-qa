import sys
import json
import random

from collections import Counter


DATA_FILE = sys.argv[1]


def read_data(filename):

  print("Start reading data...")
  with open(filename) as json_file:
    data = json.load(json_file)

  return data



def print_stats(data):

  print("Stats:")
  print("  Number of Questions: %d" % len(data["questions"]))

  count_by_type = Counter([q["type"] for q in data["questions"]])
  for question_type, count in count_by_type.items():
    print("    %s: %d" % (question_type, count))


def print_question_sample(data, count):

  print("Question sample:")
  questions = data["questions"]

  for _ in range(count):
    index = random.randint(0, len(questions))
    print_question(questions[index])


def print_question(question):

  print("  %s" % question["body"])
  if "exact_answer" in question:
    print("    Answer: %s" % str(question["exact_answer"]))
  else:
    print("    Answer: N/A")

  #print("    Snippets:")
  #for snippet in question["snippets"]:
  #  print("      * %s" % snippet["text"])


if __name__ == "__main__":

  data = read_data(DATA_FILE)
  print_stats(data)
  print_question_sample(data, 10)
