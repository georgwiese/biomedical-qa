import os
import string
import sys
import json
from concurrent.futures import ProcessPoolExecutor

import time

msmarco_json = sys.argv[1]
output_json = sys.argv[2]
answer_types = sys.argv[3].split(",")
with_answers = sys.argv[4] != "no_answers" if len(sys.argv) > 4 else True

for answer_type in answer_types:
    assert answer_type in ["factoid", "yesno"]

dataset = []
squad_style_dataset = {"data": dataset, "version": "1"}

msmarco = []
with open(msmarco_json, "r") as f:
    for l in f:
        msmarco.append(json.loads(l))

from nltk.corpus import stopwords
from nltk import RegexpTokenizer
stop_word_set = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

def normalize_answer(s):
    return " ".join(w for w in tokenizer.tokenize(s.lower()) if w not in stop_word_set)

def process(d):
    passages = "\n".join(p["passage_text"] for p in d["passages"])
    passages_lower = passages.lower()
    answers = [a for a in d.get("answers", [])]
    extractive_answers = []
    starts = []
    is_yesno = False
    is_yes = None
    if with_answers:
        for a in answers:
            if a.lower() == "yes" or a.lower() == "no":
                is_yesno = True
                is_yes = a.lower() == "yes"
                break
            index = passages_lower.find(a.lower())
            if index >= 0:
                extractive_answers.append(a)
                starts.append(index)
            else:
                # remove punctuation at the end of answer and try again
                if a[-1] in string.punctuation:
                    a = a[:-1].strip()
                index = passages_lower.find(a.lower())
                if index >= 0:
                    extractive_answers.append(a)
                    starts.append(index)

    if with_answers and is_yesno and "yesno" in answer_types:
        return {
            "title": d["query_id"],
            "paragraphs": [
                {
                    "context": passages,
                    "qas": [{
                        "question": d["query"],
                        "question_type": "yesno",
                        "id": d["query_id"],
                        "answers": [],
                        "answer_is_yes": is_yes
                    }]
                }
            ]}
    elif not with_answers or (extractive_answers and "factoid" in answer_types):
        example = {"title": d["query_id"], "paragraphs": [
            {
                "context": passages,
                "qas": [{
                    "question": d["query"],
                    "question_type": "factoid",
                    "id": d["query_id"],
                    "answers": [{
                                    "answer_start": s,
                                    "text": a
                                } for a, s in zip(extractive_answers, starts)]
                }]
            }
        ]}
        return example
    else:
        return None

counter = 0
extracted = 0
pool = ProcessPoolExecutor()
start = time.time()
chunk_size = 10
for example in pool.map(process, msmarco, chunksize=chunk_size):
    counter += 1
    if example is not None:
        dataset.append(example)
        extracted += 1

    if counter % chunk_size == 0:
        sys.stdout.write("\r%d examples processed, %.3f extracted. %.3f s/example..." % (counter,
                                                                                         extracted/counter,
                                                                                         (time.time()-start)/counter))
        sys.stdout.flush()

print()
print("Recovered %d of %d questions." % (len(dataset), len(msmarco)))

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump(squad_style_dataset, f)
