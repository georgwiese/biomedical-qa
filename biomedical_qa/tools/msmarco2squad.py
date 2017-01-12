import string
import sys
import json
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import time

msmarco_json = sys.argv[1]
output_json = sys.argv[2]
with_answers = sys.argv[3] != "no_answers" if len(sys.argv) > 3 else True


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
    if with_answers:
        for a in answers:
            if a == "Yes" or a == "No":
                continue
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

    if not with_answers or extractive_answers:
        example = {"title": d["query_id"], "paragraphs": [
            {
                "context": passages,
                "qas": [{
                    "question": d["query"],
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

with open(output_json, "w") as f:
    json.dump(squad_style_dataset, f)
