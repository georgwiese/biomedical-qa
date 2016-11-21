import json
import os
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer
import random

from biomedical_qa.models import QASetting


class SQuADSampler:
    def __init__(self, dir, filenames, batch_size, vocab,
                 instances_per_epoch=None, shuffle=True):
        self.__batch_size = batch_size
        self.unk_id = vocab["<UNK>"]
        self.start_id = vocab["<S>"]
        self.end_id = vocab["</S>"]
        self.vocab = vocab
        self._instances_per_epoch = instances_per_epoch
        self.num_batches = 0
        self.epoch = 0
        self._rng = random.Random(28739)
        # load json
        with open(os.path.join(dir, filenames[0])) as dataset_file:
            dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
        self._qas = []

        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

        def trfm(s):
            idxs = []
            offsets = []
            offset = 0
            for t in tokenizer.tokenize(s):
                offset = s.index(t, offset)
                offsets.append(offset)
                i = vocab.get(t, self.unk_id)
                idxs.append(i)
                offset += len(t)
            return idxs, offsets

        for article in dataset:
            for paragraph in article["paragraphs"]:
                context, offsets = trfm(paragraph["context"])
                for qa in paragraph["qas"]:
                    answers = []
                    answer_spans = []
                    for a in qa["answers"]:
                        answer = trfm(a["text"])[0]
                        if a["answer_start"] in offsets:
                            start = offsets.index(a["answer_start"])
                            if (start, start + len(answer)) in answer_spans:
                                continue
                            answer_spans.append((start, start + len(answer)))
                            answers.append(answer)
                    self._qas.append(QASetting(trfm(qa["question"])[0], answers, context, answer_spans))

        if shuffle:
            self._rng.shuffle(self._qas)
        if instances_per_epoch is not None:
            self._qas = self._qas[:instances_per_epoch]
        self._idx = 0

    def get_batch(self):
        qa_settings = [self._qas[i+self._idx] for i in range(min(self.__batch_size, len(self._qas) - self._idx))]
        self._idx += len(qa_settings)

        if self._idx == len(self._qas):
            self.epoch += 1
            self._rng.shuffle(self._qas)
            self._idx = 0

        return qa_settings

    def reset(self):
        self._idx = 0
