import json
import os

from biomedical_qa.models import QASetting
from biomedical_qa.sampling.base import BaseSampler


class SQuADSampler(BaseSampler):

    def __init__(self, dir, filenames, batch_size, vocab,
                 instances_per_epoch=None, shuffle=True, dataset_json=None,
                 types=None):

        if dataset_json is None:
            # load json
            with open(os.path.join(dir, filenames[0])) as dataset_file:
                dataset_json = json.load(dataset_file)
        self.dataset = dataset_json['data']
        self.types = types if types is not None else ["factoid", "list"]

        BaseSampler.__init__(self, batch_size, vocab, instances_per_epoch, shuffle)


    def build_questions(self):

        char_offsets = dict()
        qas = []

        for article in self.dataset:
            for paragraph in article["paragraphs"]:
                context, offsets = self.get_ids_and_offsets(paragraph["context"])
                for qa in paragraph["qas"]:
                    answers = []
                    answer_spans = []
                    answers_json = qa["answers"] if "answers" in qa else []
                    for a in answers_json:
                        answer, _ = self.get_ids_and_offsets(a["text"])
                        if answer and a["answer_start"] in offsets:
                            start = offsets.index(a["answer_start"])
                            if (start, start + len(answer)) in answer_spans:
                                continue
                            answer_spans.append((start, start + len(answer)))
                            answers.append(answer)

                    q_type = qa["question_type"] if "question_type" in qa else None
                    is_yes = qa["answer_is_yes"] if "answer_is_yes" in qa else None
                    if q_type is None or q_type in self.types:
                        question_tokens = self.get_ids_and_offsets(qa["question"])[0]
                        # TODO: Pass Multiple Contexts
                        qas.append(QASetting(question_tokens, answers,
                                             [context], answer_spans,
                                             id=qa["id"],
                                             q_type=q_type,
                                             is_yes=is_yes,
                                             paragraph_json=paragraph,
                                             question_json=qa))

                    char_offsets[qa["id"]] = offsets

        return qas, char_offsets
