import json
import os

from biomedical_qa.models import QASetting
from biomedical_qa.sampling.base import BaseSampler


class SQuADSampler(BaseSampler):

    def __init__(self, dir, filenames, batch_size, vocab,
                 instances_per_epoch=None, shuffle=True, dataset_json=None,
                 types=None, split_contexts_on_newline=False):

        if dataset_json is None:
            # load json
            with open(os.path.join(dir, filenames[0])) as dataset_file:
                dataset_json = json.load(dataset_file)
        self.dataset = dataset_json['data']
        self.types = types if types is not None else ["factoid", "list"]
        self.split_contexts_on_newline = split_contexts_on_newline

        BaseSampler.__init__(self, batch_size, vocab, instances_per_epoch, shuffle)


    def build_questions(self):

        char_offsets = dict()
        qas = []

        for article in self.dataset:
            for paragraph in article["paragraphs"]:

                context_str_all = paragraph["context"]
                assert "\n\n" not in context_str_all
                context_strs = context_str_all.split("\n") \
                    if self.split_contexts_on_newline else [context_str_all]

                # Compute <char offset> -> (<context index>, <token index>) map
                char_offset_to_token_index = {}
                contexts = []
                previous_contexts_length = 0
                for context_index, context_str in enumerate(context_strs):

                    context, offsets = self.get_ids_and_offsets(context_str)
                    # Add previous contexts length to offset -> offset in context_str_all
                    offsets = [o + previous_contexts_length for o in offsets]
                    # Add current context length + 1 (for "\n" token)
                    previous_contexts_length += len(context_str) + 1

                    contexts.append(context)

                    for token_index, offset in enumerate(offsets):
                        char_offset_to_token_index[offset] = (context_index, token_index)

                for qa in paragraph["qas"]:

                    answers = []
                    answers_spans = []
                    answers_json = qa["answers"] if "answers" in qa else []
                    answers_json_list = answers_json if len(answers_json) == 0  \
                                                     or isinstance(answers_json[0], list) \
                                                     else [answers_json]
                    for answer_list in answers_json_list:
                        current_answer_spans = []
                        current_answers = []

                        for a in answer_list:
                            answer, _ = self.get_ids_and_offsets(a["text"])
                            if answer and a["answer_start"] in char_offset_to_token_index:
                                context_index, start = char_offset_to_token_index[a["answer_start"]]
                                end = start + len(answer)
                                if (context_index, context_index, end) in answers_spans:
                                    continue
                                current_answer_spans.append((context_index, start, end))
                                current_answers.append(answer)

                        answers_spans.append(current_answer_spans)
                        answers.append(current_answers)

                    q_type = qa["question_type"] if "question_type" in qa else None
                    is_yes = qa["answer_is_yes"] if "answer_is_yes" in qa else None
                    if q_type is None or q_type in self.types:
                        question_tokens = self.get_ids_and_offsets(qa["question"])[0]
                        qas.append(QASetting(question_tokens, answers,
                                             contexts, answers_spans,
                                             id=qa["id"],
                                             q_type=q_type,
                                             is_yes=is_yes,
                                             paragraph_json=paragraph,
                                             question_json=qa))

                    char_offsets[qa["id"]] = {(context_index, token_index) : char_offset
                                              for char_offset, (context_index, token_index)
                                              in char_offset_to_token_index.items()}

        return qas, char_offsets
