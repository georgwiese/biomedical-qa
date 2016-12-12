import logging
import re

from nltk import RegexpTokenizer


class BioAsqSquadBuilder(object):
    """Converts BioASQ JSON objects to (enriched) SQuAD JSON objects."""


    def __init__(self, bioasq_json, context_token_limit=-1, include_answers=True):

        self._bioasq_json = bioasq_json
        self._tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        self._context_token_limit = context_token_limit
        self._include_answers = include_answers
        self._paragraphs = None
        self._stats = {
            "contexts_truncated": 0,
            "max_context_length": 0,
        }


    def get_paragraphs(self):

        assert self._paragraphs is not None, "Call build() first!"
        return self._paragraphs


    def get_reult_object(self, name):

        return self.build_result_object(name, self._paragraphs)


    def get_stats(self):

        assert self._paragraphs is not None, "Call build() first!"
        return self._stats


    def build_result_object(self, name, paragraphs):

        return {
            "version": "1.0",
            "data": [
                {
                    "title": name,
                    "paragraphs": paragraphs
                }
            ]
        }


    def build(self):

        assert self._paragraphs is None, "Call build() only once!"

        paragraphs = [self.build_paragraph(question)
                      for question in self.filter_questions(
                            self._bioasq_json["questions"])]
        paragraphs = [p for p in paragraphs if p is not None]

        self._paragraphs = paragraphs

        return self


    def filter_questions(self, questions):

        result = []

        for question in questions:

            if not question["type"] in ["factoid", "list"]:
                continue

            if len(question["snippets"]) == 0:
                logging.warning("Skipping question %s. No snippets." % question["id"])
                continue

            result.append(question)

        return result


    def build_paragraph(self, question):

        context = self.get_context(question)

        paragraph = {
            "context": context.lower(),
            "qas": [
                {
                    "id": question["id"],
                    "question": question["body"].lower(),
                    "question_type": question["type"]
                }
            ]
        }

        if self._include_answers:
            answers = self.get_answers(question, context)

            if answers is None:
                return None

            paragraph["qas"][0]["answers"] = answers
            paragraph["qas"][0]["original_answers"] = question["exact_answer"]


        return paragraph


    def get_context(self, question):

        num_tokens = 0
        snippets_set = set()
        snippets = [snippet["text"] for snippet in question["snippets"]]
        filtered_snippets = []

        for snippet in snippets:

            # Deduplicate Snippets
            if snippet in snippets_set:
                continue
            snippets_set.add(snippet)

            # Keep token limit
            snippet_length = len(self._tokenizer.tokenize(snippet))
            if self._context_token_limit > 0 and \
                        num_tokens + snippet_length > self._context_token_limit:
                self._stats["contexts_truncated"] += 1
                break
            num_tokens += snippet_length

            filtered_snippets.append(snippet)

        self._stats["max_context_length"] = max(self._stats["max_context_length"],
                                                num_tokens)

        return " ".join(filtered_snippets)


    def get_answers(self, question, context):

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

            answer = self.clean_answer(answer)

            for start_position in self.find_all_substring_positions(context, answer):
                answer_objects += [
                    {
                        "answer_start": start_position,
                        "text": answer
                    }
                ]

        if not len(answer_objects):
            # Skip question
            logging.warning("Skipping question %s. No matching answer." %
                            question["id"])
            return None

        return answer_objects


    def clean_answer(self, answer):

        answer = answer.strip().lower()
        if answer.startswith("the "):
            answer = answer[4:]
        if re.search(r"[^\w]$", answer) is not None:
            # Ends with punctuation
            answer = answer[:-1]

        return answer


    def find_all_substring_positions(self, string, substring):

        if not len(substring):
            return []

        search_strings = ["\\W(%s)\\W" % re.escape(substring),
                          "^(%s)\\W" % re.escape(substring),
                          "\\W(%s)$" % re.escape(substring)]
        return [m.span(1)[0]
                for search_string in search_strings
                for m in re.finditer(search_string, string)]
