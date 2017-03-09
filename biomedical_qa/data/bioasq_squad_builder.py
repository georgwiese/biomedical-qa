import logging
import re

from nltk import RegexpTokenizer



def ensure_list_depth_2(l):

    max_depth = max_list_depth(l)

    if max_depth == 1:
        return [l]
    elif max_depth == 2:
        return l
    else:
        raise ValueError("Unexpected list depth: %d (%s)" %
                         (max_depth, str(l)))


def max_list_depth(l):

    if not isinstance(l, str):
        item_depths = [max_list_depth(x) for x in l]
        if min(item_depths) != max(item_depths):
            logging.warning("Inconsistent answer list: %s" % str(l))
        return max(item_depths) + 1
    return 0


class BioAsqSquadBuilder(object):
    """Converts BioASQ JSON objects to (enriched) SQuAD JSON objects."""


    def __init__(self, bioasq_json, context_token_limit=-1,
                 types=None, include_synonyms=False):
        """
        Creates the BioAsqSquadBuilder.
        :param bioasq_json: The BioASQ JSON object.
        :param context_token_limit: If larger than 0, contexts will only be
                added as long as the token limit is not exceeded.
        :param types: Question types to include
        :param include_synonyms: If True, the answers object is a list of lists
                (which is NOT the SQuAD format) with the outer list containing
                the answers (i.e., correct answers of the list question) and
                inner list containing the synonyms. If False, the answers object
                is a flat list and only one synonym is included.
        """

        self._bioasq_json = bioasq_json
        self._types = types
        if self._types is None:
            self._types = ["factoid", "list"]

        self._tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        self._context_token_limit = context_token_limit
        self._include_synonyms = include_synonyms
        self._paragraphs = None
        self._stats = {
            "contexts_truncated": 0,
            "max_context_length": 0,
        }


    def get_paragraphs(self):

        assert self._paragraphs is not None, "Call build() first!"
        return self._paragraphs


    def get_result_object(self, name="BioASQ"):

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

            if not question["type"] in self._types:
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
            "context_original_capitalization": context,
            "qas": [
                {
                    "id": question["id"],
                    "question": question["body"].lower(),
                    "question_original_capitalization": question["body"],
                    "question_type": question["type"]
                }
            ]
        }

        if "exact_answer" in question:

            if question["type"] in ["factoid", "list"]:
                answers = self.get_extractive_answers(question, context)

                if answers is None:
                    return None

                paragraph["qas"][0]["answers"] = answers
                paragraph["qas"][0]["original_answers"] = question["exact_answer"]

            if question["type"] == "yesno":

                is_yes = question["exact_answer"].lower().strip() in ["yes", "yes."]
                paragraph["qas"][0]["answer_is_yes"] = is_yes

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

        return "\n".join(filtered_snippets)


    def get_extractive_answers(self, question, context):

        context = context.lower()

        def find_best_synonym(answers):

            for answer in answers:
                if answer.lower() in context and len(answer):
                    return answer

            return answers[0]

        answers = ensure_list_depth_2(question["exact_answer"])

        assert len(answers)

        answer_objects = []

        for answer in answers:

            if not self._include_synonyms:
                # Just use one synonym
                answer = [find_best_synonym(answer)]

            answer_object_list = []

            for synonym in answer:

                synonym = self.clean_answer(synonym)

                for start_position in self.find_all_substring_positions(context, synonym):
                    answer_object_list += [
                        {
                            "answer_start": start_position,
                            "text": synonym
                        }
                    ]

            if len(answer_object_list) == 0:
                continue

            if self._include_synonyms:
                # Add a list of answer objects
                answer_objects.append(answer_object_list)
            else:
                # Just add the answer objects to a flat list:
                answer_objects += answer_object_list

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
        if re.search(r"[^\w\)]$", answer) is not None:
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
