import abc
import json
import requests

import tensorflow as tf

from biomedical_qa.data.umls import build_term2types, build_concept2types


# Entity tagger settings
tf.app.flags.DEFINE_string("entity_tagger", None, "[dictionary, olelo, ctakes], or None.")
tf.app.flags.DEFINE_string("olelo_url", "https://ares.epic.hpi.uni-potsdam.de/CJosfa64Kz46H7M6/rest/api1/analyze", "Olelo URL.")
tf.app.flags.DEFINE_string("ctakes_url", "http://localhost:9876/ctakes", "CTakes URL.")
tf.app.flags.DEFINE_string("entity_blacklist_file", None, "Blacklist file.")
tf.app.flags.DEFINE_string("terms_file", None, "UML Terms file (MRCONSO.RRF).")
tf.app.flags.DEFINE_string("types_file", None, "UMLS Types file (MRSTY.RRF).")

FLAGS = tf.app.flags.FLAGS


# Maximum entity length in tokens
MAX_ENTITY_LENGTH = 10


class EntityTagger(object):


    def initialize_properties(self, types_set):
        """
        Initializes common properties
        :param types_set: set<type_string>
        """

        self.types_set = types_set
        self.num_types = len(self.types_set)
        self.type2id = {type: i for i, type in enumerate(sorted(self.types_set))}
        self.id2type = {id: type for type, id in self.type2id.items()}


    @abc.abstractmethod
    def tag(self, text, tokenizer):
        """
        Tags a given text.
        :param text: String.
        :param tokenizer: tokenizer.tokenize(text) should return the tokens.
        :return: (tags, tag_ids, found_entities) where
            - tags is a list<set<type_string>> with one entry per token
            - tag_ids is a list<set<type_id>> with one entry per token
            - found_entities is a set<string> of all the entities found
        """
        raise NotImplementedError()


    def _get_token_offsets(self, text, tokenizer):
        offsets = []
        offset = 0
        for token in tokenizer.tokenize(text):
            offset = text.index(token, offset)
            offsets.append((offset, offset + len(token)))
            offset += len(token)
        return offsets


class DictionaryEntityTagger(EntityTagger):


    def __init__(self, terms_file, types_file, case_sensitive=False, blacklist_file=None):

        self.case_sensitive = case_sensitive
        self.term2types, self.types_set = build_term2types(terms_file, types_file, case_sensitive)
        self.blacklist = self._read_blacklist_file(blacklist_file) \
                            if blacklist_file is not None else set()

        self.initialize_properties(self.types_set)


    def _read_blacklist_file(self, blacklist_file):

        print("Reading blacklist file...")

        with open(blacklist_file) as f:
            lines = f.readlines()

        return set([l.strip() for l in lines])


    def tag(self, text, tokenizer):
        if not self.case_sensitive:
            text = text.lower()
        token_offsets = self._get_token_offsets(text, tokenizer)
        tags = [set() for _ in token_offsets]
        tag_ids = [set() for _ in token_offsets]
        found_entities = set()

        for entity_length in range(1, MAX_ENTITY_LENGTH):
            for i in range(len(token_offsets)):

                if i + entity_length > len(token_offsets):
                    continue

                start_offset, _ = token_offsets[i]
                _, end_offset = token_offsets[i + entity_length - 1]
                candidate_string = text[start_offset:end_offset]

                if candidate_string.lower() in self.blacklist:
                    continue

                if candidate_string in self.term2types:
                    found_entities.add(candidate_string)
                    for token_index in range(i, i + entity_length):
                        types = self.term2types[candidate_string]
                        type_ids = set([self.type2id[type] for type in types])
                        tags[token_index].update(types)
                        tag_ids[token_index].update(type_ids)

        return tags, tag_ids, found_entities


class ApiEntityTagger(EntityTagger):


    def __init__(self, types_file, url):

        self.url = url
        self.cui2types, self.types_set = build_concept2types(types_file)
        self.initialize_properties(self.types_set)
        self.session = None


    def tag(self, text, tokenizer):
        offsets = self._get_token_offsets(text, tokenizer)

        tags = [set() for _ in offsets]
        tag_ids = [set() for _ in offsets]
        found_entities = set()

        for token_range, entity_string in self._query_api(text, offsets):

            if entity_string not in self.cui2types:
                # Unknown entity type
                continue

            start_index, end_index = token_range
            types = self.cui2types[entity_string]
            type_ids = set([self.type2id[type] for type in types])

            for token_index in range(start_index, end_index + 1):
                tags[token_index].update(types)
                tag_ids[token_index].update(type_ids)

            char_start = offsets[start_index][0]
            char_end = offsets[end_index][1]
            found_entities.add(text[char_start:char_end])

        return tags, tag_ids, found_entities


    def _find_token_range(self, offsets, entity_start, entity_end):

        for start_index in range(len(offsets)):

            start, _ = offsets[start_index]

            if start == entity_start:
                for end_index in range(start_index, len(offsets)):

                    _, end = offsets[end_index]
                    if end == entity_end:
                        return (start_index, end_index)

        return None


    @abc.abstractmethod
    def _query_api(self, text, token_offsets):
        """Yields (token_range, entity_string) tuples."""

        raise NotImplementedError()


    def _request_json_with_retry(self, params):

        if self.session is None:
            self.session = requests.Session()

        response = self.session.get(self.url, params=params)

        if response.status_code != 200:
            print("Params:", str(params))
            for _ in range(3):
                print("Got status code %d: %s" % (response.status_code, response.text))
                print("Retrying...")
                response = self.session.get(self.url, params=params)
                if response.status_code == 200:
                    print("Recovered!")
                    break

        response.raise_for_status()

        return json.loads(response.text)


class OleloEntityTagger(ApiEntityTagger):


    def __init__(self, types_file, olelo_url):

        ApiEntityTagger.__init__(self, types_file, olelo_url)


    def _query_api(self, text, token_offsets):

        olelo_response = self._query_olelo(text)

        for entity in olelo_response["entities"]:

            token_range = self._find_token_range(token_offsets,
                                                 entity["offset"],
                                                 entity["offset"] + len(entity["text"]))

            if token_range is None:
                # Tokens not found in text
                continue

            yield token_range, entity["normalizedForm"]


    def _query_olelo(self, text):

        params = {"question": text}
        return self._request_json_with_retry(params)["umls"]


class CtakesEntityTagger(ApiEntityTagger):


    def __init__(self, types_file, ctakes_url):

        ApiEntityTagger.__init__(self, types_file, ctakes_url)


    def _query_api(self, text, token_offsets):

        ctakes_response = self._query_ctakes(text)

        for entry in ctakes_response:

            if not "annotation" in entry:
                continue

            annotation = entry["annotation"]

            concepts = annotation.get("ontologyConceptArr", None)

            if concepts is None:
                continue

            token_range = self._find_token_range(token_offsets,
                                                 annotation["begin"],
                                                 annotation["end"])

            if token_range is None:
                continue

            for concept in concepts:

                entity_string = concept["annotation"]["cui"]

                yield token_range, entity_string


    def _query_ctakes(self, text):

        params = {"text": text}
        return self._request_json_with_retry(params)


def get_entity_tagger():

    tagger = None
    if FLAGS.entity_tagger == "dictionary":
        print("Adding Dictionary Tagger")
        tagger = DictionaryEntityTagger(FLAGS.terms_file, FLAGS.types_file,
                                        case_sensitive=True,
                                        blacklist_file=FLAGS.entity_blacklist_file)
    elif FLAGS.entity_tagger == "olelo":
        print("Adding Olelo Tagger")
        tagger = OleloEntityTagger(FLAGS.types_file, FLAGS.olelo_url)
    elif FLAGS.entity_tagger == "ctakes":
        print("Adding CTakes Tagger")
        tagger = CtakesEntityTagger(FLAGS.types_file, FLAGS.ctakes_url)
    elif FLAGS.entity_tagger is not None:
        raise ValueError("Unrecognized entity tagger: %s" % FLAGS.entity_tagger)

    return tagger
