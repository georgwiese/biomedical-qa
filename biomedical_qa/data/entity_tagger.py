import abc
import json
import requests


# Maximum entity length in tokens
MAX_ENTITY_LENGTH = 10


UMLS_TERMS_FILE_COLUMNS = {
    "cui": 0,
    "term": 14,
}

UMLS_TYPES_FILE_COLUMNS = {
    "cui": 0,
    "type": 3,
}


def umls_read_columns(filepath, columns_list, case_sensitive=True):

    with open(filepath) as f:
        file_lines = f.readlines()

    rows = []
    for line in file_lines:
        if not case_sensitive:
            line = line.lower()
        cells = line.split("|")
        rows.append([cells[i] for i in columns_list])

    return rows


def group_by_key(key_value_pairs):
    """
    Group key-value pairs by key.
    :param key_value_pairs: Iterable of (key, value) pairs
    :return: {key -> set<values>} object
    """

    # For string pooling
    value_strings = {}

    # string -> set<string>
    result = {}

    for key, value in key_value_pairs:

        if value not in value_strings:
            value_strings[value] = value

        if key not in result:
            result[key] = set()

        result[key].add(value_strings[value])

    return result


class EntityTagger(object):


    def initialize_properties(self, types_set):
        """
        Initializes common properties
        :param types_set: set<type_string>
        """

        self.types_set = types_set
        self.num_types = len(self.types_set)
        self.type2id = {type: i for i, type in enumerate(self.types_set)}
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
        pass


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
        self.term2types, self.types_set = self._build_term2types(terms_file, types_file)
        self.blacklist = self._read_blacklist_file(blacklist_file) \
                            if blacklist_file is not None else set()

        self.initialize_properties(self.types_set)


    def _build_term2types(self, terms_file, types_file):

        # Build string -> set<CUI> map
        print("Reading terms file...")
        columns = [UMLS_TERMS_FILE_COLUMNS["term"], UMLS_TERMS_FILE_COLUMNS["cui"]]
        term_rows = umls_read_columns(terms_file, columns, self.case_sensitive)
        term2concepts = group_by_key(term_rows)
        del term_rows

        # Build CUI -> set<Type> map
        print("Reading types file...")
        columns = [UMLS_TYPES_FILE_COLUMNS["cui"], UMLS_TYPES_FILE_COLUMNS["type"]]
        types_rows = umls_read_columns(types_file, columns, self.case_sensitive)
        concepts2types = group_by_key(types_rows)
        del types_rows

        print("Joining terms & types...")
        term2types = {}
        types_set = set()
        for term, cuis in term2concepts.items():
            term2types[term] = set()
            for cui in cuis:
                for type in concepts2types[cui]:
                    term2types[term].add(type)
                    types_set.add(type)

        print("Done mapping %d terms to %d types" % (
                len(term2types), len(types_set)))

        return term2types, types_set


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


class OleloEntityTagger(EntityTagger):


    def __init__(self, types_file, olelo_host):

        self.olelo_host = olelo_host
        self.cui2types, self.types_set = self._read_types_files(types_file)
        self.initialize_properties(self.types_set)


    def _read_types_files(self, types_file):

        print("Reading types file...")
        columns = [UMLS_TYPES_FILE_COLUMNS["cui"], UMLS_TYPES_FILE_COLUMNS["type"]]
        types_rows = umls_read_columns(types_file, columns)
        concepts2types = group_by_key(types_rows)

        types_set = set()
        for types in concepts2types.values():
            types_set.update(types)

        print("Done loading %d types." % len(types_set))

        return concepts2types, types_set


    def tag(self, text, tokenizer):

        olelo_response = self._query_olelo(text)
        offsets = self._get_token_offsets(text, tokenizer)

        tags = [set() for _ in offsets]
        tag_ids = [set() for _ in offsets]
        found_entities = set()

        for entity in olelo_response["entities"]:

            token_range = self._find_token_range(offsets, entity["offset"], len(entity["text"]))

            if token_range is None:
                # Tokens not found in text
                continue

            if entity["normalizedForm"] not in self.cui2types:
                # Unknown entity type
                continue

            start_index, end_index = token_range
            types = self.cui2types[entity["normalizedForm"]]
            type_ids = set([self.type2id[type] for type in types])

            for token_index in range(start_index, end_index + 1):
                tags[token_index].update(types)
                tag_ids[token_index].update(type_ids)

            char_start = offsets[start_index][0]
            char_end = offsets[end_index][1]
            found_entities.add(text[char_start:char_end])

        return tags, tag_ids, found_entities


    def _query_olelo(self, text):

        url = "http://" + self.olelo_host + "/text_analysis/analyze.xsjs"
        params = {"question": text}
        response = requests.get(url, params)

        return json.loads(response.text)["umls"]


    def _find_token_range(self, offsets, entity_start, entity_length):

        entity_end = entity_start + entity_length

        for start_index in range(len(offsets)):

            start, _ = offsets[start_index]

            if start == entity_start:
                for end_index in range(start_index, len(offsets)):

                    _, end = offsets[end_index]
                    if end == entity_end:
                        return (start_index, end_index)

        return None
