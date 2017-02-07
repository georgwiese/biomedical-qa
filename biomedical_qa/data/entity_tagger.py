
# Maximum entity length in tokens
MAX_ENTITY_LENGTH = 10


class EntityTagger(object):


    def __init__(self, terms_file, types_file, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.term2types, self.types_set = self._build_term2types(terms_file, types_file)

        self.num_types = len(self.types_set)
        self.type2id = {type: i for i, type in enumerate(self.types_set)}
        self.id2type = {id: type for type, id in self.type2id.items()}


    def _build_term2types(self, terms_file, types_file):

        # Build string -> set<CUI> map
        print("Reading terms file...")
        term_rows = self._read_columns(terms_file, [14, 0])
        term2concepts = self._group_by_key(term_rows)
        del term_rows

        # Build CUI -> set<Type> map
        print("Reading types file...")
        types_rows = self._read_columns(types_file, [0, 3])
        concepts2types = self._group_by_key(types_rows)
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


    def _read_columns(self, filepath, columns_list):

        with open(filepath) as f:
            file_lines = f.readlines()

        rows = []
        for line in file_lines:
            if not self.case_sensitive:
                line = line.lower()
            cells = line.split("|")
            rows.append([cells[i] for i in columns_list])

        return rows


    def _group_by_key(self, key_value_pairs):

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

                if candidate_string in self.term2types:
                    found_entities.add(candidate_string)
                    for token_index in range(i, i + entity_length):
                        types = self.term2types[candidate_string]
                        type_ids = set([self.type2id[type] for type in types])
                        tags[token_index].update(types)
                        tag_ids[token_index].update(type_ids)

        return tags, tag_ids, found_entities

    def _get_token_offsets(self, text, tokenizer):
        offsets = []
        offset = 0
        for token in tokenizer.tokenize(text):
            offset = text.index(token, offset)
            offsets.append((offset, offset + len(token)))
            offset += len(token)
        return offsets
