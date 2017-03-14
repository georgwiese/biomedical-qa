
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


def build_term2types(terms_file, types_file, case_sensitive=True):

    # Build string -> set<CUI> map
    print("Reading terms file...")
    columns = [UMLS_TERMS_FILE_COLUMNS["term"], UMLS_TERMS_FILE_COLUMNS["cui"]]
    term_rows = umls_read_columns(terms_file, columns, case_sensitive)
    term2concepts = group_by_key(term_rows)
    del term_rows

    # Build CUI -> set<Type> map
    print("Reading types file...")
    columns = [UMLS_TYPES_FILE_COLUMNS["cui"], UMLS_TYPES_FILE_COLUMNS["type"]]
    types_rows = umls_read_columns(types_file, columns, case_sensitive)
    concept2types = group_by_key(types_rows)
    del types_rows

    print("Joining terms & types...")
    term2types = {}
    types_set = set()
    for term, cuis in term2concepts.items():
        term2types[term] = set()
        for cui in cuis:
            for type in concept2types[cui]:
                term2types[term].add(type)
                types_set.add(type)

    print("Done mapping %d terms to %d types" % (
        len(term2types), len(types_set)))

    return term2types, types_set


def build_concept2types(types_file):

    print("Reading types file...")
    columns = [UMLS_TYPES_FILE_COLUMNS["cui"], UMLS_TYPES_FILE_COLUMNS["type"]]
    types_rows = umls_read_columns(types_file, columns)
    concept2types = group_by_key(types_rows)

    types_set = set()
    for types in concept2types.values():
        types_set.update(types)

    print("Done loading %d types." % len(types_set))

    return concept2types, types_set
