import logging

logging.getLogger().setLevel(logging.INFO)

UMLS_TERMS_FILE_COLUMNS = {
    "cui": 0,
    "lat": 1,
    "ts": 2,
    "stt": 4,
    "ispref": 6,
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
        rows.append(tuple([cells[i] for i in columns_list]))

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
    logging.info("Reading terms file...")
    columns = [UMLS_TERMS_FILE_COLUMNS["term"], UMLS_TERMS_FILE_COLUMNS["cui"]]
    term_rows = umls_read_columns(terms_file, columns, case_sensitive)
    term2concepts = group_by_key(term_rows)
    del term_rows

    # Build CUI -> set<Type> map
    logging.info("Reading types file...")
    columns = [UMLS_TYPES_FILE_COLUMNS["cui"], UMLS_TYPES_FILE_COLUMNS["type"]]
    types_rows = umls_read_columns(types_file, columns, case_sensitive)
    concept2types = group_by_key(types_rows)
    del types_rows

    logging.info("Joining terms & types...")
    term2types = {}
    types_set = set()
    for term, cuis in term2concepts.items():
        term2types[term] = set()
        for cui in cuis:
            for type in concept2types[cui]:
                term2types[term].add(type)
                types_set.add(type)

    logging.info("Done mapping %d terms to %d types" % (
        len(term2types), len(types_set)))

    return term2types, types_set


def build_concept2types(types_file):

    logging.info("Reading types file...")
    columns = [UMLS_TYPES_FILE_COLUMNS["cui"], UMLS_TYPES_FILE_COLUMNS["type"]]
    types_rows = umls_read_columns(types_file, columns)
    concept2types = group_by_key(types_rows)

    types_set = set()
    for types in concept2types.values():
        types_set.update(types)

    logging.info("Done loading %d types." % len(types_set))

    return concept2types, types_set


def build_term2preferred(terms_file, case_sensitive=True):
    """Build a <term> -> <preferred term> dictionary.

    Finding the preferred term is implemented as in SQL Query #7 on this site:
    https://www.nlm.nih.gov/research/umls/implementation_resources/query_diagrams/er1.html
    """

    term2preferred = {}

    columns = [UMLS_TERMS_FILE_COLUMNS["cui"],
               UMLS_TERMS_FILE_COLUMNS["lat"],
               UMLS_TERMS_FILE_COLUMNS["ts"],
               UMLS_TERMS_FILE_COLUMNS["stt"],
               UMLS_TERMS_FILE_COLUMNS["ispref"],
               UMLS_TERMS_FILE_COLUMNS["term"],]
    logging.info("Reading terms file...")
    term_rows = umls_read_columns(terms_file, columns, case_sensitive)
    logging.info("Filter for English terms...")
    term_rows = [row for row in term_rows if row[1].lower() == "eng"]
    logging.info("Grouping by concept...")
    term_rows_per_concept = group_by_key([(row[0], row) for row in term_rows])

    def is_preferred(row):
        _, _, ts, stt, ispref, _ = row
        return ts.lower() == "p" and stt.lower() == "pf" and ispref.lower() == "y"

    logging.info("Filling term2preferred...")
    for cui, concept_term_rows in term_rows_per_concept.items():
        preferred_rows = [row for row in concept_term_rows if is_preferred(row)]
        assert len(preferred_rows) != 0, "No preferred term found for CUI %s" % cui
        assert len(preferred_rows) <= 1, "Found multiple preferred terms for CUI %s" % cui
        preferred_term = preferred_rows[0][5]

        for row in concept_term_rows:
            _, _, _, _, _, term = row
            if term in term2preferred and term2preferred[term] != preferred_term:
                # Multiple preferred terms for this term. In that case, use the term itself as its preferred term.
                del term2preferred[term]
                term2preferred[term] = term
            else:
                term2preferred[term] = preferred_term

    return term2preferred
