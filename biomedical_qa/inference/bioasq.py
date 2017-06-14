from biomedical_qa.inference.postprocessing import NullPostprocessor, DeduplicatePostprocessor, ProbabilityThresholdPostprocessor, TopKPostprocessor, PreferredTermPostprocessor


def clean_bioasq_json(bioasq_json):
    """Removes empty snippets, handles no snippets."""

    all_olelo_snippets = {}
    for q in bioasq_json["questions"]:
        all_olelo_snippets[q["id"]] = q["snippets"]

    filtered_all_olelo_snippets = {}
    for q_id, snippets in all_olelo_snippets.items():
        filtered_snippets = []
        for snippet in snippets:
            snippet["text"] = snippet["text"].strip()
            if snippet["text"]:
                filtered_snippets.append(snippet)

        if not len(filtered_snippets):
            filtered_snippets = [{"text": "This is a dummy snippet.", "is_dummy": True}]

        filtered_all_olelo_snippets[q_id] = filtered_snippets

    all_olelo_snippets = filtered_all_olelo_snippets

    for question in bioasq_json["questions"]:
        question["snippets"] = all_olelo_snippets[question["id"]]

    return bioasq_json


def insert_answers(bioasq_json, answers, list_answer_prob_threshold,
                   use_preferred_terms=False, terms_file=None):
    """Inserts answers into bioasq_json from a
    <question id> -> <(<answer_string>, <prob>) iterable>."""

    questions = []

    base_postprocessor = NullPostprocessor()
    if use_preferred_terms:
        base_postprocessor = base_postprocessor.chain(PreferredTermPostprocessor(terms_file))
    base_postprocessor = base_postprocessor.chain(DeduplicatePostprocessor())

    factoid_postprocessor = base_postprocessor.chain(TopKPostprocessor(5))
    list_postprocessor = base_postprocessor.chain(ProbabilityThresholdPostprocessor(list_answer_prob_threshold, 1))

    for question in bioasq_json["questions"]:
        q_id = question["id"]

        if q_id in answers:

            if question["type"] in ["factoid", "list"]:

                if question["type"] == "list":
                    answer_strings = [answer_string
                                      for answer_string, answer_prob in list_postprocessor.process(answers[q_id])]
                else:
                    answer_strings = [answer_string
                                      for answer_string, answer_prob in factoid_postprocessor.process(answers[q_id])]

                if len(answer_strings) == 0:
                    answer_strings = [answers[q_id].answer_strings[0]]

                question["exact_answer"] = [[s] for s in answer_strings]

        if question["type"] == "yesno":

            # Strong baseline :)
            question["exact_answer"] = "yes"

        question["ideal_answer"] = ""
        questions.append(question)

    return {"questions": questions}
