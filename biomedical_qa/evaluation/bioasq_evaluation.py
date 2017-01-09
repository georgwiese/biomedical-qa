import logging
import sys

import numpy as np

from biomedical_qa.data.bioasq_squad_builder import ensure_list_depth_2

def bioasq_evaluation(sampler, inferrer, verbosity_level=0,
                      list_answer_count=5, list_answer_prob_threshold=0.5):

    if inferrer.beam_size < 5:
        logging.warning("Beam size should be at least 5 in order to get 5 ranked answers.")

    questions = sampler.get_questions()

    # Assuming one question per paragraph
    count = len(questions)
    subsample = sampler.instances_per_epoch
    print("  (Questions: %d, using %d)" %
          (count, subsample if subsample is not None else count))

    factoid_correct, factoid_total = 0, 0
    factoid_reciprocal_rank_sum = 0
    list_f1_sum, list_total = 0, 0

    if verbosity_level > 0:
      print("  Doing predictions...")
    predictions = inferrer.get_predictions(sampler)

    for question in questions:

        prediction = predictions[question.id]

        correct_answers = question.question_json["original_answers"]
        correct_answers = ensure_list_depth_2(correct_answers)
        question_type = question.q_type

        answers = list(prediction)

        if verbosity_level > 1:
            print("-------------")
            print("  ID:", question.id)
            print("  Given: ", answers)
            print("  Correct: ", correct_answers)

        if question_type == "factoid":
            factoid_total += 1
            exact_math_found = False
            rank = sys.maxsize
            for correct_answer in correct_answers[0]:
                # Compute exact match
                if not exact_math_found and \
                        answers[0][0].lower() == correct_answer.lower():
                    if verbosity_level > 1:
                        print("  Correct!")
                    factoid_correct += 1
                    exact_math_found = True
                # Compute rank
                for k in range(min(len(answers), 5)):
                    if answers[k][0].lower() == correct_answer.lower():
                        rank = min(rank, k + 1)

            if verbosity_level > 1:
                print("  Rank: %d" % (rank if rank <= 5 else -1))
            factoid_reciprocal_rank_sum += 1 / rank if rank <= 5 else 0


        if question_type == "list":

            if inferrer.model.start_output_unit == "sigmoid":
                # We get individual probabilities for each answer, can threshold.
                filtered_answers = [(a, prob) for a, prob in answers
                                    if prob >= list_answer_prob_threshold]
                answers = filtered_answers if len(filtered_answers) > 0 else answers[:1]
            else:
                # We can't apply an absolute threshold, so use a fixed count.
                answers = answers[:list_answer_count]

            if verbosity_level > 1:
                print("  Using answers:", [a for a, _ in answers])

            list_total += 1
            answer_correct = np.zeros([len(answers)], dtype=np.bool)

            for answer_option in correct_answers:
                for correct_answer in answer_option:
                    for k in range(len(answers)):

                        # Count answer if it hasn't yet been counted as correct.
                        if not answer_correct[k] and \
                                answers[k][0].lower() == correct_answer.lower():
                            answer_correct[k] = True
                            # Only count one synonym.
                            break

            tp = np.count_nonzero(answer_correct)
            precision = tp / len(answers)
            recall = tp / len(correct_answers)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            if verbosity_level > 1:
                print("F1: %f, precision: %f, recall: %f" % (f1, precision, recall))
            list_f1_sum += f1

    if verbosity_level > 0:
        print("Factoid correct: %d / %d" % (factoid_correct, factoid_total))
        print("Factoid MRR: %f" % (factoid_reciprocal_rank_sum / factoid_total))
        print("List mean F1: %f (%d Questions)" % (list_f1_sum / list_total,
                                                   list_total))
