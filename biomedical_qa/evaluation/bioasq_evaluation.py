import logging
import sys

import numpy as np

from biomedical_qa.data.bioasq_squad_builder import ensure_list_depth_2

class BioAsqEvaluator(object):


    def __init__(self, sampler, inferrer):

        self.sampler = sampler
        self.inferrer = inferrer
        self.predictions = None


    def initialize_predictions_if_needed(self, verbosity_level=0):

        if self.predictions is not None:
            # Nothing to do
            return

        if verbosity_level > 0:
            print("  Doing predictions...")

        self.predictions = self.inferrer.get_predictions(self.sampler)

        if verbosity_level > 0:
            print("  Done.")


    def find_optimal_threshold(self, threshold_search_step, verbosity_level=0):

        self.initialize_predictions_if_needed(verbosity_level)

        best_f1 = -1
        best_threshold = -1

        if verbosity_level > 0:
            print("Trying thresholds...")

        for threshold in np.arange(0.0, 1.0, threshold_search_step):

            _, _, f1, precision, recall = self.evaluate(list_answer_prob_threshold=threshold)

            if verbosity_level > 1:
                print("%f\t%f1\t%f\t%f" % (threshold, f1, precision, recall))

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        if verbosity_level > 0:
            print("Found best threshold: %f (F1: %f)" % (best_threshold, best_f1))

        return best_threshold, best_f1


    def evaluate(self, verbosity_level=0, list_answer_count=5,
                 list_answer_prob_threshold=0.5):

        self.initialize_predictions_if_needed(verbosity_level)

        if self.inferrer.beam_size < 5:
            logging.warning("Beam size should be at least 5 in order to get 5 ranked answers.")

        # Assuming one question per paragraph
        count = len(self.sampler.get_questions())
        subsample = self.sampler.instances_per_epoch
        if verbosity_level > 0:
            print("  (Questions: %d, using %d)" %
                  (count, subsample if subsample is not None else count))

        factoid_correct, factoid_total = 0, 0
        factoid_reciprocal_rank_sum = 0
        list_f1_sum, list_precision_sum, list_recall_sum, list_total = 0, 0, 0, 0

        for question in self.sampler.get_questions():

            prediction = self.predictions[question.id]

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

                first_correct, rank = self.evaluate_factoid_question(
                        answers, correct_answers)

                factoid_total += 1

                if first_correct:
                    if verbosity_level > 1:
                        print("  Correct!")
                    factoid_correct += 1

                if verbosity_level > 1:
                    print("  Rank: %d" % (rank if rank <= 5 else -1))
                factoid_reciprocal_rank_sum += 1 / rank if rank <= 5 else 0


            if question_type == "list":

                used_answers, f1, precision, recall = self.evaluate_list_question(
                        answers, correct_answers, list_answer_count,
                        list_answer_prob_threshold)
                list_total += 1

                if verbosity_level > 1:
                    print("  Using answers:", [a for a, _ in used_answers])

                if verbosity_level > 1:
                    print("F1: %f, precision: %f, recall: %f" % (f1, precision, recall))
                list_f1_sum += f1
                list_precision_sum += precision
                list_recall_sum += recall

        factoid_acc = factoid_correct / factoid_total if factoid_total else 0
        factoid_mrr = factoid_reciprocal_rank_sum / factoid_total if factoid_total else 0
        list_f1 = list_f1_sum / list_total if list_total else 0
        list_precision = list_precision_sum / list_total if list_total else 0
        list_recall = list_recall_sum / list_total if list_total else 0

        if verbosity_level > 0:
            print("Factoid correct: %d / %d" % (factoid_correct, factoid_total))
            print("Factoid MRR: %f" % factoid_mrr)
            print("List mean F1: %f (%d Questions)" % (list_f1, list_total))

        return factoid_acc, factoid_mrr, list_f1, list_precision, list_recall


    def evaluate_factoid_question(self, answers, correct_answers):

        first_correct = False
        rank = sys.maxsize
        for correct_answer in correct_answers[0]:
            # Compute exact match
            if not first_correct and \
                    answers[0][0].lower() == correct_answer.lower():
                first_correct = True
            # Compute rank
            for k in range(min(len(answers), 5)):
                if answers[k][0].lower() == correct_answer.lower():
                    rank = min(rank, k + 1)

        return first_correct, rank


    def evaluate_list_question(self, answers, correct_answers,
                               list_answer_count, list_answer_prob_threshold):

        if self.inferrer.model.start_output_unit == "sigmoid":
            # We get individual probabilities for each answer, can threshold.
            filtered_answers = [(a, prob) for a, prob in answers
                                if prob >= list_answer_prob_threshold]
            answers = filtered_answers if len(filtered_answers) > 0 else answers[:1]
        else:
            # We can't apply an absolute threshold, so use a fixed count.
            answers = answers[:list_answer_count]

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

        return answers, f1, precision, recall
