import logging
import sys

import numpy as np

from biomedical_qa.data.bioasq_squad_builder import ensure_list_depth_2


def element_wise_mean(list_of_tuples):

    lists = list(zip(*list_of_tuples))
    return [sum(l) / len(l) for l in lists]


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


    def find_optimal_answer_count(self, verbosity_level=0):

        self.initialize_predictions_if_needed(verbosity_level)

        best_f1 = -1
        best_answer_count = -1

        if verbosity_level > 0:
            print("Trying answer counts...")

        for count in range(1, 20):

            _, _, f1, precision, recall = self.evaluate(list_answer_count=count)

            if verbosity_level > 1:
                print("%d\t%f1\t%f\t%f" % (count, f1, precision, recall))

            if f1 > best_f1:
                best_f1 = f1
                best_answer_count = count

        if verbosity_level > 0:
            print("Found best answer count: %d (F1: %f)" % (best_answer_count, best_f1))

        return best_answer_count, best_f1


    def evaluate_questions(self, verbosity_level=0, list_answer_count=None,
                           list_answer_prob_threshold=None):

        self.initialize_predictions_if_needed(verbosity_level)

        if self.inferrer.beam_size < 5:
            logging.warning("Beam size should be at least 5 in order to get 5 ranked answers.")

        # Assuming one question per paragraph
        count = len(self.sampler.get_questions())
        subsample = self.sampler.instances_per_epoch
        if verbosity_level > 0:
            print("  (Questions: %d, using %d)" %
                  (count, subsample if subsample is not None else count))

        # <question id> -> (is_correct, reciprocal_rank)
        factoid_performances = {}
        # <question id> -> (f1, precision, recall)
        list_performances = {}

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

                rank = self.evaluate_factoid_question(
                    answers, correct_answers,
                    verbosity_level=verbosity_level)

                reciprocal_rank = 1 / rank if rank <= 5 else 0
                factoid_performances[question.id] = (rank == 1, reciprocal_rank)


            if question_type == "list":

                f1, precision, recall = self.evaluate_list_question(
                    answers, correct_answers, list_answer_count,
                    list_answer_prob_threshold,
                    verbosity_level=verbosity_level)

                list_performances[question.id] = (f1, precision, recall)

        return factoid_performances, list_performances


    def evaluate(self, verbosity_level=0, list_answer_count=None,
                 list_answer_prob_threshold=None):

        factoid_performances, list_performances = self.evaluate_questions(
                verbosity_level, list_answer_count, list_answer_prob_threshold)

        factoid_acc, factoid_mrr, factoid_correct = 0.0, 0.0, 0
        list_f1, list_precision, list_recall = 0.0, 0.0, 0.0

        if len(factoid_performances):
            factoid_acc, factoid_mrr = element_wise_mean(factoid_performances.values())
            factoid_correct = int(factoid_acc * len(factoid_performances))

        if len(list_performances):
            list_f1, list_precision, list_recall = element_wise_mean(list_performances.values())

        if verbosity_level > 0:
            print("Factoid correct: %d / %d" % (factoid_correct, len(factoid_performances)))
            print("Factoid MRR: %f" % factoid_mrr)
            print("List mean F1: %f (%d Questions)" % (list_f1, len(list_performances)))

        return factoid_acc, factoid_mrr, list_f1, list_precision, list_recall


    def evaluate_factoid_question(self, answers, correct_answers,
                                  verbosity_level=0):

        rank = sys.maxsize
        for correct_answer in correct_answers[0]:
            # Compute rank
            for k in range(min(len(answers), 5)):
                if answers[k][0].lower() == correct_answer.lower():
                    rank = min(rank, k + 1)


        if verbosity_level > 1:
            if rank == 1:
                print("  Correct!")
            print("  Rank: %d" % (rank if rank <= 5 else -1))

        return rank


    def evaluate_list_question(self, answers, correct_answers,
                               list_answer_count=None,
                               list_answer_prob_threshold=None,
                               verbosity_level=0):

        if self.inferrer.models[0].start_output_unit == "sigmoid" and \
                list_answer_prob_threshold is not None:
            # We get individual probabilities for each answer, can threshold.
            filtered_answers = [(a, prob) for a, prob in answers
                                if prob >= list_answer_prob_threshold]
            answers = filtered_answers if len(filtered_answers) > 0 else answers[:1]
        elif list_answer_count is not None:
            # We can't apply an absolute threshold, so use a fixed count.
            answers = answers[:list_answer_count]
        else:
            # We find the best possible cutoff
            f1, precision, recall = None, None, None
            for count in range(1, 20):
                _f1, _precision, _recall = self.evaluate_list_question(
                        answers, correct_answers, list_answer_count=count)
                if f1 is None or _f1 > f1:
                    f1, precision, recall = _f1, _precision, _recall
            return f1, precision, recall

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
            print("  Using answers:", [a for a, _ in answers])
            print("F1: %f, precision: %f, recall: %f" % (f1, precision, recall))

        return f1, precision, recall
