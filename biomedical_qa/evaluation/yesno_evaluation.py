import numpy as np


class YesNoEvaluator(object):


    def __init__(self, sess, model, sampler):

        self._sess = sess
        self._model = model
        self._sampler = sampler

        self._predictions = None


    def get_predictions(self):

        predictions = {}

        self._sampler.reset()
        epoch = self._sampler.epoch

        while self._sampler.epoch == epoch:
            batch = self._sampler.get_batch()
            [probs] = self._sess.run([self._model.yesno_probs], self._model.get_feed_dict(batch))

            for question, prob in zip(batch, probs):
                predictions[question.id] = {
                    "question": question,
                    "prob": prob,
                }

        return predictions


    def initialize_predictions_if_needed(self):

        if self._predictions is None:
            self._predictions = self.get_predictions()


    def get_yes_no_probs(self):

        self.initialize_predictions_if_needed()

        yes_probs = [p["prob"] for p in self._predictions.values()
                     if p["question"].is_yes]
        no_probs = [p["prob"] for p in self._predictions.values()
                    if not p["question"].is_yes]

        return yes_probs, no_probs


    def measure_accuracy(self, threshold=0.5):

        yes_probs, no_probs = self.get_yes_no_probs()
        yes_correct = len([p for p in yes_probs if p > threshold])
        no_correct = len([p for p in no_probs if p <= threshold])

        yes_acc = yes_correct / len(yes_probs)
        no_acc = no_correct / len(no_probs)
        acc = (yes_correct + no_correct) / len(yes_probs + no_probs)

        return acc, yes_acc, no_acc


    def find_optimal_threshold(self):

        best_acc = -1.0
        best_threshold = -1.0

        for threshold in np.arange(0.0, 1.0, 0.01):

            acc, _, _ = self.measure_accuracy(threshold)

            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        return best_threshold, best_acc
