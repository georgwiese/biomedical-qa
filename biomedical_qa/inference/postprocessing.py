import abc
import itertools


class AbstractPostprocessor(object):


    def __init__(self):

        self._chained_postprocessors = []


    def chain(self, post_processor):

        self._chained_postprocessors.append(post_processor)
        return self


    def process(self, answers_probs):

        answers_probs = self._process(answers_probs)
        for postprocessor in self._chained_postprocessors:
            answers_probs = postprocessor.process(answers_probs)
        return answers_probs


    @abc.abstractmethod
    def _process(self, answers_probs):
        """Given a (answer_string, prob) iterable, yields a modified (answer_string, prob) iterable."""

        raise NotImplementedError("Subclass Responsibility")


class NullPostprocessor(AbstractPostprocessor):


    def __init__(self):

        AbstractPostprocessor.__init__(self)


    def _process(self, answers_probs):

        return answers_probs


class DeduplicatePostprocessor(AbstractPostprocessor):


    def __init__(self):

        AbstractPostprocessor.__init__(self)


    def _process(self, answers_probs):

        answer_strings = set()

        for answer_string, prob in answers_probs:
            # If an answer string occurs multiple times, keep the first (i.e. the one with higher probability)
            if answer_string.lower() not in answer_strings:
                answer_strings.add(answer_string.lower())
                yield (answer_string, prob)


class ProbabilityThresholdPostprocessor(AbstractPostprocessor):


    def __init__(self, prob_threshold, min_count=0):

        self.prob_threshold = prob_threshold
        self.min_count = min_count
        AbstractPostprocessor.__init__(self)


    def _process(self, answers_probs):

        for (answer_string, prob), i in zip(answers_probs, itertools.count()):
            if i < self.min_count or prob > self.prob_threshold:
                yield (answer_string, prob)


class TopKPostprocessor(AbstractPostprocessor):

    def __init__(self, k):

        self.k = k
        AbstractPostprocessor.__init__(self)

    def _process(self, answers_probs):

        for (answer_string, prob), i in zip(answers_probs, itertools.count()):

            if i >= self.k:
                break

            yield (answer_string, prob)
