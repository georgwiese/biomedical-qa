import abc
import itertools
from biomedical_qa.data.umls import build_term2preferred


class AbstractPostprocessor(object):


    def chain(self, post_processor):

        return ChainedPostprocessor([self, post_processor])


    @abc.abstractmethod
    def process(self, answers_probs):
        """Given a (answer_string, prob) iterable, yields a modified (answer_string, prob) iterable."""

        raise NotImplementedError("Subclass Responsibility")


class ChainedPostprocessor(AbstractPostprocessor):


    def __init__(self, postprocessors):

        self._postprocessors = postprocessors


    def process(self, answers_probs):

        for postprocessor in self._postprocessors:
            answers_probs = postprocessor.process(answers_probs)
        return answers_probs


class NullPostprocessor(AbstractPostprocessor):


    def process(self, answers_probs):

        return answers_probs


class DeduplicatePostprocessor(AbstractPostprocessor):


    def process(self, answers_probs):

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


    def process(self, answers_probs):

        for (answer_string, prob), i in zip(answers_probs, itertools.count()):
            if i < self.min_count or prob > self.prob_threshold:
                yield (answer_string, prob)


class TopKPostprocessor(AbstractPostprocessor):


    def __init__(self, k):

        self.k = k


    def process(self, answers_probs):

        for (answer_string, prob), i in zip(answers_probs, itertools.count()):

            if i >= self.k:
                break

            yield (answer_string, prob)


class PreferredTermPreprocessor(AbstractPostprocessor):


    def __init__(self, terms_file, case_sensitive=True):

        self.case_sensitive = case_sensitive
        self.term2preferred = build_term2preferred(terms_file, case_sensitive)


    def process(self, answers_probs):

        for answer_string, prob in answers_probs:

            if not self.case_sensitive:
                answer_string = answer_string.lower()

            if answer_string in self.term2preferred:
                answer_string = self.term2preferred[answer_string]

            yield (answer_string, prob)
