import sys
import tensorflow as tf
import numpy as np

from biomedical_qa import tfutil
from biomedical_qa.evaluation.bioasq_evaluation import BioAsqEvaluator
from biomedical_qa.inference.inference import Inferrer
from biomedical_qa.models.crf import crf_log_likelihood
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.training.trainer import GoalDefiner

SIGMOID_NEGATIVE_SAMPLES = -1


class ExtractionGoalDefiner(GoalDefiner):

    def __init__(self, model, device):
        assert isinstance(model, ExtractionQAModel), "ExtractionQATrainer can only work with ExtractionQAModel"
        GoalDefiner.__init__(self, model, device)

    @property
    def name(self):
        return "ExtractionGoalDefiner"

    def _init(self):
        self.answer_starts = tf.placeholder(tf.int32, shape=[None], name="answer_start")
        self.answer_ends = tf.placeholder(tf.int32, shape=[None], name="answer_end")

        # Maps each answer alternative index to a question index
        self.question_partition = tf.placeholder(tf.int32, shape=[None], name="question_partition")
        # Maps each answer alternative index to a answer index
        self.answer_partition = tf.placeholder(tf.int32, shape=[None], name="answer_partition")

        # Maps each answer index to a question index. Works because within one
        # answer partition, all question IDs should be the same.
        self.answer_question_partition = tf.segment_max(self.question_partition, self.answer_partition)

        model = self.model

        if model.start_output_unit == "softmax":
            start_loss = self.softmax_start_loss(model)
        elif model.start_output_unit == "sigmoid":
            start_loss = self.sigmoid_start_loss(model)
        else:
            raise ValueError("Unknown start output unit: %s" % model._start_output_unit)

        end_loss = self.end_loss(model)
        self._loss = self.reduce_per_answer_loss(start_loss + end_loss)

        total = tf.cast(self.answer_ends - self.answer_starts + 1, tf.int32)

        context_indices = tf.gather(model.predicted_context_indices, self.question_partition)
        start_pointer = tf.cast(model.predicted_answer_starts, tf.int32)
        end_pointer = tf.cast(model.predicted_answer_ends, tf.int32)

        missed_from_start = start_pointer - self.answer_starts
        missed_from_end = self.answer_ends - end_pointer
        tp = tf.cast(total - tf.minimum(total, tf.maximum(0, missed_from_start) + tf.maximum(0, missed_from_end)), tf.float32)
        fp = tf.cast(tf.maximum(0, -missed_from_start) + tf.maximum(0, -missed_from_end), tf.float32)

        total = tf.cast(total, tf.float32)
        self.recall = tp / total
        self.precision = tp / (tp + fp + 1e-10)

        f1_per_answer = 2 * self.precision * self.recall / (self.precision + self.recall + 1e-10)

        # Set f1 to 0 if the predicted context index is not equal to the actual context index
        contexts_equal = tf.equal(context_indices, self.model.answer_context_indices)
        f1_per_answer = tf.select(contexts_equal,
                                  f1_per_answer,
                                  tf.zeros(tf.shape(f1_per_answer)))

        self.f1 = tf.segment_max(f1_per_answer, self.question_partition)
        self.mean_f1 = tf.reduce_mean(self.f1)

        starts_equal = tf.logical_and(tf.equal(start_pointer, self.answer_starts),
                                      contexts_equal)
        ends_equal = tf.logical_and(tf.equal(end_pointer, self.answer_ends),
                                    contexts_equal)
        spans_equal = tf.logical_and(starts_equal, ends_equal)
        self.exact_matches = tf.segment_max(tf.cast(spans_equal, tf.int32),
                                            self.question_partition)

        with tf.name_scope("summaries"):
            self._train_summaries = [
                tf.scalar_summary("loss", self._loss),
                tf.scalar_summary("start_loss", self.reduce_per_answer_loss(start_loss)),
                tf.scalar_summary("end_loss", self.reduce_per_answer_loss(end_loss)),

                tf.scalar_summary("train_f1_mean", self.mean_f1),
                tf.histogram_summary("train_f1", self.f1),
                tf.scalar_summary("correct_starts",
                                  tf.reduce_sum(tf.cast(starts_equal, tf.int32))),
                tf.scalar_summary("correct_ends",
                                  tf.reduce_sum(tf.cast(ends_equal, tf.int32))),
            ]

    def reduce_per_answer_loss(self, loss):

        # Get any of the alternatives right
        loss = tf.segment_min(loss, self.answer_partition)
        # Get all of the answers right
        loss = tf.segment_mean(loss, self.answer_question_partition)
        return tf.reduce_mean(loss)

    def softmax_start_loss(self, model):

        start_probs = tf.gather(model.start_probs, model.answer_context_indices)
        correct_start_probs = tfutil.gather_rowwise_1d(start_probs, tf.cast(self.answer_starts, tf.int64))

        # Prevent NaN losses
        correct_start_probs = tf.clip_by_value(correct_start_probs, 1e-10, 1.0)

        return - tf.log(correct_start_probs)

    def sigmoid_start_loss(self, model):

        correct_start_indices = tf.transpose(tf.pack([tf.cast(model.answer_context_indices, tf.int32),
                                                      self.answer_starts]))
        correct_start_values = tf.ones([tf.shape(self.answer_starts)[0]], dtype=tf.float32)
        is_start_correct = tf.scatter_nd(correct_start_indices,
                                         correct_start_values,
                                         tf.shape(model.start_scores))

        # Get relevant scores
        correct_start_scores = tf.gather_nd(model.start_scores, correct_start_indices)
        correct_start_mask = -1000.0 * is_start_correct
        incorrect_start_scores = model.start_scores + correct_start_mask
        if SIGMOID_NEGATIVE_SAMPLES > 0:
            incorrect_start_scores, _ = tf.nn.top_k(incorrect_start_scores,
                                                    k=SIGMOID_NEGATIVE_SAMPLES)

        # Compute Cross Entropy Loss
        correct_start_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                correct_start_scores, tf.ones(tf.shape(correct_start_scores)))
        incorrect_start_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                incorrect_start_scores, tf.zeros(tf.shape(incorrect_start_scores)))

        # Bring incorrect_start_loss into [Q] shape
        incorrect_start_loss = tf.segment_mean(tf.reduce_mean(incorrect_start_loss, axis=1),
                                               model.context_partition)
        # Now, expand to [len(answers)] shape to match correct_start_loss
        incorrect_start_loss = tf.gather(incorrect_start_loss, self.question_partition)

        with tf.name_scope("summaries"):
            self._train_summaries += [
                tf.scalar_summary("sigmoid_correct_start_loss", tf.reduce_mean(correct_start_loss)),
                tf.scalar_summary("sigmoid_incorrect_start_loss", tf.reduce_mean(incorrect_start_loss))
            ]


        return correct_start_loss + incorrect_start_loss

    def end_loss(self, model):

        return tf.nn.sparse_softmax_cross_entropy_with_logits(model.end_scores,
                                                              self.answer_ends)

    @property
    def loss(self):
        return self._loss

    @property
    def train_summaries(self):
        return tf.summary.merge(self._train_summaries)

    def eval(self, sess, sampler, subsample=-1, after_batch_hook=None, verbose=False):
        self.model.set_eval(sess)
        total = 0
        f1 = 0.0
        exact = 0.0
        e = sampler.epoch
        sampler.reset()
        while sampler.epoch == e and (subsample < 0 or total < subsample):
            batch = sampler.get_batch()
            _f1, _exact = self.run(sess, [self.f1, self.exact_matches], batch)
            exact += sum(_exact)
            f1 += sum(_f1)
            total += len(batch)
            if verbose:
                sys.stdout.write("\r%d - F1: %.3f, Exact: %.3f" % (total, f1 / total, exact / total))
                sys.stdout.flush()

        f1 = f1 / total
        exact = exact / total
        if verbose:
            print("")

        summary = tf.Summary()
        summary.value.add(tag="valid_f1_mean", simple_value=f1)
        summary.value.add(tag="valid_exact_mean", simple_value=exact)

        return f1, summary

    def get_feed_dict(self, qa_settings):
        answer_context_indices = []
        answer_starts = []
        answer_ends = []
        question_partition = []
        answer_partition = []

        question_index = 0
        answer_index = 0
        filtered_qa_settings = list()
        start_context_index = 0
        for qa_setting in qa_settings:
            for answer_spans in qa_setting.answers_spans:
                for span in answer_spans:
                    (context_index, start, end) = span
                    answer_context_indices.append(start_context_index + context_index)
                    answer_starts.append(start)
                    answer_ends.append(end - 1)
                    question_partition.append(question_index)
                    answer_partition.append(answer_index)
                if len(answer_spans):
                    answer_index += 1

            # Filter Question entirely if there are no answers
            if question_partition and question_partition[-1] == question_index:
                question_index += 1
                filtered_qa_settings.append(qa_setting)
                start_context_index += len(qa_setting.contexts)

        feed_dict = self.model.get_feed_dict(filtered_qa_settings)
        feed_dict[self.model.correct_start_pointer] = answer_starts
        feed_dict[self.model.answer_context_indices] = answer_context_indices
        feed_dict[self.answer_starts] = answer_starts
        feed_dict[self.answer_ends] = answer_ends
        feed_dict[self.question_partition] = question_partition
        feed_dict[self.answer_partition] = answer_partition
        # start weights are given when computing end-weights
        #if not is_eval:
        #    feed_dict[self.model.predicted_start_pointer] = answer_starts

        return feed_dict


class BioAsqGoalDefiner(ExtractionGoalDefiner):
    """Overwrite eval() Method to optimize BioASQ measures."""


    def __init__(self, model, device,
                 beam_size=10, list_answer_count=5):

        self._beam_size = beam_size
        self._list_answer_count = list_answer_count
        ExtractionGoalDefiner.__init__(self, model, device)


    def eval(self, sess, sampler, subsample=-1, after_batch_hook=None, verbose=False):

        if subsample > 0 or after_batch_hook is not None:
            raise NotImplementedError()

        inferrer = Inferrer(self.model, sess, self._beam_size)
        evaluator = BioAsqEvaluator(sampler, inferrer)

        list_threshold, _ = evaluator.find_optimal_threshold(0.001)
        _, factoid_mrr, list_f1, _, _ = evaluator.evaluate(list_answer_prob_threshold=list_threshold,
                                                           list_answer_count=self._list_answer_count)

        performance = (factoid_mrr + list_f1) / 2

        if verbose:
            print("MRR: %.3f, F1: %.3f -> performance: %.3f" %
                  (factoid_mrr, list_f1, performance))


        summary = tf.Summary()
        summary.value.add(tag="valid_mrr", simple_value=factoid_mrr)
        summary.value.add(tag="valid_list_f1", simple_value=list_f1)
        summary.value.add(tag="valid_performance", simple_value=performance)

        return performance, summary

