import sys
import tensorflow as tf
import numpy as np

from biomedical_qa import tfutil
from biomedical_qa.evaluation.bioasq_evaluation import BioAsqEvaluator
from biomedical_qa.inference.inference import Inferrer
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.training.trainer import GoalDefiner

SIGMOID_NEGATIVE_SAMPLES = -1


class ExtractionGoalDefiner(GoalDefiner):


    def __init__(self, model, device, forgetting_loss_factor=0.0,
                 original_weights_loss_factor=0.0):
        assert isinstance(model, ExtractionQAModel), "ExtractionQATrainer can only work with ExtractionQAModel"
        self.forgetting_loss_factor = forgetting_loss_factor
        self.original_weights_loss_factor = original_weights_loss_factor
        GoalDefiner.__init__(self, model, device)


    @property
    def name(self):
        return "ExtractionGoalDefiner"


    def _init(self):

        self.original_predictions = None
        self.original_weights = None

        self.answer_starts = tf.placeholder(tf.int32, shape=[None], name="answer_start")
        self.answer_ends = tf.placeholder(tf.int32, shape=[None], name="answer_end")

        self.original_start_probs = tf.placeholder(tf.float32, shape=[None, None], name="original_start_probs")
        self.original_end_probs = tf.placeholder(tf.float32, shape=[None, None], name="original_end_probs")

        original_start_probs = self.original_start_probs[:,:tf.shape(self.model.start_probs)[1]]
        original_end_probs = self.original_end_probs[:,:tf.shape(self.model.end_probs)[1]]

        self.original_weights_tensors = {v.name: tf.placeholder(v.dtype,
                                                                shape=v.get_shape(),
                                                                name=("original_weights__" + v.name.replace(":", "_")))
                                         for v in self.model.train_variables}

        # Maps each answer alternative index to a question index
        self.question_partition = tf.placeholder(tf.int32, shape=[None], name="question_partition")
        # Maps each answer alternative index to a answer index
        self.answer_partition = tf.placeholder(tf.int32, shape=[None], name="answer_partition")

        # Maps each answer index to a question index. Works because within one
        # answer partition, all question IDs should be the same.
        self.answer_question_partition = tf.segment_max(self.question_partition, self.answer_partition)

        model = self.model
        self._train_summaries = []

        if model.start_output_unit == "softmax":
            start_loss = self.softmax_start_loss(model)
            start_forgetting_loss = self.softmax_cross_entropy(
                self.model.start_probs, original_start_probs) \
                if self.forgetting_loss_factor > 0.0 else 0.0
        elif model.start_output_unit == "sigmoid":
            start_loss = self.sigmoid_start_loss(model)
            start_forgetting_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.model.start_scores, labels=original_start_probs) \
                if self.forgetting_loss_factor > 0.0 else 0.0
        else:
            raise ValueError("Unknown start output unit: %s" % model._start_output_unit)

        end_loss = self.end_loss(model)
        end_forgetting_loss = self.softmax_cross_entropy(self.model.end_probs, original_end_probs) \
                                if self.forgetting_loss_factor > 0.0 else 0.0

        loss = self.reduce_per_answer_loss(start_loss + end_loss)

        if self.forgetting_loss_factor > 0.0:
            # Transform both losses to [Q] shape, then take mean
            # Start loss is per context.
            start_forgetting_loss = tf.reduce_mean(
                    tf.segment_mean(start_forgetting_loss, self.model.context_partition))
            # End loss is per answer option. Note this punishes contexts more that appear multiple times
            end_forgetting_loss = tf.reduce_mean(
                    tf.segment_mean(end_forgetting_loss, self.question_partition))
        forgetting_loss = self.forgetting_loss_factor * (start_forgetting_loss + end_forgetting_loss)

        original_weights_loss = 0.0
        if self.original_weights_loss_factor:
            for variable in self.model.train_variables:
                original_weights = self.original_weights_tensors[variable.name]
                weight_diff = original_weights - variable
                original_weights_loss += tf.reduce_sum(tf.square(weight_diff))

            original_weights_loss *= self.original_weights_loss_factor

        self._loss = loss + forgetting_loss + original_weights_loss

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
        f1_per_answer = tf.where(contexts_equal,
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
            self._train_summaries += [
                tf.summary.scalar("loss", self._loss),
                tf.summary.scalar("start_loss", self.reduce_per_answer_loss(start_loss)),
                tf.summary.scalar("end_loss", self.reduce_per_answer_loss(end_loss)),

                tf.summary.scalar("train_f1_mean", self.mean_f1),
                tf.summary.histogram("train_f1", self.f1),
                tf.summary.scalar("correct_starts",
                                  tf.reduce_sum(tf.cast(starts_equal, tf.int32))),
                tf.summary.scalar("correct_ends",
                                  tf.reduce_sum(tf.cast(ends_equal, tf.int32))),
                tf.summary.scalar("start_forgetting_loss", start_forgetting_loss),
                tf.summary.scalar("end_forgetting_loss", end_forgetting_loss),
                tf.summary.scalar("forgetting_loss", forgetting_loss),
                tf.summary.scalar("plain_loss", loss),
                tf.summary.scalar("original_weight_loss", original_weights_loss),
            ]


    def initialize(self, sess, train_sampler, valid_sampler):

        self.original_predictions = self.get_original_predictions(sess, train_sampler, valid_sampler) \
                                    if self.forgetting_loss_factor > 0 else None
        self.original_weights = self.get_original_weights(sess) \
                                    if self.original_weights_loss_factor > 0 else None


    def get_original_predictions(self, sess, train_sampler, valid_sampler):

        print("Getting original predictions...")

        original_predictions = {}

        self.model.set_eval(sess)
        batches = list(train_sampler.get_all_batches()) +\
                    list(valid_sampler.get_all_batches())
        for batch in batches:
            start_probs, end_probs = sess.run(
                [self.model.start_probs, self.model.end_probs],
                self.get_feed_dict(batch)
            )
            context_index = 0
            answer_index = 0
            for question in batch:
                n_contexts = len(question.contexts)
                n_answers = sum([len(a) for a in question.answers_spans])
                question_start_probs = start_probs[context_index:(context_index + n_contexts), :]
                question_end_probs = end_probs[answer_index:(answer_index + n_answers), :]
                context_index += n_contexts
                answer_index += n_answers

                assert len(question_start_probs) == n_contexts
                assert len(question_end_probs) == n_answers

                original_predictions[question.id] = (
                    question_start_probs,
                    question_end_probs,
                )

        return original_predictions


    def get_original_weights(self, sess):

        print("Getting original weights...")
        weights = sess.run(self.model.train_variables)
        return {v.name: w for v, w in zip(self.model.train_variables, weights)}


    def reduce_per_answer_loss(self, loss):

        # Get any of the alternatives right
        loss = tf.segment_min(loss, self.answer_partition)
        # Get all of the answers right
        loss = tf.segment_mean(loss, self.answer_question_partition)
        return tf.reduce_mean(loss)

    def softmax_cross_entropy(self, probs, targets):
        # Prevent NaN losses
        probs = tf.clip_by_value(probs, 1e-10, 1.0)

        log_probs = tf.log(probs)
        losses = - tf.mul(targets, log_probs)

        return tf.reduce_mean(losses, axis=1)

    def softmax_start_loss(self, model):

        start_probs = tf.gather(model.start_probs, model.answer_context_indices)
        correct_start_probs = tfutil.gather_rowwise_1d(start_probs, tf.cast(self.answer_starts, tf.int64))

        # Prevent NaN losses
        correct_start_probs = tf.clip_by_value(correct_start_probs, 1e-10, 1.0)

        return - tf.log(correct_start_probs)

    def sigmoid_start_loss(self, model):

        correct_start_indices = tf.transpose(tf.stack([tf.cast(model.answer_context_indices, tf.int32),
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
                logits=correct_start_scores, labels=tf.ones(tf.shape(correct_start_scores)))
        incorrect_start_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=incorrect_start_scores, labels=tf.zeros(tf.shape(incorrect_start_scores)))

        # Bring incorrect_start_loss into [Q] shape
        incorrect_start_loss = tf.segment_sum(tf.reduce_sum(incorrect_start_loss, axis=1),
                                               model.context_partition)
        # Now, expand to [len(answers)] shape to match correct_start_loss
        incorrect_start_loss = tf.gather(incorrect_start_loss, self.question_partition)

        with tf.name_scope("summaries"):
            self._train_summaries += [
                tf.summary.scalar("sigmoid_correct_start_loss", tf.reduce_mean(correct_start_loss)),
                tf.summary.scalar("sigmoid_incorrect_start_loss", tf.reduce_mean(incorrect_start_loss))
            ]


        return correct_start_loss + incorrect_start_loss

    def end_loss(self, model):

        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.end_scores,
                                                              labels=self.answer_ends)

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

        original_start_probs = []
        original_end_probs = []

        question_index = 0
        answer_index = 0
        filtered_qa_settings = list()
        start_context_index = 0
        for qa_setting in qa_settings:

            if self.original_predictions is not None:
                start_probs, end_probs = self.original_predictions[qa_setting.id]
                assert len(start_probs) == len(qa_setting.contexts)
                assert len(end_probs) == sum([len(a) for a in qa_setting.answers_spans])
                original_start_probs.append(start_probs)
                original_end_probs.append(end_probs)

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

        if len(original_start_probs):
            feed_dict[self.original_start_probs] = self.merge_array_slices(original_start_probs)
            feed_dict[self.original_end_probs] = self.merge_array_slices(original_end_probs)

        if self.original_weights is not None:
            for variable_name, tensor in self.original_weights_tensors.items():
                feed_dict[tensor] = self.original_weights[variable_name]

        return feed_dict


    def merge_array_slices(self, slices):
        # Merges a list of 2D arrays of varying size into a single matrix

        width = max([s.shape[1] for s in slices])
        padded_slices = [np.pad(s, [(0, 0), (0, width - s.shape[1])], "constant")
                         for s in slices]
        return np.concatenate(padded_slices)


class BioAsqGoalDefiner(ExtractionGoalDefiner):
    """Overwrite eval() Method to optimize BioASQ measures."""


    def __init__(self, model, device,
                 beam_size=10, forgetting_loss_factor=0.0,
                 original_weights_loss_factor=0.0):

        self._beam_size = beam_size
        ExtractionGoalDefiner.__init__(self, model, device,
                                       forgetting_loss_factor=forgetting_loss_factor,
                                       original_weights_loss_factor=original_weights_loss_factor)


    def eval(self, sess, sampler, subsample=-1, after_batch_hook=None, verbose=False):

        if subsample > 0 or after_batch_hook is not None:
            raise NotImplementedError()

        inferrer = Inferrer(self.model, sess, self._beam_size)
        evaluator = BioAsqEvaluator(sampler, inferrer)

        list_threshold, _ = evaluator.find_optimal_threshold(0.01)
        answer_count, _ = evaluator.find_optimal_answer_count()
        _, factoid_mrr, list_f1, _, _ = evaluator.evaluate(list_answer_prob_threshold=list_threshold,
                                                           list_answer_count=answer_count)

        performance = (factoid_mrr + list_f1) / 2

        if verbose:
            print("MRR: %.3f, F1: %.3f -> performance: %.3f" %
                  (factoid_mrr, list_f1, performance))


        summary = tf.Summary()
        summary.value.add(tag="valid_mrr", simple_value=factoid_mrr)
        summary.value.add(tag="valid_list_f1", simple_value=list_f1)
        summary.value.add(tag="valid_performance", simple_value=performance)

        return performance, summary

