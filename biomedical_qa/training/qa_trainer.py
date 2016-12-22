import sys
import tensorflow as tf
import numpy as np

from biomedical_qa import tfutil
from biomedical_qa.models.crf import crf_log_likelihood
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.training.trainer import Trainer


class ExtractionQATrainer(Trainer):

    def __init__(self, learning_rate, model, device, train_variable_prefixes=[]):
        self._train_variable_prefixes = train_variable_prefixes
        assert isinstance(model, ExtractionQAModel), "ExtractionQATrainer can only work with ExtractionQAModel"
        Trainer.__init__(self, learning_rate, model, device)

    def _init(self):
        self.answer_starts = tf.placeholder(tf.int64, shape=[None], name="answer_start")
        self.answer_ends = tf.placeholder(tf.int64, shape=[None], name="answer_end")
        self.answer_partition = tf.placeholder(tf.int64, shape=[None], name="answer_partition")

        model = self.model
        self._opt = tf.train.AdamOptimizer(self.learning_rate)

        start_probs = tf.gather(model.start_probs, model.answer_context_indices)
        end_probs = model.end_probs
        correct_start_probs = tfutil.gather_rowwise_1d(start_probs, self.answer_starts)
        correct_end_probs = tfutil.gather_rowwise_1d(end_probs, self.answer_ends)

        loss = - tf.log(correct_start_probs) - tf.log(correct_end_probs)

        loss = tf.segment_min(loss, self.model.answer_partition)
        self._loss = tf.reduce_mean(loss)

        total = tf.cast(self.answer_ends - self.answer_starts + 1, tf.int32)

        context_indices = model.predicted_context_indices
        start_pointer = model.predicted_answer_starts
        end_pointer = model.predicted_answer_ends

        missed_from_start = tf.cast(start_pointer - self.answer_starts, tf.int32)
        missed_from_end = tf.cast(self.answer_ends - end_pointer, tf.int32)
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

        self.f1 = tf.segment_max(f1_per_answer, self.model.answer_partition)
        self.mean_f1 = tf.reduce_mean(self.f1)

        pointers_equal = tf.logical_and(tf.equal(start_pointer, self.answer_starts),
                                        tf.equal(end_pointer, self.answer_ends))
        spans_equal = tf.logical_and(contexts_equal, pointers_equal)
        self.exact_matches = tf.segment_max(tf.cast(spans_equal, tf.int32),
                                            self.model.answer_partition)

        if len(self._train_variable_prefixes):
            train_variables = [v for v in model.train_variables
                               if any([v.name.startswith(prefix)
                                       for prefix in self._train_variable_prefixes])]
        else:
            train_variables = model.train_variables

        print("Training variables: %d / %d" % (len(train_variables),
                                               len(model.train_variables)))
        grads = tf.gradients(self.loss, train_variables, colocate_gradients_with_ops=True)
        self.grads = grads
        #, _ = tf.clip_by_global_norm(grads, 5.0)

        self._update = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.grads, train_variables), global_step=self.global_step)

        self._all_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        with tf.name_scope("summaries"):
            tf.scalar_summary("loss", self._loss)
            tf.scalar_summary("train_f1_mean", self.mean_f1)
            tf.histogram_summary("train_f1", self.f1)

    @property
    def loss(self):
        return self._loss

    @property
    def update(self):
        return self._update

    @property
    def all_saver(self):
        return self._all_saver

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
        return f1, exact

    def get_feed_dict(self, qa_settings):
        answer_context_indices = []
        answer_starts = []
        answer_ends = []
        answer_partition = []

        k = 0
        filtered_qa_settings = list()
        start_context_index = 0
        for i, qa_setting in enumerate(qa_settings):
            for j, span in enumerate(qa_setting.answer_spans):
                (context_index, start, end) = span
                answer_context_indices.append(start_context_index + context_index)
                answer_starts.append(start)
                answer_ends.append(end - 1)
                answer_partition.append(k)
            start_context_index += len(qa_setting.answer_spans)

            if answer_partition and answer_partition[-1] == k:
                k += 1
                filtered_qa_settings.append(qa_setting)

        feed_dict = self.model.get_feed_dict(filtered_qa_settings)
        feed_dict[self.model.correct_start_pointer] = answer_starts
        feed_dict[self.model.answer_context_indices] = answer_context_indices
        feed_dict[self.answer_starts] = answer_starts
        feed_dict[self.answer_ends] = answer_ends
        feed_dict[self.answer_partition] = answer_partition
        # start weights are given when computing end-weights
        #if not is_eval:
        #    feed_dict[self.model.predicted_start_pointer] = answer_starts

        return feed_dict

    def run(self, sess, goal, qa_settings):
        return sess.run(goal, feed_dict=self.get_feed_dict(qa_settings))
