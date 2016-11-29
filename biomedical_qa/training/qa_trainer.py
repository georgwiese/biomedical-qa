import sys
import tensorflow as tf
import numpy as np

from biomedical_qa.models.crf import crf_log_likelihood
from biomedical_qa.models.qa_model import ExtractionQAModel
from biomedical_qa.training.trainer import Trainer


class ExtractionQATrainer(Trainer):

    def __init__(self, learning_rate, model, device, transfer_model_lr=0.0, use_mean_f1=False):
        self._initial_transfer_model_lr = transfer_model_lr
        self._use_mean_f1 = use_mean_f1
        assert isinstance(model, ExtractionQAModel), "ExtractionQATrainer can only work with ExtractionQAModel"
        Trainer.__init__(self, learning_rate, model, device)

    def _init(self):
        self.answer_starts = tf.placeholder(tf.int64, shape=[None], name="answer_start")
        self.answer_ends = tf.placeholder(tf.int64, shape=[None], name="answer_end")

        model = self.model
        self._opt = tf.train.AdamOptimizer(self.learning_rate)
        if self._initial_transfer_model_lr > 0.0:
            self.transfer_model_lr = tf.get_variable("embedder_lr", initializer=float(self._initial_transfer_model_lr), trainable=False)
            self._lr_decay_op = tf.group(self._lr_decay_op,
                                         self.transfer_model_lr.assign(self.transfer_model_lr * self._lr_decay))

        start_scores = model.start_scores
        end_scores = model.end_scores
        if isinstance(start_scores, list):
            losses = list()
            for s, e in zip(start_scores, end_scores):
                losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(s, self.answer_starts) + \
                       tf.nn.sparse_softmax_cross_entropy_with_logits(e, self.answer_ends))

            loss = tf.add_n(losses) / len(losses)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(start_scores, self.answer_starts) + \
                   tf.nn.sparse_softmax_cross_entropy_with_logits(end_scores, self.answer_ends)

        loss = tf.segment_min(loss, self.model.answer_partition)
        self._loss = tf.reduce_mean(loss)

        total = tf.cast(self.answer_ends - self.answer_starts + 1, tf.int32)


        start_pointer = model.predicted_answer_starts
        end_pointer = model.predicted_answer_ends

        missed_from_start = tf.cast(start_pointer - self.answer_starts, tf.int32)
        missed_from_end = tf.cast(self.answer_ends - end_pointer, tf.int32)
        tp = tf.cast(total - tf.minimum(total, tf.maximum(0, missed_from_start) + tf.maximum(0, missed_from_end)), tf.float32)
        fp = tf.cast(tf.maximum(0, -missed_from_start) + tf.maximum(0, -missed_from_end), tf.float32)

        total = tf.cast(total, tf.float32)
        self.recall = tp / total
        self.precision = tp / (tp + fp + 1e-10)

        self.f1 = tf.segment_max(2 * self.precision * self.recall / (self.precision + self.recall + 1e-10),
                                 self.model.answer_partition)
        self.mean_f1 = tf.reduce_mean(self.f1)

        self.exact_matches = tf.segment_max(tf.cast(tf.logical_and(tf.equal(start_pointer, self.answer_starts),
                                            tf.equal(end_pointer, self.answer_ends)), tf.int32),
                                            self.model.answer_partition)

        if self._initial_transfer_model_lr > 0.0:
            grads = tf.gradients(self.loss, model.train_variables + model.embedder.train_variables,
                                       colocate_gradients_with_ops=True)

            embedder_grads = grads[len(model.train_variables):]
            grads = grads[:len(model.train_variables)]
            self.grads = grads
            #, _ = tf.clip_by_global_norm(grads, 5.0)
            self.embedder_grads = embedder_grads
            #, _ = tf.clip_by_global_norm(embedder_grads, 5.0)

            self._update = tf.group(tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.grads, model.train_variables), global_step=self.global_step),
                                    tf.train.AdamOptimizer(self.transfer_model_lr).\
                apply_gradients(zip(self.embedder_grads, model.embedder.train_variables)))
        else:
            grads = tf.gradients(self.loss, model.train_variables, colocate_gradients_with_ops=True)
            self.grads = grads
            #, _ = tf.clip_by_global_norm(grads, 5.0)

            self._update = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.grads, model.train_variables), global_step=self.global_step)

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
        answer_starts = []
        answer_ends = []
        answer_partition = []

        k = 0
        filtered_qa_settings = list()
        for i, qa_setting in enumerate(qa_settings):
            if qa_setting.answer_spans:
                for j, span in enumerate(qa_setting.answer_spans):
                    if span:
                        (start, end) = span
                        answer_starts.append(start)
                        answer_ends.append(end - 1)
                        answer_partition.append(k)
            else:
                # search for offsets
                for a in qa_setting.answers:
                    for position in range(len(qa_setting.context)-len(a)):
                        has_answer = True
                        for j in range(len(a)):
                            if a[j] != qa_setting.context[position+j]:
                                has_answer = False
                                break
                        if has_answer:
                            answer_starts.append(position)
                            answer_ends.append(position+len(a)-1)
                            answer_partition.append(k)

            if answer_partition and answer_partition[-1] == k:
                k += 1
                filtered_qa_settings.append(qa_setting)

        feed_dict = self.model.get_feed_dict(filtered_qa_settings)
        feed_dict[self.model.correct_start_pointer] = answer_starts
        feed_dict[self.answer_starts] = answer_starts
        feed_dict[self.answer_ends] = answer_ends
        feed_dict[self.model.answer_partition] = answer_partition
        # start weights are given when computing end-weights
        #if not is_eval:
        #    feed_dict[self.model.predicted_start_pointer] = answer_starts

        return feed_dict

    def run(self, sess, goal, qa_settings):
        return sess.run(goal, feed_dict=self.get_feed_dict(qa_settings))
