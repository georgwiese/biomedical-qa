import sys

import tensorflow as tf

from biomedical_qa.training.trainer import GoalDefiner


class YesNoGoalDefiner(GoalDefiner):


    def __init__(self, model, device):
        assert model.yesno_added
        GoalDefiner.__init__(self, model, device)


    @property
    def name(self):

        return "YesNoGoalDefiner"


    def _init(self):

        with tf.variable_scope("yesno_trainer"):

            self.correct_is_yes = tf.placeholder(tf.bool, [None], "correct_is_yes")

            model = self.model
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                model.yesno_scores, tf.cast(self.correct_is_yes, tf.float32)))

            correctly_predicted_yes = tf.logical_and(self.correct_is_yes,
                                                     tf.greater_equal(model.yesno_probs, 0.5))
            correctly_predicted_no = tf.logical_and(tf.logical_not(self.correct_is_yes),
                                                    tf.logical_not(tf.greater_equal(model.yesno_probs, 0.5)))
            self.num_correct_yes = tf.reduce_sum(tf.cast(correctly_predicted_yes, tf.int32))
            self.num_correct_no = tf.reduce_sum(tf.cast(correctly_predicted_no, tf.int32))
            self.num_correct = self.num_correct_yes + self.num_correct_no

            self.num_yes = tf.reduce_sum(tf.cast(self.correct_is_yes, tf.int32))
            self.num_no = tf.reduce_sum(1 - tf.cast(self.correct_is_yes, tf.int32))
            self.accuracy = tf.cast(self.num_correct, tf.float32) / \
                            tf.cast(self.num_yes + self.num_correct_no, tf.float32)
            self.yes_accuracy = tf.cast(self.num_correct_yes, tf.float32) / \
                                tf.cast(self.num_yes, tf.float32)
            self.no_accuracy = tf.cast(self.num_correct_no, tf.float32) / \
                               tf.cast(self.num_no, tf.float32)

            self.yes_accuracy = tf.cond(tf.equal(self.num_yes, 0),
                                        lambda: tf.zeros([]),
                                        lambda: self.yes_accuracy)
            self.no_accuracy = tf.cond(tf.equal(self.num_no, 0),
                                       lambda: tf.zeros([]),
                                       lambda: self.no_accuracy)

            self.mean_yes_prob = tf.reduce_sum(tf.where(self.correct_is_yes,
                                                        model.yesno_probs,
                                                        tf.zeros(tf.shape(model.yesno_probs)))) \
                                 / tf.cast(self.num_yes, tf.float32)
            self.mean_no_prob = tf.reduce_sum(tf.where(tf.logical_not(self.correct_is_yes),
                                                       model.yesno_probs,
                                                       tf.zeros(tf.shape(model.yesno_probs)))) \
                                / tf.cast(self.num_no, tf.float32)

            with tf.name_scope("summaries"):
                self._train_summaries = [
                    tf.scalar_summary("yesno_loss", self._loss),
                    tf.scalar_summary("yesno_acc", self.accuracy),
                    tf.scalar_summary("yesno_yes_acc", self.yes_accuracy),
                    tf.scalar_summary("yesno_no_acc", self.no_accuracy),
                    tf.scalar_summary("yesno_yes_prob", self.mean_yes_prob),
                    tf.scalar_summary("yesno_no_prob", self.mean_no_prob),
                    tf.histogram_summary("yesno_probs", model.yesno_probs),
                ]

    def eval(self, sess, sampler, subsample=-1, after_batch_hook=None, verbose=False):

        self.model.set_eval(sess)
        total = 0
        num_correct_yes = 0
        num_correct_no = 0
        num_correct = 0
        num_yes = 0
        num_no = 0
        loss = 0.0
        e = sampler.epoch
        sampler.reset()
        while sampler.epoch == e and (subsample < 0 or total < subsample):
            batch = sampler.get_batch()
            _num_correct_yes, _num_correct_no, _num_yes, _num_no, _loss = self.run(
                sess,
                [self.num_correct_yes, self.num_correct_no, self.num_yes, self.num_no, self.loss],
                batch)
            num_correct_yes += _num_correct_yes
            num_correct_no += _num_correct_no
            num_correct += _num_correct_yes + _num_correct_no
            num_yes += _num_yes
            num_no += _num_no
            loss += _loss
            total += len(batch)

            if verbose:
                sys.stdout.write("\r%d - Acc: %.3f, Yes Acc: %.3f, No Acc: %.3f, Loss: %3f" %
                                 (total, num_correct / total, num_correct_yes / num_yes,
                                  num_correct_no / num_no, loss / total))
                sys.stdout.flush()

        acc = num_correct / total
        yes_acc = num_correct_yes / num_yes
        no_acc = num_correct_no / num_no
        loss /= total
        if verbose:
            print("")

        summary = tf.Summary()
        summary.value.add(tag="valid_yesno_acc", simple_value=acc)
        summary.value.add(tag="valid_yesno_yes_acc", simple_value=yes_acc)
        summary.value.add(tag="valid_yesno_no_acc", simple_value=no_acc)
        summary.value.add(tag="valid_yesno_loss", simple_value=loss)

        return acc, summary

    def get_feed_dict(self, qa_settings):

        correct_answers = []
        for qa_setting in qa_settings:
            assert qa_setting.is_yes is not None
            correct_answers.append(qa_setting.is_yes)

        feed_dict = self.model.get_feed_dict(qa_settings)
        feed_dict[self.correct_is_yes] = correct_answers

        return feed_dict


    @property
    def loss(self):
        return self._loss

    @property
    def train_summaries(self):
        return tf.summary.merge(self._train_summaries)
