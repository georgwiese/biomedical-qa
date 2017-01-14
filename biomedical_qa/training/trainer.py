import tensorflow as tf

class GoalDefiner:
    """Provides loss and eval method."""

    def __init__(self, model, device, **kwargs):
        self.model = model

        with tf.device(device):
            with tf.variable_scope("train/%s/%s" % (model.name, self.name)):
                self._init()

    def _init(self):
        pass

    @property
    def loss(self):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def eval(self, sess, sampler, subsample=-1, after_batch_hook=None, verbose=False):
        raise NotImplementedError()

    def get_feed_dict(self, qa_settings):
        raise NotImplementedError()

    def run(self, sess, goal, qa_settings):
        return sess.run(goal, feed_dict=self.get_feed_dict(qa_settings))


class Trainer(object):
    """Optimizes losses of GoalDefiners."""


    def __init__(self, model, learning_rate, goal_definers, device,
                 train_variable_prefixes):

        self.model = model
        self.init_learning_rate = learning_rate
        self.goal_definers = goal_definers
        self._train_variable_prefixes = train_variable_prefixes

        if train_variable_prefixes is None:
            self._train_variable_prefixes = []

        with tf.device(device):
            with tf.variable_scope("train/%s" % model.name):
                self.learning_rate = tf.get_variable("lr", initializer=float(learning_rate), trainable=False)
                self.global_step = tf.get_variable("step", initializer=0, trainable=False)
                self._lr_decay = tf.placeholder(tf.float32, [], "lr_decay")
                self._lr_decay_op = self.learning_rate.assign(self.learning_rate * self._lr_decay)
                self._all_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
                self._init()


    def _init(self):

        self._loss = tf.ones([])
        for trainer in self.goal_definers:
            self._loss += trainer.loss

        self._opt = tf.train.AdamOptimizer(self.learning_rate)

        if len(self._train_variable_prefixes):
            train_variables = [v for v in self.model.train_variables
                               if any([v.name.startswith(prefix)
                                       for prefix in self._train_variable_prefixes])]
        else:
            train_variables = self.model.train_variables

        print("Training variables: %d / %d" % (len(train_variables),
                                               len(self.model.train_variables)))
        grads = tf.gradients(self.loss, train_variables, colocate_gradients_with_ops=True)
        self.grads = grads
        #, _ = tf.clip_by_global_norm(grads, 5.0)

        self._update = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.grads, train_variables), global_step=self.global_step)


    def run_train_steps(self, sess, samplers, with_summaries):
        """Runs a train step for each goal definer.

        For each train step, all losses are fed to be 0, except the loss of
        the current goal definer. This way, all goals are optimized
        individually.
        """

        loss = 0.0
        summaries = []

        for goal_definer, train_sampler in zip(self.goal_definers, samplers):
            batch = train_sampler.get_batch()
            goals = [self._update, goal_definer.loss]
            if with_summaries:
                goals += [goal_definer.train_summaries]

            feed_dict = self.get_feed_dict(goal_definer)
            feed_dict.update(goal_definer.get_feed_dict(batch))
            results = sess.run(goals, feed_dict)

            loss += results[1]
            if with_summaries:
                summaries.append(results[2])

        return loss, summaries


    def get_feed_dict(self, current_goal_definer):
        """Sets all losses to 0, except current_goal_definer's loss."""

        feed_dict = {}
        for goal_definer in self.goal_definers:
            if goal_definer != current_goal_definer:
                feed_dict[goal_definer.loss] = 0.0

        return feed_dict


    def eval(self, sess, samplers, subsample=-1, after_batch_hook=None, verbose=False):

        performances, summaries = zip(*[goal_definer.eval(sess, sampler,
                                                          subsample, after_batch_hook,
                                                          verbose)
                                        for goal_definer, sampler
                                        in zip(self.goal_definers, samplers)])
        performance = sum(performances)

        return performance, summaries


    @property
    def loss(self):
        return self._loss


    @property
    def all_saver(self):
        return self._all_saver


    def decay_learning_rate(self, sess, rate):
        sess.run(self._lr_decay_op, feed_dict={self._lr_decay: rate})


