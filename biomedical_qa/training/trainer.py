import tensorflow as tf

class Trainer:
    def __init__(self, learning_rate, model, device, **kwargs):
        self.init_learning_rate = learning_rate
        self.model = model
        with tf.device(device):
            with tf.variable_scope("train/%s" % model.name):
                self.learning_rate = tf.get_variable("lr", initializer=float(learning_rate), trainable=False)
                self.global_step = tf.get_variable("step", initializer=0, trainable=False)
                self._lr_decay = tf.placeholder(tf.float32, [], "lr_decay")
                self._lr_decay_op = self.learning_rate.assign(self.learning_rate * self._lr_decay)
                self._init()
                self._all_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def _init(self):
        pass

    @property
    def loss(self):
        raise NotImplementedError()

    @property
    def update(self):
        raise NotImplementedError()

    @property
    def all_saver(self):
        return self._all_saver

    def eval(self, sess, sampler, subsample=-1, after_batch_hook=None, verbose=False):
        raise NotImplementedError()

    def decay_learning_rate(self, sess, rate):
        sess.run(self._lr_decay_op, feed_dict={self._lr_decay: rate})

    def run(self, sess, goal, inputs):
        raise NotImplementedError()
