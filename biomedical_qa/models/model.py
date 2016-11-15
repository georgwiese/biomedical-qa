import tensorflow as tf


class ConfigurableModel:

    @staticmethod
    def create_from_config(config, **kwargs):
        raise NotImplementedError()

    def get_config(self, **kwargs):
        raise NotImplementedError()

    def model_saver(self):
        raise NotImplementedError()

    @property
    def save_variables(self):
        raise NotImplementedError()

    @property
    def train_variables(self):
        raise NotImplementedError()

    def get_feed_dict(self, **kwargs):
        raise NotImplementedError()

    @property
    def model_saver(self):
        if not hasattr(self, "_model_saver"):
            self._model_saver = tf.train.Saver(self.save_variables, max_to_keep=2)
        return self._model_saver

    def run(self, sess, goal, inputs):
        raise NotImplementedError()

    def set_eval(self, sess):
        raise NotImplementedError()

    def set_train(self, sess):
        raise NotImplementedError()

    @property
    def layer_outputs(self):
        return list()