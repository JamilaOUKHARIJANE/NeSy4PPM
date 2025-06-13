import tensorflow as tf
from tensorflow import sigmoid, transpose, matmul
from tensorflow.python.keras.layers import Layer, Multiply, concatenate




class Modulator(tf.keras.layers.Layer):
    def __init__(self, attr_idx, num_attrs, time, **kwargs):
        self.attr_idx = attr_idx
        self.num_attrs = num_attrs  # Number of extra attributes used in the modulator (other than the event)
        self.time_step = time

        super(Modulator, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name="Modulator_W", shape=(self.num_attrs, (self.num_attrs + 2) * self.time_step), #self.time_step),
                                 initializer="uniform", trainable=True)
        self.b = self.add_weight(name="Modulator_b", shape=(self.num_attrs, 1), initializer="zeros", trainable=True)

        self.build = True

    def call(self, x):
        # split input to different representation vectors
        representations = []
        for i in range(self.num_attrs + 1):
            representations.append(x[:, (i * self.time_step):((i + 1) * self.time_step), :])

        # Calculate element-wise products between activities and resources representations
        tmp = []
        for elem_product in range(self.num_attrs + 1):
            if elem_product != self.attr_idx:
                tmp.append(tf.multiply(representations[self.attr_idx], representations[elem_product],
                                    name="Modulator_repr_mult_" + str(elem_product)))
        # Add original representations
        for attr_idx in range(self.num_attrs + 1):
            tmp.append(representations[attr_idx])
        z = tf.concat(tmp, axis=1, name="Modulator_concatz")

        # Calculate b-vectors
        b = sigmoid(matmul(self.w, transpose(z), name="Modulator_matmulb") + self.b, name="Modulator_sigmoid")
        # Use b-vectors to output
        tmp = transpose(tf.multiply(b[0, :], transpose(
            x[:, (self.attr_idx * self.time_step):((self.attr_idx + 1) * self.time_step), :])), name="Modulator_mult_0")
        for i in range(1, self.num_attrs + 1):
            tmp = tmp + transpose(
                tf.multiply(b[i, :], transpose(x[:, (i * self.time_step):((i + 1) * self.time_step), :])),
                name="Modulator_mult_" + str(i))
        return tmp
        #return tf.matmul(inputs, self.w) + self.b

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.time_step, input_shape[-1])

    def get_config(self):
        config = {'attr_idx': self.attr_idx, 'num_attrs': self.num_attrs, 'time': self.time_step}
        base_config = super(Modulator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
