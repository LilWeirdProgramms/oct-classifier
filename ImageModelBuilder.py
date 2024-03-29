import logging

import tensorflow as tf
import tensorflow.keras as k
import mil_pooling

class ImageModel:
    def __init__(self, string_list_of_par: tuple):
        self.string_list_of_par = string_list_of_par
        self.ave_pool = False
        self.max_pool = False
        self.stride = 1
        self.activation = None
        self.batchnorm = False
        self.num_layers = None
        self.firstdropout = None
        self.seconddropout = None
        self.regularizer = None
        self.reduction = None
        self.first_layer_nodes = None
        self.binit = None
        self.residual = False
        self.label_smoothing = False
        self.convert_string_to_par()

    def model(self, output_to=None, input_shape=(1444, 1448, 1)):
        init, binit = self.initilizer()
        inp = k.layers.Input(shape=input_shape)
        out = k.layers.GaussianNoise(0.01)(inp)
        out = k.layers.Conv2D(self.first_layer_nodes,
                                  3,
                                  strides=self.stride,
                                  activation=self.activation,
                                  padding="same",
                                  kernel_initializer=init,
                                  bias_initializer=binit,
                                  kernel_regularizer=self.regularizer
                                  )(out)
        if self.firstdropout:
            out = k.layers.Dropout(0.05)(out)
        for i in range(1, self.num_layers + 1):
            if self.residual:
                out_short = out
            additional_nodes = i
            out = k.layers.Conv2D(self.first_layer_nodes * additional_nodes,
                                  3,
                                  strides=self.stride,
                                  activation=self.activation,
                                  padding="same",
                                  kernel_initializer=init,
                                  bias_initializer=binit,
                                  kernel_regularizer=self.regularizer
                                  )(out)
            out = k.layers.Conv2D(self.first_layer_nodes * additional_nodes,
                                  3,
                                  strides=self.stride,
                                  activation=self.activation,
                                  padding="same",
                                  kernel_initializer=init,
                                  bias_initializer=binit,
                                  kernel_regularizer=self.regularizer
                                  )(out)
            if self.batchnorm:
                out = k.layers.BatchNormalization()(out)
            if self.residual:
                if self.firstdropout:
                    out = k.layers.Dropout(0.01)(out)
                out = k.layers.Add()([out, out_short])
                out = k.layers.Conv2D(self.first_layer_nodes * (additional_nodes+1),
                                      3,
                                      strides=self.stride,
                                      activation=self.activation,
                                      padding="same",
                                      kernel_initializer=init,
                                      bias_initializer=binit,
                                      kernel_regularizer=self.regularizer
                                      )(out)
            if i < self.num_layers:
                if self.ave_pool:
                    out = k.layers.AveragePooling2D(2)(out)
                if self.max_pool:
                    out = k.layers.MaxPooling2D(2)(out)
        out = self.reduction(out)
        if self.seconddropout:
            out = k.layers.Dropout(0.1)(out)
        out = k.layers.Dense(1, activation="linear")(out)

        model = k.Model(inp, out)
        model.summary(print_fn=output_to)
        if self.label_smoothing:
            model.compile(
            loss=k.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.02),
                #loss=MilLoss(from_logits=True, label_smoothing=0.02),
                          optimizer=k.optimizers.Adam(learning_rate=1e-4),
                          metrics=["accuracy"
                                   , TrueNegatives(from_logits=True)
                                   , TruePositives(from_logits=True)
                                   , FalseNegatives(from_logits=True)
                                   , FalsePositives(from_logits=True)
                              , Precision(from_logits=True)
                              , MilMetric()
                                   ])
        else:
            model.compile(loss=k.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=k.optimizers.Adam(learning_rate=1e-4),
                          metrics=["accuracy"
                                      , TrueNegatives(from_logits=True)
                                      , TruePositives(from_logits=True)
                                      , FalseNegatives(from_logits=True)
                                      , FalsePositives(from_logits=True)
                                   , Precision(from_logits=True)
                                   ])

        return model

    def initilizer(self):
        match self.activation:
            case "selu":
                init = k.initializers.LecunNormal()
            case "relu":
                init = k.initializers.HeNormal()
            case _:
                raise NotImplementedError("Not implemented Choice of Activation Function Initializer")
        if self.binit == "zeros":
            binit = k.initializers.Zeros()
        else:
            binit = init
        return init, binit

    def convert_string_to_par(self):
        for parameter in self.string_list_of_par:
            match parameter:
                case "ave_pool":
                    self.ave_pool = True
                case "max_pool":
                    self.max_pool = True
                case "stride":
                    self.stride = 2
                case "relu":
                    self.activation = "relu"
                case "relu_norm":
                    self.activation = "relu"
                    self.batchnorm = True
                case "selu":
                    self.activation = "selu"
                case "lay2" | "lay3" | "lay4" | "lay6" | "lay7" | "lay8" | "lay5":
                    self.num_layers = int(parameter[-1])
                case "no_drop":
                    self.firstdropout = False
                    self.seconddropout = False
                case "little_drop":
                    self.firstdropout = False
                    self.seconddropout = True
                case "lot_drop":
                    self.firstdropout = True
                    self.seconddropout = True
                case "l2":
                    self.regularizer = k.regularizers.l2()
                case "little_l2":
                    self.regularizer = k.regularizers.l2(l2=0.001)
                case "l1_l2":
                    self.regularizer = k.regularizers.L1L2(l2=0.005, l1=0.001)
                case "non":
                    self.regularizer = k.regularizers.l2(l2=0)
                case "flatten":
                    self.reduction = lambda x: k.layers.Flatten()(x)
                case "global_ave_pooling":
                    self.reduction = lambda x: k.layers.GlobalAveragePooling2D()(x)
                case "ave_pooling_little":
                    self.reduction = lambda x: k.layers.Flatten()(k.layers.AveragePooling2D(3)(x))
                case "ave_pooling_large":
                    self.reduction = lambda x: k.layers.Flatten()(k.layers.AveragePooling2D(6)(x))
                case "n32":
                    self.first_layer_nodes = 32
                case "n64":
                    self.first_layer_nodes = 64
                case "n8":
                    self.first_layer_nodes = 8
                case "n128":
                    self.first_layer_nodes = 128
                case "zeros":
                    self.binit = "zeros"
                case "same":
                    self.binit = "same"
                case "residual":
                    self.residual = True
                case "label_smoothing":
                    self.label_smoothing = True

class MilMetric(tf.keras.metrics.AUC):
    def __init__(self, name=None, **kwargs):
        super(MilMetric, self).__init__(name="mil_metric")

    def update_state(self, y_true, y_pred, sample_weight=None):
        prediction = tf.reshape(y_pred, (-1, 100))
        pooled_predicition = tf.reduce_max(prediction, axis=1)
        #print(pooled_predicition.numpy())
        #print(tf.nn.sigmoid(pooled_predicition).numpy())


        label = tf.reshape(y_true, (-1, 100))
        pooled_label = tf.cast(tf.reduce_mean(label, axis=1), tf.int32)
        #print(pooled_label.numpy())
        super().update_state(pooled_label, tf.nn.sigmoid(pooled_predicition), sample_weight)


class MilLoss(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, from_logits, label_smoothing, name="custom_binary_crossentropy"):
        super().__init__(name=name, from_logits=from_logits, label_smoothing=label_smoothing)

    def call(self, y_true, y_pred):
        prediction = tf.reshape(y_pred, (-1, 100))
        pooled_predicition = tf.reduce_max(prediction, axis=1)
        label = tf.reshape(y_true, (-1, 100))
        pooled_label = tf.cast(tf.reduce_mean(label, axis=1), tf.int32)
        return super().call(pooled_label, pooled_predicition)


# class MyCustomMetric(tf.keras.metrics.Metrics):
#
#     def __init__(self, **kwargs):
#         # Initialise as normal and add flag variable for when to run computation
#         super(MyCustomMetric, self).__init__(**kwargs)
#         self.metric_variable = self.add_weight(name='metric_varaible', initializer='zeros')
#         self.update_metric = tf.Variable(False)
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Use conditional to determine if computation is done
#         if self.update_metric:
#             # run computation
#             self.metric_variable.assign_add(computation_result)
#
#     def result(self):
#         return self.metric_variable
#
#     def reset_states(self):
#         self.metric_variable.assign(0.)
#
# class ToggleMetrics(tf.keras.callbacks.Callback):
#     '''On test begin (i.e. when evaluate() is called or
#      validation data is run during fit()) toggle metric flag '''
#     def on_test_begin(self, logs):
#         for metric in self.model.metrics:
#             if 'MilMetric' in metric.name:
#                 metric.on.assign(True)
#     def on_test_end(self,  logs):
#         for metric in self.model.metrics:
#             if 'MilMetric' in metric.name:
#                 metric.on.assign(False)

class TruePositives(tf.keras.metrics.TruePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TruePositives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TruePositives, self).update_state(y_true, y_pred, sample_weight)


class FalsePositives(tf.keras.metrics.FalsePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalsePositives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalsePositives, self).update_state(y_true, y_pred, sample_weight)


class TrueNegatives(tf.keras.metrics.TrueNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TrueNegatives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TrueNegatives, self).update_state(y_true, y_pred, sample_weight)


class FalseNegatives(tf.keras.metrics.FalseNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalseNegatives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalseNegatives, self).update_state(y_true, y_pred, sample_weight)


class Precision(tf.keras.metrics.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs, name="precision")
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)

if __name__ == "__main__":
    import numpy as np
    import sklearn.metrics as metrics

    import matplotlib.pyplot as plt

    with tf.device("cpu:0"):
        mm = MilMetric()
        prediction = np.arange(0, 80, 0.1) - 35
        label = np.ones((800, ))
        label[:200] = 0
        label[-300:-100] = 0
        prediction = prediction.astype("float32")
        auc = metrics.roc_auc_score(label, prediction)
        print(auc)
        for i in range(8):
            mm.update_state(label[:int(100*(i+1))], prediction[:int(100*(i+1))])
            print(mm.result().numpy())
        mm = MilMetric()
        mm.update_state(label, prediction)
        print(mm.result().numpy())
        tpr, fpr, threshold = metrics.roc_curve(label, prediction)
        plt.plot(tpr, fpr)
        plt.show()

