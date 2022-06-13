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
        self.convert_string_to_par()

    def model(self, output_to=None, input_shape=(1444, 1448, 1)):
        init, binit = self.initilizer()
        inp = k.layers.Input(shape=input_shape)
        out = inp
        if self.firstdropout:
            out = k.layers.Dropout(0.2)(out)
        for i in range(1, self.num_layers + 1):
            if self.residual:
                out_short = out
            out = k.layers.Conv2D(self.first_layer_nodes,
                                  3,
                                  strides=self.stride,
                                  activation=self.activation,
                                  padding="same",
                                  kernel_initializer=init,
                                  bias_initializer=binit,
                                  kernel_regularizer=self.regularizer
                                  )(out)
            if not i % 2 and self.batchnorm:
                out = k.layers.BatchNormalization()(out)
            if self.residual:
                out = k.layers.Add()([out, out_short])
            if self.ave_pool:
                out = k.layers.AveragePooling2D(2)(out)
            if self.max_pool:
                out = k.layers.MaxPooling2D(2)(out)
        out = self.reduction(out)
        if self.seconddropout:
            out = k.layers.Dropout(0.3)(out)
        out = k.layers.Dense(1, activation="linear")(out)

        model = k.Model(inp, out)
        model.summary(print_fn=output_to)
        model.compile(loss=k.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=k.optimizers.Adam(learning_rate=1e-4),
                      metrics=["accuracy"])
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
                case "lay3" | "lay4" | "lay6" | "lay7" | "lay8" | "lay5":
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




