import tensorflow.keras as keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
from datetime import datetime
import logging

class CustomCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.x = []
        self.loss = []
        self.val_loss = []

        self.keys = list(logs.keys())
        self.values = dict.fromkeys(self.keys, [])


    # def on_batch_end(self, batch, logs=None):
    #     self.x.append(batch)
    #     clear_output(wait=True)
    #     plt.figure(figsize=(10, 8))
    #     ax1 = plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
    #     for key in self.keys:
    #         self.values[key].append(logs[key])
    #         ax1.plot(self.x, self.values[key], lw=4, label=key)
    #     ax1.legend(fontsize=16)
    #     ax1.set_xlabel("Epoche", fontsize=16)
    #     ax1.set_ylabel("Loss", fontsize=16)
    #     plt.tight_layout()
    #     plt.show()

    # def on_batch_end(self, batch, logs=None):
    #     self.x.append(batch)
    #     self.loss.append(logs["loss"])
    #     clear_output(wait=True)
    #     plt.figure(figsize=(10, 8))
    #     ax1 = plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
    #     ax1.plot(self.x, self.loss, lw=4, label="Training")
    #     ax1.legend(fontsize=16)
    #     ax1.set_xlabel("Epoche", fontsize=16)
    #     ax1.set_ylabel("Loss", fontsize=16)
    #     plt.tight_layout()
    #     plt.show()

    def on_epoch_end(self, epoch, logs:dict=None):
        plt.close()
        self.x.append(epoch)
        self.loss.append(logs["loss"])
        clear_output(wait=True)
        self.ax1 = plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
        self.val_loss.append(logs["val_loss"])
        self.ax1.plot(self.x, self.loss, lw=4, label="Training")
        self.ax1.plot(self.x, self.val_loss, lw=4, label="Validation")
        self.ax1.legend(fontsize=16)
        self.ax1.set_xlabel("Epoch", fontsize=16)
        self.ax1.set_ylabel("Loss", fontsize=16)
        plt.tight_layout()
        plt.show()


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/best_model_image",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

last_epoch_callback = keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/last_model",
    save_weights_only=True,
    save_freq='epoch',
    save_best_only=False)


history_checkpoint_callback = keras.callbacks.CSVLogger("checkpoints/log.csv", separator=",", append=True)
tb_callback = keras.callbacks.TensorBoard('checkpoints/tensorboard/logs', update_freq=20)

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = keras.callbacks.TensorBoard(log_dir=logs,
                                              histogram_freq=1,
                                              profile_batch='1,20')
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S_mnist")
tboard_callback_mnist = keras.callbacks.TensorBoard(log_dir=logs,
                                              histogram_freq=1,
                                              profile_batch='1,20')

my_callbacks = [
    CustomCallback(),
    model_checkpoint_callback,
    last_epoch_callback,
    history_checkpoint_callback,
    tboard_callback
]

my_mnist_callbacks = [
    CustomCallback(),
    history_checkpoint_callback,
    tboard_callback_mnist,
]

my_image_callbacks = [
    CustomCallback(),
    model_checkpoint_callback,
    ]


def hyper_image_callback(name: str):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=f"results/hyperparameter_study/supervised/models/{name}",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(f"results/hyperparameter_study/histories/{name}.csv",
                                  separator=",",
                                  append=True),
    ]

def mil_pooling_callback(name: str):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=f"results/hyperparameter_study/mil/models/{name}",
            save_weights_only=False,
            monitor='val_precision',
            mode='max',
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(f"results/hyperparameter_study/histories/{name}.csv",
                                  separator=",",
                                  append=True),
    ]

def raw_callback(name: str):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=f"results/hyperparameter_study/mil/models/{name}",
            save_weights_only=False,
            monitor='val_precision',
            mode='max',
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(f"results/hyperparameter_study/histories/{name}.csv",
                                  separator=",",
                                  append=True),
        tboard_callback
    ]


def vgg_callback(name: str):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=f"results/vgg/models/{name}",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(f"results/histories/{name}.csv",
                                  separator=",",
                                  append=True),
    ]
