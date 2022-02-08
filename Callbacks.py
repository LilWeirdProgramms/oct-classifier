import tensorflow.keras as keras
from IPython.display import clear_output
import matplotlib.pyplot as plt


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

    def on_epoch_end(self, epoch, logs=None):
        self.x.append(epoch)
        self.loss.append(logs["loss"])
        clear_output(wait=True)
        self.val_loss.append(logs["val_loss"])
        plt.figure(figsize=(10, 8))
        ax1 = plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
        ax1.plot(self.x, self.loss, lw=4, label="Training")
        ax1.plot(self.x, self.val_loss, lw=4, label="Validation")
        ax1.legend(fontsize=16)
        ax1.set_xlabel("Epoche", fontsize=16)
        ax1.set_ylabel("Loss", fontsize=16)
        plt.tight_layout()
        plt.show()


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/best_model",
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

history_checkpoint_callback = keras.callbacks.CSVLogger("checkpoints/log.csv", separator=",", append=True)
tb_callback = keras.callbacks.TensorBoard('checkpoints/tensorboard/logs', update_freq=20)

my_callbacks = [
    CustomCallback(),
    model_checkpoint_callback,
    history_checkpoint_callback,
    tb_callback
]
