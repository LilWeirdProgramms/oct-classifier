import tensorflow as tf
from importlib import reload
import InputList
import BinaryReader
import Preprocessor
import models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
reader = BinaryReader.BinaryReader()  # TODO: Normalizer
training_dataset, validation_dataset = reader.create_training_datasets(InputList.training_files)
preprocesser = Preprocessor.Preprocessor(training_dataset)


model = models.classiRaw3D(training_dataset.element_spec[0].shape)

history = model.fit(
    preprocesser.batch(20).take(2),
    epochs=10,
    validation_data=Preprocessor.Preprocessor(validation_dataset).batch(20).take(2),
    # callbacks=Callbacks.my_callbacks
)
model.save('savedModels/first')
