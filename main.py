from BinaryReader import BinaryReader
from Preprocessor import Preprocessor
from models import classiRaw3D

reader = BinaryReader()
dataset = reader.create_training_datasets()
dataset.
preprocesser = Preprocessor(dataset)

# model = classiRaw3D(dataset.element_spec[0].shape.as_list(), preprocesser.normalize())
model = classiRaw3D(dataset.element_spec[0].shape)
model.fit(preprocesser.batch(20), epochs=10)  # TODO: Validation Split

#%%

model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(
    train_dataset,
    epochs=1,
    # Only run validation using the first 10 batches of the dataset
    # using the `validation_steps` argument
    validation_data=val_dataset,
    validation_steps=10,
)