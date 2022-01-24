from BinaryReader import BinaryReader
from Preprocessor import Preprocessor
from models import classiRaw3D

reader = BinaryReader()
dataset = reader.create_training_datasets()
preprocesser = Preprocessor(dataset)

# model = classiRaw3D(dataset.element_spec[0].shape.as_list(), preprocesser.normalize())
model = classiRaw3D(dataset.element_spec[0].shape.as_list())
model.fit(preprocesser.batch(), validation_split=0.2, epochs=10)

#%%
