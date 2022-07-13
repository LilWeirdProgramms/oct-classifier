from hyperparameterStudy.image_dataset import ImageDataset
from tensorflow.keras.applications import VGG16
import tensorflow as tf
import random
import tensorflow.keras as k
from tensorflow.keras.optimizers import Adam

# def create_vgg_model(input_shape=(1444, 1448, 3), train_from_layer=13):
#     network_input = k.layers.Input(shape=input_shape)
#     model = VGG16(weights='imagenet', input_tensor=network_input, include_top=False)
#
#     feature_extraction_model = k.models.Model(inputs=model.input, outputs=model.layers[train_from_layer].output)
#     feature_extraction_model.trainable = False
#     classify_model = k.models.Model(inputs=model.layers[train_from_layer].output, outputs=model.output)
#
#     pooling_layer = k.layers.GlobalAveragePooling2D()
#     dropout_layer = k.layers.Dropout(0.1)
#     dense_layer1 = k.layers.Dense(64, activation="relu")
#     dense_layer2 = k.layers.Dense(1, activation="linear")
#     model = k.Sequential([feature_extraction_model, classify_model, pooling_layer, dropout_layer, dense_layer1,
#                           dense_layer2])
#     model.summary()
#     return model

# def create_vgg_model(input_shape=(1444, 1448, 3), train_from_layer=14):
#     network_input = k.layers.Input(shape=input_shape)
#     vgg_model = VGG16(weights='imagenet', include_top=False)
#     #vgg_model = VGG16(weights='imagenet', input_tensor=network_input, include_top=False)
#
#     model = k.models.Sequential()
#     model.add(k.layers.Input(shape=input_shape))
#     for i, layer in enumerate(vgg_model.layers):
#         # if i < train_from_layer:
#         #     layer.trainable = False
#         model.add(layer)
#         if i == 5:
#             break
#
#     model.add(k.layers.GlobalAveragePooling2D())
#     model.add(k.layers.Dropout(0.1))
#     model.add(k.layers.Dense(64, activation='relu'))
#     model.add(k.layers.Dense(1, activation='linear'))
#     model.compile(optimizer=Adam(learning_rate=1e-4), loss=k.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=["accuracy"]
#     )
#     model.summary()
#     return model

def create_vgg_model(input_shape=(None, 1444, 1448, 3), train_from_layer=14):
    network_input = k.layers.Input(shape=input_shape)
    #vgg_model = VGG16(weights='imagenet', include_top=False)
    vgg_model = VGG16(weights="imagenet", input_tensor=network_input, include_top=False)

    model = k.models.Sequential()
    #inp = k.layers.Input(shape=input_shape)
    #inp = vgg_model.layers[0]
    #out = inp
    for i, layer in enumerate(vgg_model.layers[:]):
        if i < train_from_layer:
            layer.trainable = False
        #out = layer(out)
        model.add(layer)

    # out = k.layers.GlobalAveragePooling2D()(out)
    # out = k.layers.Dropout(0.1)(out)
    # out = k.layers.Dense(64, activation='relu')(out)
    # out = k.layers.Dense(1, activation='linear')(out)
    # model = k.Model(inputs=inp, outputs=out)
    model.add(k.layers.GlobalAveragePooling2D())
    model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(64, activation='relu'))
    model.add(k.layers.Dense(1, activation='linear'))
    #model = k.Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=k.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"]
    )
    #model2 = k.Model(inputs=inp, outputs=out_bkp)
    #model2 = k.Model(model.layers[1].input, model.layers[6].output)
    model.build((None, 1444, 1448, 3))
    model.summary()
    return model


if __name__ == "__main__":
    file_list = ImageDataset.load_file_list("train")[-200:]
    dataset = ImageDataset(data_list=file_list, validation_split=True, mil=False, rgb=True)
    for elem in dataset.dataset_train.take(1):
        print(elem[1])
        print(elem[0].shape)
    # TODO: Memory Problem sind die 3 Channel

    # Taken From:
    # https://tvst.arvojournals.org/article.aspx?articleid=2770240

    vgg_model = create_vgg_model()
    vgg_model.fit(dataset.dataset_train.batch(2), validation_data=dataset.dataset_val)


