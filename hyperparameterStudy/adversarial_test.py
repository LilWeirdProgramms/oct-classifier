import os

import matplotlib.pyplot as plt

from hyperparameterStudy.mil_postprocessing import MilPostprocessor
from PreprocessMILImageData import PreprocessMILImageData
from PreprocessData import PreprocessData
import tensorflow.keras as k
import tensorflow as tf
import matplotlib.pyplot as plt

os.chdir("../")
data_type = "test"
model_name = "ave_pool_selu_lay5_no_drop_little_l2_global_ave_pooling_n32_zeros_afalse_no_noise_fft_denoise5_residual_mil_cfalse_nfalse_images_lfalse"
model_path = "results/hyperparameter_study/mil/models"
file_list = PreprocessData.load_file_list(data_type, angio_or_structure="images")
pid = PreprocessMILImageData(input_file_list=file_list, rgb=False, crop=False, data_type=data_type)
pid.preprocess_data_and_save()
ds = pid.create_dataset_for_calculation()

# visualize_file_list = PreprocessMILImageData.load_file_list("test", angio_or_structure="images")
# visualize_file_list = sorted(visualize_file_list, key=lambda file: (int(file[1]),
#                                                                     int(file[0].split("_")[-1][:-4])))
pretrained_model = k.models.load_model(os.path.join(model_path, model_name))
ds_norm = ds.batch(1)
image_probs = pretrained_model.predict(ds_norm, verbose=1)

loss_object = k.losses.BinaryCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  eps = 0.01
  adv_image = input_image + eps*signed_grad
  return adv_image, input_label

i = 1
def skiptake(ds, i):
  i += 1
  if i >= 120:
    return ds.skip(25).take(1)
  else:
    return skiptake(ds.skip(25).take(1), i)

# for elem in ds.skip(4).take(1):
#   plt.figure()
#   plt.imshow(elem[0], "gray")
#   plt.show()

#ds = ds.batch(1).map(create_adversarial_pattern)
ds_adv = ds.batch(1).map(create_adversarial_pattern)
image_probs_adv = pretrained_model.predict(ds_adv, verbose=1)

ins, adv, norm = 0, 0, 0
for elem in zip(ds_norm, image_probs_adv, image_probs):
  ins += 1
  adv += (image_probs_adv > 0 and ds_norm[1] == 1.) or (image_probs_adv < 0 and ds_norm[1] == 0.)
  norm += (image_probs > 0 and ds_norm[1] == 1.) or (image_probs < 0 and ds_norm[1] == 0.)

print(adv/ins, norm/ins)

# for elem in ds.skip(4).take(1):
#   plt.figure()
#   plt.imshow(elem[0][0], "gray")
#   plt.show()


# hp = MilPostprocessor(model_name, ds, visualize_file_list, crop=False)
# hp.mil_postprocessing()
