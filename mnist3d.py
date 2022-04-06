import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import copy

class MNISTDataHandler:
    """
    Is it the number 5 MNIST
    """

    def __init__(self, data_size=2000, frequency=False):
        use_every = 2
        self.use_every = use_every
        self.bagging = False
        with h5py.File("data/3d_mnist/full_dataset_vectors.h5", "r") as hf:
            self.x_train = hf["X_train"][:data_size].reshape((data_size, 16, 16, 16, 1))[::use_every]
            self.y_train = hf["y_train"][:data_size].reshape((data_size,))[::use_every]
            self.x_val = hf["X_train"][data_size:int(data_size/10)+data_size].reshape((int(data_size/10), 16, 16, 16, 1))[::use_every]
            self.y_val = hf["y_train"][data_size:int(data_size/10)+data_size].reshape((int(data_size/10),))[::use_every]
            self.x_test = hf["X_test"][:int(data_size/10)].reshape((int(data_size/10), 16, 16, 16, 1))[::use_every]
            self.y_test = hf["y_test"][:int(data_size/10)].reshape((int(data_size/10),))[::use_every]
        self.x_train_bag = []
        self.y_train_bag = []
        self.x_test_bag = []
        self.y_test_bag = []
        self.train_size = int(data_size/use_every)
        self.val_size = int(self.train_size/10)
        with h5py.File("data/3d_mnist/train_point_clouds.h5", "r") as hf:
            a = hf["0"]
            parsing = (a["img"][:], a["points"][:], a.attrs["label"])
            self.random_number1 = parsing[0]
            self.random_transform1 = None
            a = hf["1"]
            parsing = (a["img"][:], a["points"][:], a.attrs["label"])
            self.random_number2 = parsing[0]
            self.random_transform2 = None
            self.train_dataset = None
            self.test_dataset = None
        self.frequency = frequency
        if self.frequency:
            self.create_frequency_dataset()

    # @ staticmethod
    # def create_mnist_bags(data, labels, bagsize=8):
    #     all_bags = []
    #     for sample, label in zip(data, labels):
    #         one_bag = []
    #         for i in range(bagsize):
    #             one_bag.append((sample, label))
    #         all_bags.append(one_bag)
    #     return all_bags

    def create_mnist_bags(self, data, labels: np.ndarray, bagsize=5):
        org_labels = labels
        original_true = np.logical_or(labels == 5, labels == 9).astype("int16")
        noise_reduced_ret = copy.deepcopy(original_true)

        labels = original_true
        if self.use_every == 1:
            yes = 0
            noise_reduced = []
            for element in original_true:
                if yes == 1:
                    yes = 0
                    noise_reduced.append(0)
                elif element == 1:
                    noise_reduced.append(1)
                    yes = 1
                else:
                    noise_reduced.append(0)
            noise_reduced_ret = np.array(noise_reduced)

        #labels = (labels == 5).astype("int16")
        if self.bagging:
            for j in range(int(org_labels.size / bagsize)):
                true_instances = np.logical_or(org_labels[bagsize*j:bagsize*j+bagsize] == 5,
                                               org_labels[bagsize*j:bagsize*j+bagsize] == 9)
                labels[bagsize*j:bagsize*j+bagsize] = np.logical_or(true_instances, np.any(true_instances))
        return labels, noise_reduced_ret

    def put_data_in_bags(self):
        self.y_train, self.y_train_org = self.create_mnist_bags(self.x_train, self.y_train)
        self.y_val, self.y_val_org = self.create_mnist_bags(self.x_val, self.y_val)
        self.y_test, self.y_test_org = self.create_mnist_bags(self.x_test, self.y_test)

    def transform_into_frequency(self, data: np.ndarray = None, only_real = True):
        """
        Transforms numpy array into fourier domain along the 0 axis (1D) (from top of number to bottom)
        :param data:
        :return:
        """
        if data is None:
            data = self.random_number2
        fourier_transformed_sample = fft(data, axis=0).real.astype("float32")
        return fourier_transformed_sample

    def create_frequency_dataset(self):
        self.x_train = np.array([self.transform_into_frequency(element) for element in self.x_train])
        self.x_val = np.array([self.transform_into_frequency(element) for element in self.x_val])
        self.x_test = np.array([self.transform_into_frequency(element) for element in self.x_test])

    def same_amount(self):
        bool_train_labels = self.y_train == 1
        pos_features = self.x_train[bool_train_labels]
        neg_features = self.x_train[~bool_train_labels]
        pos_labels = self.y_train[bool_train_labels]
        neg_labels = self.y_train[~bool_train_labels]
        BUFFER_SIZE = 100000
        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
            ds = ds.shuffle(BUFFER_SIZE)
            return ds
        pos_ds = make_ds(pos_features, pos_labels).repeat()
        neg_ds = make_ds(neg_features, neg_labels)
        resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], stop_on_empty_dataset=True)

        bool_train_labels = self.y_test == 1
        pos_features = self.x_test[bool_train_labels]
        neg_features = self.x_test[~bool_train_labels]
        pos_labels = self.y_test[bool_train_labels]
        neg_labels = self.y_test[~bool_train_labels]
        BUFFER_SIZE = 100000
        pos_ds = make_ds(pos_features, pos_labels).repeat()
        neg_ds = make_ds(neg_features, neg_labels)
        resampled_ds2 = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], stop_on_empty_dataset=True)
        return resampled_ds, resampled_ds2

    def standardize(self):
        scaler = StandardScaler()
        scaler.fit(self.x_train.reshape(self.train_size, 16 * 16 * 16))
        self.x_train = scaler.transform(self.x_train.reshape(self.train_size, 16 * 16 * 16))\
            .reshape(self.train_size, 16, 16, 16, 1)
        self.x_val = scaler.transform(self.x_val.reshape(self.val_size, 16 * 16 * 16))\
            .reshape(self.val_size, 16, 16, 16, 1)
        self.x_test = scaler.transform(self.x_test.reshape(self.val_size, 16 * 16 * 16))\
            .reshape(self.val_size, 16, 16, 16, 1)

    def create_dataset(self):
        self.put_data_in_bags()
        self.standardize()
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        #train_dataset, test_dataset = self.same_amount()
        all_labels = []
        for feature, label in train_dataset:
            all_labels.append(label.numpy())
        print(np.mean(all_labels))
        for element in train_dataset.take(1):
            print("Dataset Dimensions are:")
            print(element[0].shape, element[1].shape)
            print("Example Label:")
            print(element[1])
        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        return train_dataset, val_dataset, test_dataset

    def plot_3dnumber(self, plot_number: np.array = None, transpose_plot=False):
        if plot_number is None:
            plot_number = self.x_train[0]
        plot_number = plot_number.reshape((16, 16, 16))
        plot_number[:, 8, 8] = np.zeros((16,))
        plot_number[:, 6, 8] = np.zeros((16,))
        plot_number[:, 10, 10] = np.zeros((16,))
        if transpose_plot == True:
            plot_number = np.transpose(plot_number, axes=[1, 2, 0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(plot_number > 0.05, alpha=1, edgecolors="k")
        plt.show()


if __name__ == "__main__":
    hello = MNISTDataHandler(frequency=False)
    x = hello.x_train[::]
    y = hello.y_train[::]
    for i in range(0, 10, 1):
        hello.plot_3dnumber(x[i], True)
        #print(y[i])



