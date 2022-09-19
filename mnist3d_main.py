import os.path

import numpy as np
import models
import mnist3d
import Callbacks
import tensorflow as tf
import tensorflow.keras as k
from Postprocessor import Postprocessing
import matplotlib.pyplot as plt

def calc_class_weights(labels: np.ndarray):
    a = labels[labels == 1].size
    b = labels[labels == 0].size
    class_weights = {0: (1 / b) * (a + b) / 2, 1: (1 / a) * (a + b) / 2}
    #class_weights = {0: 1, 1: 1}
    print(class_weights)
    return class_weights

import utils

def image_run(batch_size, epochs, name):
    mnist_data = mnist3d.MNISTDataHandler(frequency=False)
    train_dataset, val_dataset, test_dataset = mnist_data.create_dataset()
    class_weights = calc_class_weights(mnist_data.y_train)
    model = models.classiRaw3Dmnist_small((16, 16, 16, 1))
    k.utils.plot_model(model, show_shapes=True, expand_nested=True, show_layer_activations=False)
    utils.calc_receptive_field3D(model)

    #vds2 = vds1.map(lambda x, y: (x, y * np.array([-0.6, 0.6]) + np.array([0.8, 0.2])))

    history = model.fit(
        train_dataset.shuffle(mnist_data.y_train.size).batch(batch_size),
        epochs=epochs,
        validation_data=val_dataset.batch(batch_size),
        callbacks=Callbacks.my_mnist_callbacks(name),
        class_weight=class_weights
    )
    model.save(f'savedModels/mnist{name}')

def image_eval():
    mnist_data = mnist3d.MNISTDataHandler(frequency=False)
    train_dataset, val_dataset, test_dataset = mnist_data.create_dataset()
    set_to_eval = test_dataset
    all_outputs = []
    for i in range(1):
        name = f"bagging3{i}"
        model = k.models.load_model(f'savedModels/mnist_{name}', custom_objects={'MilMetric':models.MilMetric})
        model_output = model.predict(
            set_to_eval.batch(2),
            verbose=1,
            use_multiprocessing=True
        )
        model_output = model_output.flatten()
        all_outputs.append(model_output)

    all_outputs = np.array(all_outputs)
    model_output = np.mean(all_outputs, axis=0)
    labels = []
    for ground_truth, predicted, gt2 in zip(set_to_eval, model_output, mnist_data.y_test_org):
        print(f"Predicted: {predicted}, Ground Truth: {ground_truth[1]}, Truly: {gt2}")
        labels.append(ground_truth[1].numpy())
    make_bag_predicition(model_output, labels)
    calculate_best_threshold(model_output,
                             np.logical_or(mnist_data.y_test_org == 4, mnist_data.y_test_org == 9), labels)
    pp = Postprocessing(model_output, np.logical_or(mnist_data.y_test_org == 4, mnist_data.y_test_org == 9).astype("int16"))
    pp.binary_confusion_matrix(threshold=1.5, name=os.path.join("/home/julius/Documents/masterarbeit/arbeit_figures/confusion_fourier", name))
    print(np.max(model_output))
    print(np.mean(model_output))
    print(np.mean(model_output[np.logical_or(mnist_data.y_test_org == 4, mnist_data.y_test_org == 9)]))
    true_labels = np.logical_or(mnist_data.y_test_org == 4, mnist_data.y_test_org == 9)
    plot_roc(model_output, labels, true_labels)
    import seaborn as sns
    sns.displot(model_output, bins=10, kde=True)
    plt.show()
    import pandas as pd
    df = pd.DataFrame(
        {"Prediction": model_output.reshape(-1, ), "MNIST value": mnist_data.y_test_org})
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(5, 5))
    sns.boxplot(x="MNIST value", y="Prediction", data=df)
    plt.title("Predictions Sorted by Underlying Label")
    plt.show()
    cntr = 0
    for a, b in zip((model_output > 0), labels):
        if a == b:
            cntr += 1
    print(f"Accuracy: {cntr / len(labels)}")
    cntr = 0
    for a, b in zip((model_output > 0), np.logical_or(mnist_data.y_test_org == 4, mnist_data.y_test_org == 9)):
        if a == b:
            cntr += 1
    print(f"Accuracy true: {cntr / len(labels)}")
    cntr2 = 0
    for a, b in zip(mnist_data.y_test_org, labels):
        if b == 1 and a != 4 and a != 9:
            cntr2 += 1
    print(f"Amount of false positive labels {cntr2}")

def make_bag_predicition(model_output, labels, ture_labels=None):
    bag_model_output = model_output.reshape((-1, 2))
    bag_labels = np.array(labels).reshape((-1, 2))
    #bag_true_labels = ture_labels[:99].reshape((-1, 3))
    bag_labels = np.mean(bag_labels, axis=1)
    #bag_model_output_pooling = np.mean(bag_model_output, axis=1)
    #bag_model_output_pooling = np.maximum(bag_model_output, 0)
    bag_model_output_pooling = np.max(bag_model_output, axis=1)
    #bag_model_output_pooling = np.maximum(bag_model_output, 0)
    #bag_model_output_pooling = np.mean(bag_model_output_pooling**2, axis=1)
    plot_roc2(bag_model_output_pooling, bag_labels)

def calculate_best_threshold(prediction, true_labels, labels):
    fpr, tpr, threshold = metrics.roc_curve(labels, prediction, pos_label=1)
    result = []
    result2 = []
    import seaborn as sns
    for i in range(100):
        fpr_limit = (i+1) / 100
        gmean = np.sqrt(tpr * (1 - fpr))
        gmean = gmean[fpr < fpr_limit]
        index = np.argmax(gmean)
        best_threshold = threshold[index]
        result2.append(best_threshold)
        cntr = 0
        for a, b in zip((prediction > best_threshold), true_labels):
            if a == b:
                cntr += 1
        result.append(cntr/len(labels))
    sns.set_theme()
    fig, ax = plt.subplots(1, 2, figsize=(7, 5))
    ax[0].plot((np.array(list(range(100)))+1) / 100, result, lw=2)
    ax[1].plot((0, 1), (0, 0), "--", color="black", lw=0.5)
    ax[1].plot((np.array(list(range(100)))+1) / 100, result2, color="orange", lw=2)
    ax[0].set_xlabel('Allowed false positive rate')
    ax[1].set_xlabel('Allowed false positive rate')
    ax[0].set_ylabel('Test Accuracy')
    ax[1].set_ylabel('Threshold Value')
    plt.tight_layout()
    plt.show()
    print(f"Max Accuracy Threshold: {result2[np.argmax(result)]}")
    print(f"Max Accuracy: {np.max(result)}")


from sklearn import metrics
def plot_roc(prediction, labels, true_labels):
    import seaborn as sns
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(5, 5))
    fpr, tpr, threshold = metrics.roc_curve(labels, prediction, pos_label=1)
    fpr_true, tpr_true, threshold_true = metrics.roc_curve(true_labels, prediction, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction)
    auc_true = metrics.roc_auc_score(true_labels, prediction)
    print(f"AUC = {round(auc, 5)}, AUC true = {round(auc_true, 5)}")
    plt.plot(fpr, tpr)
    plt.plot(fpr_true, tpr_true)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic (ROC)")
    plt.legend(["Actual ROC Curve", "Underlying ROC Curve"], loc="lower right")
    #plt.legend([f"AUC = {round(auc, 3)}"], loc="lower right")
    plt.show()



    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean[fpr < 0.1])
    #index = np.argmax(gmean)
    best_threshold = threshold[index]
    print("Calculated Threshold:" + str(best_threshold))
    #precision = tpr / (tpr + fpr)
    #print(precision)
    #precision = tpr_true / (tpr_true + fpr_true)
    #print(precision)

def plot_roc2(prediction, labels):
    import seaborn as sns
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(5, 5))
    fpr, tpr, threshold = metrics.roc_curve(labels, prediction, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve, AUC = {auc}")
    plt.show()
    gmean = np.sqrt(tpr * (1 - fpr))
    # np.argmax(gmean[fpr < 0.5])
    index = np.argmax(gmean)
    best_threshold = threshold[index]
    print("Threshold:" + str(best_threshold))
    #precision = tpr / (tpr + fpr)
    #print(precision)
    #precision = tpr_true / (tpr_true + fpr_true)
    #print(precision)
    cntr = 0
    for a, b in zip((prediction > 2.3), labels):
        if a == b:
            cntr += 1
    print(f"MIL Accuracy: {cntr / len(labels)}")


def plot_data():
    mnist_data = mnist3d.MNISTDataHandler(frequency=False)
    train_dataset, val_dataset, test_dataset = mnist_data.create_dataset()
    # for elem, label in train_dataset.take(10):
    #     print(label)
    #     mnist_data.show_3dnumber(np.transpose(elem.numpy()[::], (1, 2, 0, 3)))
    #     # plt.figure(figsize=(5, 5))
    #     # plt.imshow(elem.numpy()[::-1, ::, 6, 0], cmap="gray")
    #     # plt.colorbar()
    #     plt.show()
    for elem, label in train_dataset.take(1):
        print(label)
        mnist_data.plot_3dnumber(np.transpose(elem.numpy()[::], (1, 2, 0, 3)))
        plt.figure(figsize=(5, 5))
        plt.imshow(elem.numpy()[::-1, ::, 6, 0], cmap="gray")
        plt.colorbar()
        plt.show()
    mnist_data = mnist3d.MNISTDataHandler(frequency=True)
    train_dataset, val_dataset, test_dataset = mnist_data.create_dataset()
    for elem, label in train_dataset.take(1):
        print(label)
        mnist_data.plot_3dnumber(np.transpose(elem.numpy()[::], (1, 2, 0, 3)))
        plt.figure(figsize=(5, 5))
        plt.imshow(elem.numpy()[::-1, ::, 6, 0], cmap="gray")
        plt.colorbar()
        plt.show()



def main(eval=False, plot=False):
    if plot:
        plot_data()
        return
    if not eval:
        for i in range(1):
            name = f"bagging3{i}"
            image_run(2, 200, name)
    else:
        image_eval()

if __name__ == "__main__":
    #main()
    #main(plot=True)
    main(eval=True)
