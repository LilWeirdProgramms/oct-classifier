import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Postprocessing:

    def __init__(self, prediction_results, belonging_labels):
        self.prediction_results = prediction_results
        self.belonging_labels = belonging_labels

    def binary_confusion_matrix(self, threshold=0, name="results/binary_confusion_matrix.png"):
        """

        :param threshold: Threshold at what probability an instance is decided as being diabetic
        :return:
        """
        self.prediction_results = self.prediction_results > threshold
        cm = confusion_matrix(self.belonging_labels, self.prediction_results, labels=(0, 1))
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax)
        plt.savefig(name)
