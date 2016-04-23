import numpy as np
from scipy.integrate import trapz
from scipy.stats import beta

# DONE:20 Separate ROC analysis from probability density model, consider puting the Beta distribution as an input class for my custom ROC analysis.
class ROC:
    def __init__(self, data, labels):
        """
        Perform ROC analysis on data.

        Input:
        data :
        labels : numpy array of length N, with entries in {0, 1}, where 1 is referred to as positive
        """
        self.N_positives = np.sum(labels)
        self.N_negatives = labels.size - self.N_positives
        self.data = data.squeeze()
        self.labels = labels.astype('f')


    def _get_mixture_fraction(self):
        z = float(self.N_negatives + self.N_positives)
        return [self.N_negatives / z, self.N_positives / z]

    # DONE:30 double check that I am computing the ROC correctly
    def roc_curve(self):
        """
        ROC curves

        Input:
        list of two states

        Return:
        python list of (FP, TP)
        """
        N = self.N_negatives + self.N_positives
        threshold = np.linspace(self.data.max()*1.1,
                                self.data.min()*0.9,
                                2*N)
        true_positive = np.zeros(2*N)
        false_positive = np.zeros(2*N)
        count = 0
        for wthresh in threshold:
            true_positive[count] = np.sum(self.labels[self.data >= wthresh]) / self.N_positives
            false_positive[count] = np.sum(1.-self.labels[self.data >= wthresh]) / self.N_negatives
            count += 1
        return [false_positive, true_positive]

    # DONE:10 precision recall calculation, plot and AUC
    def precision_recall(self):
        """
        Precision and recall

        Returns:
        python list [precision, recall (TPR)]
        """
        N = self.N_negatives + self.N_positives
        threshold = np.linspace(self.data.max()*1.1,
                                self.data.min()*0.9,
                                2*N)
        recall = np.zeros(2*N)
        precision = np.zeros(2*N)
        count = 0
        for wthresh in threshold:
            recall[count] = np.sum(self.labels[self.data >= wthresh]) / self.N_positives
            FPR = np.sum(1. - self.labels[self.data >= wthresh]) / self.N_negatives
            precision[count-1] = recall[-1] / (recall[-1] + FPR)
        nans = precision + recall
        if np.isnan(np.sum(nans)):
            precision = precision[~np.isnan(nans)]
            recall = recall[~np.isnan(nans)]
        return [recall, precision]

    def auprc(self):
        recall, precision = self.precision_recall()
        return trapz(precision, recall)

    def auroc(self):
        fp, tp = self.roc_curve()
        return trapz(tp, fp)
