import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

class Metrics():

    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes

    def compute_roc_auc(self, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()
        for i in range(self.n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])

        # Compute micro-average ROC curve and ROC area
        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        self.roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])

    def compute_macro_auc(self):
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])
        print('macro AUC: ', self.roc_auc["macro"])

    def plot_roc(self, y_test, y_prob, model_name):
        lw = 2
        # Plot all ROC curves
        plt.figure()
        plt.plot(
            self.fpr["micro"],
            self.tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            self.fpr["macro"],
            self.tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(
                self.fpr[i],
                self.tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, self.roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(f'figures/auc_score_{model_name}.png', dpi=300)

        macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
        weighted_roc_auc_ovo = roc_auc_score(
            y_test, y_prob, multi_class="ovo", average="weighted"
        )
        macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="weighted"
        )
        print(
            "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        print(
            "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )