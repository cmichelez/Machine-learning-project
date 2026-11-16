from matplotlib import pyplot as plt
from sklearn import metrics


def accuracy_score(y_true, y_pred) -> float:
    """
    Computes the proba that the prediction is equal to the label.
    """
    return sum(y_true == y_pred) / len(y_true)


def recall_score(y_true, y_pred, neg_label, pos_label) -> float:
    """
    Computes the proba that the prediction is 1 knowing that the label is 1.
    """
    y_1_ = [(true, pred) for true, pred in zip(y_true, y_pred, strict=True) if true == pos_label]
    y_11 = [(true, pred) for true, pred in y_1_ if pred == pos_label]
    recall = 0.0 if len(y_1_) == 0 else len(y_11) / len(y_1_)
    return recall


def precision_score(y_true, y_pred, neg_label, pos_label) -> float:
    """
    Computes the proba that the label is 1 knowing that the prediction is 1.
    """
    y__1 = [(true, pred) for true, pred in zip(y_true, y_pred, strict=True) if pred == pos_label]
    y_11 = [(true, pred) for true, pred in y__1 if true == pos_label]
    prec = 0.0 if len(y__1) == 0 else len(y_11) / len(y__1)
    return prec


def specificity_score(y_true, y_pred, neg_label, pos_label) -> float:
    """
    Computes the proba that the label is 0 knowing that the prediction is 0.
    """
    y__0 = [(true, pred) for true, pred in zip(y_true, y_pred, strict=True) if pred == neg_label]
    y_00 = [(true, pred) for true, pred in y__0 if true == neg_label]
    spec = 0.0 if len(y__0) == 0 else len(y_00) / len(y__0)
    return spec


def fpr_score(y_true, y_pred, neg_label, pos_label) -> float:
    """
    Computes the proba that the prediction is 1 knowing that the label is 0.
    """
    y_0_ = [(true, pred) for true, pred in zip(y_true, y_pred, strict=True) if true == neg_label]
    y_01 = [(true, pred) for true, pred in y_0_ if pred == pos_label]
    fpr = 0.0 if len(y_0_) == 0 else len(y_01) / len(y_0_)
    return fpr


def f1_score(y_true, y_pred, neg_label, pos_label) -> float:
    """
    Computes the f1-score.
    """
    recc = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        neg_label=neg_label,
        pos_label=pos_label,
    )
    prec = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        neg_label=neg_label,
        pos_label=pos_label,
    )
    s = float(recc + prec)
    return 0.0 if s == 0.0 else (2 * recc * prec / s)


def binary_classif_perf(y_true, y_pred, neg_label, pos_label) -> None:
    """
    Evaluate the performance of a binary classification model.
    """
    print("Accuracy : {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("Recall : {:.3f}".format(recall_score(y_true, y_pred, neg_label, pos_label)))
    print("Precision : {:.3f}".format(precision_score(y_true, y_pred, neg_label, pos_label)))
    print("Specificity: {:.3f}".format(specificity_score(y_true, y_pred, neg_label, pos_label)))
    print("False Positive Rate: {:.3f}".format(fpr_score(y_true, y_pred, neg_label, pos_label)))
    print("F1 score : {:.3f}".format(f1_score(y_true, y_pred, neg_label, pos_label)))
    return


def precision_recall_curve_with_threshold(clf, X, y, threshold=0.5):  # noqa: N803
    class_names = clf.classes_.tolist()
    probas = clf.predict_proba(X)[:, 1]
    targets = [class_names.index(t) for t in y]
    precision, recall, thresholds = metrics.precision_recall_curve(targets, probas)

    # calculate precision-recall AUC
    auc_val = metrics.auc(recall, precision)
    print("Area Under Curve (AUC) : {:.3f}".format(auc_val))
    # plot no skill
    if threshold is not None:
        index = len([y for y in thresholds if y <= threshold])  # thresholds are sorted

        print("Recall with threshold = {}: {:.2f}%".format(threshold, recall[index] * 100))
        print("Precision with threshold = {} : {:.2f}%".format(threshold, precision[index] * 100))

        plt.plot([0, 1], [precision[index], precision[index]], linestyle="--", color="grey")
        plt.plot([recall[index], recall[index]], [0, 1], linestyle="--", color="grey")

        # plot the precision-recall curve for the model
        plt.plot(recall[:index], precision[:index], color="navy", marker=".")
        plt.plot(recall[index:], precision[index:], color="orange", marker=".")

        # show the plot
        plt.title("Precision-Recall curve with threshold = {}".format(threshold))
        plt.xlabel(r"Recall $P(\, \widehat{1}\, | 1)$")
        plt.ylabel(r"Precision $P(\, 1\, | \widehat{1})$                                ", rotation="horizontal")
    else:
        # plot the precision-recall curve for the model
        plt.plot(recall, precision, color="orange", marker=".")

        # show the plot
        plt.title("Precision-Recall curve")
        plt.xlabel(r"Recall $P(\, \widehat{1}\, | 1)$")
        plt.ylabel(r"Precision $P(\, 1\, | \widehat{1})$                                ", rotation="horizontal")

    plt.show()
    return
