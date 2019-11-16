import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix

def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

def plot_learning_curves(output_dir, curve_type, train_losses, valid_losses, valid=True):
    # create list of epochs
    epoch_num = len(train_losses)
    epoch_list = [i for i in range(1,epoch_num+1)]

    # Generate the Loss Curve
    plt.figure()
    plt.plot(epoch_list, train_losses, label="Training {}".format(curve_type))

    if valid:
        plt.plot(epoch_list, valid_losses, label="Validation {}".format(curve_type))
    plt.title("{} Curve".format(curve_type))
    plt.xlabel("Epoch")
    plt.ylabel(curve_type)
    if valid:
        plt.legend(loc="best")

    # save the plot
    plot_filename = os.path.join(output_dir,"{} curve at {} epochs.png".format(curve_type, epoch_num))
    plt.savefig(plot_filename)

def plot_confusion_matrix(output_dir, y_true, y_pred, class_names, normalized=True):
    # Note that this code was heavily inspired by the Sklearn examples for confusion matrix plots
    # Reference code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    # calculate the normalized confusion_matrix using sklearn function
    cm = confusion_matrix(y_true, y_pred)

    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)
        plot_title = "Normalized Confusion Matrix"
    else:
        plot_title = "Confusion Matrix"

    # set print options up to 2 digits
    np.set_printoptions(precision=2)

    # setup the plot for confusion matrix
    fig, ax = plt.subplots(figsize=(8,6))

    # set the colormap to Blues
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # adjust the colormap ticks and boundaries to range from 0 to 1
    ax.figure.colorbar(im, ax=ax 
                    # boundaries = np.linspace(0.,1.,100), 
                    # ticks=np.linspace(0.,1.,6)
                    )

    # Label all with appropriate class names
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title=plot_title,
        ylabel='True',
        xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plot_filename = os.path.join(output_dir,"{}.png".format(plot_title))
    fig.savefig(plot_filename)

def plot_roc_over_epochs(output_dir, auroc):
    df_auc = pd.DataFrame(auroc)
    df_auc['MeanAUC'] = df_auc.mean(axis=1)
    df_auc.index = df_auc.index + 1
    df_auc.plot()
    plt.legend(loc="best", bbox_to_anchor=(1.1,1))
    plt.title("Validation AUC ROC over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC ROC")

    # save the plot
    plot_filename = os.path.join(output_dir,"Valid AUROC.png")
    plt.savefig(plot_filename)