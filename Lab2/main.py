from Lab2 import dataset

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def model(clf_model, predictors_train, response_train, predictors_test, response_test):

    print('Calculate:', clf_model.__class__.__name__)

    if isinstance(clf_model, QuadraticDiscriminantAnalysis):
        parameters = {'reg_param': np.arange(0.0, 1, 0.05).tolist()}
    elif isinstance(clf_model, KNeighborsClassifier):
        parameters = {
            'n_neighbors': list(range(1, 10))
        }
    elif isinstance(clf_model, LogisticRegression):
        parameters = {
            'penalty': ['l1', 'l2'],
            'C': np.arange(0.5, 1.5, 0.05).tolist()
        }

    clf = GridSearchCV(clf_model, parameters)

    probas = clf.fit(predictors_train, response_train).predict_proba(predictors_test)
    response_pred = clf.best_estimator_.predict(predictors_test).tolist()

    roc = roc_curve(response_test, probas[:, 1])
    conf = confusion_matrix(response_test, response_pred)

    return clf, roc, conf


if __name__ == '__main__':

    data, data_true, data_false = dataset.read()
    dataset.plot(data_true, data_false)

    predictors = list(data[:, 0:2])
    response = list(data[:, -1])

    data_split = train_test_split(predictors, response, test_size=.1, random_state=1)
    predictors_train, predictors_test, response_train, response_test = data_split

    models = {
        'knn': model(KNeighborsClassifier(), predictors_train, response_train, predictors_test, response_test),
        'log': model(LogisticRegression(), predictors_train, response_train, predictors_test, response_test),
        'qda': model(QuadraticDiscriminantAnalysis(), predictors_train, response_train, predictors_test, response_test),
    }

    classes = ['rainy', 'dry']
    roc_fig_data = []
    roc_fig_data_01 = []
    titles = 'Confusion matrix for %s'
    for key, value in models.items():

        clf, roc, conf = value

        roc_auc = auc(roc[0], roc[1])
        roc_fig_data.append([roc[0], roc[1], roc_auc, key])

        index = 0
        for _index, _x in enumerate(roc[0]):
            if _x < 0.1:
                index = _index
            else:
                break

        roc_auc = np.trapz(roc[0][:index], roc[1][:index])
        roc_fig_data_01.append([roc[0], roc[1], roc_auc, key])

        plot_confusion_matrix(conf / conf.sum(), classes, title=titles % key)

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    for _x, _y, _auc, _title in roc_fig_data:
        plt.plot(_x, _y, label='ROC curve %s (area = %0.2f)' % (_title, _auc))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    for _x, _y, _auc, _title in roc_fig_data_01:
        plt.plot(_x, _y, label='ROC curve %s' % _title)

    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 0.5])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print('Done')