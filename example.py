from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scoco import blending

X, y = load_iris(True)

CV_FOLDER = 'iris_cv_folder'
PREDICTION_FOLDER = 'iris_prediction_folder'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifiers = [MLPClassifier(), SVC(probability=True), GaussianNB(), GaussianProcessClassifier(), KNeighborsClassifier(),
                DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier()]


if not os.path.exists(CV_FOLDER):
    os.makedirs(CV_FOLDER)
else:
    for f in os.listdir(CV_FOLDER):
        os.remove(os.path.join(CV_FOLDER, f))

if not os.path.exists(PREDICTION_FOLDER):
    os.makedirs(PREDICTION_FOLDER)
else:
    for f in os.listdir(PREDICTION_FOLDER):
        os.remove(os.path.join(PREDICTION_FOLDER, f))

# binarize to dependant variable
y_train = label_binarize(y_train, classes=[0, 1, 2])
y_test = label_binarize(y_test, classes=[0, 1, 2])


# persist train data set
df_train_x = pd.DataFrame(X_train, columns=['feature_{}'.format(i) for i in range(X_train.shape[1])])
df_train_y = pd.DataFrame(y_train, columns=['y_{}'.format(i) for i in range(y_train.shape[1])])
df_train = pd.concat([df_train_x, df_train_y], axis=1)
df_train['id'] = list(range(df_train.shape[0]))
df_train.to_csv('train.csv', index=False)

# persist test data set
df_test_x = pd.DataFrame(X_test, columns=['feature_{}'.format(i) for i in range(X_test.shape[1])])
df_test_y = pd.DataFrame(y_test, columns=['y_{}'.format(i) for i in range(y_test.shape[1])])
df_test = pd.concat([df_test_x, df_test_y], axis=1)
df_test['id'] = list(range(df_test.shape[0]))
df_test.to_csv('test.csv', index=False)

# make prediction data and persist them
ids_train, ids_test = list(range(y_train.shape[0])), list(range(y_test.shape[0]))
for classifier in classifiers:
    res_cv, res_pred = {}, {}
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for i in range(y_train.shape[1]): #iterate through the columns
        y_pred_cv = cross_val_predict(classifier, X_train, y_train[:,i], cv=cv, method='predict_proba')[:,1]
        res_cv['y_{}'.format(i)] = y_pred_cv
        res_cv['id'] = ids_train
        classifier.fit(X_train, y_train[:,i])
        res_pred['y_{}'.format(i)] = classifier.predict_proba(X_test)[:,1]
        res_pred['id'] = ids_test
    res_cv = pd.DataFrame(res_cv)
    res_pred = pd.DataFrame(res_pred)
    res_cv.to_csv(os.path.join(CV_FOLDER, '{}.csv'.format(classifier.__class__.__name__)), index=False)
    res_pred.to_csv(os.path.join(PREDICTION_FOLDER, '{}.csv'.format(classifier.__class__.__name__)), index=False)

categories = ['y_0', 'y_1', 'y_2']

methods = ['simple', 'coeff', 'poly', 'exp']
# create blend file according to all possible features
for method in methods:
    blending(categories, 'train.csv', 'iris_cv_folder', 'iris_prediction_folder', method=method, power=15)

methods, scores = [], []
train = pd.read_csv('train.csv')
# compute the average roc_auc score over the categories to predict for all the cv (oof) predictions
for f in os.listdir(CV_FOLDER):
    if f.endswith('.csv'):
        methods.append(f.replace('.csv', ''))
        df = pd.read_csv(os.path.join(CV_FOLDER, f))
        scores.append(np.mean([roc_auc_score(train[category], df[category]) for category in categories]))

cv_result = pd.DataFrame({'method': methods, 'score': scores})


methods, scores = [], []
test = pd.read_csv('test.csv')
# compute the average roc_auc score over the categories to predict for all the test predictions
for f in os.listdir(PREDICTION_FOLDER):
    if f.endswith('.csv'):
        methods.append(f.replace('.csv', ''))
        df = pd.read_csv(os.path.join(PREDICTION_FOLDER, f))
        scores.append(np.mean([roc_auc_score(test[category], df[category]) for category in categories]))

prediction_result = pd.DataFrame({'method': methods, 'score': scores})


print(cv_result)
print(prediction_result)



