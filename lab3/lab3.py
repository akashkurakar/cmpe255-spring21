import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing


class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin',
                     'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0,
                                names=col_names, usecols=col_names)
        print(self.pima.corr().T)
        self.X_test = None
        self.y_test = None

    def define_feature(self):
        # ------Solution 1---------
        # feature_cols = ['pregnant', 'glucose', 'bmi', 'age']
        # --------Solution 2 & Solution 3---------
        # feature_cols = ['pregnant', 'glucose', 'pedigree', 'bmi', 'age']
        feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label

        return X, y

    def train(self):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(tol=10)
        # -----------_Solution 3------------
        #logreg = LogisticRegression(tol=10)
        # --------Solution 2 ---------
        # train a logistic regression model on the training set
        # logreg = LogisticRegression(
        #    C=0.30971587230022724, penalty='l2', solver='saga', max_iter=5000)
        # grid_values = {'penalty': ['l1', 'l2'], 'C': [
        #    0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # space = dict()
        # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
        # space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
        # space['C'] = loguniform(1e-5, 100)
        # search = RandomizedSearchCV(
        #    logreg, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
        # search.fit(X_train, y_train)
        # print(clf.best_score_)
        # model_lr = GridSearchCV(logreg, param_grid=grid_values)
        # model_lr.fit(X_train, y_train)
        # print(search.best_params_)
        # --------Solution 2 & Solution 3 end---------

        logreg.fit(X_train, y_train)
        return logreg

    def predict(self):
        model = self.train()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)

    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()

    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)


if __name__ == "__main__":
    classifer = DiabetesClassifier()
    result = classifer.predict()
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
