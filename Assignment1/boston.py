from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator

boston = pd.read_csv('housing.csv', names=[
                     'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])

# Understanding Data
boston_dataframe = pd.DataFrame(boston)
boston_dataframe.isnull().sum()


X = boston_dataframe['RM']
y = boston_dataframe['MEDV']

X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Train Model


def linearRegression():
    reg_1 = LinearRegression()
    reg_1.fit(X_train_1, Y_train_1)
    y_train_predict_1 = reg_1.predict(X_train_1)

    rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
    r2 = round(reg_1.score(X_train_1, Y_train_1), 2)
    y_train_predict_1 = reg_1.predict(X_train_1)
    #plt.scatter(X_train_1, Y_train_1)
    #ax = sns.regplot(x=X_train_1, y=y_train_predict_1)
    #ax.set(xlabel='RM', ylabel='Predicted Price', title='Train Data Set')
    plt.show()
    print("The Lineat model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

    y_test_pred_1 = reg_1.predict(X_test_1)
    #plt.scatter(Y_test_1, y_test_pred_1)
    #ax = sns.regplot(x=Y_test_1, y=y_test_pred_1)
    #ax.set(xlabel='RM', ylabel='Predicted Price', title='Test Data Set')
    plt.show()
    rmse = (np.sqrt(mean_squared_error(Y_test_1, y_test_pred_1)))
    r2 = round(reg_1.score(X_test_1, Y_test_1), 2)

    print("The Linear model performance for test set")
    print("--------------------------------------")
    print("Root Mean Squared Error: {}".format(rmse))
    print("R^2: {}".format(r2))
    print("\n")
    print(len(X))
    y_train_predict = reg_1.predict(X)
    plt.scatter(X, y)
    ax = sns.regplot(x=X, y=y_train_predict)
    ax.set(xlabel='RM', ylabel='Predicted Price', title='Room vs Price')
    plt.show()


def polynomialRegression(degree):
    polynomial_features = PolynomialFeatures(degree)
    X = boston_dataframe['RM']
    y = boston_dataframe['MEDV']
    print(X.head())

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    x_poly = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    print(rmse)
    print(r2)

    plt.scatter(X, y, s=10)

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_poly_pred), key=sort_axis)
    X, y_poly_pred = zip(*sorted_zip)
    plt.plot(X, y_poly_pred, color='m')
    plt.xlabel("Original Price")
    plt.ylabel("Predicted Price")
    plt.title("Polynomial curve for degree"+str(degree))
    plt.show()


def multipleLinearRegression():
    X = boston_dataframe[['RM', 'LSTAT', 'PTRATIO']]
    y = boston_dataframe['MEDV']

    X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(
        X, y, test_size=0.2, random_state=42)
    reg_1 = LinearRegression()
    print(len(y))
    reg_1.fit(X_train_1, Y_train_1)
    print(reg_1.coef_)
    print(reg_1.intercept_)

    y_poly_pred = reg_1.predict(X_test_1)

    rmse = (np.sqrt(mean_squared_error(Y_test_1, y_poly_pred)))

    r2 = round(reg_1.score(X_test_1, Y_test_1), 2)
    adjusted_r_squared = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    print(rmse)
    print(r2)
    print(adjusted_r_squared)


def test() -> None:
    linearRegression()
    polynomialRegression(2)
    polynomialRegression(20)
    multipleLinearRegression()


if __name__ == "__main__":
    # execute only if run as a script
    test()
