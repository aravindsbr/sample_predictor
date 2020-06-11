import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


from sklearn.preprocessing import StandardScaler as ss
from sklearn import linear_model
from scipy.signal import savgol_filter
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


class RegressionModels:

    def __int__(self):
        pass

    def data_split(self):
        datasets = pd.read_csv('C:/Users/praveen.ram.kannan/Desktop/data/healthyfy-me-updated.csv')
        X = datasets.iloc[:, 5:12].values
        Y = datasets.iloc[:, -1].values
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=1 / 3, random_state=0)
        return X_Train, X_Test, Y_Train, Y_Test


    def multilinear_model(self, user_data):
        X_Train, X_Test, Y_Train, Y_Test = data_split()
        regressor = LinearRegression()
        regressor.fit(X_Train, Y_Train)
        Y_Pred = regressor.predict(user_data)
        return Y_Pred

    def polynomial_model(self, user_data):
        X_Train, X_Test, Y_Train, Y_Test = data_split()
        poly1 = PolynomialFeatures(degree=2)
        X_poly = poly1.fit_transform(X_Train)
        poly = LinearRegression()
        poly.fit(X_poly, Y_Train)
        Y_Pred = poly.predict(poly1.fit_transform(user_data))
        return Y_Pred

    def support_vector_model(self, user_data):
        X_Train, X_Test, Y_Train, Y_Test = data_split()
        sc = ss()
        X_Train = sc.fit_transform(X_Train)
        X_Test = sc.transform(X_Test)
        regressor = SVR(kernel='rbf')
        regressor.fit(X_Train, Y_Train)
        Y_Pred = regressor.predict(user_data)
        return Y_Pred

    def PCA_regression_model(self, user_data):
        pca = PCA()
        X_Train, X_Test, Y_Train, Y_Test = data_split()
        d1X = savgol_filter(X_Train, 5, polyorder=2, deriv=1)
        Xstd = StandardScaler().fit_transform(d1X[:, :])
        Xreg = pca.fit_transform(Xstd)[:, :]
        regr = linear_model.LinearRegression()
        regr.fit(Xreg, Y_Train)
        Y_Pred = regr.predict(user_data)
        return Y_Pred

    def PLS_regression_model(self, user_data):
        pls = PLSRegression(n_components=5)
        X_Train, X_Test, Y_Train, Y_Test = data_split()
        pls.fit(X_Train, Y_Train)
        Y_Pred = pls.predict(user_data)
        return Y_Pred

    def health_suggestions(self, percent):
        #to-do: suggest health suggestions
        pass

    def dict_to_df(self, user_data):
        # to-do: dictionary to data frame conversion
        pass
