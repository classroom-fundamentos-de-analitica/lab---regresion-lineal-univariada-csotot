"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def pregunta_01():
    df = pd.read_csv('gm_2008_region.csv')
    y = df['life'].values
    X = df['fertility'].values
    print(y.shape)
    print(X.shape)
    y_reshaped = y.reshape(-1, 1)
    X_reshaped = X.reshape(-1, 1)
    print(y_reshaped.shape)
    print(X_reshaped.shape)

def pregunta_02():
    df = pd.read_csv('gm_2008_region.csv')
    print(df.shape)
    print(round(df['life'].corr(df['fertility']), 4))
    print(round(df['life'].mean(), 4))
    print(df['fertility'].dtype)
    print(round(df['GDP'].corr(df['life']), 4))

def pregunta_03():
    df = pd.read_csv('gm_2008_region.csv')
    X_fertility = df['fertility'].values
    y_life = df['life'].values
    reg = LinearRegression()
    prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1, 1)
    reg.fit(X_fertility.reshape(-1, 1), y_life)
    y_pred = reg.predict(prediction_space)
    print(reg.score(X_fertility.reshape(-1, 1), y_life).round(4))

def pregunta_04():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('gm_2008_region.csv')
    X_fertility = df['fertility'].values
    y_life = df['life'].values
    X_train, X_test, y_train, y_test = train_test_split(X_fertility.reshape(-1, 1), y_life, test_size=0.2, random_state=53)
    linearRegression = LinearRegression()
    linearRegression.fit(X_train, y_train)
    y_pred = linearRegression.predict(X_test)
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
