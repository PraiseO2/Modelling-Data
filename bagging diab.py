# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:14:28 2023

@author: CEO
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
sum = df.isnull().sum()
#print(df.describe())

X = df.drop('Outcome', axis = 1)
y = df['Outcome']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xsc = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xsc, y, test_size=0.2, random_state=10, stratify=y)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv = 5).mean()

from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100,
                  max_samples=0.8, oob_score=True, random_state=0)
bag_model.fit(X_train, y_train)
score = bag_model.oob_score_
scoreb = bag_model.score(X_test, y_test)


























