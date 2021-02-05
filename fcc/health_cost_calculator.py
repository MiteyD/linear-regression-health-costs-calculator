# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:35:24 2020

@author: HP
"""

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('insurance.csv')

# Replacing string values to numbers
dataset['sex'] = dataset['sex'].apply({'male':0, 'female':1}.get) 
dataset['smoker'] = dataset['smoker'].apply({'yes':1, 'no':0}.get)
dataset['region'] = dataset['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
dataset = np.array(ct.fit_transform(dataset))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = 0)

# To dataframe
train_dataset = pd.DataFrame(train_dataset)
test_dataset = pd.DataFrame(test_dataset)

# Pop the 'expenses' column
train_labels = train_dataset.pop(11)
test_labels = test_dataset.pop(11)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_dataset = sc.fit_transform(train_dataset)
test_dataset = sc.transform(test_dataset)


# Part 2 - Building the Model

# Initializing the Model
model = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Part 3 - Training the ANN

# Compiling the ANN
from keras.optimizers import SGD
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics = ['mae', 'mse'])

# Training the ANN on the Training set
model.fit(train_dataset, train_labels,
          batch_size = 32,
          epochs = 100,
          validation_data = (test_dataset, test_labels),
          verbose = 2)


loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
