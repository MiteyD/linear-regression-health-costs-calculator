---
title: "Linear Regression Health Costs Calculator"
date: 2020-09-04
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Linear Regression Health Costs Calculator

In this project I predicted healthcare costs using a regression algorithm using the data to predict healthcare costs based on new data.

The given dataset contains information about different people including their healthcare costs.

## Project Instructions

*Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containing both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*

---

In this challenge, you will predict healthcare costs using a regression algorithm.

You are given a dataset that contains information about different people including their healthcare costs. Use the data to predict healthcare costs based on new data.

The first two cells of this notebook import libraries and the data.

Make sure to convert categorical data to numbers. Use 80% of the data as the `train_dataset` and 20% of the data as the `test_dataset`.

`pop` off the "expenses" column from these datasets to create new datasets called `train_labels` and `test_labels`. Use these labels when training your model.

Create a model and train it with the `train_dataset`. Run the final cell in this notebook to check your model. The final cell will use the unseen `test_dataset` to check how well the model generalizes.

To pass the challenge, `model.evaluate` must return a Mean Absolute Error of under 3500. This means it predicts health care costs correctly within $3500.

The final cell will also predict expenses using the `test_dataset` and graph the results.


```python
# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
```


```python
# Import data
dataset_path = keras.utils.get_file("insurance.csv", "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv")
dataset = pd.read_csv(dataset_path)
dataset.tail()
```


```python
# Replacing string values to numbers
dataset['sex'] = dataset['sex'].apply({'male':0, 'female':1}.get) 
dataset['smoker'] = dataset['smoker'].apply({'yes':1, 'no':0}.get)
dataset['region'] = dataset['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
```


```python
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
dataset = np.array(ct.fit_transform(dataset))
```


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = 0)
```


```python
# To dataframe in order to pop
train_dataset = pd.DataFrame(train_dataset)
test_dataset = pd.DataFrame(test_dataset)
```


```python
# Pop the 'expenses' column
train_labels = train_dataset.pop(11)
test_labels = test_dataset.pop(11)
```


```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_dataset = sc.fit_transform(train_dataset)
test_dataset = sc.transform(test_dataset)
```


```python
# Building the Model

# Initializing the Model
model = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
model.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
```


```python
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
```


```python
# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)

```
