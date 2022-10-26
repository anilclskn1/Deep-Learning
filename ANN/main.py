import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


tf.__version__

dataset = pd.read_csv(
    '/Users/anilytugce/Desktop/Projects/Deep Learning/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep '
    'Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

print(x)
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])])
x = np.array(ct.fit_transform(x))

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, 0.2, 0)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(6, 'relu'))

ann.add(tf.keras.layers.Dense(6, 'relu'))

ann.add(tf.keras.layers.Dense(1, 'sigmoid'))

ann.compile('adam', 'binary_crossentropy', ['accuracy'])

ann.fit(x_train, y_train, 32, 100)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
