import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,classification_report


# Leemos datos
data = pd.read_csv('C:/Users/Jorge/Documents/Master/IAI/PraticaFinal/mushrooms.csv')

# Agrupamos variables de entrada y salida
# y transformamos cada letra en un numero
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Al ser np-array el algoritmo va más rapido.
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = np.array(X_train)
y_train = np.array(y_train)

for i in range (0,2):
    classifier = Sequential()
    if i > 0:
        classifier.add(Dense(25, kernel_initializer='uniform', activation= 'relu', input_dim = X.shape[1]))

    classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid'))
    classifier.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])

    history = classifier.fit(X_train,y_train,batch_size=2,epochs=6)



    y_pred=classifier.predict(X_test)
    y_pred=(y_pred>0.5)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred_train=classifier.predict(X_train)
    y_pred_train=(y_pred_train>0.5)

    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))
