import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Coste(theta, X, Y, lmbda):

    m = len(Y)
    hipotesis = sigmoid(np.dot(X, theta))
    A = np.dot(-Y, np.log(hipotesis))
    B = np.dot((1 - Y), np.log(1 - hipotesis))
    C = np.dot((lmbda/(2*m)), np.sum(np.power(theta, 2)))
    return ((A-B)/m)+C

def Gradiente_descendiente(theta, X, Y, lmbda):
    m = len(Y)
    hipotesis = sigmoid(np.dot(X, theta))
    g = np.zeros(X.shape[1])
    J= Coste(theta, X, Y, lmbda)
    for j in range(0, X.shape[1]):
        if (j == 0):
            g[j] = np.dot((hipotesis - Y), X[:, j])/m
        else:
            A = np.dot((hipotesis - Y), X[:, j])/m
            g[j] = A + (np.dot(lmbda, theta[j])/m)
    return J, g


def evaluador(X, theta):
    hipotesis = sigmoid(np.dot(X, theta))
    ev = []
    for h in hipotesis:
        if (h > 0.5):
            ev.append(1)
        else:
            ev.append(0)

    return np.array(ev)

def mostrar(pred,pred_train,lmd,grad):
    plt.figure()
    plt.title("Acierto según lambdas para un grado %i" %grad)
    plt.scatter(lmd, pred, c="blue", marker='+', label='testeo')
    plt.scatter(lmd, pred_train, c="red", marker='x', label='entrenamiento')
    plt.xlabel("Tasa aprendizaje")
    plt.ylabel("Acierto")
    plt.legend(loc=1)
    plt.savefig('C:/Users/Jorge/Documents/Master/IAI/PraticaFinal/Resultados_Regresion/Fig_Grado%i' %i)

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

for i in range(1,4):
    v_res = np.array([])
    v_res_train = np.array([])
    st = time.time()
    p = PolynomialFeatures(i)
    X_poly_train = p.fit_transform(X_train)
    X_poly_test = p.fit_transform(X_test)
    theta = np.zeros(X_poly_train.shape[1])

    for j in np.linspace(0, 1.5, num=4):

        #result = opt.fmin_tnc(func=Coste, x0=theta.flatten(), fprime=Gradiente_descendiente,
        #                        args = (X_poly_train, y_train.flatten(), j))
        #theta_opt = result[0]
        theta_opt= minimize(fun=Gradiente_descendiente, x0=theta.flatten(), args=(X_poly_train, y_train.flatten(), j), method='TNC', jac=True, options={'maxiter': 70})
        y_pred = sigmoid(np.dot(X_poly_test, theta_opt.x))
        ev = evaluador(X_poly_test, theta_opt.x)
        res =  np.sum(ev == y_test.flatten()) * 100.0 / len(y_test.flatten())
        v_res = np.append(v_res,res)

        y_pred_train = sigmoid(np.dot(X_poly_train, theta_opt.x))
        ev_train = evaluador(X_poly_train, theta_opt.x)
        res_train =  np.sum(ev_train == y_train.flatten()) * 100.0 / len(y_train.flatten())
        v_res_train = np.append(v_res_train,res_train)

        file = open('C:/Users/Jorge/Documents/Master/IAI/PraticaFinal/Resultados_Regresion/Res_%i.txt' %i,'a')

        file.write("Para un polinomio de grado, %i con una tasa de aprendizaje %f ha tardado %f segundos" %(i,j,time.time()-st))
        file.write("\nPorcentaje de ejemplos acertados: %f" %res)
        file.write("\nPorcentaje de ejemplos acertados con conjunto de entrenamiento: %f" %res_train)
        file.write("\n\n")

    mostrar(v_res,v_res_train,np.linspace(0, 1.5, num=4),i)
file.close()
