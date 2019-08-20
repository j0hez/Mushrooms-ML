import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
import time


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidGrad(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))


def pesosAleatorios(L_in, L_out):
    epsilon = 0.12
    return np.random.rand(L_out, L_in+1) * (2 * epsilon) - epsilon


def cost(params_rn, theta1, theta2, num_etiquetas, X, yk, reg):

    # Inicializamos parametros para calculo de h
    m = X.shape[0]
    ones = np.ones((m, 1))
    a1 = np.hstack((ones, X))
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    A = np.multiply(yk, np.log(h))
    B = np.multiply(1-yk, np.log(1-h))
    C = np.sum(A + B)
    coste = np.sum(C/(-m))

    # Regulacion
    D = np.sum(np.sum(np.power(theta1[:, 1:], 2)))
    E = np.sum(np.sum(np.power(theta2[:, 1:], 2)))
    r = (D+E) * reg / (2 * m)

    coste = coste + r

    return coste


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, yk, reg):

    X = np.matrix(X)
    yk = np.matrix(yk)

    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, (num_ocultas + 1)))

    # Calculo de coste
    J = cost(params_rn, theta1, theta2, num_etiquetas, X, yk, reg)

    # Inicializamos parametros para calculo de h
    m = X.shape[0]

    # Inicializamos Deltas
    delta2 = np.zeros(theta1.shape) # (25L, 401L)
    delta3 = np.zeros(theta2.shape) # (10L, 26L)

    a1 = np.insert(X, 0, values=np.ones(m), axis=1) # (5000L, 401L)
    z2 = np.dot(a1, theta1.T) # (5000L, 25L)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) # (5000L, 26L)
    z3 = np.dot(a2, theta2.T) # (5000L, 10L)
    a3 = sigmoid(z3) # (1L, 10L)

    # Propagacion hacia atras
    for t in range(m):
        a1t = a1[t, :]  # (1L, 401L)
        z2t = z2[t, :]  # (1L, 25L)
        a2t = a2[t, :]  # (1L, 26L)
        a3t = a3[t, :]  # (1L, 10L)

        # Calculo de deltas
        d3 = a3t - yk[t, :] # (1L, 10L)

        z2t = np.insert(z2t, 0, values=np.ones(1)) # (1L, 26L)
        d2 = np.multiply(np.dot(theta2.T, d3.T).T, sigmoidGrad(z2t)) # (1L, 26L)

        # Ignoraos la primera componente de c2
        d2 = d2[:, 1:] # (1L, 25L)

        # Acumulamos Deltas
        delta2 = delta2 + np.dot(d2.T, a1t) # (25L, 401L)
        delta3 = delta3 + np.dot(d3.T, a2t) # (10L, 26L)

    delta3 = delta3 / m
    delta2 = delta2 / m

    # Regularizar solo para j>=1
    delta2[:, 1:] = delta2[:, 1:] + theta1[:, 1:] * reg / m
    delta3[:, 1:] = delta3[:, 1:] + theta2[:, 1:] * reg / m

    grad = np.concatenate((np.ravel(delta2), np.ravel(delta3)))

    return J, grad


def predict(theta1, theta2, X):
    m = len(X)
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = np.dot(a1, theta1.T)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return np.argmax(h, axis=1) + 1


def mostrar(pred, pred_train, lmd, oc):
    plt.figure()
    plt.title("Acierto según lambdas para un numero de capas ocultas %i" %oc)
    plt.scatter(lmd, pred, c="blue", marker='+', label='testeo')
    plt.scatter(lmd, pred_train, c="red", marker='x', label='entrenamiento')
    plt.xlabel("Tasa aprendizaje")
    plt.ylabel("Acierto")
    plt.legend(loc=1)
    plt.savefig('C:/Users/Jorge/Documents/Master/IAI/PraticaFinal/Resultados_Neuronal/Fig_Grado%i' %oc)

# Leemos datos
data = pd.read_csv('C:/Users/Jorge/Documents/Master/IAI/PraticaFinal/mushrooms.csv')

# Agrupamos variables de entrada y salida
# y transformamos cada letra en un numero
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Al ser np-array el algoritmo va más rapido.
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

num_entradas = X.shape[1]

for oc in np.linspace(10, 100, num=2):
    r = np.array([])
    r_train = np.array([])
    st = time.time()

    num_ocultas = int(oc)
    num_etiquetas = 2
    reg = 1

    theta1 = pesosAleatorios(num_entradas, num_ocultas)
    theta2 = pesosAleatorios(num_ocultas, num_etiquetas)

    # parametros de la red en vector columna
    params_rn = np.concatenate((theta1.ravel(), theta2.ravel()))

    yk = np.zeros((X_train.shape[0], num_etiquetas))

    # Lo transformamos a un array
    y_train = np.array(y_train)

    for i in range(X_train.shape[0]):
        yk[i, y_train[i]-1] = 1

    j = 100
    ############ Aprendizaje de parametros
    for lmb in np.linspace(0, 1, num=3): # lambdas
        thetas_opt = minimize(fun=backprop, x0=params_rn, args=(num_entradas, num_ocultas, num_etiquetas, X_train, yk, lmb), method='TNC', jac=True, options={'maxiter': int(j)})
        theta1_opt = np.reshape(thetas_opt.x[:num_ocultas*(num_entradas+1)], (num_ocultas, num_entradas+1))
        theta2_opt = np.reshape(thetas_opt.x[num_ocultas*(num_entradas+1):], (num_etiquetas, num_ocultas+1))

        pred = predict(theta1_opt, theta2_opt, X_test)
        res = np.mean(pred == y_test.flatten()) * 100
        r = np.append(r, res)

        pred_train = predict(theta1_opt, theta2_opt, X_train)
        res_train = np.mean(pred_train == y_train.flatten()) * 100
        r_train = np.append(r_train, res_train)

        file = open('C:/Users/Jorge/Documents/Master/IAI/PraticaFinal/Resultados_Neuronal/Res_num_oc_%i.txt' %oc,'a')

        file.write("Para un lambda=%f, numero de iteraciones= %i y capas ocultas=%i" %(lmb,int(j),oc))
        file.write("\nEl porcentaje de acierto es de un: %f, en un tiempo de ejecución %f segundos" %(res,time.time()-st))
        file.write("\nEl resultado con el conjunto de entrenamiento es %f" %res_train)
        file.write("\n\n")

        print("Para un lambda=%f, numero de iteraciones= %i y capas ocultas=%i" %(lmb,int(j),oc))
        print("\nEl porcentaje de acierto es de un: %f, en un tiempo de ejecución %f segundos" %(res,time.time()-st))
        print("\nEl resultado con el conjunto de entrenamiento es %f" %res_train)
    mostrar(r,r_train,np.linspace(0, 1, num=3),oc)
file.close()
