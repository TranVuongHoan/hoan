import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv("")

x= data["chieu cao"].to_numpy()
y= data["can nang"].to_numpy() #label

#visualize data
plt.scatter(x, y)
plt.xlabel("chieu cao")
plt.ylabel("can nang")
plt.savefig("data.png")
# x * x: element-wise/ hadarmard product(same shape)
# A @ B: matrix multiplication ( so cot A = so hang B)
# tim w: y= w * x + b = f(x)
#x= x[None, :] #x = x.reshape(1, -1) -> (1, N) (row: so sample, col: so feature)
#w= np.linalg.pinv(x @ x.T) @ x @ y
x= x[:, None] #(N, 1)
x_ones= np.ones_like(x) #(N,1)
x= np.concatenate((x, x_ones), axis=1) #(N, 2)
w=np.linalg.pinv


plt.figure()
plt.scatter(x, y)
plt.xlabel("chieu cao")
plt.ylabel("can nang")
plt.plot(x.squeeze(), w.T @ x)
plt.savefig("linear_regression.png")

import numpy as np
x= 10
#y= (x-2)*2 + 2x
n_iter= 1000
lr= 0.01
y_grad= 10000
i=0
for i in range(n_iter):
    if y_grid < 0.1:
        break
y_grad= 2 * x - 2
x= x - lr * y_grad
print(x, i, y_grad)
i += 1

def sigmoid(x):
    t= 1 / (1 + np.exp(-x))
    return t

def loss(y, p):
    return np.mean(-y * np.log(p) - (1-y) * np.log(1-p))

def gradient_loss(y_gr, p, x):
    return x.T @ (y_gr - p)

import pandas as pd
data= pd.read_csv("c:\Users\hoan\Downloads\04_dataset.csv").values
x= data[:, :2]
y= data[:, 2].reshape(-1, 1)
x.shape, y.shape

#def predict(X, w)
x= data[:, :2]
y= data[:, 2].reshape(-1, 1)
n_iter= 1000
threshold= 0.1
w= np.array([0., 0.1, 0.1]).reshape(-1, 1)
x= np.concatenate((x, np.ones(x.shape[0], 1)), axis= 1)
#x: (20,2); y; (20,1)
#y = sigmoid(x*W+b)
for i in range(n_iter):
    if (gradient < threshold).all():
        break
    y_predict= sigmoid(np.dot(x, w))
    #print(y_predict)
    gradient= gradient_loss(y, y_predict, x)
    w = w - lr * gradient

import matplotlib.pyplot as plt
x_tu_choi= x[y[:,0]==0]
x_cho_vay= x[y[:,0]==1]
plt.show()
plt.legend(loc=1)
plt.scatter(x_cho_vay[:,0], x_cho_vay[:, 1], c= "red", edgecolors= "none")
