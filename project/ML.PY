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
x= x[None, :]#x = x.reshape(1, -1)
w= np.linalg.pinv(x @ x.T) @ x @ y
print(w)
plt.figure()
plt.scatter(x, y)
plt.xlabel("chieu cao")
plt.ylabel("can nang")
plt.plot(x.squeeze(), w.T @ x)
plt.savefig("linear_regression.png")

