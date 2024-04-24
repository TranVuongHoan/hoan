import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data= pd.read_csv("")
x= data["chieu cao"].to_numpy()
y= data["can nang"].to_numpy() #label

#visualize data
plt.scatter(x, y)
plt.xlabel("chieu cao")
plt.ylabel("can nang")
plt.savefig("data.png")

x= x[:, None]
model = LinearRegression(fit_intercept=False) #w: coef_, b: intercept
model.fit(x, y)
w= model.coef_

print(w)
plt.figure()
plt.scatter(x, y)
plt.xlabel("chieu cao")
plt.ylabel("can nang")
plt.plot(x.squeeze(), w.T @ x)
plt.savefig("linear_regression_sklearn.png")


#python .\linear_regression_with_bias.py
#python .\linear_regression_sklearn.py
