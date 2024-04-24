#introduction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#get the data

import csv

# Open the text file in read mode
with open('c:\Users\hoan\Downloads\Ecommerce Customers (1).txt', 'r') as infile:
    # Create a CSV file in write mode
    with open('output.csv', 'w', newline='') as outfile:
        # Create a CSV writer object
        writer = csv.writer(outfile)

        # Write the header row to the CSV file
        writer.writerow(["Email", "Address", "Avatar", "Avg", "Session Length", "Time on App" ,"Time on Website","Length of Membership","Yearly Amount Spent"])

        # Loop through each line in the text file
        for line in infile:
            # Split the line into fields using the comma character as the delimiter
            fields = line.strip().split(',')

            # Write the fields to the CSV file
            writer.writerow(fields)

customers = pd.read_csv("output.csv")

#change style of chart
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}

sns.set_style("darkgrid", rc=custom_style)
print(customers)

#make joint plot based on "time on website" and "yearly amount spent"
jointplot_web = sns.jointplot(x="Time on Website", y="Yearly Amount Spent",data=customers, size=10)
plt.show()

#make jointplot based on "Time on app" and "yearly amount spent"
jointplot_app = sns.jointplot(x="Time on App", y="Yearly Amount Spent",data=customers, size=10)
plt.show()

#make jointplot (type: hex) based on "time on app" and "length of membership"
jointplot_hex = sns.jointplot(x="Time on App", y="Length of Membership", data=customers, size=10, kind="hex")

#các kind: scatter, reg, resid, kde, hex

#make pairplot based on customers
sns.pairplot(data=customers)

#make linear model plot
sns.lmplot(x="Length of Membership", y = "Yearly Amount Spent", data=customers, size=10)

#train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

#training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
pd.DataFrame(lm.coef_,X_train.columns,columns=["Coeff"])

#predicting test data
predictions = lm.predict(X_test)
sns.regplot(x=y_test,y=predictions)

#evaluating the model
from sklearn import metrics
sns.distplot((y_test - predictions), bins = 50)
pd.DataFrame(lm.coef_, X.columns, columns= ["Coeffecient"])