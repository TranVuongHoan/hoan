#introduction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Get the Data
import csv

# Open the text file in read mode
with open('c:\\Users\\hoan\\Downloads\\KNN_Project_Data.txt', 'r') as infile:
    # Create a CSV file in write mode
    with open('output.csv', 'w', newline='') as outfile:
        # Create a CSV writer object
        writer = csv.writer(outfile)

        # Write the header row to the CSV file
        writer.writerow(["XVPM","GWYH","TRAT","TLLZ","IGGA","HYKR","EDFS","GUUB","MGJM","JHZC","TARGET CLASS"])

        # Loop through each line in the text file
        for line in infile:
            # Split the line into fields using the comma character as the delimiter
            fields = line.strip().split(',')

            # Write the fields to the CSV file
            writer.writerow(fields)
knn_df = pd.read_csv("c:\\Users\\hoan\\Downloads\\KNN_Project_Data.txt")

#Check the head of the dataframe.
print(knn_df.head())

#EDA
sns.pairplot(data=knn_df, hue= "TARGET CLASS")

#Standardize the Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
knn_df_notc = knn_df.drop('TARGET CLASS', axis=1)
scaler.fit(knn_df_notc)
scaled_features = scaler.transform(knn_df_notc)
df_scale = pd.DataFrame(scaled_features, columns=knn_df.columns[:-1])
df_scale.head()

#Train Test Split
from sklearn.model_selection import train_test_split
X = df_scale
y = knn_df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 101)

#Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#Predictions and Evaluations
predictions = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Choosing a K Value
error_rate = []
for i in range(1,40):
    knn_loop = KNeighborsClassifier(n_neighbors=i)
    knn_loop.fit(X_train,y_train)
    pred_i = knn_loop.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1,40), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red')

#Retrain with new K Value
knn_new = KNeighborsClassifier(n_neighbors=31)
knn_new.fit(X_train, y_train)
pred_new = knn_new.predict(X_test)

print(confusion_matrix(y_test,pred_new))
print('/n')
print(classification_report(y_test,pred_new))