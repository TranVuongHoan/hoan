from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from visualize_utils import visualize_digit, visualize_multiple_digits

dataset= load_digits()
#1
print(dataset.images[0].shape)
visualize_digit(dataset.data[0], "digit.png")

#2
print(dataset.images[0].shape, dataset.target[0])
visualize_digit(dataset.data[0], "digit.png")

visualize_multiple_digits(list(dataset.images[0:10]), dataset.target[0:10], "digits.png")

#.\kmeans_digit.py

X= dataset.data

#fit: train kmeans -> centroid, label
#transfrom: infer -> label
#fit_transform: train, infer -> centroid, label
model= KMeanas(n_clusters= 10)
model.fit(X)

label= model.labels_
visualize_multiple_digits(list(X[:10]), label[:10], "test_kmean.png")
