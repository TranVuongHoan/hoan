import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

image = mpl.image.imread("c:\\Users\\hoan\\Downloads\\03.image.jpg")
plt.imshow(image)

"""image.shape
X= image.reshape(-1, 3)
kmeans= KMeans(n_clusters= 2, n_init= 10)
kmeans.fit(X)

segmented_img= kmeans.cluster_centers_[kmeans.labels_]
segmented_img= segmented_img.reshape(image.shape)

plt.imshow(segmented_img/ 255)

import cv2
cv2.imwrite("c:\Users\hoan\Downloads\03.image.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite("c:\Users\hoan\Downloads\03.image.jpg", cv2.cvtColor(segmented_img.astype("uint8"), cv2.COLOR_BGR2RGB))"""
