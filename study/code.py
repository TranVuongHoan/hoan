from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
X, y = make_blobs(n_samples=500, n_features=2, cluster_std=0.7, centers=3)

#chia tập dữ liệu X thành hai phần: tập huấn luyện (X_train) và tập kiểm tra (X_test), tập huấn luyện sẽ chiếm 80% dữ liệu ban đầu.
X_train, X_test= train_test_split(X, train_size=0.8)

# phân thành 2 cụm
model=KMeans(n_clusters=2)

#xác định trung tâm cụm
model.fit(X_train)
import matplotlib.pyplot as plt

#truy cập các trung tâm cụm tìm được từ quá trình huấn luyện.
model.cluster_centers_

#truy cập các nhãn của mỗi điểm dữ liệu trong tập huấn luyện.
model.labels_
model_inertia_

#dự đoán nhãn của các điểm dữ liệu trong tập kiểm tra X_test
y_pred= model.predict(X_test)
plt.scatter(X[:, 0], X[:, 1] , c="y")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c=["g", "b"])
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred+2)
plt.show()

sil_list=[]
sse_list=[]
for k in range(2, 10):

#khởi tạo đối tượng kmeans với k cụm
model= KMeans(n_clusters=k)

#Huấn luyện mô hình KMeans trên dữ liệu huấn luyện X_train.
model.fit(X_train)

#Thêm giá trị SSE của mô hình vào danh sách SSE. SSE là tổng các bình phương khoảng cách từ các điểm dữ liệu tới trung tâm của cụm gần nhất.
see_list.append(model.inertia_)

y_pred= model.predict(X_test)

# Tính giá trị Silhouette Score cho dữ liệu kiểm tra( đo lường độ tương đồng trong cụm so với các cụm khác)
sil= silhouette_score(X_test, y_pred)

#Thêm giá trị Silhouette Score vào danh sách Silhouette Score
sil_list.append(sil)

#Tạo một subplot trong hình vẽ, có hai cột và một hàng, và đặt trục hiện tại là trục thứ nhất.
ax=plt.subplot(1,2,1)

#Vẽ đường cong SSE trên subplot hiện tại, trong đó trục x là số lượng cụm và trục y là giá trị SSE tương ứng.
plt.plot(range(2, 10), sse_list)
ax.set_ylabel()
ax.set_xlabel()
ax= plt.subplot(1, 2, 1)

#Vẽ đường cong Silhouette Score trên subplot hiện tại, trong đó trục x là số lượng cụm và trục y là giá trị Silhouette Score tương ứng.
plt.plot(range(2,10), sil_list)
ax.set_xlabel()
ax.set_ylabel()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data= pd.read_csv("C:/Users/thengoc/Downloads/advertising.csv")

#inplace= True -> thay đổi DataFrame gốc
data.dropna(inplace= True)
x= data[["TV", "Newspaper", "Radio"]]
y= data["Sales"]
print(abc.corr())

#Xây dựng mô hình hồi quy tuyến tính
from sklearn.linear_model import LinearRegression

#chia dữ liệu thành các tập huấn luyện và tập kiểm tra để đánh giá mô hình.
from sklearn.model_selection import train_test_split

#đánh giá hiệu suất của mô hình hồi quy.
from sklearn.metrics import mean_squared_error, r2_score

#tạo một đối tượng mô hình hồi quy tuyến tính từ lớp LinearRegression
model= LinearRegression()

#Mô hình sẽ học các hệ số để dự đoán y dựa trên các giá trị trong x.
model.fit(x, y)

# lấy các hệ số hồi quy từ mô hình đã được huấn luyện và lưu chúng vào biến coefficient
coefficient= model.coef_

# dự đoán giá trị y dựa trên dữ liệu x sử dụng mô hình đã được huấn luyện.
predicted_values = model.predict(x)
plt.scatter(y, predicted_values)
plt.show()

#model.coef_ la a
#model.intercept_ la b
#metrics: re_score / mse

#bc 1: chon k -> bc2 : chon random k điểm từ dataset -> bc3: phân nhóm -> bc4: tính lại centroid
#Hàm này được sử dụng để tạo dữ liệu mô phỏng theo các cụm
from sklearn.datasets._samples_generator import make_blobs

#đánh giá chất lượng phân cụm (clustering) bằng cách tính điểm silhouette.
from sklearn.metrics import silhouette_score

#chia dữ liệu thành các tập huấn luyện và tập kiểm tra.
from sklearn.model_selection import train_test_split

#import lớp KMeans từ module sklearn.cluster.
from sklearn.cluster import KMeans

#500 mẫu, số lượng : 2, độ lệch chuẩn: 0.7, số cụm: 3
X, _ = make_blobs(n_samples=500, n_features=2, cluster_std=0.7, centers=3)
plt.scatter(X[:, 0], X[:, 1] , c="y")
plt.show()


import pandas as pd
data= pd.read_csv("C:/Users/thengoc/Downloads/FuelConsumption.csv")
data.dropna(inplace= True)
abc= data.groupby("MAKE")[['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']].mean()
correlation =abc.corr()
import matplotlib.pyplot as plt
plt.scatter(abc["FUELCONSUMPTION_CITY"], abc["FUELCONSUMPTION_HWY"])
plt.xlabel("fuelconsumption_city")
plt.ylabel("fuelconsumption_hwy")
plt.title("fuelconsumption_city vs fuelconsumption_hwy")
plt.scatter(abc["FUELCONSUMPTION_CITY"],
abc["FUELCONSUMPTION_COMB"])
plt.xlabel("fuelconsumption_city")
plt.ylabel("fuelconsumption_comb")
plt.title("fuelconsumption_city vs fuelconsumption_comb")
plt.scatter(abc["FUELCONSUMPTION_HWY"], abc["FUELCONSUMPTION_COMB"])
plt.xlabel("fuelconsumption_hwy")
plt.ylabel("fuelconsumption_comb")
plt.title("fuelconsumption_hwy vs fuelconsumption_comb")
plt.show()
comb= data["FUELCONSUMPTION_COMB"]

city= data["FUELCONSUMPTION_CITY"]

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

#chia tập dữ liệu comb và city thành hai phần: tập huấn luyện (train_x, train_y) và tập kiểm tra (test_x, test_y). Tập huấn luyện chiếm 80% dữ liệu ban đầu.

train_x, test_x, train_y, test_y = train_test_split(comb, city, train_size=0.8)

model= LinearRegression()

#train_x được chuyển đổi thành DataFrame bằng to_frame(), train_y là giá trị mục tiêu tương ứng với train_x.

model.fit(train_x.to_frame(), train_y)

#dự đoán giá trị đầu ra (y) cho tập kiểm tra test_x

y_pred= model.predict(test_x.to_frame())

print(y_pred)

y_pred=model.coef_ + test_x.to_frame() + model.intercept_

print(y_pred)

sns.lmplot(data= abc, x = "FUELCONSUMPTION_CITY", y= "FUELCONSUMPTION_COMB")

sample_columns= abc.sample(2, axis=1)

sns.scatterplot(data= sample_columns)

plt.show()

#hang la sample, cot la feature

#corr chay tu -1 den 1 , cang ve 0 cang k lquan

#y=ax + b x: 1 cot ~ series (nhieu cot) y: 1 cot

#crawl data -> clean data -> modelize (du doan) (chon features, model, train, test) -> deploy

#from sklearn.linear_model import LinearRegression

#from sklearn.model_selection import train_test_split

#from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

data= pd.read_csv("C:/Users/thengoc/Downloads/weather.csv")

data[["Precipitation amount (mm)", "Snow depth (cm)", "Air temperature (degC)"]].corr()

data.dropna(inplace= True)

temp = data["Air temperature (degC)"]

depth= data["Snow depth (cm)"]

train_x, test_x, train_y, test_y = train_test_split(temp, depth, train_size=0.8)

model= LinearRegression()

model.fit(train_x.to_frame(), train_y)

y_pred= model.predict(test_x.to_frame())

print(y_pred)

y_pred=model.coef_ + test_x.to_frame() + model.intercept_

print(y_pred)

# Tính giá trị Mean Squared Error (MSE) giữa giá trị dự đoán 'y_pred' và giá trị thực tế 'test_y' bằng hàm 'mean_squared_error' từ thư viện sci-kit learn.

mse = mean_squared_error(y_pred, test_y)

# Tính giá trị R-squared (R2) giữa 'test_y' và 'y_pred' bằng hàm 'r2_score' từ thư viện sci-kit learn.

r2= r2_score(test_y, y_pred)

import matplotlib.pyplot as plt

plt.scatter(train_x, train_y, c= "b")

plt.scatter(test_x, test_y, c ="y")

plt.plot(test_x, y_pred, c= "r")

plt.show()

#####################################

#3.1

import pandas as pd

data= pd.read_csv("C:/Users/thengoc/Downloads/automobileEDA.csv")

data["stroke"]= data["stroke"].replace("", np.nan)

data["horsepower-binned"]= data["horsepower-binned"].replace("", np.nan)

missing_count= data.isnull().sum()

print(missing_count)

#3.2 a

import pandas as pd

data= pd.read_csv("C:/Users/thengoc/Downloads/automobileEDA.csv")

original_rows= data.shape[0]

drop_rows= data.dropna()

num_rows_after_drop= original_rows - drop_rows.shape[0]

print(num_rows_after_drop)

#3.2 b

import pandas as pd

data= pd.read_csv("C:/Users/thengoc/Downloads/automobileEDA.csv")

original_columns= data.shape[1]

drop_columns= data.dropna()

num_columns_after_drop= original_columns - drop_columns.shape[1]

print(num_columns_after_drop)

#3.3 a

#3.4

import numpy as np

import pandas as pd

import seaborn as sns

data= pd.read_csv("C:/Users/thengoc/Downloads/automobileEDA.csv")

data.dropna(inplace= True)

x= data["price"]

y= data["highway-mpg"]

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

model= LinearRegression()

model.fit(train_x.to_frame(), train_y)

y_pred= model.predict(test_x.to_frame())

print(y_pred)

y_pred=model.coef_ + test_x.to_frame() + model.intercept_

print(y_pred)

sns.lmplot(data= data, x = "price", y="highway-mpg")

sample_columns= data.sample(2, axis=1)

sns.scatterplot(data= sample_columns)

plt.show()

#4

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

data= pd.read_csv("C:/Users/thengoc/Downloads/Mall_Customers.csv")

data.dropna(inplace= True)

from sklearn.datasets._samples_generator import make_blobs

from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, n_features=2, cluster_std=0.7, centers=5)

X_train, X_test= train_test_split(X, train_size=0.8)

model=KMeans(n_clusters=5)

model.fit(X_train)

model.cluster_centers_

model.labels_

y_pred= model.predict(X_test)

plt.scatter(X[:, 0], X[:, 1] , c="y")

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c=["g", "b", "y", "r", "w"])

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred+2)

plt.show()

sil_list=[]

sse_list=[]

for k in range(2, 10):
    model= KMeans(n_clusters=k)
    model.fit(X_train)

sse_list.append(model_inertia_)

y_pred= model.predict(X_test)

sil= silhouette_score(X_test, y_pred)

sil_list.append(sil)

ax=plt.subplot(1,2,1)

plt.plot(range(2, 10), sse_list)

ax.set_ylabel()

ax.set_xlabel()

ax= plt.subplot(1, 2, 1)

plt.plot(range(2,10), sil_list)

ax.set_xlabel()

ax.set_ylabel()

ax= plt.subplot(1, 2, 1)

plt.plot(range(2,10), sil_list)

ax.set_xlabel()

ax.set_ylabel()

plt.show()

#2 a

import numpy as np

random_array= np.random.random((5, 10))

random_array[:, [2,5]]= random_array[:, [5, 2]]

print(random_array)

# 2 b

import numpy as np

random_array = np.random.random((5, 10))

max_value = np.max(random_array, axis=0)

normalized_array = random_array / max_value

print(normalized_array)

#1

test_input = """4

bcdef

abcdefg

bcde

bcdef

"""

input_lines = test_input.strip().split('\n')

n = int(input_lines[0])

words = input_lines[1:]

distinct_count, occurrences = word_occurrences(n, words)

print(distinct_count)

print(" ".join(map(str, occurrences)))

######################################################################

#NETFLIX

#most common genre

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

genre_counts = data['Genre'].str.split(', ').explode().value_counts()

top_5_genres= genre_counts.head(5)

plt.bar(top_5_genres.index, top_5_genres.values)

plt.xlabel("Genres")

plt.ylabel("Number of Occurences")

plt.title("Top 5 most common genre")

plt.show()

#distribution of movies by languages

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: country_counts = data['Languages'].str.split(', ').explode().value_counts()

...: top_5_country = country_counts.head(10)

...: plt.bar(top_5_country.index, top_5_country.values)

...: plt.xlabel("Country")

...: plt.ylabel("Number of Films")

...: plt.tight_layout()

...: plt.xticks(rotation=45)

...: plt.show()

#% movies is series/movies

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-r

...: otten-tomatoes-metacritic-imdb.csv")

...: movies_count= data[data["Series or Movie"] == "Movie"].

...: shape[0]

...: series_count= data[data["Series or Movie"] == "Series"]

...: .shape[0]

...: total_count = movies_count + series_count

...: movies_percentage= (movies_count / total_count) * 100

...: series_percentage= (series_count / total_count) * 100

...: labels= ["Movies", "Series"]

...: sizes= [movies_percentage, series_percentage]

...: plt.pie(sizes, labels=labels, autopct='%1.1f%%')

...: plt.title("Percentage of movies ( movies or series)")

...: plt.axis("equal")

...: plt.show()

# average imdb score

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data.dropna(subset=['IMDb Score'], inplace=True)

...: average_score= data["IMDb Score"].mean()

...: print(average_score)

# average rotten tomatoes score

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data.dropna(subset=['Rotten Tomatoes Score'], inplace=True)

...: average_score= data["Rotten Tomatoes Score"].mean()

...: print(average_score)

# average metacritic score

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data.dropna(subset=['Metacritic Score'], inplace=True)

...: average_score= data["Metacritic Score"].mean()

...: print(average_score)

56.813653136531364

# average hidden gem score

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data.dropna(subset=['Hidden Gem Score'], inplace=True)

...: average_score= data["Hidden Gem Score"].mean()

...: print(average_score)

# runtime

import numpy as np

...: import pandas as pd

...: import matplotlib.pyplot as plt

...: import seaborn as sns

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data.dropna(subset=['Runtime'], inplace=True)

...: runtime_counts = data['Runtime'].value_counts()

...: runtime_order = ['< 30 minutes', '30-60 mins', '1-2 hour', '> 2 hrs']

...: plt.bar(runtime_order, runtime_counts[runtime_order])

...: plt.xlabel('Runtime')

...: plt.ylabel('Number of films')

...: plt.title('Number of films according to runtime')

...: plt.show()

#film according to year

import matplotlib.pyplot as plt

...: import pandas as pd

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data.dropna(subset=['Release Date'], inplace=True)

...: data['Release Date'] = pd.to_datetime(data['Release Date'], format='%d-%b-%y')

...: data["release year"] = data["Release Date"].dt.year

...: data_filter= data[data["release year"] <= 2029]

...: movies_count= data_filter["release year"].value_counts().sort_index()

...: plt.bar(movies_count.index, movies_count.values)

...: plt.title("Number of films in Netflix according to year")

...: plt.ylabel("Number of Films")

...: plt.xlabel("Year")

...: plt.show()

#NUMBER OF FILMS / COUNTRY

...: df = pd.read_csv('C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv')

...: df_country = df['Country Availability']

...: countries = df_country.str.split(',', expand=True).stack().reset_index(level=0, drop=True).str.strip()

...: country_counts = countries.value_counts()

...: fig, ax = plt.subplots(figsize=(12, 6))

...: country_counts.plot(kind='bar', ax=ax)

...: ax.set_xlabel('Quốc gia')

...: ax.set_ylabel('Số lượng')

...: ax.set_title('Số lượng quốc gia được chiếu phim')

...: plt.show()

#revenue film

import matplotlib.pyplot as plt

...: import pandas as pd

...: data= pd.read_csv("C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv")

...: data_sorted= data.sort_values('Boxoffice', ascending=False)

...: top_10_movies= data_sorted.head(5)

...: fig, ax = plt.subplots(figsize=(12, 6))

...: ax.bar(top_10_movies['Title'], top_10_movies['Boxoffice'])

...: ax.set_xlabel('Phim')

...: ax.set_ylabel('Doanh thu')

...: ax.set_title('Top 5 phim có doanh thu cao nhất')

...: plt.xticks(rotation=45)

...: ax.xaxis.set_tick_params(labelsize=8)

...: plt.show()

#FILM-RELEASE DATE

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv')

df['Year'] = df['Netflix Release Date'].str.split('/').str[-1]

movies_per_year = df['Year'].value_counts().reset_index()

movies_per_year.columns = ['Year', 'Number of Movies']

movies_per_year = movies_per_year.sort_values('Year')

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(movies_per_year['Year'], movies_per_year['Number of Movies'])

ax.set_xlabel('Năm')

ax.set_ylabel('Số lượng phim')

ax.set_title('Số lượng phim mỗi năm')

plt.show()

#PRODUCTION HOUSE

import pandas as pd

...: import matplotlib.pyplot as plt

...:

...: # Đọc dữ liệu từ file CSV vào DataFrame

...: df = pd.read_csv('C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv')

...:

...: # Tách chuỗi trong cột 'production house' thành các nhà sản xuất riêng biệt

...: production_houses = df['Production House'].str.split(',').explode().str.strip()

...:

...: # Đếm số lượng xuất hiện của từng nhà sản xuất

...: production_house_counts = production_houses.value_counts()

...:

...: # Lấy 10 nhà sản xuất xuất hiện nhiều nhất

...: top_10_production_houses = production_house_counts.head(10)

...:

...: # Tạo biểu đồ pie chart

...: fig, ax = plt.subplots()

...: ax.pie(top_10_production_houses, labels=top_10_production_houses.index, autopct='%1.1f%%')

...: ax.set_title('Phân phối nhà sản xuất')

plt.show()

f= open("C:/Users/thengoc/Downloads/Ecommerce-Customers.csv", "r")

In [3]: for i, line in enumerate(f.readlines()):

...: print(line)

...: if i == 5: break

f= open("C:/Users/thengoc/Downloads/Ecommerce-Customers.csv", "r")

...: o= open("output.csv", "w")

...: new_line= ""

...: for i, line in enumerate(f.readlines()):

...: print(line)

...: if i == 0:

...: o.write(line)

...: continue

...: if i % 2 != 0 :

...: new_line = line.strip()

...: else:

...: new_line += line

...: o.write(new_line)

...: new_line = ""

...: o.close()

import psycopg2 as pg2

...: conn= pg2.connect(databasse= 'dvdrental', user= 'postgres', password= 'password', host= 'localhost', port= '

...: cur= conn.cursor()

...: cur.execute('select * from film')

...: cur.fetchone()

import psycopg2 as pg2

...: conn= pg2.connect(database= 'dvdrental', user= 'postgres', password= 'hoan2303')

...: conn.autocommit= True

...: cur= conn.cursor()

...: cur.execute('select * from film')

...: print(cur.fetchmany(2))

...: cur.close()