"""import pandas as pd

s= pd.Series([1, 4, 5, 2, 5, 2])

s.name = "Grades"

print(f"Name: {s.name}, dtype: {s.dtype}, size= {s.size}")

print(s[1])"""

"""import pandas as pd

s3= pd.Series([1, 4, 5, 2, 5, 2], index = list("abcdef"))

print(s3)

print(s3.index)

print(s3["b"])

print(s3["b" : "e"])"""

"""import pandas as pd

s4 = pd.Series(["Jack", "Jones", "James"], index = [1, 2, 3])

print(s4)

print(s4.index)

print(s4.loc[3]) #so 1 la gia tri dau

print(s4.iloc[1]) #so 0 la gia tri dau"""

"""import pandas as pd

import numpy as np

df = pd.DataFrame(np.random.randn(2, 3), columns = ["First", "Second", "Third"], index = ["a", "b"])

print(df)"""

#series -> 1d

#dataframe -> 2d

#pd.series(data, index, dtype)

#loc -> lay theo index

#iloc -> lay theo number

#pd.Series(s.index, index= s.values)

#df.index

#df.columns

"""import pandas as pd

df = pd.DataFrame([[100, 1000], [200, 2000]], columns= ["Population", "Total area"], index =["Hel", "abc"])

print(df)"""

"""import pandas as pd

def power_of_series(series, k):

data= {}

for i in range(1, k+1):

data[i] = series ** i

df = pd.DataFrame(data)

return df

series= pd.Series([1, 2, 3, 4, 5])

result = power_of_series(series, 4)

print(result)"""

"""def verify_credit_card(card_number):

card_number = card_number.replace("-", "")

if len(card_number) == 16 and card_number.isdigit():

if card_number[0] in ['4', '5', '6']:

if not any(card_number[i] == card_number[i+1] == card_number[i+2] == card_number[i+3] for i in range(len(card_number)-3)):

return True

return False

def verify_credit_card(card_number):

card_number = card_number.replace("-", "")

if not card_number[0] in ['4', '5', '6']:

return False

if len(card_number) != 16:

return False

if not card_number.isdigit():

return False

if '-' in card_number:

groups = card_number.split('-')

if len(groups) != 4:

return False

for group in groups:

if len(group) != 4:

return False

for i in range(len(card_number) - 3):

if card_number[i] == card_number[i+1] == card_number[i+2] == card_number[i+3]:

return False

return True

credit_card_number = "1234-5678-9012-3456"

print(verify_credit_card(credit_card_number))"""

"""def calculate_depth_of_parentheses(string):

depths = []

count = 0

for char in string:

if char == "(":

count += 1

elif char == ")":

depths.append(count)

count -= 1

return depths"""

'''import numpy as np

array = np.array([1, 2, np.nan, 4, 5])

mean_value= np.nanmean(array)

array[np.isnan(array)] = mean_value

print(array)'''

"""import pandas as pd

df = pd.read_csv("C:/Users/thengoc/Downloads/FuelConsumption.csv")

unique_model = df['MODEL'].unique()

print(unique_model)

num_vehicle_classes = len(df["VEHICLECLASS"].unique())

print(num_vehicle_classes)

abc= df.dropna(subset = ["VEHICLECLASS"])

print(abc)

import pandas as pd

df = pd.read_excel('data.xlsx')

merged_column = []

for index, row in df.iterrows():

merged_value = str(row['Thoi_Gian'])[:5] + '_' + str(row['Ngay']) + '-' + str(row['Thang']) + '-' + str(row['Nam'])

merged_column.append(merged_value)

df['merged_column'] = merged_column

print(df)"""

#astype(str) hoac dung .join([df["d"].astype(str), df["d"].astype(str), df...])

"""import re

def verify_credit_card(card_number):

if not re.match(r'^[4-6]', card_number):

return False

if len(card_number) == 16 or len(card_number) == 19:

if '-' in card_number:

groups = card_number.split('-')

if len(groups) != 4:

return False

for group in groups:

if len(group) != 4 or not group.isdigit():

return False

card_number = card_number.replace("-", "")

if not card_number.isdigit():

return False

return True

credit_card_number = "4253-9999-5611-5786"

print(verify_credit_card(credit_card_number))"""

#plt.savefig("chart.jpg")

#plt.figure()

"""import matplotlib.pyplot as plt

def main():

x1 = [2, 4, 6, 7]

y1 = [4, 3, 5, 1]

x2 = [1, 2, 3, 4]

y2 = [4, 2, 3, 1]

plt.plot(x1, y1, label='Graph 1')

plt.plot(x2, y2, label='Graph 2')

plt.title('Two Graphs in One Axes')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.show()

if __name__ == '__main__':

main()"""

"""import matplotlib.pyplot as plt

def main():

x1= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

y1= [2, 5, 7, 4, 7, 0.2, 3, 1, 8.5, 2]

plt.plot(x1, y1, label= "abc")

plt.show()

if __name__ == '__main__':

main()"""

"""import matplotlib.pyplot as plt

x = ["orange", "Apples", "Banana"]

y = [10, 20, 30]

plt.barh(x, y)

plt.title("...")

plt.xlabel("asda")

plt.ylabel("sdaskd")

plt.show()"""

"""import matplotlib.pyplot as plt

x = ["orange", "Apples", "Banana"]

y = [10, 20, 30]

plt.bar(x, y)

plt.title("...")

plt.xlabel("asda")

plt.ylabel("sdaskd")

plt.show()"""

"""import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]

y= [2, 4, 6, 8, 10]

plt.fill_between(x, y, color= "blue", alpha= 0.5)

plt.title("hoan")

plt.xlabel("x-axis")

plt.ylabel("y-axis")

plt.show()"""

"""import matplotlib.pyplot as plt

slices= ["Apples", "Banana", "Cheese"]

values = [10, 20, 30]

plt.pie(values, labels= slices, autopct= "%1.1f%%")

plt.title("pie chart")

plt.axis("equal")

plt.show()"""

"""import matplotlib.pyplot as plt

import numpy as np

x= np.random.rand(1, 100)

y= np.random.rand(1, 100)

plt.scatter(x, y)

plt.title("scatter plot")

plt.xlabel("x-axis")

plt.ylabel("y-axis")

plt.show()"""

"""import matplotlib.pyplot as plt

import numpy as np

data= np.random.rand(1000)

plt.hist(data)

plt.title("")

plt.xlabel("")

plt.ylabel("")

plt.show()"""

"""import matplotlib.pyplot as plt

import numpy as np

data= np.random.randn(1000)

plt.boxplot(data)

plt.title("")

plt.xlabel("")

plt.ylabel("")

plt.show()"""

"""import matplotlib.pyplot as plt

import numpy as np

data= np.random.randn(10, 10)

plt.imshow(data)

plt.title("heatmap")

plt.xlabel("x-axis")

plt.ylabel("y-axis")

plt.show()"""

"""import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

data = [[1, 2, 3, 4, 5], [6, 7, 8, 9 , 10]]

sns.boxplot(data= data)

plt.title("")

plt.xlabel("")

plt.ylabel("")

plt.show()"""

"""import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

data= np.random.randn(10, 10)

sns.heatmap(data, cmap= 'hot')

plt.title("")

plt.xlabel("")

plt.ylabel("")

plt.show()"""

"""import numpy as np

x= np.arange(10, 20)

y= np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

r= np.corrcoef(x, y)

print(r[0, 1])"""

"""import numpy as np

import scipy.stats

x = np.arange(10, 20)

y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

correlation_coefficient, p_value = scipy.stats.pearsonr(x, y)

correlation_coefficient, p_value = scipy.stats.spearmanr(x, y)

correlation_coefficient, p_value = scipy.stats.kendalltau(x, y)

print("Correlation coefficient:", correlation_coefficient)

print("p-value:", p_value)"""

"""import pandas as pd

x= pd.Series(range(10, 20))

y= pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

print(x.corr(y))

print(y.corr(x))

print(x.corr(y, method ="spearman"))

print(x.corr(y, method="kendall"))"""

"""train_x, test_x, train_y, test_y = train_test_split(temp, depth, train_size=0.8)

model= LinearRegression()

model.fit(train_x, train_y)

model.fit(train_x.to_frame(), train_y)

y_pred= model.predict(test_x.to_frame())

model.coef_, model.intercept_"""

"""import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/thengoc/Downloads/netflix-rotten-tomatoes-metacritic-imdb.csv')

df.dropna(subset=['Country Availability'], inplace=True)

country_counts = df['Country Availability'].str.split(', ').explode().value_counts()

plt.bar(country_counts.index, country_counts.values)

plt.xlabel("Country Availability")

plt.ylabel('Movie Count')

plt.title('Number of Movies Available on Netflix by Country')

plt.xticks(rotation=90)

plt.show()"""