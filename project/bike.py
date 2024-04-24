#introduction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('c:\\Users\\hoan\\Downloads\\JC-201810-citibike-tripdata.csv\\JC-201810-citibike-tripdata.csv')

df.info()

df.sample()

#find top 5 most popular start station
top5_start= df['start station name'].value_counts()[:5]
print(top5_start)

#bar chart : 5 most popular start station in oct 2019
plt.bar(top5_start.index, top5_start.values)
plt.ylabel('total rentals')
plt.title('5 most popular start station in Oct, 2019')
plt.xticks(rotation= 15)
plt.show()

#find top 5 most popular end stations
top5_end = df['end station name'].value_counts()[:5]
print(top5_end)

#bar chart : 5 most popular end station in oct 2019
plt.bar(top5_end.index, top5_end.values)
plt.xticks(rotation=15)
plt.ylabel('total rentals')
plt.title('5 most popular end station in Oct. 2019')
plt.show()

#look at the gender and usertype to establish a reference point for later analysis
n = df.shape[0]
df['gender'].value_counts()/n

#bar chart: user gender
gender_list = ['Male', 'Female', 'Unspecified']
plt.bar(gender_list, df['gender'].value_counts()/n * 100, color = "base_color")
plt.ylabel('Percentage [%]')
plt.title('User gender')
plt.show()

#exclude unspecified rows and see adjusted percentage
n2 = df.query('gender !=0').shape[0]
(df['gender'].value_counts()/n2)[:2]

#bar chart : user gender
gender_list = ['Male', 'Female']
plt.bar(gender_list, (df['gender'].value_counts()/n2)[:2]* 100, color = "base_color")
plt.ylabel('Percentage [%]')
plt.title('User gender')
plt.show()

df['usertype'].value_counts()/n

#bar chart : user type
usertype_list = ['Subscriber', 'Customer'];
plt.bar(usertype_list, (df['usertype'].value_counts()/n)* 100, color = "base_color")
plt.ylabel('Percentage [%]')
plt.title('User Type')
plt.show()

df['usertype'].value_counts()

import folium

# Retrieve lat/long information
top100_list = df['start station name'].value_counts()[:100].index.values

lat_list = []
long_list = []
total_traffic = []
female_list = []
male_list = []
customer_list = []
subscriber_list = []

for i in range(100):
    df_temp = df[(df['start station name'] == top100_list[i]) | (df['end station name'] == top100_list[i])]

    total_traffic.append(len(df_temp)) # used later in multivariate exploration

    gender_specified = df_temp.query('gender != 0').shape[0] # total of number of entries with specified gender

    male_list.append(df_temp.query('gender == 1').shape[0]/gender_specified)
    female_list.append(df_temp.query('gender == 2').shape[0]/gender_specified)
    subscriber_list.append(df_temp.query('usertype == "Subscriber"').shape[0]/len(df_temp))
    customer_list.append(df_temp.query('usertype == "Customer"').shape[0]/len(df_temp))

    df_samp = df[df['start station name'] == top100_list[i]]

    lat_list.append(df_samp['start station latitude'].value_counts().index.values[0])

    long_list.append(df_samp['start station longitude'].value_counts().index.values[0])

# Create new dataframe

data = {'station_name': top100_list,
        'total_traffic': total_traffic,
        'male_percentage': male_list,
        'subscriber_percentage': subscriber_list,
        'latitude': lat_list,
        'longitude': long_list}

df_top100 = pd.DataFrame(data)
df_top100