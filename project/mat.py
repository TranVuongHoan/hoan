import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

csvpath_city = os.path.join('raw_data','c:\Users\hoan\Downloads\city_data.csv')
pyber_city_df = pd.read_csv(csvpath_city)
csvpath_ride = os.path.join('raw_data','c:\Users\hoan\Downloads\ride_data.csv')
pyber_ride_df = pd.read_csv(csvpath_ride)

pyber_city_df.head()
pyber_city_df['type'].value_counts()
pyber_gcity = pyber_ride_df.groupby('city')

pyber_gcity.head()
city_fare = pyber_gcity['fare'].sum()
city_count = pyber_gcity['city'].count()

avg_city = pyber_gcity['fare'].sum()/pyber_gcity['city'].count()