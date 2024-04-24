#introduction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('c:\\Users\\hoan\\Downloads\\movies.csv')

df.head(1)

#top 5 in top 100 most profitable film
df_top100 = df.sort_values(by=['revenue_adj'], ascending=False).head(100).reset_index()
df_top100.head(5)

#line chart: revenue of top 100 blockbusters
plt.plot(df_top100.index, df_top100['revenue_adj']/1E9)
plt.ylabel("Billion [USD]")
plt.xlabel("Rank")
plt.title('Revenue of top 100 blockbusters')
plt.show()

#extract rows concerning the specific actor
df_temp = df.dropna(subset=["cast"])
def actor_search(name):
    return df_temp[df_temp['cast'].str.contains(name)]

df_cruise = actor_search("Tom Cruise")
df_hanks = actor_search("Tom Hanks")

# not every column is useful

#extract only interested columns
interested = ['original_title','genres','revenue_adj']
df_cruise = df_cruise[interested]
df_hanks = df_hanks[interested]

#data cleaning: remove film with revenue = 0
def drop_zero_revenue_rows (df):
    return df[df.revenue_adj != 0]

df_cruise_adj = drop_zero_revenue_rows(df_cruise)
df_hanks_adj = drop_zero_revenue_rows(df_hanks)

#number of movies appeared in (cruise and hanks)
cruise_tot = df_cruise.shape[0]
hanks_tot = df_hanks.shape[0]

names = ("Cruise", "Hanks")
plt.bar(names,(cruise_tot, hanks_tot))
plt.ylabel("# of movies")
plt.xlabel('Actor name')
plt.title('Number of movies appeared in')
plt.show()

print("Tom Cruise has appeared in "+ str(cruise_tot) + " movies.")
print("Tom Hanks has appeared in "+ str(hanks_tot) + " movies.")
print("Tom Hanks has appeared in "+ str(hanks_tot - cruise_tot) + " more movies than Tom Cruise.")

#who more romantic tom (most number/ percentage of films with romantic tags)
def romance_search(df):
    return df[df['genres'].str.contains('Romance')]

romance_search(df_hanks)

#bar chart : number of romantic movies appeared in
cruise_tot_rom = romance_search(df_cruise).shape[0]
hanks_tot_rom = romance_search(df_hanks).shape[0]
cruise_rom_pro = cruise_tot_rom/cruise_tot
hanks_rom_pro = hanks_tot_rom/hanks_tot

plt.bar(names,(cruise_tot_rom, hanks_tot_rom))
plt.title('Number of romantic movies appeared in')
plt.ylabel('# of movies')
plt.xlabel('Actor name')
plt.show()

print("Tom Cruise has appeared in "+ str(cruise_tot_rom) + " romantic movies.")
print("Tom Hanks has appeared in "+ str(hanks_tot_rom) + " romantic movies.")

#bar chart: percentage of romantic movies appeared in
plt.bar(names,(cruise_rom_pro, hanks_rom_pro))
plt.title('Percentage of romantic movies appeared in')
plt.ylabel('% of movies')
plt.xlabel('Actor name')
plt.show()

print(str(round(cruise_rom_pro,2)) + "% of the movie Tom Cruise appeared in are romantic movies.")
print(str(round(hanks_rom_pro,2)) + "% of the movie Tom hanks appeared in are romantic movies")

# average revenue of movies appeared in
hanks_rev = df_hanks_adj['revenue_adj'].mean()
cruise_rev = df_cruise_adj['revenue_adj'].mean()

plt.bar(names,(cruise_rev/1000000, hanks_rev/1000000))
plt.ylabel("Million USD")
plt.xlabel('Actor name')
plt.title('Average revenue of movies appeared in')
plt.show()