import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.metrics import  mean_squared_error
import seaborn as sns


mov_data_1 = pd.read_csv(r"C:\Users\Personal\Desktop\New folder\movie_metadata.csv")
# print(mov_data_1.head())
print(mov_data_1.info())

mov_data = mov_data_1.iloc[:,4:6]
#print(mov_data)
mov_data = mov_data.dropna()
#print(mov_data)

plt.scatter(mov_data['director_facebook_likes'], mov_data['actor_3_facebook_likes'])
plt.xlabel('Director')
plt.ylabel('Actor')
plt.show()

model = KMeans(n_clusters=5)
print(model)

model = model.fit(mov_data)