import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.metrics import  mean_squared_error
import seaborn as sns


sal_data = pd.read_csv(r"C:\Users\Personal\Desktop\New folder\Salary_age_exp.csv")
print(sal_data.head())

# plt.scatter(sal_data['Age'],sal_data['Salary'])
# plt.xlabel('AGE')
# plt.ylabel('SALARY IN $')
# plt.show()

model = KMeans(n_clusters=4)
predict = model.fit_predict(sal_data[['Age', 'Salary']])
#print(predict)

sal_data['myclusters'] = predict
#print(sal_data.tail())

df0 = sal_data[sal_data['myclusters'] == 0]
print(df0)
df1 = sal_data[sal_data['myclusters'] == 1]
df2 = sal_data[sal_data['myclusters'] == 2]
df3 = sal_data[sal_data['myclusters'] == 3]

plt.scatter(df0['Age'], df0['Salary'], color = 'red')
plt.scatter(df1['Age'], df1['Salary'], color = 'blue')
plt.scatter(df2['Age'], df2['Salary'], color = 'green')
plt.scatter(df3['Age'], df3['Salary'], color = 'yellow')

plt.xlabel('AGE')
plt.ylabel('SALARY IN $')

plt.show()




print(model.cluster_centers_)

#SSE

sse = []
k_range = range(1,10)

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(sal_data[['Age','Salary']])
    sse.append(km.inertia_)

print(sse)

plt.scatter(k_range, sse )
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()