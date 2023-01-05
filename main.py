import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import  mean_squared_error
import seaborn as sns


zoo_data = pd.read_csv(r"C:\Users\Personal\Desktop\New folder\zoo.csv")
# print(zoo_data)
# print(zoo_data.shape)

labels = zoo_data['class_type']
#print(labels)

print(np.unique(labels.values))

# fig , ax = plt.subplots()
# (labels.value_counts()).plot(ax=ax, kind='bar')
# plt.show()

features = zoo_data.drop(['class_type', 'animal_name'], axis=1)
print(features)

model = AgglomerativeClustering(n_clusters=7, linkage='average')
model.fit(features)
print(model)
newlabels = model.labels_
print(np.unique(newlabels))
labels = labels-1
score = mean_squared_error(labels, newlabels)
res = np.sqrt(score)
print("error is :", res)

zoo_data['cluster'] = newlabels
print(zoo_data)

# frame0 = zoo_data[zoo_data.cluster == 0]
# frame1 = zoo_data[zoo_data.cluster == 1]
# frame2 = zoo_data[zoo_data.cluster == 2]
# frame3= zoo_data[zoo_data.cluster == 3]
# frame4 = zoo_data[zoo_data.cluster == 4]
# frame5 = zoo_data[zoo_data.cluster == 5]
# frame6 = zoo_data[zoo_data.cluster == 6]
#
# plt.scatter(frame0['backbone'], frame0['class_type'], color='red')
# plt.scatter(frame1['backbone'], frame1['class_type'], color='blue')
# plt.scatter(frame2['backbone'], frame2['class_type'], color='green')
# plt.scatter(frame3['backbone'], frame3['class_type'], color='yellow')
# plt.scatter(frame3['backbone'], frame3['class_type'], color='black')
# plt.scatter(frame3['backbone'], frame3['class_type'], color='purple')
# plt.scatter(frame3['backbone'], frame3['class_type'], color='orange')
# plt.xlabel('backbone')
# plt.ylabel('classtype')
# plt.legend()
# plt.show()


# sns.set_style('whitegrid')
# sns.lmplot(data=zoo_data, x= "class_type", y='eggs',  hue='cluster', palette = 'coolwarm', fit_reg=False)
# plt.show()


fig , ax = plt.subplots()
(zoo_data['cluster'].value_counts()).plot(ax=ax, kind='bar')
plt.show()