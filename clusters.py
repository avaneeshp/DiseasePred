import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split





#creates a one hot encoding matrix called symptomarr (2d numpy array) 5362 rows long (number of data) and 131 columns wide (number of unique symptoms)
# symptomarr = np.zeros((len(labels),len(uniquesymptoms)))
# for i in range(len(symptomarr)):
#   for j in range(len(symptomlists[i])):
#     for k in range(len(uniquesymptoms)):
#       if symptomlists[i][j] == uniquesymptoms[k]:
#         symptomarr[i][k] = 1
#print(symptomarr)



uniqueprognoses=np.unique(y)#contains all unique sickness
avgFeature=np.zeros((len(uniqueprognoses),len(uniquesymptoms)))#makes a matrix with the same amt of rows as uniqueprognoses and num_symptoms columns
for i in range(len(symptomarr)):#iterate thru all rows of our one hot encoded matrix/labels (they have the same amt of rows)
  for j in range(len(uniqueprognoses)):#iterate thru all rows of uniqueprognoses/avgFeature (they have the same amt of rows)
    if(labels[i]==uniqueprognoses[j]):#if the label of a row in our symptommatrix is the same as the label of a row in our avg features
      avgFeature[j]=avgFeature[j]+symptomarr[i]#then add all entries into avgfeatures

for i in range(len(avgFeature)):
  avgFeature[i]=avgFeature[i]/labels.count(uniqueprognoses[i])#divide each row by the amt of times that disease appeared, because average


XElbow = avgFeature
distorsions = []
for k in range(1, 41):
    kmeans = KMeans(k, random_state=0, n_init="auto")
    kmeans.fit(XElbow)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(1, 41), distorsions)
plt.grid(True)
plt.title('Elbow curve')


n_clusters=4
kmeans = KMeans(n_clusters, random_state=0, n_init="auto").fit(avgFeature)
print(uniqueprognoses)
groups=kmeans.labels_
print(groups)


clusters=[]#list of lists, the index is the cluster, the data is the diseases in that cluster
for i in range(n_clusters):
  my_list=[]
  clusters.append(my_list)
for i in range(len(uniqueprognoses)):
  j=groups[i]
  clusters[j].append(uniqueprognoses[i])




y_clust=[]#contains the cluster that each label is put in
for i in range(len(labels)):#iterate thru labels
  for j in range(len(clusters)):#look for label in clusters
    for k in range(len(clusters[j])):
      if(clusters[j][k]==labels[i]):
        y_clust.append(j)#append the cluster for that row of data to y_test
        
print(accuracy_score(y_clust, kmeans.predict(X)))#test kmeans on all our data



n_components=3 #best number of pcas according to function above
pca = PCA(n_components) #call PCA class
avg_pca=pca.fit_transform(avgFeature)
print("The first",n_components,"PCAs explain",pca.explained_variance_ratio_.sum()*100,"% of the variance in the data")


from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

XElbow1 = avg_pca
distorsions = []
for k in range(1, 41):
    kmeans = KMeans(k, random_state=0, n_init="auto")
    kmeans.fit(XElbow1)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(1, 41), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.xticks(range(1,41))
plt.show()


n_clusters=4
kmeans_pca = KMeans(n_clusters, random_state=0, n_init="auto").fit(avg_pca)
groups_pca=kmeans_pca.labels_


clusters_pca=[]
for i in range(n_clusters):
  my_list=[]
  clusters_pca.append(my_list)
for i in range(len(uniqueprognoses)):
  j=groups_pca[i]
  clusters_pca[j].append(uniqueprognoses[i])
clusters_pca
#0213


y_clust_pca=[]#contains the cluster that each label is put in
for i in range(len(labels)):#iterate thru labels
  for j in range(len(clusters_pca)):#look for label in clusters
    for k in range(len(clusters_pca[j])):
      if(clusters_pca[j][k]==labels[i]):
        y_clust_pca.append(j)#append the cluster for that row of data to y_test
        
X_pca=pca.fit_transform(X)
print(accuracy_score(y_clust_pca, kmeans_pca.predict(X_pca)))#test kmeans on all our data



# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
colors=['#ff0000', '#00ff00', '#0000ff', 'black']
col=[]
for i in range(len(groups_pca)):
  col.append(colors[groups_pca[i]])


ax.scatter3D(avg_pca[:,0], avg_pca[:,1], avg_pca[:,2], c=col, marker="D")
plt.title("Average Features Plot")
 
# show plot
plt.show()





