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


#We settled on 8 because of the number of k vs Error Rate graphs a couple cells lower
#knn does not use gridsearch CV as nearest neighbors is its only hyperparameter and the graphs decide for us
knn = KNeighborsClassifier(n_neighbors = 8)
knnPCA = KNeighborsClassifier(n_neighbors = 8)

knn.fit(X_train, y_train)
knnPCA.fit(X_train_pca, y_train)

print(accuracy_score(y_test, knn.predict(X_test)))
print(accuracy_score(y_train, knn.predict(X_train)))
print(accuracy_score(y_test, knnPCA.predict(X_test_pca)))
print(accuracy_score(y_train, knnPCA.predict(X_train_pca)))

print("Scores for k-nearest neighbors that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, knn.predict(X_test)))
print(balanced_accuracy_score(y_train, knn.predict(X_train)))
print("precision scores")
print(precision_score(y_test, knn.predict(X_test), average = 'weighted'))
print(precision_score(y_train, knn.predict(X_train), average ='weighted'))
print("recall scores")
print(recall_score(y_test, knn.predict(X_test), average = 'weighted'))
print(recall_score(y_train, knn.predict(X_train), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, knn.predict(X_test), average = 'weighted'))
print(f1_score(y_train, knn.predict(X_train), average ='weighted'))



print("Scores for k-nearest neighbors that uses PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, knnPCA.predict(X_test_pca)))
print(balanced_accuracy_score(y_train, knnPCA.predict(X_train_pca)))
print("precision scores")
print(precision_score(y_test, knnPCA.predict(X_test_pca), average = 'weighted'))
print(precision_score(y_train, knnPCA.predict(X_train_pca),average = 'weighted'))
print("recall scores")
print(recall_score(y_test, knnPCA.predict(X_test_pca), average = 'weighted'))
print(recall_score(y_train, knnPCA.predict(X_train_pca),average = 'weighted'))
print("f1 scores")
print(f1_score(y_test, knnPCA.predict(X_test_pca), average = 'weighted'))
print(f1_score(y_train, knnPCA.predict(X_train_pca),average = 'weighted'))


error_rate = []  #error rate list
#complete this. Use a for loop of range(1,40) for n_neighbors = i in KNeighborsClassifier classifier
for i in range(1,40):
  knn2 = KNeighborsClassifier(n_neighbors=i, 
                           p=2, 
                           metric='minkowski')
  knn2.fit(X_train_pca, y_train)
  error_rate.append(1-accuracy_score(y_test, knn2.predict(X_test_pca)))



plt.figure(figsize=(15,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o', markerfacecolor='red', markersize='10')
plt.xlabel('no. of K')
plt.ylabel('Error Rate')
#label the y-axis as "Error Rate"



error_rate = []  #error rate list
#complete this. Use a for loop of range(1,40) for n_neighbors = i in KNeighborsClassifier classifier
for i in range(1,40):
  knn2 = KNeighborsClassifier(n_neighbors=i, 
                           p=2, 
                           metric='minkowski')
  knn2.fit(X_train, y_train)
  error_rate.append(1-accuracy_score(y_test, knn2.predict(X_test)))
  

plt.figure(figsize=(15,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o', markerfacecolor='red', markersize='10')
plt.xlabel('no. of K')
plt.ylabel('Error Rate')
#label the y-axis as "Error Rate"




nn100 = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', max_iter=100, verbose=True, validation_fraction=0.3)
#Neural network with 1 hidden layer and 100 nodes on all data
nn100.fit(X_train, y_train)


nn100PCA = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', max_iter=500, verbose=True, validation_fraction=0.3)
#Neural network with 1 hidden layer and 100 nodes on PCA optimized data
nn100PCA.fit(X_train_pca, y_train)


import matplotlib.pyplot as plt
#loss curve for neural network using 1 hidden layer, 100 nodes, and all data
plt.plot(range(nn100.n_iter_), nn100.loss_curve_)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()


y_test_pred = nn100.predict(X_test) 
y_train_pred = nn100.predict(X_train) 
from sklearn.metrics import accuracy_score
accTest = accuracy_score(y_test, y_test_pred)
print('Test accuracy: %.2f%%' % (accTest * 100))
accTrain = accuracy_score(y_train, y_train_pred)
print('Train accuracy: %.2f%%' % (accTrain * 100))


print("Scores for nn100 that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, y_test_pred))
print(balanced_accuracy_score(y_train, y_train_pred))
print("precision scores")
print(precision_score(y_test, y_test_pred, average = 'weighted'))
print(precision_score(y_train, y_train_pred, average ='weighted'))
print("recall scores")
print(recall_score(y_test, y_test_pred, average = 'weighted'))
print(recall_score(y_train, y_train_pred, average ='weighted'))
print("f1 scores")
print(f1_score(y_test, y_test_pred, average = 'weighted'))
print(f1_score(y_train, y_train_pred, average ='weighted'))


y_test_pred = nn100PCA.predict(X_test_pca) 
y_train_pred = nn100PCA.predict(X_train_pca) 
from sklearn.metrics import accuracy_score
accTest = accuracy_score(y_test, y_test_pred)
print('Test accuracy: %.2f%%' % (accTest * 100))
accTrain = accuracy_score(y_train, y_train_pred)
print('Train accuracy: %.2f%%' % (accTrain * 100))


print("Scores for nn100 that uses PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, y_test_pred))
print(balanced_accuracy_score(y_train, y_train_pred))
print("precision scores")
print(precision_score(y_test, y_test_pred, average = 'weighted'))
print(precision_score(y_train, y_train_pred, average ='weighted'))
print("recall scores")
print(recall_score(y_test, y_test_pred, average = 'weighted'))
print(recall_score(y_train, y_train_pred, average ='weighted'))
print("f1 scores")
print(f1_score(y_test, y_test_pred, average = 'weighted'))
print(f1_score(y_train, y_train_pred, average ='weighted'))


nn5050 = MLPClassifier(hidden_layer_sizes=(50,50), activation='relu', solver='adam', max_iter=100, verbose=True, validation_fraction=0.3)
#Neural network with 2 hidden layers and 50 nodes on all data
nn5050.fit(X_train, y_train)



import matplotlib.pyplot as plt
#loss curve for neural network using 2 hidden layer, 50 nodes, and all data
plt.plot(range(nn5050.n_iter_), nn5050.loss_curve_)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()





y_test_pred = nn5050.predict(X_test) 
y_train_pred = nn5050.predict(X_train) 
from sklearn.metrics import accuracy_score
accTest = accuracy_score(y_test, y_test_pred)
print('Test accuracy: %.2f%%' % (accTest * 100))
accTrain = accuracy_score(y_train, y_train_pred)
print('Train accuracy: %.2f%%' % (accTrain * 100))





print("Scores for nn5050 that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, y_test_pred))
print(balanced_accuracy_score(y_train, y_train_pred))
print("precision scores")
print(precision_score(y_test, y_test_pred, average = 'weighted'))
print(precision_score(y_train, y_train_pred, average ='weighted'))
print("recall scores")
print(recall_score(y_test, y_test_pred, average = 'weighted'))
print(recall_score(y_train, y_train_pred, average ='weighted'))
print("f1 scores")
print(f1_score(y_test, y_test_pred, average = 'weighted'))
print(f1_score(y_train, y_train_pred, average ='weighted'))




params = {'criterion': ["gini", "entropy", "log_loss"], 'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10]} # create list of params for grid search cv
tree1 = GridSearchCV(DecisionTreeClassifier(), params, cv=5) # do 5 fold CV
#grid-search CV provides us with optimal hyperparameters

tree1.fit(X_train, y_train) #fit the model from grid-search CV
print(tree1.best_estimator_)
print(tree1.best_params_)
print(tree1.best_score_)

print(accuracy_score(y_train, tree1.predict(X_train)))
print(accuracy_score(y_test, tree1.predict(X_test)))





params = {'criterion': ["gini", "entropy", "log_loss"], 'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10]} # create list of params for grid search cv
treePCA = GridSearchCV(DecisionTreeClassifier(), params, cv=5) # do 5 fold CV
#grid-search CV provides us with optimal hyperparameters
treePCA.fit(X_train_pca, y_train) #fit the model from grid-search CV
print(treePCA.best_estimator_)
print(treePCA.best_params_)
print(treePCA.best_score_)
print(accuracy_score(y_train, treePCA.predict(X_train_pca)))
print(accuracy_score(y_test, treePCA.predict(X_test_pca)))





params = {'criterion': ["gini", "entropy", "log_loss"], 'n_estimators': [40,50,60,70]} # create list of params for grid search cv
forest1 = GridSearchCV(RandomForestClassifier(), params, cv=5) # do 5 fold CV
#grid-search CV provides us with optimal hyperparameters
forest1.fit(X_train, y_train) #fit the model from grid-search CV
print(forest1.best_estimator_)
print(forest1.best_params_)
print(forest1.best_score_)
print(accuracy_score(y_train, forest1.predict(X_train)))
print(accuracy_score(y_test, forest1.predict(X_test)))





params = {'criterion': ["gini", "entropy", "log_loss"], 'n_estimators': [40,50,60,70]} # create list of params for grid search cv
forestPCA = GridSearchCV(RandomForestClassifier(), params, cv=5) # do 5 fold CV
#grid-search CV provides us with optimal hyperparameters
forestPCA.fit(X_train_pca, y_train) #fit the model from grid-search CV
print(forestPCA.best_estimator_)
print(forestPCA.best_params_)
print(forestPCA.best_score_)
print(accuracy_score(y_train, forestPCA.predict(X_train_pca)))
print(accuracy_score(y_test, forestPCA.predict(X_test_pca)))




print("train/test accuracy of tree model using PCA")
print(accuracy_score(y_train, treePCA.predict(X_train_pca)))
print(accuracy_score(y_test, treePCA.predict(X_test_pca)))

print("train/test accuracy of tree model on all data")
print(accuracy_score(y_train, tree1.predict(X_train)))
print(accuracy_score(y_test, tree1.predict(X_test)))


print("train/test accuracy of forest model using PCA")
print(accuracy_score(y_train, forestPCA.predict(X_train_pca)))
print(accuracy_score(y_test, forestPCA.predict(X_test_pca)))

print("train/test accuracy of forest model on all data")
print(accuracy_score(y_train, forest1.predict(X_train)))
print(accuracy_score(y_test, forest1.predict(X_test)))



print("Scores for tree model that uses PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, treePCA.predict(X_test_pca)))
print(balanced_accuracy_score(y_train, treePCA.predict(X_train_pca)))
print("precision scores")
print(precision_score(y_test, treePCA.predict(X_test_pca), average = 'weighted'))
print(precision_score(y_train, treePCA.predict(X_train_pca), average ='weighted'))
print("recall scores")
print(recall_score(y_test, treePCA.predict(X_test_pca), average = 'weighted'))
print(recall_score(y_train, treePCA.predict(X_train_pca), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, treePCA.predict(X_test_pca), average = 'weighted'))
print(f1_score(y_train, treePCA.predict(X_train_pca), average ='weighted'))



print("Scores for tree model that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, tree1.predict(X_test)))
print(balanced_accuracy_score(y_train, tree1.predict(X_train)))
print("precision scores")
print(precision_score(y_test, tree1.predict(X_test), average = 'weighted'))
print(precision_score(y_train, tree1.predict(X_train), average ='weighted'))
print("recall scores")
print(recall_score(y_test, tree1.predict(X_test), average = 'weighted'))
print(recall_score(y_train, tree1.predict(X_train), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, tree1.predict(X_test), average = 'weighted'))
print(f1_score(y_train, tree1.predict(X_train), average ='weighted'))


print("Scores for forest model that uses PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, forestPCA.predict(X_test_pca)))
print(balanced_accuracy_score(y_train, forestPCA.predict(X_train_pca)))
print("precision scores")
print(precision_score(y_test, forestPCA.predict(X_test_pca), average = 'weighted'))
print(precision_score(y_train, forestPCA.predict(X_train_pca), average ='weighted'))
print("recall scores")
print(recall_score(y_test, forestPCA.predict(X_test_pca), average = 'weighted'))
print(recall_score(y_train, forestPCA.predict(X_train_pca), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, forestPCA.predict(X_test_pca), average = 'weighted'))
print(f1_score(y_train, forestPCA.predict(X_train_pca), average ='weighted'))


print("Scores for forest model that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, forest1.predict(X_test)))
print(balanced_accuracy_score(y_train, forest1.predict(X_train)))
print("precision scores")
print(precision_score(y_test, forest1.predict(X_test), average = 'weighted'))
print(precision_score(y_train, forest1.predict(X_train), average ='weighted'))
print("recall scores")
print(recall_score(y_test, forest1.predict(X_test), average = 'weighted'))
print(recall_score(y_train, forest1.predict(X_train), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, forest1.predict(X_test), average = 'weighted'))
print(f1_score(y_train, forest1.predict(X_train), average ='weighted'))










