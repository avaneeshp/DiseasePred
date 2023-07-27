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

# #tests many test size and PCA amount combos to figure out which one gives the best test accuracy
# maxTestAcc=0
# bestSplit=0
# bestPCA=0
# X, y = mldf.values, labels
# for i in range(3,5):
#   #split dataset into i*10% test split, tests 10%-90% test splits in increments of 10
#   for j in range(1,131):
#     #tests reasonable PCAs
#     pca = PCA(j) #call PCA class
#     X_pca=pca.fit_transform(X)
#     #split dataset into 40% test split, best test size according to function above
#     X_train_pca, X_test_pca, y_train, y_test = \
#         train_test_split(X_pca, y, test_size=i/10, 
#                         random_state=0)
#     #clf = LogisticRegression(random_state=0).fit(X_train_pca, y_train) 
#     clf = LogisticRegression()
#     clf.fit(X_train_pca, y_train)
#     currAcc=clf.score(X_test_pca,y_test)
#     if(currAcc>maxTestAcc):
#       maxTestAcc=currAcc
#       bestSplit=i
#       bestPCA=j
#     print("At a test size of",i*10,"% and the top",j,"PCAs being used, \
# \nthe model's test accuracy is",currAcc*100,"% compared to the current best, which is",maxTestAcc*100,"%")
    
# print("The best test accuracy,",maxTestAcc*100,", was obtained with a test size of",bestSplit*10,"% and using the top",bestPCA,"PCAs")
# #The best test accuracy, 73.98601398601399 %, was obtained with a test size of 40 % and using the top 10 PCAs



X, y = mldf.values, labels

#split dataset into 40% test split, best test size according to function above
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, 
                     random_state=0)
    

#best number of pcas according to function above
pca = PCA(n_components = 10) #call PCA class
X_pca=pca.fit_transform(X)
n_components=len(X_pca[0])
#split dataset into 40% test split, best test size according to function above
X_train_pca, X_test_pca, y_train_pca, y_test_pca = \
    train_test_split(X_pca, y, test_size=0.4, 
                     random_state=0)
print("The first",n_components,"PCAs explain",pca.explained_variance_ratio_.sum()*100,"% of the variance in the data")  

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print("Logistic Regression")
print("Train nonPCA, test nonPCA")
print(clf.score(X_train, y_train),clf.score(X_test, y_test))
clfPCA = LogisticRegression(random_state=0).fit(X_train_pca, y_train)
print("Train PCA, test PCA")
print(clfPCA.score(X_train_pca, y_train),clfPCA.score(X_test_pca, y_test))

print("Scores for Logistic Regression that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, clf.predict(X_test)))
print(balanced_accuracy_score(y_train, clf.predict(X_train)))
print("precision scores")
print(precision_score(y_test, clf.predict(X_test), average = 'weighted'))
print(precision_score(y_train, clf.predict(X_train), average ='weighted'))
print("recall scores")
print(recall_score(y_test, clf.predict(X_test), average = 'weighted'))
print(recall_score(y_train, clf.predict(X_train), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, clf.predict(X_test), average = 'weighted'))
print(f1_score(y_train, clf.predict(X_train), average ='weighted'))

print("Scores for Logistic Regression that does not use PCAs")
print("test data score on top, training data score on bottom")
print("balanced accuracy scores")
print(balanced_accuracy_score(y_test, clfPCA.predict(X_test_pca)))
print(balanced_accuracy_score(y_train, clfPCA.predict(X_train_pca)))
print("precision scores")
print(precision_score(y_test, clfPCA.predict(X_test_pca), average = 'weighted'))
print(precision_score(y_train, clfPCA.predict(X_train_pca), average ='weighted'))
print("recall scores")
print(recall_score(y_test, clfPCA.predict(X_test_pca), average = 'weighted'))
print(recall_score(y_train, clfPCA.predict(X_train_pca), average ='weighted'))
print("f1 scores")
print(f1_score(y_test, clfPCA.predict(X_test_pca), average = 'weighted'))
print(f1_score(y_train, clfPCA.predict(X_train_pca), average ='weighted'))


cov_mat = np.cov(X_train.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)  #total is sum of eigenvalues
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]  #compute explained variance for each PC 
cum_var_exp = np.cumsum(var_exp) #compute cumulative sum variance

plt.bar(range(1, 132), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 132), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

