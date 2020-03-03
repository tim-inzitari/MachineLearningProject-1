# reading and writing data
import pandas as pd


#import pokemon data
pokemon = pd.read_csv('Pokemon.csv', sep=',')
pokemon.columns = pokemon.columns.str.replace(' ', '_')
pokemon.head()

X = pokemon[['Type_1', 'Type_2', 'Total', 'HP', 'Attack', 'Defense', 'Sp._Atk', 'Sp._Def', 'Speed']]
Type_1 = X['Type_1']

y = pokemon['Legendary']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

seeData = True
if seeData:
    # plotting a scatter matrix
    from matplotlib import cm
    from pandas.plotting import scatter_matrix
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

    # plotting a 3D scatter plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d   # must keep
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X_train['Total'], X_train['Attack'], X_train['HP'], c = y_train, marker = 'o', s=100)
    ax.set_xlabel('Total')
    ax.set_ylabel('Attack')
    ax.set_zlabel('HP')
    plt.show()

# Create classifier object: kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))

# Create classifier object: Create a linear SVM classifier
# C: Regularization parameter. Default C=1
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=100, random_state=10, tol=1e-4)
lsvc.fit(X_train, y_train)
print("Linear SVM Training set score: {:.2f}%".format(100*lsvc.score(X_train, y_train)))
print("Linear SVM Test set score: {:.2f}%".format(100*lsvc.score(X_test, y_test)))

#
lsvc.predict(X_test)
print(lsvc.coef_)
print(lsvc.intercept_)

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default=â€™rbfâ€™ = radial basis function
# if poly, default degree = 3
from sklearn.svm import SVC
svc = SVC(degree=2, kernel='poly', random_state=1)
svc.fit(X_train, y_train)
print("SVM Poly Training set score: {:.2f}%".format(100*svc.score(X_train, y_train)))
print("SVM Poly Test set score: {:.2f}%".format(100*svc.score(X_test, y_test)))

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default=â€™rbfâ€™ = radial basis function
from sklearn.svm import SVC
svc = SVC(C=10)
svc.fit(X_train, y_train)
print("SVM Gaussian Training set score: {:.2f}%".format(100*svc.score(X_train, y_train)))
print("SVM Gaussian Test set score: {:.2f}%".format(100*svc.score(X_test, y_test)))

# SVM for multiple classes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# SVM with linear kernel
from sklearn.svm import SVC
svc = SVC(C=10, degree=1, kernel='poly')
svc.fit(X_train, y_train)
print("SVM Gaussian Training set score: {:.2f}%".format(100*svc.score(X_train, y_train)))
print("SVM Gaussian Test set score: {:.2f}%".format(100*svc.score(X_test, y_test)))

# kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))