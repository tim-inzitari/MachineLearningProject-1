# reading and writing data
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Import pokemon data
pokemon = pd.read_csv('Pokemon.csv', sep=',')
pokemon.columns = pokemon.columns.str.replace(' ', '_')
X = pokemon[['Type_1', 'Type_2', 'Total', 'HP', 'Attack', 'Defense', 'Sp._Atk', 'Sp._Def', 'Speed']]
X_scaled = preprocessing.scale(X)
y = pokemon['Legendary']

# Random_state: set seed for random# generator
# Test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.10, random_state=42)
print("Our data set consists of " + str(len(pokemon.columns) - 1) + " attributes, however we only use 9 of them.")
print("Distribution of Legendary Pokemon:")
print(pokemon['Legendary'].value_counts())
print("Testing Split: 10% testing and 90% training")
seeData = False
if seeData:
    # plotting a scatter matrix
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
    # plotting a 3D scatter plot
    from mpl_toolkits.mplot3d import axes3d   # must keep
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train['Total'], X_train['Attack'], X_train['Defense'], c=y_train, marker='o', s=100)
    ax.set_xlabel('Total')
    ax.set_ylabel('Attack')
    ax.set_zlabel('Defense')
    plt.show()

# kNN
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)
print("\nkNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))

# Create classifier object: Create a linear SVM classifier
# C: Regularization parameter. Default C=1
svcL = LinearSVC(C=5, random_state=10, tol=1e-4, max_iter=10000)
svcL.fit(X_train, y_train)
print("Linear SVM Training set score: {:.2f}%".format(100*svcL.score(X_train, y_train)))
print("Linear SVM Test set score: {:.2f}%".format(100*svcL.score(X_test, y_test)))

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default= radial basis function
# if poly, default degree = 3
svcP = SVC(C=10, degree=3, kernel='poly', random_state=1)
svcP.fit(X_train, y_train)
print("SVM Poly Training set score: {:.2f}%".format(100*svcP.score(X_train, y_train)))
print("SVM Poly Test set score: {:.2f}%".format(100*svcP.score(X_test, y_test)))


# Create classifier object: Create a nonlinear SVM classifier
# kernel, default = radial basis function
svcG = SVC(C=10)
svcG.fit(X_train, y_train)
print("SVM Gaussian Training set score: {:.2f}%".format(100*svcG.score(X_train, y_train)))
print("SVM Gaussian Test set score: {:.2f}%".format(100*svcG.score(X_test, y_test)))


# Logistic Regression
LR = LogisticRegression(random_state=5)
LR.fit(X_train, y_train)
print("Logistic Regression set score: {:.2f}%".format(100*LR.score(X_train, y_train)))
print("Logistic Regression Test set score: {:.2f}%".format(100*LR.score(X_test, y_test)))

# Plot non-normalized confusion matrix
# Knn Matrix
titles_options = [("KNN", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn, X_test, y_test,
                                 display_labels=["Legendary", "Non-Legendary"],
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
plt.show()

# Linear Matrix
titles_options = [("SVM Linear", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svcL, X_test, y_test,
                                 display_labels=["Legendary", "Non-Legendary"],
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
plt.show()

# Poly Matrix
titles_options = [("SVM Poly", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svcP, X_test, y_test,
                                 display_labels=["Legendary", "Non-Legendary"],
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
plt.show()

# Gaussian Matrix
titles_options = [("SVM Gaussian", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svcG, X_test, y_test,
                                 display_labels=["Legendary", "Non-Legendary"],
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
plt.show()

# Logistic Regression
titles_options = [("Logistic Regression", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(LR, X_test, y_test,
                                 display_labels=["Legendary", "Non-Legendary"],
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
plt.show()