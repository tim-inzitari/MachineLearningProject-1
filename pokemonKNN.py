
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy import stats


pokemon = pd.read_csv('Pokemon.csv', sep=',')
pokemon.columns = pokemon.columns.str.replace(' ', '_')
pokemon.head()

X = pokemon[['Type_1', 'Type_2', 'Total', 'HP', 'Attack', 'Defense', 'Sp._Atk', 'Sp._Def', 'Speed']]
Type_1 = X['Type_1']

y = pokemon['Legendary']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# DESCRIPTION OF EACH STATS MEAN MEDIAN MODE ECT

# type1
print("-------------------------------------------------------------------")
print("Stat:Type1 - Descriptions of DataSet ")
print(pokemon['Type_1'].describe())
print("Median of Type 1: %d" % np.median(pokemon['Type_1']))
print("Mode of Type 1: ",  stats.mode(pokemon['Type_1']))
print("-------------------------------------------------------------------\n")

# type2
print("-------------------------------------------------------------------")
print("Stat:Type - Descriptions of DataSet ")
print(pokemon['Type_2'].describe())
print("Median of Type_2: %d" % np.median(pokemon['Type_2']))
print("Mode of Type 2: ",  stats.mode(pokemon['Type_2']))
print("-------------------------------------------------------------------\n")

# total
print("-------------------------------------------------------------------")
print("Stat:Total - Descriptions of DataSet ")
print(pokemon['Total'].describe())
print("Median of Total: %d" % np.median(pokemon['Total']))
print("Mode of Total: ",  stats.mode(pokemon['Total']))
print("-------------------------------------------------------------------\n")

# HP
print("-------------------------------------------------------------------")
print("Stat:HP - Descriptions of DataSet ")
print(pokemon['HP'].describe())
print("Median of HP: %d" % np.median(pokemon['HP']))
print("Mode of HP: ",  stats.mode(pokemon['HP']))
print("-------------------------------------------------------------------\n")

# Attack
print("-------------------------------------------------------------------")
print("Stat:Attack - Descriptions of DataSet ")
print(pokemon['Attack'].describe())
print("Median of Attack: %d" % np.median(pokemon['Attack']))
print("Mode of Attack: ",  stats.mode(pokemon['Attack']))
print("-------------------------------------------------------------------\n")

# Defense
print("-------------------------------------------------------------------")
print("Stat:Defense - Descriptions of DataSet ")
print(pokemon['Defense'].describe())
print("Median of Defense: %d" % np.median(pokemon['Defense']))
print("Mode of Defense: ",  stats.mode(pokemon['Defense']))
print("-------------------------------------------------------------------\n")

# Special Attack
print("-------------------------------------------------------------------")
print("Stat:Special Attack - Descriptions of DataSet ")
print(pokemon['Sp._Atk'].describe())
print("Median of Special Attack: %d" % np.median(pokemon['Sp._Atk']))
print("Mode of Special Attack: ",  stats.mode(pokemon['Sp._Atk']))
print("-------------------------------------------------------------------\n")

# Special Defense
print("-------------------------------------------------------------------")
print("Stat:Special Defense - Descriptions of DataSet ")
print(pokemon['Sp._Def'].describe())
print("Median of Special Defense: %d" % np.median(pokemon['Sp._Def']))
print("Mode of Special Defense: ",  stats.mode(pokemon['Sp._Def']))
print("-------------------------------------------------------------------\n")

# Speed
print("-------------------------------------------------------------------")
print("Stat:Speed - Descriptions of DataSet ")
print(pokemon['Speed'].describe())
print("Median of Speed: %d" % np.median(pokemon['Speed']))
print("Mode of Speed: ",  stats.mode(pokemon['Speed']))
print("-------------------------------------------------------------------\n")

# END STAT DESCRIPTIONS

# count number of legendary and not legendary pokemon
print("-------------------------------------------------------------------")
print("Count of Legendary Pokemon vs Not Legendary, True is Legendary False is not")
print(pokemon['Legendary'].value_counts())
print("-------------------------------------------------------------------\n")

# plotting a scatter matrix

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)



# plotting a 3D scatter plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['Total'], X_train['Attack'], X_train['HP'], c=y_train, marker='o', s=100)
ax.set_xlabel('Total')
ax.set_ylabel('Attack')
ax.set_zlabel('HP')
plt.show()

# Create classifier object

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Train the classifier (fit the estimator) using the training data

knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# ## Use the trained k-NN classifier model to classify new, previously unseen objects
# this example is similar is stats to the pokemon Palkia
#
pokemon_prediction = knn.predict([[11, 15, 684, 91, 121, 99, 151, 121, 101]])
print("Are the stats of Pokemon 1 that of a legendary Pokemon?: %s " % pokemon_prediction[0])

# second example: chosen similar to that of a non-legendary pokemon
pokemon_prediction = knn.predict([[7, 3, 390, 55, 35, 50, 55, 110, 85]])
print("Are the stats of Pokemon 2 that of a legendary Pokemon?: %s " % pokemon_prediction[0])

# Third Example:
pokemon_prediction = knn.predict([[1, 3, 349, 63, 60, 55, 50, 50, 71]])
print("Are the stats of Pokemon 3 that of a legendary Pokemon?: %s " % pokemon_prediction[0])

# Fourth Pokemon
pokemon_prediction = knn.predict([[6, 5, 580, 75, 100, 105, 95, 85, 120]])
print("Are the stats of Pokemon 4 that of a legendary Pokemon?: %s " % pokemon_prediction[0])

# USER INPUT TEST AMOUNT ##
# kth Pokemon tests, change to Y to enable
another = "N"
while another == "Y":
    type1, type2, HP, Att, Def, spAtt, spDef, Speed = input("Type1 Type2 HP Att Def spAtt spDef Speed: ").split()
    Total = str(int(HP) + int(Att) + int(Def) + int(spAtt) + int(spDef) + int(Speed))
    pokemon_prediction = knn.predict([[type1, type2, Total, HP, Att, Def, spAtt, spDef, Speed]])
    print("Total: %s" % Total)
    print("Are the stats of this Pokemon that of a legendary Pokemon?: %s " % pokemon_prediction[0])
    another = input("Another? Y for Yes: ")

# END USER INPUT AMOUNT #

bestSplit = 0
bestK = 0
# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(3, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
bestK = scores.index(max(scores)) + 1
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([3, 5, 10, 15, 20])
plt.show()

# How sensitive is k-NN classification accuracy to the train/test split proportion?

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=bestK)
plt.figure()
previousLargestSplit = 0
for s in t:
    scores = []   
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
    if np.mean(scores) > previousLargestSplit:
        previousLargestSplit = np.mean(scores)
        bestSplit = s

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()

dist = ['euclidean', 'manhattan']
bestDist = ''
previousScore = 0
scores = []
plt.figure()
for s in dist:
    knn = KNeighborsClassifier(n_neighbors=bestK, metric=s)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-bestSplit)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
    if previousScore < knn.score(X_test, y_test):
        previousScore = knn.score(X_test, y_test)
        bestDist = s
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Distance Used')
plt.ylabel('accuracy')
plt.show()

# using best K and best split
knnBest = KNeighborsClassifier(n_neighbors=bestK, metric=bestDist)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-bestSplit)
knnBest.fit(X_train, y_train)
print('Using best K(' + str(bestK) + ') and best split(' + str(bestSplit) + ') and best dist(' + str(bestDist) + ')')
print('the accuracy is ' + str(knnBest.score(X_test, y_test)))
print(confusion_matrix(knnBest.predict(X_test), y_test))







