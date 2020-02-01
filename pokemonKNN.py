
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

pokemon = pd.read_csv('Pokemon.csv', sep=',')
pokemon.columns = pokemon.columns.str.replace(' ', '_')
pokemon.head()


# create a mapping from fruit label value to fruit name to make results easier to interpret
# lookup_pokemon_name = dict(zip(pokemon.name.unique(), pokemon.))
# print(lookup_pokemon_name)

X = pokemon[['Type_1', 'Type_2', 'Total', 'HP', 'Attack', 'Defense', 'Sp._Atk', 'Sp._Def', 'Speed']]
Type_1 = X['Type_1']

y = pokemon['Legendary']

kinds = np.array([dt.kind for dt in X.dtypes])
all_columns = X.columns.values
is_number = kinds != 'O'
number_cols = all_columns[is_number]
string_cols = all_columns[~is_number]

# Making pipeline for strings
string_imputer_step = ('string_imputer', SimpleImputer(strategy='constant', fill_value='MISSING'))
string_encoder_step = ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
steps = [string_imputer_step, string_encoder_step]
string_pipe = Pipeline(steps)

# Make pipeline for numbers
number_imputer_step = ('number_imputer', SimpleImputer(strategy='median'))
number_scaler_step = ('scaler', StandardScaler())
number_steps = [number_imputer_step, number_scaler_step]
number_pipe = Pipeline(number_steps)

# Make ColumnTransformer
columns = ['Type_1', 'Type_2']
transformers = [('cat', string_pipe, string_cols), ('num', number_pipe, number_cols)]
ct = ColumnTransformer(transformers=transformers)
X__transformed = ct.fit_transform(X)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X__transformed, y_encoded, random_state=0)

# plotting a scatter matrix

# cmap = cm.get_cmap('gnuplot')
# scatter = scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)

# plotting a 3D scatter plot

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_train['Total'], X_train['Attack'], X_train['HP'], c=y_train, marker='o', s=100)
# ax.set_xlabel('Total')
# ax.set_ylabel('Attack')
# ax.set_zlabel('HP')
# plt.show()

# Create classifier object

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Train the classifier (fit the estimator) using the training data


knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# ## Use the trained k-NN classifier model to classify new, previously unseen objects
# first example: a small fruit with mass 20g, color_score = 5.5, width 4.3 cm, height 5.5 cm
pokemon_prediction = knn.predict(ct.transform(['Normal', 'Flying', 349, 63, 60, 55, 50, 50, 71]))
print(pokemon_prediction[0])
# print(lookup_pokemon_name[pokemon_prediction[0]])

# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm, color_score 6.3
pokemon_prediction = knn.predict([['Bug', 'Flying', 390, 55, 35, 50, 55, 110, 85]])
print(pokemon_prediction[0])
# print(lookup_pokemon_name[pokemon_prediction[0]])

# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()

# How sensitive is k-NN classification accuracy to the train/test split proportion?

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=5)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()
