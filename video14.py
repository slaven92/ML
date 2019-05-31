import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

# %% load data
df = pd.read_csv("Y:/Slaven/pythonStuff/projects/ML/breast-cancer-wisconsin.data.txt")
df_copy = df.copy()
# %%
df = df_copy.copy()
df.replace('?', -99999, inplace=True)
df.drop('id', 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# %% score
accuracy = clf.score(X_test, y_test)
accuracy

# %% prediction
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[30,2,1,0,1,2,3,2,1]])
prediction = clf.predict(example_measures)
print(prediction)
