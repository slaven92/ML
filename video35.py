import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# %%
df = pd.read_excel('titanic.xls')
df_copy = df.copy()
# %%
df = df_copy.copy()
df.drop(['body', 'name'], axis=1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def hadle_non_numeic_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int,df[column]))
    return df

df = hadle_non_numeic_data(df)
df.drop(['ticket'],1, inplace=True)
# %%

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0]==y[i]:
        correct +=1

print(correct/len(X))


# df.head()
