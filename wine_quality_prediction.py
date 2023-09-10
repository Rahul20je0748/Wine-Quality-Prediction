import pandas as pd

data = pd.read_csv('winequality-red.csv')

"""### 1. Display Top 5 Rows of The Dataset"""

data.head()

"""### 2. Check Last 5 Rows of The Dataset"""

data.tail()

"""### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)"""

data.shape

print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

"""### 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement"""

data.info()

"""### 5.Check Null Values In The Dataset"""

data.isnull().sum()

"""### 6. Get Overall Statistics About The Dataset"""

data.describe()

"""### 7. Quality Vs. Fixed Acidity"""

data.columns

import matplotlib.pyplot as plt

plt.bar(data['quality'],data['fixed acidity'])
plt.xlabel('Quality')
plt.ylabel('fixed acidity')
plt.show()

"""### 8. Volatile acidity Vs. Quality"""

plt.bar(data['quality'],data['volatile acidity'])
plt.xlabel('Quality')
plt.ylabel('Volatile acidity')
plt.show()

"""### 9. Residual sugar Vs. Quality"""

data.columns

plt.bar(data['quality'],data['residual sugar'])
plt.xlabel('Quality')
plt.ylabel('residual sugar')
plt.show()

"""### 10. Chlorides Vs. Quality"""

data.columns

plt.bar(data['quality'],data['chlorides'])
plt.xlabel('Quality')
plt.ylabel('chlorides')
plt.show()

"""### 11. Total sulfur dioxide Vs. Quality"""

data.columns

plt.bar(data['quality'],data['total sulfur dioxide'])
plt.xlabel('Quality')
plt.ylabel('total sulfur dioxide')
plt.show()

"""### 12. Alcohol  Vs. Quality"""

data.columns

plt.bar(data['quality'],data['alcohol'])
plt.xlabel('Quality')
plt.ylabel('alcohol')
plt.show()

"""### 13. Correlation Matrix"""

import seaborn as sns

plt.figure(figsize=(10,5))
sns.heatmap(data.corr(),annot=True,fmt='0.1f')

"""### 14. Binarizaton of target variable"""

data['quality'].unique()

data['quality']=[1 if x>=7 else 0 for x in data['quality']]

data['quality'].unique()

"""### Not Handling Imbalanced"""

data['quality'].value_counts()

import seaborn as sns

sns.countplot(data['quality'])
"""### 15. Store Feature Matrix In X And Response (Target) In Vector y"""

X = data.drop('quality',axis=1)
y = data['quality']

y

X

"""### 16. Handling Imbalanced Dataset"""

from imblearn.over_sampling import SMOTE

X_res,y_res = SMOTE().fit_resample(X,y)

y_res.value_counts()


"""### 17. Splitting The Dataset Into The Training Set And Test Set"""

# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,random_state=42)

"""### 18 . Feature Scaling"""

from sklearn.preprocessing import StandardScaler

st  =StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

X_train

"""### 19. Applying PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components=0.90)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

sum(pca.explained_variance_ratio_)

pca.explained_variance_ratio_

"""### 20. Logistic Regression"""

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train,y_train)

y_pred1 = log.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred1)

accuracy_score(y_test,y_pred1)

from sklearn.metrics import precision_score,recall_score,f1_score

precision_score(y_test,y_pred1)

precision_score(y_test,y_pred1)

recall_score(y_test,y_pred1)

recall_score(y_test,y_pred1)

f1_score(y_test,y_pred1)

f1_score(y_test,y_pred1)

"""### 21. SVC"""

from sklearn import svm

svm = svm.SVC()

svm.fit(X_train,y_train)

y_pred2 = svm.predict(X_test)

accuracy_score(y_test,y_pred2)

precision_score(y_test,y_pred2)

f1_score(y_test,y_pred1)

"""### 22. KNeighbors Classifier"""

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred3 = knn.predict(X_test)

accuracy_score(y_test,y_pred3)

precision_score(y_test,y_pred3)

recall_score(y_test,y_pred3)

f1_score(y_test,y_pred3)

"""### 23. Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

dt =DecisionTreeClassifier()

dt.fit(X_train,y_train)

y_pred4 = dt.predict(X_test)

accuracy_score(y_test,y_pred4)

precision_score(y_test,y_pred4)

f1_score(y_test,y_pred4)

"""### 24. Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred5 = rf.predict(X_test)

accuracy_score(y_test,y_pred5)

precision_score(y_test,y_pred5)

f1_score(y_test,y_pred5)

"""### 25. Gradient Boosting Classifier"""

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)

y_pred6 = gbc.predict(X_test)

accuracy_score(y_test,y_pred6)

precision_score(y_test,y_pred6)

f1_score(y_test,y_pred6)

import pandas as pd

final_data = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GBC'],
             'ACC':[accuracy_score(y_test,y_pred1)*100,
                   accuracy_score(y_test,y_pred2)*100,
                   accuracy_score(y_test,y_pred3)*100,
                   accuracy_score(y_test,y_pred4)*100,
                   accuracy_score(y_test,y_pred5)*100,
                   accuracy_score(y_test,y_pred6)*100]})

final_data

"""### Save The Model"""

X = data.drop('quality',axis=1)
y = data['quality']

from imblearn.over_sampling import SMOTE
X_res,y_res = SMOTE().fit_resample(X,y)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X = st.fit_transform(X_res)

X = pca.fit_transform(X)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y_res)

import joblib

joblib.dump(rf,'wine_quality_prediction')

model = joblib.load('wine_quality_prediction')

"""### Prediction on New Data"""

import pandas as pd
new_data = pd.DataFrame({
    'fixed acidity':7.3,
    'volatile acidity':0.65,
    'citric acid':0.00,
    'residual sugar':1.2,
    'chlorides':0.065,
    'free sulfur dioxide':15.0,
    'total sulfur dioxide':21.0,
    'density':0.9946,
    'pH':3.39,
    'sulphates':0.47,
    'alcohol':10.0,

},index=[0])

new_data

test = pca.transform(st.transform(new_data))

p = model.predict(test)

if p[0] == 1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")