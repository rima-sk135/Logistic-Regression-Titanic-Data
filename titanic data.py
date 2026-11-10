import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv(r"C:\Users\user\Documents\test-julia\Titanic - train.csv")
data_test = pd.read_csv(r"C:\Users\user\Documents\test-julia\Titanic - test.csv")

print('Titanic train data\n', data_train.head(10))
print(data_train.columns)
print('Titanic test data\n', data_test.head(10))
print(data_train.columns)



print("Here parch is the no. of parents or children aboard the Titanic, and sibsp is the no. of spouses or siblings.")
#print(data_train.isnull())
# this is way too huge so we take the sum
print(data_train.isnull().sum())
print("we can see that age has a lot of missing/null values.")

sns.countplot(x="Survived", data= data_train)
#plt.show()

sns.countplot(x="Survived", hue="Sex", data=data_train)
#plt.show()
print("From the graph we can see that females' survival rate were way more than men.")

sns.countplot(x="Survived", hue="Pclass", data=data_train)
#plt.show()
print("From the graph we can see that the non-survival/death rate for the cls3 passengers was more than 3 times less than that of the cls1 passengers.\nIn the case of survival rate, though not much the difference is obvious of cls3 from cls1 passengers.")

data_train["Age"].plot.hist()
#plt.show()
print("From the graph we can see that maximum no. of passengers belong to 20 to 40 years of age group.")
print("So far we can conclude that we don't need PassengerId, SibSp, Parch, Ticket, Fare, Cabin, Embarked")

sex_dummies = pd.get_dummies(data_train["Sex"], drop_first=True)
print(sex_dummies)
pclass_dummies = pd.get_dummies(data_train["Pclass"], drop_first=True)
print(pclass_dummies)
embarked_dummies = pd.get_dummies(data_train["Embarked"], drop_first=True)
print(embarked_dummies)

titanic_data_train = pd.concat([data_train, sex_dummies, pclass_dummies, embarked_dummies], axis=1)
print(titanic_data_train.columns)
titanic_data_train.rename(columns = {2:'Pclass2', 3:'Pclass3', "male":"Sex_male"}, inplace = True)
print(titanic_data_train.columns)

sex_dummies1 = pd.get_dummies(data_train["Sex"], drop_first=True)
print(sex_dummies1)
pclass_dummies1 = pd.get_dummies(data_train["Pclass"], drop_first=True)
print(pclass_dummies1)
embarked_dummies1 = pd.get_dummies(data_train["Embarked"], drop_first=True)
print(embarked_dummies1)

titanic_data_test = pd.concat([data_test, sex_dummies1, pclass_dummies1, embarked_dummies1], axis=1)
print(titanic_data_test.columns)
titanic_data_test.rename(columns = {2:'Pclass2', 3:'Pclass3', "male":"Sex_male"}, inplace = True)
print(titanic_data_test.columns)



x = titanic_data_train[['Sex_male', 'Pclass2', 'Pclass3', 'Q', 'S']]
y = titanic_data_train["Survived"]
print(x.shape)
print(y.shape)
x_t = titanic_data_test[['Sex_male', 'Pclass2', 'Pclass3', 'Q', 'S']]

model = LogisticRegression().fit(x, y.values.reshape(-1))
y_pred = model.predict(x_t)
print(y_pred)
print('model score is: ', model.score(x, y))

from sklearn.feature_selection import chi2
scores, pvalues = chi2(x, y)
print(pvalues[0])
print(pvalues[1])
print(pvalues[2])
print(pvalues[3])
print(pvalues[4])
print("Here we see that the p-value of Embarked from Q is very high, so we'll do the model fitting again without that particular variable")

x1 = titanic_data_train[['Sex_male', 'Pclass2', 'Pclass3']]
x_t1 = titanic_data_test[['Sex_male', 'Pclass2', 'Pclass3']]

model1 = LogisticRegression().fit(x1, y.values.reshape(-1))
y_pred1 = model1.predict(x_t1)
print("y_pred1", y_pred1)
print('model score is: ', model1.score(x1, y))
print(pvalues[0])
print(pvalues[1])
print(pvalues[2])
print(pvalues[3])


