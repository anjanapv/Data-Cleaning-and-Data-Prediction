# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,

# Libraries for data visualization
import matplotlib.pyplot as pplt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Import scikit_learn module for the algorithm/model: Linear Regression
from sklearn.linear_model import LogisticRegression
# Import scikit_learn module to split the dataset into train.test sub-datasets
from sklearn.model_selection import train_test_split
# Import scikit_learn module for k-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# import the metrics class
from sklearn import metrics
# import stats for accuracy
# import statsmodels.api as sm

#load the dataset provided
df = pd.read_csv('C:/FDM/adult.csv')

# salary dataset info to find columns and count of the data
df.info()

#replacing some special character columns names with proper names
df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country','hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)
# print(salary_dataset.columns)

#Finding the special characters in the data frame
print(df.isin(['?']).sum(axis=0))

# code will replace the special character to nan and then drop the columns
df['country'] = df['country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)
#dropping the NaN rows now
df.dropna(how='any',inplace=True)

# print(df)

#running a loop of value_counts of each column to find out unique values.
for c in df.columns:
    print ("---- %s ---" % c)
    print (df[c].value_counts())

#dropping based on uniquness of data from the dataset
df.drop(['educational-num','age', 'hours per week', 'fnlwgt', 'capital gain','capital loss', 'country'], axis=1, inplace=True)
# df.info()

#mapping the data into numerical data using map function
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

#gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
#race
df['race'] = df['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
#marital
df['marital'] = df['marital'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
#workclass
df['workclass'] = df['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
#education
df['education'] = df['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)
#occupation
df['occupation'] = df['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2, 'Adm-clerical': 3, 'Handlers-cleaners': 4,
 'Prof-specialty': 5,'Machine-op-inspct': 6, 'Exec-managerial': 7,'Priv-house-serv': 8,'Craft-repair': 9,'Sales': 10, 'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13,'Protective-serv':14}).astype(int)
#relationship
df['relationship'] = df['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)

# print(df)

#plotting a bar graph for Education against Income to see the co-relation between these columns
df.groupby('education').income.mean().plot(kind='bar')
# pplt.show()
df.groupby('gender').income.mean().plot(kind='bar')
# pplt.show()
df.groupby('race').income.mean().plot(kind='bar')
# pplt.show()
df.groupby('marital').income.mean().plot(kind='bar')
# pplt.show()
df.groupby('workclass').income.mean().plot(kind='bar')
# pplt.show()
df.groupby('occupation').income.mean().plot(kind='bar')
# pplt.show()
df.groupby('relationship').income.mean().plot(kind='bar')
# pplt.show()
# pplt.title('education2 vs income2')
# pplt.xlabel('education1')
# pplt.ylabel('income1')

#Transform the data set into a data frame
#X axis = We concatenate the Relationship, Education,Race,Occupation columns concate using np.c_ provided by the numpy library
df_x = pd.DataFrame(np.c_[df['relationship'], df['education'], df['race'],df['occupation'],df['gender'],df['marital'],df['workclass']], columns = ['relationship','education','race','occupation','gender','marital','workclass'])
#Y axis = Our dependent variable or the income of adult i.e Income
df_y = pd.DataFrame(df.income)

#Initialize the linear regression model
reg = LogisticRegression()
#Split the data into 77% training and 33% testing data
#NOTE: We have to split the dependent variables (x) and the target or independent variable (y)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
#Train our model with the training data
reg.fit(x_train, y_train)
#print our price predictions on our test data

#feeding the predict function with our test values in the format
y_pred = reg.predict(x_test)
[['relationship','education','race','occupation','gender','marital','workclass']]
print('Result is =',reg.predict([[1,7,3,7,0,2,0]]))
#printing the accuracy values
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))