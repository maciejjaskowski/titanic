from __future__ import division
import csv as csv
import numpy as np

import pandas as pd
from pandas import get_dummies
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split


def transform_gender(df):
  df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
  return df

def transform_age(df, mean):
  df['AgeFilledMean'] = df['Age']
  df['AgeFilledMean'][df['AgeFilledMean'].isnull()] = mean
  return df

def extract_last_name(df):
  df['LastName']  = df['Name'].apply(lambda c: c[0:c.index(',')])
  return df


def create_age_model(df):
  forest = RandomForestRegressor(n_estimators = 100)  
  data = df[['Age', 'Gender', 'Pclass', 'SibSp', 'Parch', 'Fare']].dropna()
  forest.fit(data.values[0::,1::], data.values[0::,0])
  return forest

def add_family_size(df):
  return pd.merge(df, df.groupby(['LastName']).size().reset_index().rename(columns = {0: 'FamilySize'}), on = 'LastName')


def add_family_size2(df):
  df['FamilySize2'] = df['SibSp'] + df['Parch']
  return df

def transform_age_with_model(df, forest):
  to_predict = df[df['Age'].isnull()][['Age', 'Gender', 'Pclass', 'SibSp', 'Parch', 'Fare']]
  df['AgeFilledModel'] = df['Age']
  df.loc[df['AgeFilledModel'].isnull(),'AgeFilledModel'] = forest.predict(to_predict.drop('Age', axis = 1))
  return df

def transform_age_with_mean(df, mean):
  df['AgeFilledMean'] = df['Age']
  df.loc[df['AgeFilledMean'].isnull(), 'AgeFilledMean'] = mean
  return df

def drop_unused(df):
  return df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'LastName', 'SibSp', 'Parch', 'Fare', 'Pclass', 'FamilySize2', 'AgeFilledModel'], axis = 1)

def model_sex_only(df):
  return df[['Survived', 'Gender']]

def extract_title(df):
  df['Title'] = df['Name'].apply(lambda c: c[c.index(',')+2 : c.index('.')]).map(
    {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Don': 4, 'Rev': 5, 'Dr': 6, 'Mme': 7, 'Ms': 8, 'Major': 9, 'Capt': 10, 'Lady': 11, 'Sir': 12, 'Mlle': 13, 'Col': 14, 'the Countess': 15, 'Jonkheer': 15})
  df['TitleMin'] = df['Name'].apply(lambda c: c[c.index(',')+2 : c.index('.')]).map(
  {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Don': 4, 'Rev': 4, 'Dr': 0, 'Mme': 5, 'Ms': 5, 'Major': 0, 'Capt': 4, 'Lady': 5, 'Sir': 0, 'Mlle': 5, 'Col': 0, 'the Countess': 5, 'Jonkheer': 4})
  df.loc[df['TitleMin'].isnull(), 'TitleMin'] = 5 # Probably some fancy title
  return df

df = add_family_size(add_family_size2(extract_last_name(transform_gender(pd.read_csv('train.csv', header=0)))))
df_test = add_family_size(add_family_size2(extract_last_name(transform_gender(pd.read_csv('test.csv', header=0)))))

mean_age = df['Age'].mean()
age_model = create_age_model(df)
  
def add_features(df):

  df = transform_age_with_model(df, age_model)
  df['AgeFilledModel*Class'] = df.AgeFilledModel * df.Pclass
  df['AgeOriginallyNaN'] = df['Age'].isnull().astype(int)
  df['Male3'] = np.all([df['Sex'] == 'male', df['Pclass'] == 3], axis = 0).astype(int)
  df['Class3'] = (df['Pclass'] == 3).astype(int) 
  df['InCabin'] = 1 - (df['Cabin'].isnull()).astype(int)
  df = extract_title(df)
  

  medians_by_title = df.groupby('TitleMin')['Age'].median().reset_index().rename(columns = {'Age': 'AgeFilledMedianByTitle'})
  df = pd.merge(df, medians_by_title, on = 'TitleMin')
  #df = get_dummies(df, columns=['TitleMin'], prefix=['Title_'])
  df['Fare>50'] = (df['Fare'] > 50).astype(int)
  df['Fare>25'] = (df['Fare'] > 25).astype(int)
  df['Fare>35'] = (df['Fare'] > 35).astype(int)
  df['Fare>40'] = (df['Fare'] > 40).astype(int)
  df['Fare>200'] = (df['Fare'] > 200).astype(int)
  return df

df = add_features(df)


def train_forest(X_train, y_train, X_validation):
  forest = RandomForestClassifier(n_estimators = 10)
  forest = forest.fit(X_train,y_train)
  gfit = forest.predict(X_validation)
  return gfit

def validate_model(train_data):
  kf = StratifiedKFold(df['Survived'] == 1, n_folds = 10) 
  return [ 1 - np.mean(abs(train_forest(X_train = train_data[train_index,1::], 
               y_train = train_data[train_index,0],
               X_validation =train_data[validation_index,1::]) - train_data[validation_index,0])) 
    for train_index, validation_index in kf ]

def compare_models(df1, df2):
  X_train, X_test, y_train, y_test = train_test_split(df1[0::, 1::], df1[0::, 0], test_size = 0.2, random_state = 42)
  result1 = train_forest(X_train = X_train, 
               y_train = y_train,
               X_validation = X_test)
  X_train, X_test, y_train, y_test = train_test_split(df2[0::, 1::], df2[0::, 0], test_size = 0.2, random_state = 42)
  result2 = train_forest(X_train = X_train, 
               y_train = y_train,
               X_validation = X_test)
  return [result1, result2]


def validate_models(df):
  np.random.seed(44)
  df = df.reindex(np.random.permutation(df.index))
  df = df.reindex(np.random.permutation(df.index))

  print "Survived ~ Gender"
  results1 = validate_model(df[['Survived', 'Gender']].values)
  print results1
  print "Mean: {m}".format(m =  np.mean(results1))
  print 

  print "Survived ~ InCabin"
  results2 = validate_model(df[['Survived', 'InCabin']].values)
  print results2
  print "Mean: {m}".format(m =  np.mean(results2))
  print 

  print "Survived ~ Gender + InCabin"
  results2 = validate_model(df[['Survived', 'Gender', 'InCabin']].values)
  print results2
  print "Mean: {m}".format(m =  np.mean(results2))
  print 


  print "Survived ~ TitleMin"
  results = validate_model(df[['Survived', 'TitleMin']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ TitleMin + AgeFilledMedianByTitle"
  results = validate_model(df[['Survived', 'TitleMin', 'AgeFilledMedianByTitle']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + Parch + SibSp"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'Parch', 'SibSp']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Male3"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Male3']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Male3 + InCabin"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Male3' + 'InCabin']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Class3"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Class3']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Male3 + Fare>25"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Male3', 'Fare>25']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Male3 + Fare>25 + Fare>40"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Male3', 'Fare>25', 'Fare>40']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Male3 + Fare>50"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Male3', 'Fare>50']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize2 + Male3 + Fare>50 + Fare>200"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize2', 'Male3', 'Fare>50', 'Fare>200']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize + Male3"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize', 'Male3']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ AgeFilledMedianByTitle"
  results = validate_model(df[['Survived', 'AgeFilledMedianByTitle']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + Title"
  results = validate_model(df[['Survived', 'Gender', 'Title']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + Title + FamilySize"
  results = validate_model(df[['Survived', 'Gender', 'Title', 'FamilySize']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 


  print "Survived ~ Gender + AgeFilledModel"
  results = validate_model(df[['Survived', 'Gender', 'AgeFilledModel']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + FamilySize"
  results = validate_model(df[['Survived', 'Gender', 'FamilySize']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + AgeFilledModel*Class"
  results = validate_model(df[['Survived', 'Gender', 'AgeFilledModel*Class', 'FamilySize']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + Pclass"
  results = validate_model(df[['Survived', 'Gender', 'Pclass']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + Male3"
  results = validate_model(df[['Survived', 'Gender', 'Male3']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + AgeFilledModel + AgeOriginallyNaN"
  results = validate_model(df[['Survived', 'Gender', 'AgeFilledModel', 'AgeOriginallyNaN']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 


forest = RandomForestClassifier(n_estimators = 10)
forest = forest.fit(df[['Survived', 'TitleMin', 'FamilySize2', 'Male3']].values[0::,1::],df[['Survived', 'TitleMin', 'FamilySize2', 'Male3']].values[0::,0])

i_tree = 0
for tree_in_forest in forest.estimators_:
   with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
       my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, feature_names = ['TitleMin', 'FamilySize2', 'Male3'])
   i_tree = i_tree + 1
#gfit = forest.predict(train_data[0::,1::])

df_test = add_features(df_test)
output = forest.predict(df_test[['TitleMin', 'FamilySize2', 'Male3']].values)

csv = pd.concat([pd.DataFrame(df_test['PassengerId']), pd.DataFrame({'Survived': output.astype(int)})], axis=1)

csv.to_csv("submission.csv", index = False)
