from __future__ import division
import csv as csv
import numpy as np

import pandas as pd
from pandas import get_dummies
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split


def extract_title(df):
  df['Title'] = df['Name'].apply(lambda c: c[c.index(',')+2 : c.index('.')]).map(
    {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Don': 4, 'Rev': 5, 'Dr': 6, 'Mme': 7, 'Ms': 8, 'Major': 9, 'Capt': 10, 'Lady': 11, 'Sir': 12, 'Mlle': 13, 'Col': 14, 'the Countess': 15, 'Jonkheer': 15})
  df['TitleMin'] = df['Name'].apply(lambda c: c[c.index(',')+2 : c.index('.')]).map(
  {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Don': 4, 'Rev': 4, 'Dr': 0, 'Mme': 5, 'Ms': 5, 'Major': 0, 'Capt': 4, 'Lady': 5, 'Sir': 0, 'Mlle': 5, 'Col': 0, 'the Countess': 5, 'Jonkheer': 4})
  df.loc[df['TitleMin'].isnull(), 'TitleMin'] = 5 # Probably some fancy title
  return df

def basic_transformation(df):
  df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
  df['LastName']  = df['Name'].apply(lambda c: c[0:c.index(',')])
  df = pd.merge(df, df.groupby(['LastName']).size().reset_index().rename(columns = {0: 'NamePopularity'}), on = 'LastName')
  df['FamilySize'] = df['SibSp'] + df['Parch']
  return df



df = basic_transformation(pd.read_csv('train.csv', header=0))
df_test = basic_transformation(pd.read_csv('test.csv', header=0))
  
df_all = pd.concat([df, df_test])
medians_by_title = extract_title(df_all).groupby('TitleMin')['Age'].median().reset_index().rename(columns = {'Age': 'AgeFilledMedianByTitle'})


def add_features(df):

  df['AgeOriginallyNaN'] = df['Age'].isnull().astype(int)
  df['Male3'] = np.all([df['Sex'] == 'male', df['Pclass'] == 3], axis = 0).astype(int)
  df['Class3'] = (df['Pclass'] == 3).astype(int) 
  df = extract_title(df)
  df = pd.merge(df, medians_by_title, on = 'TitleMin')

  #df = get_dummies(df, columns=['TitleMin'], prefix=['Title_'])
  return df


def train_forest(X_train, y_train, X_validation):
  forest = RandomForestClassifier(n_estimators = 50)
  forest = forest.fit(X_train,y_train)
  gfit = forest.predict(X_validation)
  return (gfit, forest)

def validate_model(train_data):
  kf = StratifiedKFold(df['Survived'] == 1, n_folds = 10) 
  return [ 1 - np.mean(abs(train_forest(X_train = train_data[train_index,1::], 
               y_train = train_data[train_index,0],
               X_validation =train_data[validation_index,1::])[0] - train_data[validation_index,0])) 
    for train_index, validation_index in kf ]

def compare_models(df1, df2):
  X_train, X_test, y_train, y_test = train_test_split(df1[0::, 1::], df1[0::, 0], test_size = 0.2, random_state = 42)
  result1, _ = train_forest(X_train = X_train, 
               y_train = y_train,
               X_validation = X_test)
  X_train, X_test, y_train, y_test = train_test_split(df2[0::, 1::], df2[0::, 0], test_size = 0.2, random_state = 42)
  result2, _ = train_forest(X_train = X_train, 
               y_train = y_train,
               X_validation = X_test)
  return [result1, result2]


def validate_models(df):
  np.random.seed(44)
  df = df.reindex(np.random.permutation(df.index))

  print "Survived ~ Gender"
  results1 = validate_model(df[['Survived', 'Gender']].values)
  print results1
  print "Mean: {m}".format(m =  np.mean(results1))
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

  print "Survived ~ Gender + TitleMin + NamePopularity"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'NamePopularity']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize + AgeFilledMedianByTitle"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 
  

  print "Survived ~ Gender + TitleMin + FamilySize + Male3"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize', 'Male3']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize + Class3"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize', 'Class3']].values)
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

  print "Survived ~ Gender + Title + NamePopularity"
  results = validate_model(df[['Survived', 'Gender', 'Title', 'NamePopularity']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 


  print "Survived ~ Gender + AgeFilledMedianByTitle"
  results = validate_model(df[['Survived', 'Gender', 'AgeFilledMedianByTitle']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + AgeFilledMedianByTitle"
  results = validate_model(df[['Survived', 'Gender', 'AgeFilledMedianByTitle', 'AgeOriginallyNaN']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + NamePopularity"
  results = validate_model(df[['Survived', 'Gender', 'NamePopularity']].values)
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


df = add_features(df)
df_test = add_features(df_test)

features = ['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle']

output, forest = train_forest(df[features].values[0::,1::], df[features].values[0::,0], df_test[features[1:]].values)


csv = pd.concat([pd.DataFrame(df_test['PassengerId']), pd.DataFrame({'Survived': output.astype(int)})], axis=1)

csv.to_csv("submission.csv", index = False)

i_tree = 0
for tree_in_forest in forest.estimators_:
   with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
       my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, feature_names = features[1:])
   i_tree = i_tree + 1
