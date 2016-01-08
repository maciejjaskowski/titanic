from __future__ import division
import csv as csv
import numpy as np

import pandas as pd
from pandas import get_dummies
from pandas import DataFrame

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn_pandas import DataFrameMapper

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
  df['EmbarkedFill'] = df['Embarked'].map( { 'S': 0, 'Q': 1, 'C': 2})
  df['FamilyID'] = df['LastName'] + df['FamilySize'].apply(str)
  df.loc[df['EmbarkedFill'].isnull(), 'EmbarkedFill'] = 0
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


def train_forest(X_train, y_train, X_validation, n_estimators = 50):
  forest = RandomForestClassifier(n_estimators = n_estimators)
  forest = forest.fit(X_train,y_train)
  gfit = forest.predict(X_validation)  
  return (gfit, forest)

def train_decision_tree(X_train, y_train, X_validation):
  clf = tree.DecisionTreeClassifier()  
  clf = clf.fit(X_train, y_train)
  gfit = clf.predict(X_validation)
  return (gfit, clf)

def find_best_hyperparameters(X_train, y_train):
  forest = RandomForestClassifier(n_estimators = 10)
  pipeline = Pipeline([("forest", forest)])
  param_grid = dict(forest__n_estimators = [16,24, 32,40, 48, 56, 64], forest__criterion = ['gini', 'entropy'])
  grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
  grid_search.fit(X_train, y_train)
  print(grid_search.best_estimator_)  

def validate_model(train_data, train = train_forest):
  kf = StratifiedKFold(train_data[0::,0] == 1, n_folds = 10) 
  #kf = KFold(len(train_data), n_folds = 10)
  return [ 1 - np.mean(abs(train(X_train = train_data[train_index,1::], 
               y_train = train_data[train_index,0],
               X_validation = train_data[validation_index,1::])[0] - train_data[validation_index,0])) 
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

  print "Survived ~ Gender + FamilyID"
  results1 = validate_model(df[['Survived', 'Gender', 'FamilyID']].values)
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

  print "Survived ~ Gender + TitleMin + FamilySize + AgeFilledMedianByTitle + EmbarkedFill"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle', 'EmbarkedFill']].values)
  print results
  print "Mean: {m}".format(m =  np.mean(results))
  print 

  print "Survived ~ Gender + TitleMin + FamilySize + AgeFilledMedianByTitle + EmbarkedFill + FamilyID"
  results = validate_model(df[['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle', 'EmbarkedFill', 'FamilyID']].values)
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

def submission(df, df_test, test_passenger_ids):
  output, forest = train_decision_tree(df[0::,1::], df[0::,0], df_test)
  csv = pd.concat([pd.DataFrame(test_passenger_ids), pd.DataFrame({'Survived': output.astype(int)})], axis=1)

  print forest.feature_importances_
  
  csv.to_csv("submission.csv", index = False)
  #i_tree = 0
  #for tree_in_forest in forest.estimators_:
  #   with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
  #       my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, feature_names = features[1:])
  #   i_tree = i_tree + 1
  #with open('tree.dot', 'w') as my_file:
  #       my_file = tree.export_graphviz(forest, out_file = my_file, feature_names = df_test.columns)


df = add_features(df)
df_test = add_features(df_test)

comb = df[['FamilyID', 'FamilySize']].append(df_test[['FamilyID', 'FamilySize']]).reset_index()
mapper = DataFrameMapper([('FamilyID', LabelEncoder())])
comb['FamilyID'] = pd.DataFrame(mapper.fit_transform(comb))
comb.loc[comb['FamilySize'] <= 1, 'FamilyID'] = 'Small'

comb = get_dummies(comb, columns=['FamilyID'], prefix=['FamilyID']).drop(['FamilySize', 'index'], axis = 1)

df [comb.columns] = comb[0:891]
df_test[comb.columns] = comb[891:]
df_test[comb.columns] = comb[891:].values

features1 = ['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle'] #=> 0.78469 on Kaggle
#features = ['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle', 'EmbarkedFill', 'FamilyID'] # ???
#features = ['Survived', 'Gender', 'FamilySize', 'AgeFilledMedianByTitle'] # ???
features = ['Survived', 'Gender', 'TitleMin', 'FamilySize', 'AgeFilledMedianByTitle'] + comb.columns.tolist()

np.random.seed(44)
df = df.reindex(np.random.permutation(df.index))
results = validate_model(df[features].values, train = train_decision_tree)
print results
print "Mean: {m}".format(m =  np.mean(results))
print 

results = validate_model(df[features1].values, train = train_decision_tree)
print results
print "Mean: {m}".format(m =  np.mean(results))
print 

submission(df[features].values, df_test[features[1::]].values, df_test['PassengerId'])

#pca = PCA(n_components=3)



#pcaX = pca.fit_transform(pd.concat([df[features[1::]], df_test[features[1::]]]).values)

#dff = np.c_[df['Survived'].values, pcaX[0:891,0::]]

#print pca.components_
#print pca.explained_variance_ratio_



#np.random.seed(44)
#df = df.reindex(np.random.permutation(df.index))
#find_best_hyperparameters(df[features].values[0::,1::], df[features].values[0::,0])

#submission(dff, pcaX[891:,0::], df_test['PassengerId'])

#results = validate_model(dff[features].values)
#print results
#print "Mean: {m}".format(m =  np.mean(results))
#print  
#results = validate_model(dff)
#print results
#print "Mean: {m}".format(m =  np.mean(results))
#print  
#results = validate_model(df[features].values, train = lambda X_train, y_train, X_validation: train_forest(X_train, y_train, X_validation, n_estimators = 64))
#print results
#print "Mean: {m}".format(m =  np.mean(results))
#print  
#results = validate_model(df[features].values, train = lambda X_train, y_train, X_validation: train_forest(X_train, y_train, X_validation, n_estimators = 16))
#print results
#print "Mean: {m}".format(m =  np.mean(results))
#print  

#results = validate_model(df[features].values)
#print results
#print "Mean: {m}".format(m =  np.mean(results))
#print  
