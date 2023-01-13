import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model, ensemble
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
import numpy as np
import json
from functools import reduce
import os

dataset_path = '/workspaces/workspace/clanci/crops/BreizhCrops/data/serbia'
dataset_name = 'data_serbia_01'

files = os.listdir(os.path.join(dataset_path, dataset_name))

num_instances = len(files) - 1
X = []
for i in range(num_instances):
    x = np.load(os.path.join(dataset_path,dataset_name,f"{i}.npy"))
    X.append(x)
    
X = np.array(X)
y = np.load(os.path.join(dataset_path,dataset_name,"y.npy"))
    
    






train, test = next(iter(strat_split.split(X,y)))

Xtrain = pd.DataFrame(X.loc[train], columns=X.columns)
ytrain = pd.Series(y.loc[train])

scaler = StandardScaler().fit(Xtrain)

Xtrain = scaler.transform(Xtrain)
Xtrain = pd.DataFrame(Xtrain, columns=X.columns)

Xtest = pd.DataFrame(X.loc[test], columns=X.columns)
ytest = pd.Series(y.loc[test])

Xtest = scaler.transform(Xtest)
Xtest = pd.DataFrame(Xtest, columns=X.columns)


estimator= ensemble.RandomForestClassifier(bootstrap=True, max_features=int(n_features/3), min_samples_split = 4, min_samples_leaf=2, oob_score=True, n_jobs=-1, random_state = 0, n_estimators=500,  max_samples = 0.7)

# make_pipeline(, )

gs_all_metric_results = estimator.fit(Xtrain, ytrain)

rf_results = [{'feature': key, 'importance': value} for key, value in zip(estimator.feature_names_in_, estimator.feature_importances_)]
rf_results = sorted(rf_results, key=lambda x: -x['importance'])
feature_importances = pd.DataFrame(rf_results)
feature_importances.to_excel('../results/randomforest_feature_importances.xlsx', index = False)

print(
    f"""
    EVALUATION METRICS:
        Train mean r2: {estimator.score(Xtrain, ytrain)},
        Test  mean r2: {estimator.score(Xtest, ytest)},
        Train mean cod: {cod(estimator.predict(Xtrain), ytrain)},
        Test  mean cod: {cod(estimator.predict(Xtest), ytest)},
        Train mean mape: {mape(estimator.predict(Xtrain), ytrain)},
        Test mean mape: {mape(estimator.predict(Xtest), ytest)},
        
    """
)
metrics = {
    'train_mean_r2': estimator.score(Xtrain, ytrain),
    'test_mean_r2': estimator.score(Xtest, ytest),
    'train_mean_cod': cod(estimator.predict(Xtrain), ytrain),
    'test_mean_cod': cod(estimator.predict(Xtest), ytest),
    'train_mean_mape': mape(estimator.predict(Xtrain), ytrain),
    'test_mean_mape': mape(estimator.predict(Xtest), ytest)
}
with open('../results/randomforest_metrics.json', 'w') as f:
    json.dump(metrics, f)