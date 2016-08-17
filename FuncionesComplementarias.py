from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer

def add_cluster_feature(df, features, total_clusters, imputer_mbkm = None):
    normalizer = Normalizer()
    
    if imputer_mbkm is None:
        mbkm = MiniBatchKMeans(n_clusters=total_clusters, verbose=0, reassignment_ratio=0.3, n_init=40)
        normalized_train = normalizer.fit_transform(df[features])
        mbkm.fit(normalized_train)
    else:
        mbkm = imputer_mbkm
        
    normalized_df = normalizer.fit_transform(df[features])
    cluster_prediction = mbkm.predict(normalized_df)
    df['cluster'] = cluster_prediction
    return df, mbkm

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

def grid_search_multiple(X, y, algorithms, parameters, preliminary_steps, scoring='roc_auc', cv=None, n_jobs=1, verbose=0):
    verboseprint = print if verbose > 0 else lambda *a, **k: None
    
    if cv is None:
        StratifiedShuffleSplit(y, n_iter=8, test_size=0.1)
    
    grids = []

    for index, classifier_steps in enumerate(algorithms):
        verboseprint("Now testing", classifier_steps[-1][0])

        clf = Pipeline(preliminary_steps + classifier_steps)
        grid = GridSearchCV(estimator = clf, param_grid=parameters[index], scoring=scoring, verbose=2, cv=cv, n_jobs=n_jobs)

        verboseprint("Preparing to fit")
        grid.fit(X, y)
        verboseprint("Fitting completed")
        grids.append(grid)
        verboseprint("Best parameters set found on development set:")
        verboseprint(grid.best_params_)
        verboseprint("Best score set found on development set:")
        verboseprint(grid.best_score_)
        verboseprint("Grid scores on development set:\r\n")
        for parameters_, mean_score, scores in grid.grid_scores_:
            verboseprint("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, parameters_))
        verboseprint()
    return grids

from sklearn.preprocessing import Imputer
import numpy as np

def impute_train_test(train_data, test_data, is_discrete=False, missing_values='NaN', strategy='mean'):
    imputer = Imputer(strategy=strategy, missing_values=missing_values)
    imputer.fit(train_data)
    train_imputation = imputer.transform(train_data)
    test_imputation = imputer.transform(test_data)
    
    if is_discrete:
        train_imputation = train_imputation.astype('int').astype('float')
        test_imputation = test_imputation.astype('int').astype('float')
    
    train_data = train_imputation
    test_data = test_imputation
    
    return train_data, test_data