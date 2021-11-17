import numpy as np
import pandas as pd


def get_bucketer(method, encoding_method=None, case_id_col=None, cat_cols=None, num_cols=None, n_clusters=None, random_state=None, n_neighbors=None):
    if method == "single":
        return NoBucketer(case_id_col=case_id_col)
    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=case_id_col)
    else:
        print("Invalid bucketer type")
        return None


class NoBucketer(object):
    def __init__(self, case_id_col):
        self.n_states = 1
        self.case_id_col = case_id_col
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X, y=None):
        return np.ones(len(X[self.case_id_col].unique()), dtype=np.int)
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


class PrefixLengthBucketer(object):
    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col
        
    
    def fit(self, X, y=None):
        
        sizes = X.groupby(self.case_id_col).size()
        self.n_states = sizes.unique()
        
        return self
    
    
    def predict(self, X, y=None):
        
        bucket_assignments = X.groupby(self.case_id_col).size()
        while sum(~bucket_assignments.isin(self.n_states)) > 0:
            bucket_assignments[~bucket_assignments.isin(self.n_states)] -= 1
        return bucket_assignments.as_matrix()
    
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

