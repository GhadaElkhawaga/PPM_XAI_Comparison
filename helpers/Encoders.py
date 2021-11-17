import numpy as np
import pandas as pd
import time
from sklearn.base import TransformerMixin


def get_encoder(method, case_id_col=None, static_cat_cols=None, static_num_cols=None, dynamic_cat_cols=None,
                dynamic_num_cols=None, fillna=True, max_events=None, activity_col=None, resource_col=None, timestamp_col=None,
                scale_model=None):
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, boolean=False, fillna=fillna)
    elif method == "bool":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, boolean=True, fillna=fillna)
    elif method == "index":
        return IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, max_events=max_events, fillna=fillna)
    else:
        print("Invalid encoder type")
        return None


class StaticTransformer(TransformerMixin):
    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        start = time.time()
        
        dt_first = X.groupby(self.case_id_col).first()
        
        # transform numeric cols
        dt_transformed = dt_first[self.num_cols]
        
        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_first[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns
        
        self.transform_time = time.time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns



class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.boolean = boolean
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time.time()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg({'mean':np.mean, 'max':np.max, 'min':np.min, 'sum':np.sum, 'std':np.std})
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
            
        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        
        self.transform_time = time.time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns
        
        
class IndexBasedTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, max_events=None, fillna=True, create_dummies=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.create_dummies = create_dummies
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time.time()
        
        grouped = X.groupby(self.case_id_col, as_index=False)
        
        if self.max_events is None:
            self.max_events = grouped.size().max()
        
        
        dt_transformed = pd.DataFrame(grouped.apply(lambda x: x.name), columns=[self.case_id_col])
        for i in range(self.max_events):
            dt_index = grouped.nth(i)[[self.case_id_col] + self.cat_cols + self.num_cols]
            dt_index.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.cat_cols] + ["%s_%s"%(col, i) for col in self.num_cols]
            dt_transformed = pd.merge(dt_transformed, dt_index, on=self.case_id_col, how="left")
        dt_transformed.index = dt_transformed[self.case_id_col]
        
        # one-hot-encode cat cols
        if self.create_dummies:
            all_cat_cols = ["%s_%s"%(col, i) for col in self.cat_cols for i in range(self.max_events)]
            dt_transformed = pd.get_dummies(dt_transformed, columns=all_cat_cols).drop(self.case_id_col, axis=1)
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        self.transform_time = time.time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns