import os
import pandas as pd
from sklearn.base import clone


class ALEProcessing:
    def __init__(self, artefacts_dir, dataset_name, cls_method, method_name, dm, ffeatures, dataset, train_y_experiment,
                 cls):
        self.dataset = dataset
        self.y = train_y_experiment
        self.cls = cls
        self.artefacts_dir = artefacts_dir
        self.dataset_name = dataset_name
        self.cls_method = cls_method
        self.bkt_enc = method_name
        self.ffeatures = ffeatures
        self.dm = dm

    def data_processing(self):
        ALE_dir = os.path.join(self.artefacts_dir, 'ALE_%s_%s_%s' % (self.dataset_name, self.cls_method, self.bkt_enc))
        if not os.path.exists(ALE_dir):
            os.makedirs(ALE_dir)
        ALE_df = pd.DataFrame(self.dataset, columns=self.ffeatures)
        counts_df = ALE_df.nunique(dropna=False)
        counts_df = counts_df.to_frame()
        counts_df.columns = ['values']
        counts_df.reset_index(level=0, inplace=True)
        # get indices of numerical features in the counts_df
        original_num_cols = self.dm.dynamic_num_cols + self.dm.static_num_cols
        feat_cat_indices = list(set([i for i, feature_name in enumerate(counts_df['index']) if not (
                    any(feat in feature_name for feat in original_num_cols) and not (
                any(s in feature_name for s in ['concept:name', 'True', 'False', 'other'])))]))
        counts_df['values_count'] = 'Numerical Variable'
        for cat_col_idx in feat_cat_indices:
            if counts_df.loc[cat_col_idx, 'values'] <= 20:
                ALE_col = counts_df.loc[cat_col_idx, 'index']
                counts_df.loc[cat_col_idx, 'values_count'] = [ALE_df[ALE_col].value_counts().to_dict()]
            else:
                counts_df.loc[cat_col_idx, 'values_count'] = 'more than 20 cat levels'
        for col in ALE_df:
            for x in counts_df['index']:
                if x == col:
                    counts_df['type'] = ALE_df.dtypes[col]

        counts_df.to_csv(os.path.join(self.artefacts_dir, 'counts_file_after_encoding_%s.csv' % (self.dataset_name)),
                         sep=';', index=False)
        with open(os.path.join(self.artefacts_dir, 'counts_file_after_encoding_%s.html' % (self.dataset_name)),
                  'w') as c:
            c.write(counts_df.to_html() + '\n\n')
        c.close()

        for index, row in counts_df.iterrows():
            if row['values'] <= 2:
                ALE_df.drop(row['index'], axis=1, inplace=True)

        ALE_training_arr = ALE_df.to_numpy(copy=True)
        self.ALE_df = ALE_df
        return ALE_dir, ALE_df, ALE_training_arr, counts_df

    def get_ALE_names(self):
        return self.ALE_df.columns, [self.dm.pos_label, self.dm.neg_label]

    def get_target_encoded(self, yy):
        if yy == self.dm.pos_label:
            return 1
        else:
            return 0

    def ALE_classifier(self, ALE_training_arr):
        ALE_cls = clone(self.cls)
        ALE_cls.fit(ALE_training_arr, self.y)
        return ALE_cls

