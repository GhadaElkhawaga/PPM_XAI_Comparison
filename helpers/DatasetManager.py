import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import Definitions
logs_dir = 'logs'


class DatasetManager:
    def __init__(self, d_name):
        self.d_name = d_name
        self.case_id_col = Definitions.case_id_col[self.d_name]
        self.activity_col = Definitions.activity_col[self.d_name]
        self.timestamp_col = Definitions.timestamp_col[self.d_name]
        self.label_col = Definitions.label_col[self.d_name]
        self.pos_label = Definitions.pos_label[self.d_name]
        self.neg_label = Definitions.neg_label[self.d_name]
        self.dynamic_cat_cols = Definitions.dynamic_cat_cols[self.d_name]
        self.static_cat_cols = Definitions.static_cat_cols[self.d_name]
        self.dynamic_num_cols = Definitions.dynamic_num_cols[self.d_name]
        self.static_num_cols = Definitions.static_num_cols[self.d_name]
        self.sorting_cols = [self.timestamp_col, self.activity_col]

    def read_dataset(self):
        dtypes = {col: 'object' for col in (self.dynamic_cat_cols + self.static_cat_cols + [
            self.case_id_col + self.label_col + self.timestamp_col])}
        for col in (self.dynamic_num_cols + self.static_num_cols):
            dtypes[col] = 'float'
        # read encoded data
        df = pd.read_csv(Definitions.filename[self.d_name], sep=';', dtype=dtypes, engine='c', encoding='ISO-8859-1',
                         error_bad_lines=False)
        return df
    
    
    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]
        
        
    # to determine the length of a case which has a positive outcome, used in determining the max prefix length for each log
    def get_pos_case_length_quantile(self, data, percentage=0.90):
        return int(
            np.ceil(data[data[self.label_col] == self.pos_label].groupby(self.case_id_col).size().quantile(percentage)))

    def split_data_strict(self, data, train_ratio, split='temporal'):
        # split into train and test using temporal split and discard events that overlap the periods
        grouped = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort').groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index().sort_values(self.timestamp_col,
                                                                                       ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio * len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                         kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                         kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)

    # function to split the training set into training and validation sets
    def split_val(self, data, val_ratio, split='random', seed=22):
        start_timestamps = data.groupby(self.case_id_col)[self.timestamp_col].min().reset_index()
        if split == 'temporal':
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        elif split == 'random':
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[-int(val_ratio * len(start_timestamps)):]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                     kind='mergesort')
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                        kind='mergesort')
        return (train, val)

    # to get the ratio of samples belonging to the positive class
    def get_class_ratio(self, data):
        frequencies = data[self.label_col].value_counts()
        return frequencies[self.pos_label] / frequencies.sum()

    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # getting the length of each process instance
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)
        # getting instances which are longer than the minimum length and getting amount of data equivalent to the min length
        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        # this is the first prefixed chunk
        dt_prefixes['prefix_nr'] = 1
        # keeping the original case id with each case
        dt_prefixes['original_case_id'] = dt_prefixes[self.case_id_col]
        # prefix-based bucketing requires certain nr_events
        # repeat the previous process while increasing the prefixed data bz the gap everytime
        for nr_events in range(min_length + gap, max_length + 1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp['original_case_id'] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: '%s_%s' % (x, nr_events))
            tmp['prefix_nr'] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        return dt_prefixes

    def get_label_numeric(self, data):
        # get the label of the first row in a process instance, as they are grouped
        y = data.groupby(self.case_id_col).first()[self.label_col]
        return [1 if label == self.pos_label else 0 for label in y]

    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        firsts = data.groupby(self.case_id_col, as_index=False).first()
        stratifiedKF = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        # using an instance of the StratifiedKFold to generate chunks of stratified train and test data, and send it at once to the calling line
        for trainIdx, testIdx in stratifiedKF.split(firsts, firsts[self.label_col]):
            current_train = firsts[self.case_id_col][trainIdx]
            train_chunk = data[data[self.case_id_col].isin(current_train)].sort_values(self.timestamp_col,
                                                                                       ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train)].sort_values(self.timestamp_col,
                                                                                       ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()['prefix_nr']

    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: '_'.join(x.split('_')[:-1]))
        return case_ids

