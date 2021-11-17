import  pandas as pd #using pandas==0.25
import numpy as np
import sys
import os
import io
from sklearn.pipeline import FeatureUnion
import warnings
warnings.simplefilter('ignore')
import csv
from helpers.Encoders import get_encoder
from helpers.Bucketers import get_bucketer
from helpers.DatasetManager import DatasetManager
import Definitions

#Defining Basic parameters
#params_dir = 'paramsDir'
n_iter = 3
method_name = 'single_agg'
#method_name = 'prefix_index'
cls_encoding = 'agg'
#cls_encoding = 'index'
#cls_method = 'xgboost'
cls_method = 'logit'
bucket_method = 'single'
#bucket_method = 'prefix'
gap = 1
train_ratio = 0.8
n_splits = 3 #number of cv_folds
random_state = 22
min_cases_for_training = 1
encoded_datasets_dir = 'encoded_datasets_%s' %(method_name)
if not os.path.exists(encoded_datasets_dir):
  os.makedirs(os.path.join(encoded_datasets_dir))
dataset_ref_to_datasets = {
    "sepsis_cases": ["sepsis1", "sepsis2", "sepsis3"],
    "bpic2017" :["BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"],
    "Hospital_Billing" :["hospital_billing_%s" %i for i in range(1,3)]
}
encoding_dict = {
    'agg' : ['static', 'agg'],
    'index' : ['static', 'index']}
if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"
bucket_method , cls_encoding = method_name.split('_')
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
outfile = os.path.join(encoded_datasets_dir,'all_datasets_info.csv')
with open(outfile, 'w') as out:
        out.write('%s;%s;%s;%s;%s;%s\n' % ('dataset', 'method', 'dataset_type', 'bucket_size', 'prefix_length', 'feature_num'))
for dataset_name in datasets:
    dm = DatasetManager(dataset_name)
    df = dm.read_dataset()
    cls_encoder_args_final = {'case_id_col': dm.case_id_col,
                              'static_cat_cols': dm.static_cat_cols,
                              'dynamic_cat_cols': dm.dynamic_cat_cols,
                              'static_num_cols': dm.static_num_cols,
                              'dynamic_num_cols': dm.dynamic_num_cols,
                              'fillna': True}
    # determine min and max (truncated) prefix lengths
    min_prefix_length_final = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length_final = 10
    elif "BPIC2017" in dataset_name:
        max_prefix_length_final = min(20, dm.get_pos_case_length_quantile(df, 0.90))
    else:
        max_prefix_length_final = min(40, dm.get_pos_case_length_quantile(df, 0.90))

    train, test = dm.split_data_strict(df, train_ratio=0.8, split='temporal')

    df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final)

    df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final)

    # Bucketing prefixes based on control flow
    bucketer_args = {'encoding_method': bucket_encoding,
                     'case_id_col': dm.case_id_col,
                     'cat_cols': [dm.activity_col],
                     'num_cols': [],
                     'random_state': random_state}

    bucketer = get_bucketer(bucket_method, **bucketer_args)
    bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
    bucket_assignment_test = bucketer.predict(df_test_prefixes)
    nr_events_all = []
    for bucket in set(bucket_assignment_test):
        relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
        relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
        df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
        test_prfx_len = dm.get_prefix_lengths(df_test_bucket)[0]
        test_y = np.array([dm.get_label_numeric(df_test_bucket)])
        nr_events_all.extend(list(dm.get_prefix_lengths(df_test_bucket)))
        if len(relevant_train_bucket) == 0:
            preds = [dm.get_class_ratio(train)] * len(relevant_test_bucket)
            current_online_times.extend([0] * len(preds))
        else:
            df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
            train_y_experiment = np.array([dm.get_label_numeric(df_train_bucket)])
            prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]
            featureCombinerExperiment = FeatureUnion(
                    [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])

            encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
            ffeatures = featureCombinerExperiment.get_feature_names()
            feat_num = len(ffeatures)
            ffeatures.append('encoded_label')
            encoded_training = np.concatenate((encoded_training,train_y_experiment.T), axis=1)
            training_set_df = pd.DataFrame(encoded_training, columns=ffeatures)
            bkt_size = training_set_df.shape[0]
            encoded_testing_bucket = featureCombinerExperiment.fit_transform(df_test_bucket)
            encoded_testing_bucket = np.concatenate((encoded_testing_bucket,test_y.T), axis=1)
            testing_set_df = pd.DataFrame(encoded_testing_bucket, columns=ffeatures)
            test_bkt_size = testing_set_df.shape[0]
            training_set_df.to_csv(os.path.join(encoded_datasets_dir, 'encoded_training_%s_%s_%s_%s_%s.csv' % (
            dataset_name, method_name, bkt_size, prfx_len, feat_num)), sep=';', columns= ffeatures, index=False)
            with open(outfile, 'w') as out:
                    out.write(
                        '%s;%s;%s;%s;%s;%s\n' % (dataset_name, method_name, 'training', bkt_size, prfx_len, feat_num))
            testing_set_df.to_csv(os.path.join(encoded_datasets_dir, 'encoded_testing_%s_%s_%s_%s_%s.csv' % (
                dataset_name, method_name, test_bkt_size, test_prfx_len, feat_num)), sep=';', columns=ffeatures,
                                      index=False)
            with open(outfile, 'w') as out:
                    out.write('%s;%s;%s;%s;%s;%s\n' % (
                    dataset_name, method_name, 'testing', test_bkt_size, test_prfx_len, feat_num))

