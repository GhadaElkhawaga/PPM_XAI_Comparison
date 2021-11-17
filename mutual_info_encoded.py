import pandas as pd #using pandas==0.25
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import io
from sklearn.pipeline import FeatureUnion, Pipeline
import warnings
warnings.simplefilter('ignore')
import csv
from helpers.Encoders import get_encoder
from helpers.Bucketers import get_bucketer
from helpers.DatasetManager import DatasetManager
import Definitions
models_dir = 'model_and_hdf5'
datasets_info_file = os.path.join(models_dir, 'all_datasets_info.csv')
output_dir = 'mutual_information'
if not os.path.exists(output_dir):
  os.makedirs(os.path.join(output_dir))
encoded_datasets_mutual_info = os.path.join(output_dir, 'after_encoding')
if not os.path.exists(encoded_datasets_mutual_info):
  os.makedirs(os.path.join(encoded_datasets_mutual_info))
original_datasets_mutual_info = os.path.join(output_dir, 'before_encoding')
if not os.path.exists(original_datasets_mutual_info):
  os.makedirs(os.path.join(original_datasets_mutual_info))
cat_cols = []
random_state = 22
datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"hospital_billing_1","hospital_billing_2", "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"]
single_list = ["hospital_billing_1","hospital_billing_2", "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled"]

def mutual_info_comp(out_dir,df,target,disc_indices,feats,d_name,method='original',prfx_len=0):
    mi = mutual_info_classif(df,target,discrete_features=disc_indices,random_state=42)
    concat_res = np.column_stack((feats,mi))
    concat_df = pd.DataFrame(concat_res, columns=['features','mi'])
    concat_df['mi'] = pd.to_numeric(concat_df['mi'])
    concat_df.sort_values(by=['mi'], ascending=False, inplace=True)
    concat_df= concat_df.reset_index()
    ax = concat_df.iloc[:21].plot(y='mi',kind='bar',color='DarkBlue', legend=False)
    ax.set_xticklabels(concat_df.features[:21])
    plt.savefig(os.path.join(out_dir, 'Mutual_imp_%s_%s_%s_%s_%s.png' % (d_name,method,df.shape[0],prfx_len,len(feats))), dpi=300, bbox_inches='tight');
    plt.clf();

for dataset_name in datasets:
    dm = DatasetManager(dataset_name)
    df = dm.read_dataset()
    df_before_enc = df.copy()

    cat_cols = dm.static_cat_cols + dm.dynamic_cat_cols
    df_before_enc[cat_cols] = df_before_enc[cat_cols].apply(lambda x: x.astype('category').cat.codes)
    discrete_feats_indices = [df_before_enc.columns.tolist().index(col) for col in cat_cols]
    target = df_before_enc[dm.label_col].astype('category').cat.codes
    df_before_enc.drop([dm.label_col,dm.case_id_col,dm.timestamp_col],axis=1, inplace=True)
    if 'lifecycle:transition' in df_before_enc.columns:
        df_before_enc.drop(['lifecycle:transition'],inplace=True, axis=1)
    
    mutual_info_comp(original_datasets_mutual_info,df_before_enc,target,discrete_feats_indices,df_before_enc.columns.tolist(),dataset_name)
    for method in ['single_agg', 'prefix_index']:
          if method == 'prefix_index':
                gap = 5
          else:
                gap = 1
          bucket_method , cls_encoding = method.split('_')
          if bucket_method == "state":
              bucket_encoding = "last"
          else:
              bucket_encoding = "agg"
          encoding_dict = {
                        "agg": ["static", "agg"],
                        "index": ["static", "index"]}
          methods = encoding_dict[cls_encoding]
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
          if not (any([x in dataset_name for x in single_list]) and method == 'prefix_index'):
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
           test_y_all = []
           for bucket in set(bucket_assignment_test):
              relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
              relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
              df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
              test_y = dm.get_label_numeric(df_test_bucket)
              test_prfx_len = dm.get_prefix_lengths(df_test_bucket)[0]
              for i in range(1, max_prefix_length_final+1 , gap):
                  if i == test_prfx_len:
                      if len(relevant_train_bucket) == 0:
                           preds = [dm.get_class_ratio(train)] * len(relevant_test_bucket)
                           current_online_times.extend([0] * len(preds))
                      else:
                           df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
                           train_y_experiment = dm.get_label_numeric(df_train_bucket)
                           prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]

                           if len(set(train_y_experiment)) < 2:
                               preds = [train_y_experiment[0]] * len(relevant_train_bucket)
                               current_online_times.extend([0] * len(preds))
                               test_y_all.extend(dm.get_label_numeric(df_test_prefixes))
                           else:
                                 featureCombinerExperiment = FeatureUnion(
                                       [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])
                                 encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
                                 ffeatures = featureCombinerExperiment.get_feature_names()
                                 #to get the indices of discrete features:
                                 discrete_feat_indices_encoded = [ffeatures.index(feat) for feat in ffeatures if any(col in feat for col in cat_cols)]
                                 feat_num = len(ffeatures)
                                 training_set_df = pd.DataFrame(encoded_training, columns=ffeatures)
                                 bkt_size = training_set_df.shape[0]
                                 encoded_testing = featureCombinerExperiment.fit_transform(df_test_bucket)
                                 testing_set_df = pd.DataFrame(encoded_testing, columns=ffeatures)
                                 test_bkt_size = testing_set_df.shape[0]
                                 encoded_df = pd.concat([training_set_df, testing_set_df], ignore_index=True)
                                 encoded_target = train_y_experiment + test_y
                                 mutual_info_comp(encoded_datasets_mutual_info,encoded_df,encoded_target,discrete_feat_indices_encoded,ffeatures,dataset_name,method,prfx_len)



                
