from numpy import percentile
import pandas as pd
import numpy as np
import os
import pickle


def get_list(grps, cols, i):
  """
  a function to retrieve all time values of a certain XAI method, in a set of rows grouped by dataset_name and cls_method
  """
  vals_list = []
  for col in cols:
       vals_list.append(grps[col].apply(list)[i])
  return vals_list
  

def retrieve_dataFrame(grouped_info, method_name, datasets):
  subject_datasets_dict = {"dataset_name":[],"method":[],"bkt_size":[],"prfx_len":[],"feat_num":[]};
  for idx, group in grouped_info:
        if idx[1] == method_name and idx[0] in datasets:
          if (method_name == 'single_agg') or (idx[0] == 'traffic_fines'): 
                gap = 1
          else:
                gap = 5
          for i in range(1, group.shape[0]+1,gap):
              for row_idx, row in group.iterrows():
                if i == row['prefix_length'] and row['bucket_size'] > 500:
                  subject_datasets_dict['dataset_name'].append(idx[0])
                  subject_datasets_dict['method'].append(idx[1])
                  subject_datasets_dict['bkt_size'].append(row['bucket_size'])
                  subject_datasets_dict['prfx_len'].append(row['prefix_length'])
                  subject_datasets_dict['feat_num'].append(row['feature_num'])
  return pd.DataFrame.from_dict(subject_datasets_dict)


def retrieve_datasets_info(dir,datasets_info, datasets, method_name):
  """
  a function to retrieve information about datasets from the all_datasets_info file containing relevant information
  """
  info_df = pd.read_csv(os.path.join(dir,datasets_info),sep=';')
  #to drop rows containing info about training datasets:
  training_info_df = info_df[info_df.dataset_type.str.contains("training")]
  info_df = info_df[~info_df.dataset_type.str.contains("training")]
  info_df.drop(['dataset_type'], inplace=True, axis=1)
  testing_grouped_info = info_df.groupby(['dataset','method'])
  training_grouped_info = training_info_df.groupby(['dataset','method'])
  training_info = retrieve_dataFrame(training_grouped_info, method_name,datasets)
  testing_info = retrieve_dataFrame(testing_grouped_info, method_name, datasets)
  return training_info, testing_info


#a function to retrieve artefacts
def retrieve_artefact(folder,file_end,*argv):
  retrieved_file = retrieve_file(folder,file_end,argv)
  if '.pickle' in file_end:
    with open(retrieved_file, 'rb') as fin:
        retrieved_artefact = pickle.load(fin)
  else:
    retrieved_artefact = pd.read_csv(retrieved_file,sep=';',encoding='ISO-8859-1')
  return retrieved_artefact


#a function to retrieve files of artefacts
def retrieve_file(folder,file_end,argv):
    sep = '_'
    file_name = sep.join([str(a) for a in argv])
    file_name += file_end 
    return os.path.join(folder, file_name)


def get_corr_files(EDA_output,dataset_name, method_name, cls_method, feat_num):
    out= 'correlations_%s_%s_%s_%s' %(dataset_name, method_name, cls_method, feat_num)
    corr_dir = os.path.join(EDA_output, out)
    cat_csv = os.path.join(corr_dir, 'Categorical_correlations_%s_%s_%s.csv' % (dataset_name, method_name, feat_num))
    num_csv = os.path.join(corr_dir, 'Numerical_correlations_%s_%s_%s.csv' % (dataset_name, method_name, feat_num))
    files = [cat_csv, num_csv]
    return files

