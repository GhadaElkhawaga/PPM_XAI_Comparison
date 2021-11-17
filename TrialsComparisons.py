import pandas as pd
import numpy as np
import os
import csv


models_dir = 'model_and_hdf5'
datasets_info_file = os.path.join(models_dir, 'all_datasets_info.csv')
TrialsComparisons = os.path.join('TrialsComparisons')
if not os.path.exists(TrialsComparisons):
  os.makedirs(os.path.join(TrialsComparisons))
datasetrefs = ['hospital', 'traffic', 'sepsis', 'bpic']
datasets_dict = {'hospital': ['hospital_billing1','hospital_billing2'], 'traffic':['traffic_fines'], 'sepsis':['sepsis1','sepsis2','sepsis3'], 'bpic':['BPIC2017_O_Accepted','BPIC2017_O_Cancelled','BPIC2017_O_Refused']}
files = []
info_df = pd.read_csv(datasets_info_file,sep=';')
#to drop rows containing info about training datasets:
training_info_df = info_df[info_df.dataset_type.str.contains("training")]
training_info_df.drop(['dataset_type'], inplace=True, axis=1)
training_grouped_info = training_info_df.groupby(['dataset','method'])
for idx, group in training_grouped_info:
    for row_idx, row in group.iterrows():
        d_name, method, bkt_size, prfx_len, feat_num = idx[0], idx[1], row['bucket_size'], row['prefix_length'], row['feature_num']
        if method == 'prefix_index' and d_name not in ['hospital_billing_1', 'hospital_billing_2', 'BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled']:
           gap = 5
           for i in range(1, group.shape[0]+1, gap):
               if i == prfx_len:
                   f_name = '%s_%s_%s_%s_%s' %(d_name, method, bkt_size, prfx_len, feat_num)
                   files.append(f_name)
        elif method == 'single_agg':
            f_name = '%s_%s_%s_%s_%s' %(d_name, method, bkt_size, prfx_len, feat_num)
            files.append(f_name)
headers = {'logit':['Details','Type','Variable','Score', 'Trial'], 'xgboost':['Details','Type1','Variable1','Score1','Details','Type2' 'Trial1','Variable2','Score2', 'Trial2']}
feats_num = {'all':5, 'case_1':2, 'case_2':4}
for cls in ['logit', 'xgboost']:
   for method in ['single_agg','prefix_index']:
      comparison_file = os.path.join(TrialsComparisons,'comparisons_%s_%s.csv' %(cls,method))
      with open(comparison_file, 'w+') as comp:
        header = headers[cls]
        writer = csv.DictWriter(comp, delimiter=';', lineterminator='\n', fieldnames=header)
        writer.writeheader()
      for f in files:
          if method in f:
             if cls == 'logit':
                  comparison_df = pd.DataFrame(columns= headers[cls])
                  f1 = os.path.join('sepsis_cases_%s_%s_gap_1' %(method,cls), 'logit_coef_%s'%(f),'logit_coefficients_%s.csv' %(f))
                  f2 = os.path.join('sepsis_cases_%s_%s_gap_2' %(method,cls), 'logit_coef_%s'%(f),'logit_coefficients_%s.csv' %(f))
                  df1 = pd.read_csv(f1,sep=';')
                  df2 = pd.read_csv(f2,sep=';')
                  comparison_df['Score'] = pd.concat((df1.iloc[:5,0],df2.iloc[:5,0]),axis=0)
                  comparison_df['Variable'] = pd.concat([df1.iloc[:5,1],df2.iloc[:5,1]],axis=0)
                  comparison_df['Details'] = f
                  comparison_df['Trial'] = ['trial1' for i in range(5)] + ['trial2' for i in range(5)]
                  comparison_df['Type'] = ['Coefficient' for i in range(10)]
                  with open(comparison_file, 'a') as comp:
                    comparison_df.to_csv(comp, header=False,sep=';', index=False)
             else:
                  compare = pd.DataFrame(columns= headers[cls])
                  f1 = os.path.join('sepsis_cases_%s_%s_gap_1' %(method,cls), 'global_Importance_xgboost_%s.csv' %(f))
                  f2 = os.path.join('sepsis_cases_%s_%s_gap_2' %(method,cls), 'global_Importance_xgboost_%s.csv' %(f))
                  df1 = pd.read_csv(f1,sep=';', header=0, names=['Dataset','ImpType','Feature','Score','Time'])
                  df2 = pd.read_csv(f2,sep=';', header=0, names=['Dataset','ImpType','Feature','Score','Time'])
                  df1_grouped = df1.groupby(['ImpType']).apply(lambda x: x.sort_values(by='Score',ascending=False))
                  df2_grouped = df2.groupby(['ImpType']).apply(lambda x: x.sort_values(by='Score',ascending=False))
                  importance_types = list(set(df1['ImpType']))
                  comparison_df1 = pd.DataFrame(columns= ['Details','Type','Variable','Score', 'Trial'])
                  comparison_df2 = pd.DataFrame(columns= ['Details','Type','Variable','Score', 'Trial'])
                  results = []
                  try:               
                    for imp in importance_types:
                        compact_df = pd.DataFrame(columns= headers[cls])
                        comparison_df1['Score'] = df1_grouped[df1_grouped['ImpType']==imp]['Score'].head(4).reset_index(drop=True)    
                        comparison_df2['Score'] = df2_grouped[df2_grouped['ImpType']==imp]['Score'].head(4).reset_index(drop=True)
                        comparison_df1['Variable'] = df1_grouped[df1_grouped['ImpType']==imp]['Feature'].head(4).reset_index(drop=True)
                        comparison_df2['Variable'] = df2_grouped[df2_grouped['ImpType']==imp]['Feature'].head(4).reset_index(drop=True)
                        comparison_df1['Details'] = comparison_df2['Details']= f
                        comparison_df1['Trial'] = ['1' for i in range(4)]
                        comparison_df2['Trial'] = ['2' for i in range(4)]
                        comparison_df1['Type'] = comparison_df2['Type'] = [imp for i in range(4)]
                        compact_df = pd.concat([comparison_df1,comparison_df2], axis=1)
                        results.append(compact_df)
                  except:
                      print('less features are available for comparison')
                  try:
                    compare = pd.concat(results, axis=0)
                    with open(comparison_file, 'a') as comp:
                         compare.to_csv(comp, header=False,sep=';', index=False)
                  except:
                    print('There were no features in this log file')

