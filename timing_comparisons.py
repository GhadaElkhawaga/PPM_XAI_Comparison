import pandas as pd #using pandas==0.25
import numpy as np
import sys
import os
import pickle
import io
import warnings
warnings.simplefilter('ignore')
import csv
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import math
from Data_retrieval import retrieve_artefact, get_list
from plots_annot import plot, autolabel

#Defining Basic parameters
method_name = 'single_agg'
#gap = 1
#cls_method = 'logit'

#method_name = 'prefix_index'
gap = 1
#cls_method = 'xgboost'

dataset_ref_to_datasets = {
    "sepsis_cases": ["sepsis1", "sepsis2", "sepsis3"],
    "bpic2017" :["BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"],
    "Hospital_Billing" :["hospital_billing_%s" %i for i in range(1,3)]
}
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
ALE_dir = 'ALE_artefacts'
Perm_dir = 'Perm_artefacts'
shap_dir = 'shap_artefacts'
pred_perf_dir = 'performance_files'
if not os.path.exists(pred_perf_dir):
  os.makedirs(os.path.join(pred_perf_dir))
comparisons_dir = 'timing_comparisons'
if not os.path.exists(comparisons_dir):
  os.makedirs(os.path.join(comparisons_dir))
saved_exec_logs = 'output_text_files'
if not os.path.exists(saved_exec_logs):
  os.makedirs(os.path.join(saved_exec_logs))
saved_artefacts = 'model_and_hdf5'
training_df_dict = {"dataset_name":[],"method":[],"cls_method":[],"bkt_size":[],"prfx_len":[],"feat_num":[],"prediction_time":[],"shap_time":[],"ale_time":[],"perm_time":[]}
for method_name in ['prefix_index', 'single_agg']:
  timings_dir = os.path.join(comparisons_dir,'%s' %(method_name))
  if not os.path.exists(timings_dir):
    os.makedirs(timings_dir)
  if method_name == 'single_agg':
    datasets = ["sepsis1", "sepsis2", "sepsis3", 'traffic_fines', "hospital_billing_1", "hospital_billing_2",
                "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"]
  else:
    datasets = ["sepsis1", "sepsis2", "sepsis3", 'traffic_fines', "BPIC2017_O_Refused"]
  for dataset_name in datasets:
      training_dataset_info = retrieve_artefact(saved_artefacts,'all_datasets_info.csv')
      training_dataset_info = training_dataset_info[~training_dataset_info.dataset_type.str.contains("testing")]
      training_dataset_info.drop(['dataset_type'], inplace=True, axis=1)
      training_info = training_dataset_info.groupby(['dataset','method'])
      for idx, group in training_info:
        if idx[1] == method_name and idx[0] == dataset_name:
          if (method_name == 'single_agg') or (idx[0] == 'traffic_fines'): 
                gap = 1
          else:
                gap = 5
          for i in range(1, group.shape[0]+1,gap):
              for row_idx, row in group.iterrows():
                if i == row['prefix_length'] and method_name == idx[1]:
                  bkt_size = row['bucket_size']
                  prfx_len = row['prefix_length']
                  feat_num = row['feature_num']
                  prediction_times_df = retrieve_artefact(pred_perf_dir,'.csv','prediction_timing', method_name)
                  for cls_method in ['logit', 'xgboost']:
                    shap_df = retrieve_artefact(shap_dir,'.csv','shap_values', cls_method, dataset_name,method_name,'training',bkt_size,prfx_len,feat_num)
                    ALE_df = retrieve_artefact(ALE_dir,'.csv','ALE_pred_explainer', cls_method, dataset_name,method_name,bkt_size,prfx_len)
                    Perm_df = retrieve_artefact(Perm_dir,'.csv','permutation_importance', dataset_name,cls_method,method_name,bkt_size,prfx_len,feat_num, 'final')
                    subject_rows = prediction_times_df.loc[(prediction_times_df['Dataset']==dataset_name) & \
                                                              (prediction_times_df['method_name']==method_name) & \
                                                              (prediction_times_df['cls_method']==cls_method) & \
                                                              (prediction_times_df['prfx_len']==prfx_len)] 
                    n_iter = len(np.delete(subject_rows['n_iter'].unique(),np.where(subject_rows['n_iter'].unique()==-1)))
                    train_time_rows = subject_rows.iloc[:n_iter]
                    train_time_rows['train_time'] = train_time_rows['train_time'].apply(lambda x: float(x))
                    try:
                      train_time = np.average(train_time_rows['train_time'])
                    except:
                      train_time = 0
                    training_df_dict['dataset_name'].append(idx[0])
                    training_df_dict['method'].append(idx[1])
                    training_df_dict['cls_method'].append(str(cls_method))
                    training_df_dict['bkt_size'].append(row['bucket_size'])
                    training_df_dict['prfx_len'].append(row['prefix_length'])
                    training_df_dict['feat_num'].append(row['feature_num'])
                    training_df_dict['shap_time'].append(shap_df.iloc[bkt_size,1])
                    training_df_dict['ale_time'].append(ALE_df.iloc[-1,1])
                    training_df_dict['perm_time'].append(Perm_df.iloc[feat_num,1])
                    training_df_dict['prediction_time'].append(train_time)
                    del shap_df, ALE_df, Perm_df

  training_df = pd.DataFrame.from_dict(training_df_dict)
  df = training_df[(training_df['method']==method_name)].drop(['method'], axis=1)
  latex_file = os.path.join(timings_dir,'table_%s.tex' %(method_name))
  with open(latex_file,'w') as tf:
     tf.write(df.to_latex(index=False))
  df = df.drop(['bkt_size', 'feat_num'],axis=1)
  df_grouped = df.groupby(['dataset_name'], as_index=False)
  labels = ['pred', 'shap', 'ale', 'perm']
  cols = ['prfx_len', 'cls_method','prediction_time','shap_time','ale_time','perm_time']
  for _,grp in df_grouped:        
      grouped_grps = grp.groupby(['cls_method'], as_index=False)
      ds_name = grp['dataset_name'].iloc[0]
      if method_name == 'prefix_index':
        for i in range(len(grouped_grps['cls_method'].apply(list))): 
          lists = get_list(grouped_grps,cols,i)  
          prfx_list, cls_list, pred_list, shap_list, ale_list, prem_list = lists[0], lists[1], lists[2], lists[3], lists[4], lists[5]
          cls = cls_list[0]
          markers = ['o', 's', 'd', 'x']
          global_min, global_max = 0, 0
          ax = plt.axes()
          #plotting a line for each (XAI method_cls_method) 
          for j in range(4):
            y = np.array(lists[j+2], dtype=float)
            current_min, current_max = min(y), max(y)
            plot(prfx_list, y, labels[j], cls, markers[j])
            if current_min < global_min:
              global_min = current_min
            if current_max > global_max:
              global_max = current_max
          plt.legend(loc = 8, bbox_to_anchor=[0.5,1], fontsize=4, ncol=4)
          plt.xticks(fontsize=8)
          plt.xlabel(xlabel='Prefix lengths',fontsize=10)
          plt.ylabel(ylabel='Execution Times',fontsize=10)
        plt.savefig(os.path.join(timings_dir,'XAI_timings_%s.png'%(ds_name)), dpi=300, bbox_inches='tight');
        plt.clf();
      else:
        #plot a barplot grouped by XAI method
        grp = grp.reset_index()
        grp = grp.drop(['index'],axis=1)
        x = np.arange(4)  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        x_locs = [x - width/2, x + width/2]
        for i in range(len(grp['cls_method'])):
          y = np.array(grp.iloc[i,3:], dtype=float)
          rects = ax.bar(x_locs[i], y, width, label=grp['cls_method'][i])
          autolabel(ax, rects)
        ax.set_ylabel('Execution Times')
        ax.set_title('XAI Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend()
        plt.savefig(os.path.join(timings_dir,'XAI_timings_%s.png'%(ds_name)), dpi=300, bbox_inches='tight');
        plt.clf();
        #plot a barplot grouped by cls_method
        current_labels =grp['cls_method']
        width = 0.1
        x = np.arange(len(grp))    
        x_locs_cls = [x - (3*width/2), x - width/2, x+width/2, x + (3*width/2)]
        fig, ax = plt.subplots()
        colors = ['firebrick','mediumblue','green','darkorange']
        for i in range(len(grp)): 
          lists = get_list(grouped_grps,cols,i)  
          prfx_list, cls_list, pred_list, shap_list, ale_list, prem_list = lists[0], lists[1], lists[2], lists[3], lists[4], lists[5]
          for j in range(4):
            y = np.array(lists[j+2], dtype=float)
            rects = ax.bar(x_locs_cls[j][i], y, width, label=grp.columns[j+3], color=colors[j])
            autolabel(ax, rects)
        handles, ls = ax.get_legend_handles_labels()
        ax.legend(handles[:4], labels[:4])
        ax.set_ylabel('Execution Times')
        ax.set_xticks(x)
        ax.set_xticklabels(current_labels, fontsize=8)
        plt.savefig(os.path.join(timings_dir,'XAI_timings_%s_classifier.png'%(ds_name)), dpi=300, bbox_inches='tight');
        plt.clf();
del training_df_dict, training_df

