import pandas as pd #using pandas==0.25
import numpy as np
import sys
import os
import pickle
import io
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb #using xgboost==0.9
import warnings
warnings.simplefilter('ignore')
import csv
import shap #using shap==0.36
from lime.lime_tabular import LimeTabularExplainer
#import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
from explainers.explain_utils import explain_and_save, shap_plots, save_plot
from helpers.Data_retrieval import retrieve_artefact, retrieve_dataFrame, retrieve_datasets_info
from helpers.clustering_params import compute_best_params


artefacts_dir = 'clustering_local_comparison'
if not os.path.exists(artefacts_dir):
  os.makedirs(os.path.join(artefacts_dir))
LIME_local = os.path.join(artefacts_dir,'LIME_local')
if not os.path.exists(LIME_local):
  os.makedirs(os.path.join(LIME_local))
SHAP_local = os.path.join(artefacts_dir,'SHAP_local')
if not os.path.exists(SHAP_local):
  os.makedirs(os.path.join(SHAP_local))
saved_artefacts = 'model_and_hdf5'
encoded_datasets_dir = 'encoded_datasets'
saved_exec_logs = 'output_text_files'
if not os.path.exists(saved_exec_logs):
  os.makedirs(os.path.join(saved_exec_logs))
output_clusters = open(os.path.join(saved_exec_logs,'output_clusters.txt'), 'w')
clusters_scores =  os.path.join(artefacts_dir,"clusters_scores.csv")
with open(clusters_scores, "w+") as fout:
          header = ['dataset', 'method', 'bkt_size', 'prfx_len', 'feat_num', 'n_clusters','Silhouette_scores']
          writer = csv.DictWriter(fout, delimiter=';', lineterminator='\n', fieldnames=header)
          writer.writeheader()
shap_dir = 'shap_artefacts'
if not os.path.exists(shap_dir):
  os.makedirs(os.path.join(shap_dir))
for method_name in ['single_agg', 'prefix_index']:
  if method_name == 'single_agg':
          datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"hospital_billing_1","hospital_billing_2", "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"]
  else:
          datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"BPIC2017_O_Refused"]
  subject_datasets_df_training, subject_datasets_df_testing = retrieve_datasets_info(saved_artefacts,'all_datasets_info.csv', datasets, method_name)
  #to retrieve the best n_clusters for each dataset and cluster its instances accordingly
  #compute_best_params(clusters_scores, encoded_datasets_dir,datasets,method_name,subject_datasets_df_testing)
  compute_best_params(clusters_scores, encoded_datasets_dir,datasets,method_name,subject_datasets_df_training)
  dataset_clusters = pd.read_csv(clusters_scores ,sep=';')
  for dataset_name in datasets:
        row = dataset_clusters.loc[(dataset_clusters['dataset']==dataset_name) & (dataset_clusters['method']==method_name)]
        if (not row.empty):
            grouped_row = row.groupby([row.columns[0],row.columns[1],row.columns[2],row.columns[3],row.columns[4]], as_index=False)
            for _,grp in grouped_row:
              dataset_training_df = retrieve_artefact(encoded_datasets_dir,'.csv','encoded_training',dataset_name, grp.iloc[0,1],grp.iloc[0,2],grp.iloc[0,3], grp.iloc[0,4])
              sep = '_'
              f = sep.join(['encoded_training',dataset_name, str(grp.iloc[0,1]), str(grp.iloc[0,2]),str(grp.iloc[0,3]), str(grp.iloc[0,4]),'.csv'])
              LIME_local_dir = os.path.join(LIME_local,'%s_%s_%s_%s_%s' %(dataset_name, str(grp.iloc[0,1]), str(grp.iloc[0,2]),str(grp.iloc[0,3]), str(grp.iloc[0,4])))
              if not os.path.exists(LIME_local_dir):
                os.makedirs(os.path.join(LIME_local_dir))
              SHAP_local_dir = os.path.join(SHAP_local,'%s_%s_%s_%s_%s' %(dataset_name, str(grp.iloc[0,1]), str(grp.iloc[0,2]),str(grp.iloc[0,3]), str(grp.iloc[0,4])))
              if not os.path.exists(SHAP_local_dir):
                os.makedirs(os.path.join(SHAP_local_dir))
              # load shap values files and explainers for both classifiers
              frmt_str = '%s_%s_%s_%s_%s_%s' % (dataset_name, method_name, 'training', grp.iloc[0,2],grp.iloc[0,3], grp.iloc[0,4])
              shap_explainer_logit = retrieve_artefact(shap_dir, '.pickle', 'shap_explainer_logit', frmt_str) 
              expected_value_logit = shap_explainer_logit.expected_value   
              shap_explainer_xgboost = retrieve_artefact(shap_dir, '.pickle', 'shap_explainer_xgboost', frmt_str)
              expected_value_xgboost = shap_explainer_xgboost.expected_value
              shap_values_xgboost = retrieve_artefact(shap_dir, '.pickle', 'shap_values_xgboost', frmt_str)
              shap_values_logit = retrieve_artefact(shap_dir, '.pickle', 'shap_values_logit', frmt_str)
              #retrieve here classifiers
              ffeatures = retrieve_artefact(saved_artefacts, '.pickle', 'ffeatures', 'xgboost', dataset_name, method_name,
                                            grp.iloc[0,2],grp.iloc[0,3])
              logit_model = retrieve_artefact(saved_artefacts, '.pickle', 'model', 'logit', dataset_name, method_name,grp.iloc[0,2],grp.iloc[0,3],grp.iloc[0,4])
              xgboost_model = retrieve_artefact(saved_artefacts, '.pickle', 'model', 'xgboost', dataset_name, method_name,grp.iloc[0,2],grp.iloc[0,3], grp.iloc[0,4])
              #call here the Lime explainer on the encoded dataset
              if dataset_name == 'sepsis3':
                pos_label = "regular"
                neg_label = "deviant"
              else:
                pos_label = "deviant"
                neg_label = "regular"
              limeexplainer = LimeTabularExplainer(dataset_training_df.to_numpy(),feature_names=ffeatures,class_names=[neg_label,pos_label],discretize_continuous=True)
              best_n_clusters = grp.loc[grp['Silhouette_scores']==grp['Silhouette_scores'].max()]['n_clusters'].iloc[0]
              scaler = StandardScaler()
              scaled_dataset = pd.DataFrame(scaler.fit_transform(dataset_training_df),columns=dataset_training_df.columns)
              clusterer = KMeans(n_clusters=best_n_clusters, random_state=42)
              clusters_labels = clusterer.fit_predict(scaled_dataset)
              labels_df = pd.DataFrame(clusterer.labels_)
              #concatenating the original dataset with labels generated for the scaled dataset
              labelled_dataset = pd.concat((dataset_training_df,labels_df),axis=1)
              labelled_dataset.rename(columns={labelled_dataset.columns[-1]: "cluster_label" }, inplace = True)
              scaled_dataset = pd.concat((scaled_dataset,labels_df),axis=1)
              scaled_dataset.rename(columns={scaled_dataset.columns[-1]: "cluster_label" }, inplace = True)
              clusters_centers = clusterer.cluster_centers_
              for col in scaled_dataset:
                scaled_dataset[col] = pd.to_numeric(scaled_dataset[col], downcast='float')
              distances_df = pd.DataFrame(columns=['cluster_label', 'distance'])
              distances_df['cluster_label'] = scaled_dataset['cluster_label']
              for i, centroid in enumerate(clusters_centers):
                distances = [np.linalg.norm(x[1]-centroid) for x in scaled_dataset.loc[scaled_dataset['cluster_label'] == i].drop(['cluster_label'],axis=1).iterrows()]
                distances_df.loc[(scaled_dataset['cluster_label']==i) & (distances_df.index==scaled_dataset.index),'distance'] = distances
              scaled_dataset['distances'] = distances_df['distance']
              labelled_dataset['distances'] = distances_df['distance'] 
              del distances_df
              for n in range(best_n_clusters):
                #get the closest and the farest samples from the centroid for each cluster in the same dataset
                cluster_block = labelled_dataset.loc[labelled_dataset['cluster_label']==n]
                cluster_block.sort_values(by=['distances'], inplace=True)
                nearest_sample = cluster_block.iloc[0,:-2]                             
                distant_sample_row = cluster_block.shape[0]-1
                distant_sample_col = cluster_block.shape[1]-2
                distant_sample = cluster_block.iloc[distant_sample_row,:distant_sample_col]
                while distant_sample.isnull().values.any():
                  distant_sample_row -= 1
                  distant_sample = cluster_block.iloc[distant_sample_row,:distant_sample_col]
                #get the index of the nearest and distant samples
                idx_nearest = nearest_sample.name
                idx_distant = distant_sample.name
                #explain them with lime for both classifiers
                explain_and_save(LIME_local_dir,frmt_str,n,limeexplainer,nearest_sample,6,logit_model,'nearest', 'logit')
                explain_and_save(LIME_local_dir,frmt_str,n,limeexplainer,distant_sample,6,logit_model,'distant','logit')
                explain_and_save(LIME_local_dir,frmt_str,n,limeexplainer,nearest_sample,6,xgboost_model,'nearest','xgboost')
                explain_and_save(LIME_local_dir,frmt_str,n,limeexplainer,distant_sample,6,xgboost_model,'distant','xgboost')
                #plotting explainations of both samples with shap for both classifiers
                #plotting force_plots and decision_plots of the nearest and distant samples from explanations of logit
                shap_plots(expected_value_logit, shap_values_logit[idx_nearest],dataset_training_df.loc[idx_nearest],ffeatures,SHAP_local_dir,'logit',frmt_str,n,'nearest')
                shap_plots(expected_value_logit, shap_values_logit[idx_distant],dataset_training_df.loc[idx_distant],ffeatures,SHAP_local_dir,'logit',frmt_str,n,'distant')
                shap_plots(expected_value_xgboost, shap_values_xgboost[idx_nearest],dataset_training_df.loc[idx_nearest],ffeatures,SHAP_local_dir,'xgboost',frmt_str,n,'nearest')
                shap_plots(expected_value_xgboost, shap_values_xgboost[idx_distant],dataset_training_df.loc[idx_distant],ffeatures,SHAP_local_dir,'xgboost',frmt_str,n,'distant')
                del cluster_block

