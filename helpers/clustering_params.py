import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_samples, silhouette_score
from helpers.Data_retrieval import retrieve_artefact


def compute_best_params(clusters_scores,encoded_datasets_dir,datasets,method_name,subject_datasets_df):
    """
    a function to compute the average Silhouette score with different cluster numbers
    inputs: dataset names, method name, directory of encoded datasets, scores file, a dataframe with information about subject datasets
    outputs: a csv file
    """
    best_params_df = pd.DataFrame(columns=['dataset', 'method', 'bkt_size', 'prfx_len', 'feat_num', 'n_clusters','Silhouette_scores'])
    for row in subject_datasets_df.itertuples():
            if row[2] == method_name and row[1] in datasets:
              dataset_df = retrieve_artefact(encoded_datasets_dir,'.csv','encoded_training',row[1], row[2], row[3],row[4], row[5])
              if dataset_df.isnull().values.any():
                  dataset_df.fillna(0)
              range_n_clusters = [2, 3, 4, 5, 6]
              for cluster_n in range_n_clusters:
                clusterer = KMeans(n_clusters=cluster_n, random_state=42)
                cluster_labels = clusterer.fit_predict(dataset_df)
                silhouette_avg = silhouette_score(dataset_df, cluster_labels)
                best_params_df = best_params_df.append(pd.DataFrame([[row[1], row[2], row[3],row[4],row[5], cluster_n, silhouette_avg]], columns=best_params_df.columns))
    best_params_df.to_csv(clusters_scores, mode='a', header=False,sep=';', index=False)
    return