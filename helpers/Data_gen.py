import pandas as pd
import numpy as np
import os


#to generate the df of instances used in shap explainer for a logit model
def generate_instances_df(df_bucket, dm, ffeatures, encoder, inst_count):
  buckets_grouped = df_bucket.groupby(dm.case_id_col)
  cases_count = 0
  gen_df = pd.DataFrame(columns=ffeatures)
  for idx, grp in buckets_grouped:
      cases_count += 1
      if cases_count <= inst_count :
          encoded_group = encoder.fit_transform(grp)
          case = np.transpose(encoded_group[0])
          gen_df.loc[idx] = case
      else:
          break
  return gen_df


#to get the percentage of instances with the same class
def get_percentage(label, X):
  return X.count(label)/len(X)

