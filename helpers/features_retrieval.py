from numpy import percentile
import pandas as pd
import numpy as np
import os
import pickle
from helpers.Data_retrieval import retrieve_artefact


#a function to get highly correlated features (together or with the target):
def get_correlated_features(corr_df,threshold,corr_type,output_file):
    if corr_type == 'others':
          #this dictionary represents features with correlation >= threshold with the other features:
          corr_artefact = {}
          try:
            for cor in corr_df.columns[1:]:
                l = corr_df.index[(abs(corr_df[cor])>=threshold)].tolist() #to get features highly correlated with others
                corr_artefact[cor] = l
            
            #to remove the feature itself and the target from highly correlated list of features
            tmp = corr_artefact.copy()
            for k,v in tmp.items():
                if (k in v):
                  v.remove(k)
                if  ('encoded_label' in v):
                  v.remove('encoded_label')

            corr_artefact = tmp.copy()
            for k in tmp.keys():
                if not tmp[k]:
                  corr_artefact.pop(k,None)
          except ValueError:
             pass
    
    else:
        #this list represents features with correlation >= threshold with the target:
        corr_artefact = [] 
        try:
            corr_artefact = corr_df.index[abs(corr_df[corr_df.columns[-1]])>=threshold].tolist() #to get features highly correlated with the target
            for col in corr_artefact:
              if 'label' in col:
                corr_artefact.remove(col)
        except:
            pass
    return corr_artefact
   

#a function to get features to be plotted
def get_important_features(folder,compared_feat,dataset_name, method_name, bkt_size, prfx_len,feat_num, gap):
    frmt_str = '%s_%s_%s_%s' %(dataset_name, method_name, bkt_size, prfx_len)
    ffeatures = retrieve_artefact(folder,'.pickle','ffeatures','xgboost',frmt_str)
    mapper = {'f{0}'.format(i):ffeatures[i] for i in range(0, len(ffeatures))}
    #get the feature with the highest importance in the xgboost model:
    xgb_model = retrieve_artefact(folder,'.pickle','model','xgboost',frmt_str,feat_num)
    xgb_scores = xgb_model.get_booster().get_score(importance_type='gain')
    mapped_df = pd.DataFrame(list({mapper[k]: v for k, v in xgb_scores.items()}.items()),columns = ['Feature','Importance'])
    mapped_df.sort_values(by=['Importance'], ascending=False, inplace=True)
    mapped_df = mapped_df.reset_index(drop=True)
    #get the feature with the highest coefficient in the logit model:
    logit_model = retrieve_artefact(folder,'.pickle','model','logit',frmt_str,feat_num)
    coefs = logit_model.coef_.reshape(-1, 1)
    coefs_df = pd.DataFrame(coefs, columns=['coef']).astype(float)
    coefs_df['Variable'] = ffeatures
    coefs_df.sort_values(by=['coef'], ascending=False, inplace=True)
    coefs_df = coefs_df.reset_index(drop=True)
    try:
        if compared_feat == 1:
          return list(set([mapped_df.iloc[0,0], coefs_df.iloc[0,1]]))
        else:
          return mapped_df.iloc[0:compared_feat,0].append(coefs_df.iloc[0:compared_feat,1], ignore_index=True)
    except:
        if mapped_df.empty or coefs_df.empty:
            return 'quit'

