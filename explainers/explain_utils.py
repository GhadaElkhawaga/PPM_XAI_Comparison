import os
import pandas as pd
import matplotlib.pyplot as plt
from lime_stability.stability import LimeTabularExplainerOvr
from lime.lime_tabular import LimeTabularExplainer
import shap


def explain_and_save(dir_name,frmt_str,i,explainer,sample,feats,model, flag,mod):
  expparams = {"data_row": sample,"predict_fn": model.predict_proba,
                     "num_features": feats,"distance_metric": "euclidean"}
  explanation = explainer.explain_instance(**expparams)
  if mod == 'logit':
    m = 'logit'
  else:
    m = 'xgboost' 
  title = "explanation_%s_%s_cluster{%s}_%s.html" % (m, flag,i, frmt_str)
  explanation.save_to_file(os.path.join(dir_name, title))
  return
  
  

def shap_plots(exp_val, shap_vals_sample,sample,feats,dir_name,model_name,frmt_str,i,flag):
  """
  a function to plot force and decision plots for distant and nearest samples in a cluster
  '"""
  shap.force_plot(exp_val, shap_vals_sample, show=False, matplotlib=True)
  fig_name = 'force_plot'
  save_plot(dir_name,model_name,fig_name,frmt_str,i,flag)

  shap.decision_plot(exp_val, shap_vals_sample, sample,feature_names=feats, show=False)
  fig_name = 'decision_plot'
  save_plot(dir_name,model_name,fig_name,frmt_str,i,flag)
  return


def save_plot(dir_name,model_name,fig_name,frmt_str,i,flag):
    """
    a function to save generated plots
    """
    plt.savefig(os.path.join(dir_name,'%s_%s_%s_cluster(%s)_%s.png' % (fig_name,model_name,flag,i, frmt_str)),\
                            dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    return

