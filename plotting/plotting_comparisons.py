import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import shap


def plot_shap_dependence(SHAP_comparisons,frmt_str, specs, shap_values_xgboost,shap_values_logit,feature,plot_type,X,ffeatures):
  if plot_type == 'Importance':
      fig_str = 'ShapDependence_impClassifiers_%s_%s'%(frmt_str,feature)
  else:
      fig_str = 'ShapDependence_%s_withOthers_%s_%s'%(plot_type,frmt_str,feature)
  c = 0
  nrows, ncols, figure_size, font_s = specs[0],specs[1],specs[2],specs[3]
  fig = plt.figure(figsize=figure_size)
  for shap_vals in [shap_values_xgboost,shap_values_logit]:
            try:
                  comparison = shap_vals == shap_values_xgboost
                  if comparison.all():
                      classifier = 'xgboost'
                  else:
                      classifier = 'logit'
                  c += 1
                  #compare shap values for the same feature in both classifiers
                  ax = fig.add_subplot(nrows,ncols,c)
                  if shap.__version__ >= str(0.37):
                      shap.plots.scatter(shap_vals[:,feature], color=shap_vals, show=False,ax=ax)
                  else:
                      shap.dependence_plot(feature, shap_vals, X, feature_names=ffeatures, show=False, ax=ax)
                  ax.set_title('Dependence_plot of (%s) in (%s)' %(feature, classifier),size=font_s)
                  ax.tick_params(labelsize=font_s-3, labeltop=False, labelright=False)
                  ax.set_xlabel(xlabel=feature,fontsize=font_s)
                  ax.set_ylabel(ylabel='shap values for %s'%(feature),fontsize=font_s)
            except:
                    pass
  plt.savefig(os.path.join(SHAP_comparisons,'%s.png'%(fig_str)), dpi=300, bbox_inches='tight');
  plt.clf();


#a function to plot permutations of features
def plot_perm(perm_comparisons,frmt_str, specs, perm_logit_df, perm_xgboost_df,feats_tobe_plotted,plot_type):
  if plot_type == 'Importance':
      fig_str = 'Perm_comparison_importantToClassifiers_%s'%(frmt_str)
  else:
      fig_str = 'Perm_comparison_%s_withOthers_%s'%(plot_type,frmt_str)
  c = 0
  nrows, ncols, figure_size, font_s = specs[0],specs[1],specs[2],specs[3]
  fig = plt.figure(figsize=figure_size);
  fig.subplots_adjust(hspace=.5, wspace =.5)
  fig.text(0.5, 0.04, 'Repetitions', ha='center',fontsize=font_s)
  fig.text(0.04, 0.5, 'Importance Values', va='center', rotation='vertical',fontsize=font_s)
  for f in feats_tobe_plotted:
        try:
              feature_importance_logit = perm_logit_df.loc[perm_logit_df['Feature']==f,'importances'].iloc[0][1:-1].split(' ')
              feature_importance_logit = [float(i) for i in feature_importance_logit if i]
              feature_importance_xgboost = perm_xgboost_df.loc[perm_xgboost_df['Feature']==f,'importances'].iloc[0][1:-1].split(' ')
              feature_importance_xgboost = [float(i)  for i in feature_importance_xgboost if i]
              c += 1
              x = np.arange(1,11)
              #compare models
              ax = fig.add_subplot(nrows,ncols,c)
              ax.plot(x,feature_importance_xgboost,linestyle='solid', label='XGB', color='b');
              ax.plot(x,feature_importance_logit,linestyle='solid', label='LR', color='r');
              ax.grid(ls=':',which='both', axis='both')
              ax.set_title('(%s)' %(f),size=font_s-2)
              ax.tick_params(labelsize=font_s-3, labeltop=False, labelright=False)
              handles, labels = ax.get_legend_handles_labels()
        except:
            pass
  #together with the last line before this comment, these two lines define a single legend for the subplots
  fig.legend(handles, labels, loc = 'lower center', bbox_to_anchor=[0.35,0], fontsize=font_s-2)
  if len(feats_tobe_plotted) < (nrows*ncols) and plot_type != 'Importance':
      diff = (nrows*ncols) - len(feats_tobe_plotted)
      for i in range(diff):
        fig.delaxes(axes[nrows-1][ncols-(i+1)])
  plt.savefig(os.path.join(perm_comparisons,'%s.png'%(fig_str)), dpi=300, bbox_inches='tight');
  plt.clf();

