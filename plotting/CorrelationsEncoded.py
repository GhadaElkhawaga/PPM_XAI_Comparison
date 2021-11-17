import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_plot_correlations(EDA_output, out, df, dataset_name, method_name, f, feat_num):
    correlation_df = df.corr()
    correlation_df = pd.DataFrame([round(correlation_df[x], 3) for x in correlation_df])
    for x in correlation_df:
        if (len(correlation_df[x].value_counts()) == 0) and (x != 'encoded_label'):
            correlation_df.drop([x], axis=1, inplace=True)
            correlation_df.drop([x], axis=0, inplace=True)
    if f:
        correlation_df.to_csv(os.path.join(EDA_output, out, 'Numerical_correlations_%s_%s_%s.csv' % (
        dataset_name, method_name, feat_num)), sep=';')
    else:
        correlation_df.to_csv(os.path.join(EDA_output, out, 'Categorical_correlations_%s_%s_%s.csv' % (
        dataset_name, method_name, feat_num)), sep=';')
        # to split columns so as not to be plot in one figure:
    if len(correlation_df.columns) >= 10:
        for i in range(0, len(correlation_df.columns), 15):
            end = i + 15
            if end < len(correlation_df.columns):
                df1 = pd.DataFrame(correlation_df.iloc[i:end, i:end].values, columns=correlation_df.columns[i:end])
                df2 = pd.DataFrame(correlation_df.iloc[-1, i:end].values.reshape(1, end - i),
                                   columns=correlation_df.columns[i:end])
                df3 = pd.DataFrame(correlation_df.iloc[i:end, -1].values.reshape(end - i, 1),
                                   columns=[correlation_df.columns[-1]])
                plot_df = pd.concat([df1, df2], axis=0, sort=False).reset_index(drop=True)
                plot_df_final = pd.concat([plot_df, df3], axis=1, sort=False).reset_index(drop=True)
                plt.figure(figsize=(16, 8))
                res = sns.heatmap(plot_df_final, annot=True, annot_kws={"size": 12}, fmt='.3f',
                                  xticklabels=plot_df_final.columns, yticklabels=plot_df_final.columns, cmap='coolwarm')
                res.set_xticklabels(res.get_xmajorticklabels(), fontsize=12)
                res.set_yticklabels(res.get_ymajorticklabels(), fontsize=12)
                # use matplotlib.colorbar.Colorbar object
                cbar = res.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=12)
            else:
                end = len(correlation_df.columns) - 1
                if i != end:
                    df1 = pd.DataFrame(correlation_df.iloc[i:end, i:end].values, columns=correlation_df.columns[i:end])
                    df2 = pd.DataFrame(correlation_df.iloc[-1, i:end].values.reshape(1, end - i),
                                       columns=correlation_df.columns[i:end])
                    df3 = pd.DataFrame(correlation_df.iloc[i:end, -1].values.reshape(end - i, 1),
                                       columns=[correlation_df.columns[-1]])

                    plot_df = pd.concat([df1, df2], axis=0, sort=False).reset_index(drop=True)
                    plot_df_final = pd.concat([plot_df, df3], axis=1, sort=False).reset_index(drop=True)
                    plt.figure(figsize=(16, 8))
                    res = sns.heatmap(plot_df_final, annot=True, annot_kws={"size": 12}, fmt='.3f',
                                      xticklabels=plot_df_final.columns, yticklabels=plot_df_final.columns,
                                      cmap='coolwarm')
                    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=12)
                    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=12)
                    # use matplotlib.colorbar.Colorbar object
                    cbar = res.collections[0].colorbar
                    # here set the labelsize by 20
                    cbar.ax.tick_params(labelsize=12)
            if f:
                plt.savefig(os.path.join(EDA_output, out, 'correlation matrix_%s_numerical_%s_%s_part_%s.png' % (
                dataset_name, method_name, feat_num, i)))
            else:
                plt.savefig(os.path.join(EDA_output, out, 'correlation matrix_%s_categorical_%s_%s_part_%s.png' % (
                dataset_name, method_name, feat_num, i)))
            plt.show()
            plt.close()
    else:
        plt.figure(figsize=(20, 10))
        res = sns.heatmap(correlation_df, annot=True, annot_kws={"size": 12}, fmt='.3f',
                          xticklabels=correlation_df.columns, yticklabels=correlation_df.columns, cmap='coolwarm')
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=12)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=12)
        # use matplotlib.colorbar.Colorbar object
        cbar = res.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=12)
        if (f == True):
            plt.savefig(os.path.join(EDA_output, out, 'correlation matrix_%s_numerical_%s_%s.png' % (
            dataset_name, method_name, feat_num)))
        else:
            plt.savefig(os.path.join(EDA_output, out, 'correlation matrix_%s_categorical_%s_%s.png' % (
            dataset_name, method_name, feat_num)))
        plt.show()
        plt.close()


def compute_correlations(cls_method, method_name, ffeatures, encoded_training, train_y_experiment, \
                         encoded_testing_bucket, test_y_all, dataset_name, cls_encoder_args_final, feat_num):
    EDA_output = 'EDA_output_%s' % (method_name)
    if not (os.path.exists(EDA_output)):
        os.makedirs(EDA_output)

    out = 'correlations_%s_%s_%s_%s' % (dataset_name, method_name, cls_method, feat_num)
    if not (os.path.exists(os.path.join(EDA_output, out))):
        os.makedirs(os.path.join(EDA_output, out))
    train_y_experiment = pd.DataFrame(train_y_experiment, columns=['encoded_label'])
    test_y_all = pd.DataFrame(test_y_all, columns=['encoded_label'])
    encoded_training = pd.DataFrame(encoded_training, columns=ffeatures)
    encoded_testing_bucket = pd.DataFrame(encoded_testing_bucket, columns=ffeatures)
    df_train = pd.concat([encoded_training, train_y_experiment], axis=1, sort=False)
    df_test = pd.concat([encoded_testing_bucket, test_y_all], axis=1, sort=False)
    total_df = pd.concat([df_train, df_test], axis=0, sort=False)
    xx = [cls_encoder_args_final['case_id_col']]
    xx.extend(cls_encoder_args_final['static_cat_cols'] + cls_encoder_args_final['dynamic_cat_cols'])
    flag = True
    num_corr_cols = []
    cat_df_cols = []
    for i in ffeatures:
        for j in xx:
            if j in i:
                flag = False
                break
        if not flag:
            cat_df_cols.append(i)
        else:
            num_corr_cols.append(i)
        flag = True
    num_corr_cols.append('encoded_label')
    cat_df_cols.append('encoded_label')
    num_df = pd.concat([total_df[x] for x in num_corr_cols], axis=1, sort=False)
    cat_df = pd.concat([total_df[x] for x in cat_df_cols], axis=1, sort=False)
    compute_plot_correlations(EDA_output, out, num_df, dataset_name, method_name, True, feat_num)
    compute_plot_correlations(EDA_output, out, cat_df, dataset_name, method_name, False, feat_num)

