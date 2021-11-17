import csv
import os
import xgboost as xgb
import time
import graphviz
import time
#from lime_stability.stability import LimeTabularExplainerOvr
from sklearn.inspection import permutation_importance
from alibi.explainers import ALE, plot_ale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import shap
import pickle
from helpers.plotting_utils import define_range, compute_histogram_bins
from helpers.Data_retrieval import get_corr_files


def xgboost_features_importance(artefacts_dir, Importance_score_file, model, dataset_name, method_name, ffeatures,
                                bkt_size, prfx_len, feat_num):
    mapper = {'f{0}'.format(i): ffeatures[i] for i in range(0, len(ffeatures))}
    with open('file.csv', 'w') as impfile:
        for k, v in mapper.items():
            impfile.write('%s;%s\n' % (k, v))
    frmt_str = '%s_%s_%s_%s_%s' % (dataset_name, method_name, bkt_size, prfx_len, feat_num)
    xgb_imp_dir = os.path.join(artefacts_dir, 'xgb_imp_%s' % (frmt_str))
    if not os.path.exists(xgb_imp_dir):
        os.makedirs(os.path.join(xgb_imp_dir))
    with open(Importance_score_file, 'a') as impf:
        for imp in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            global_imp_time = time.time()
            scores = model.get_booster().get_score(importance_type=imp)
            generation_time = time.time() - global_imp_time
            mapped = {mapper[k]: v for k, v in scores.items()}
            for key, value in mapped.items():
                impf.write('%s;%s;%s;%s;%s\n' % (dataset_name, imp, key, value, np.nan))
            impf.write('%s;%s;%s;%s;%s\n' % (dataset_name, imp, 0, 0, generation_time))
            try:
                fig_title = 'XGboost Feature Importance using %s as importance type _%s' % (imp, frmt_str)
                fig, ax = plt.subplots(1, 1, figsize=(18, 8))
                xgb.plot_importance(mapped, importance_type=imp, title=fig_title, ax=ax, max_num_features=20)
                plt.savefig(os.path.join(xgb_imp_dir, fig_title + '.png'), dpi=300, bbox_inches='tight');
                plt.clf()
                plt.close()
            except:
                pass


def Permutation_importance_analysis(artefacts_dir, cls, method_name, ffeatures, encoded_training, train_y_experiment, \
                                    encoded_testing_bucket, test_y_all, dataset_name, cls_method, bkt_size, prfx_len,
                                    test_bkt_size, test_prfx_len, feat_num):
    Permutation_dir = os.path.join(artefacts_dir, 'Permutation_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num))
    if not os.path.exists(Permutation_dir):
        os.makedirs(os.path.join(Permutation_dir))
    permutation_file_name = 'permutation_importance_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num)
    permutation_test_file_name = 'permutation_importance_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, test_bkt_size, test_prfx_len, feat_num)
    start_calc_train = time.time()
    try:
        training_result = permutation_importance(cls, encoded_training,
                                                 train_y_experiment, n_repeats=10, random_state=42, n_jobs=-1)
    except:
        encoded_training = pd.DataFrame(encoded_training, columns=ffeatures)
        training_result = permutation_importance(cls, encoded_training,
                                                 train_y_experiment, n_repeats=10, random_state=42, n_jobs=-1)
    perm_time_train = time.time() - start_calc_train
    cols = ['Feature', 'importances(mean)', 'importances(std)', 'importances']
    df_res_train = pd.DataFrame(zip(ffeatures, training_result.importances_mean,
                                    training_result.importances_std, training_result.importances),
                                columns=cols)
    df_train_sorted = df_res_train.sort_values('importances(mean)', ascending=False)
    df_train_sorted.to_csv(os.path.join(Permutation_dir, '%s_training.csv' %permutation_file_name), sep=';',
                           index=False)
    with open(os.path.join(Permutation_dir, '%s_training.csv' %permutation_file_name), 'a') as fout:
        fout.write('%s;%s\n' % ('Calculcation time (Train)', perm_time_train))

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.boxplot(df_train_sorted.iloc[:20, 3].T, vert=False, labels=df_train_sorted.iloc[:20, 0])
    ax.set_title("Permutation Importances (train set)")
    plt.savefig(os.path.join(Permutation_dir, '%s_training.png' %permutation_file_name), dpi=300,
                bbox_inches='tight')
    plt.figure(figsize=(12, 8))
    plt.barh(np.arange(0, 20), df_train_sorted.iloc[:20, 1], align='center', alpha=0.5)
    plt.yticks(np.arange(0, 20), df_train_sorted.iloc[:20, 0])
    plt.xlabel('Importances')
    plt.title("Permutation Importances (train set)")
    plt.savefig(os.path.join(Permutation_dir, '%s_training2.png' %permutation_file_name), dpi=300,
                bbox_inches='tight')
    start_calc_test = time.time()
    try:
        testing_result = permutation_importance(cls, encoded_testing_bucket,
                                                test_y_all, n_repeats=10, random_state=42, n_jobs=-1)
    except:
        encoded_testing_bucket = pd.DataFrame(encoded_testing_bucket, columns=ffeatures)
        testing_result = permutation_importance(cls, encoded_testing_bucket,
                                                test_y_all, n_repeats=10, random_state=42, n_jobs=-1)
    perm_time_test = time.time() - start_calc_test
    df_res_test = pd.DataFrame(zip(ffeatures, testing_result.importances_mean,
                                   testing_result.importances_std, testing_result.importances),
                               columns=cols)
    df_test_sorted = df_res_test.sort_values('importances(mean)', ascending=False)
    df_test_sorted.to_csv(os.path.join(Permutation_dir, '%s_testing.csv' %permutation_test_file_name), sep=';',
                          index=False)
    with open(os.path.join(Permutation_dir, '%s_testing.csv' %permutation_test_file_name), 'a') as fout:
        fout.write('%s;%s\n' % ('Calculcation time (Test)', perm_time_test))
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.boxplot(df_test_sorted.iloc[:20, 3].T, vert=False, labels=df_test_sorted.iloc[:20, 0])
    ax.set_title("Permutation Importances (test set)")
    plt.savefig(os.path.join(Permutation_dir, '%s_testing.png' %permutation_test_file_name), dpi=300,
                bbox_inches='tight');
    plt.clf()
    plt.close()
    plt.figure(figsize=(12, 8))
    plt.barh(np.arange(0, 20), df_test_sorted.iloc[:20, 1], align='center', alpha=0.5)
    plt.yticks(np.arange(0, 20), df_test_sorted.iloc[:20, 0])
    plt.xlabel('Importances')
    plt.title("Permutation Importances (test set)");
    plt.savefig(os.path.join(Permutation_dir, '%s_testing2.png' %permutation_test_file_name), dpi=300,
                bbox_inches='tight');
    plt.clf()
    plt.close()
    with open(os.path.join(Permutation_dir, '%s_training.csv' %permutation_file_name), 'a+',
              newline='') as originalcsv:
        with open(os.path.join(Permutation_dir, '%s_testing.csv' %permutation_test_file_name), 'r') as mergedcsv:
            freader = csv.reader(mergedcsv, delimiter=';')
            fwriter = csv.writer(originalcsv, delimiter=';')
            for row in freader:
                fwriter.writerow(row)
    os.rename(os.path.join(Permutation_dir, '%s_training.csv' %permutation_file_name),
              os.path.join(Permutation_dir, '%s_final.csv' %permutation_file_name))
    os.remove(os.path.join(Permutation_dir, '%s_testing.csv' %permutation_test_file_name))


def explain_local_instances(artefacts_dir, dataset_ref, resexpfile, explanationfile, cls_experiment, limeexplainer,
                            test_instance, case_id, grouppred, round_count, dm, pipeline_final, method_name):
    dev_pos_count = 0
    reg_neg_count = 0
    deviant_exp_dict = {}
    regular_exp_dict = {}
    dataset_name = dm.d_name
    Lime_dir = os.path.join(artefacts_dir, 'LIME_%s_%s' % (dataset_name, method_name))
    if not os.path.exists(Lime_dir):
        os.makedirs(os.path.join(Lime_dir))
    if method_name == 'single_agg':
        c = 20
        j = 1000 if (dataset_ref == 'sepsis_cases') else 10000
    else:
        c = 5
        j = 25 if (dataset_ref == 'sepsis_cases') else 250
    # generating explanations for 1/20 of the number of traces
    if round_count % c == 0:
        start_explanation = time.time()
        ActPred = dm.get_label_numeric(grouppred)
        expparams = {"data_row": test_instance, "predict_fn": cls_experiment.predict_proba,
                     "num_features": 6, "distance_metric": "euclidean"}
        explanation = limeexplainer.explain_instance(**expparams)
        exportedexp = explanation.as_list()
        probability_result = cls_experiment.predict_proba([test_instance])[0]
        PredClass = pipeline_final.predict(grouppred)
        # generating stability indices for 1 of each 10 explanations
        if (round_count % j) == 0:
            csi, vsi = limeexplainer.check_stability(n_calls=10, **expparams, index_verbose=False)
            explanation.show_in_notebook(show_table=True)
            xx = "explanation_%s_%s.html" % (case_id[0], dataset_name)
            explanation.save_to_file(os.path.join(Lime_dir, xx))
        exptime = time.time() - start_explanation
        if ActPred == [1]:
            dev_pos_count += 1
            deviant_exp_dict[tuple(case_id)] = exportedexp
            # dev_df = pd.DataFrame(deviant_exp_dict.items, columns=['Case Id', 'Explanation'])
            with open(resexpfile, 'a') as resf:
                resf.write('%s;%s;%s\n' % (case_id.values, exportedexp, 'Deviant'))
            if (round_count % j) == 0:
                with open(explanationfile, 'a') as expf:
                    expf.write('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                    dataset_name, case_id[0], ActPred, exportedexp, probability_result, PredClass, 'Deviant', exptime,
                    csi, vsi))
            else:
                with open(explanationfile, 'a') as expf:
                    expf.write('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                    dataset_name, case_id[0], ActPred, exportedexp, probability_result, PredClass, 'Deviant', exptime,
                    0, 0))
        else:
            reg_neg_count += 1
            regular_exp_dict[tuple(case_id)] = exportedexp
            # reg_df = pd.DataFrame(regular_exp_dict.items, columns=['Case Id', 'Explanation'])
            with open(resexpfile, 'a') as resf:
                resf.write('%s;%s;%s\n' % (case_id.values, exportedexp, 'Regular'))
            if (round_count % j) == 0:
                with open(explanationfile, 'a') as expf:
                    expf.write('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                    dataset_name, case_id[0], ActPred, exportedexp, probability_result, PredClass, 'Regular', exptime,
                    csi, vsi))
            else:
                with open(explanationfile, 'a') as expf:
                    expf.write('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                    dataset_name, case_id[0], ActPred, exportedexp, probability_result, PredClass, 'Regular', exptime,
                    0, 0))


def shap_global(artefacts_dir, cls, X, dataset_name, cls_method, method_name, ffeatures, bkt_size, prfx_len, feat_num,
                X_other=None, flag=None):
    shap_values_dir = os.path.join(artefacts_dir, 'shap_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num))
    if not os.path.exists(shap_values_dir):
        os.makedirs(os.path.join(shap_values_dir))
    typ = flag
    shap_time = time.time()
    if shap.__version__ >= str(0.37):
        explainer = shap.Explainer(cls, X, feature_names=ffeatures)
    else:
        if cls_method == 'xgboost':
            explainer = shap.TreeExplainer(cls)
        else:
            explainer = shap.LinearExplainer(cls, X)
    if cls_method == 'xgboost':
            shap_values = explainer.shap_values(X, check_additivity=False)
    else:
            shap_values = explainer.shap_values(X)
    shap_time_end = time.time() - shap_time

    frmt_str = '%s_%s_%s_%s_%s_%s_%s' % (cls_method, dataset_name, method_name, typ, bkt_size, prfx_len, feat_num)
    out1 = os.path.join(shap_values_dir,
                        'shap_explainer_%s.pickle' %frmt_str)
    with open(out1, 'wb') as output:
        pickle.dump(explainer, output)
    shap_data = os.path.join(shap_values_dir,
                             'shap_values_%s.pickle' %frmt_str)
    with open(shap_data, 'wb') as fout:
        pickle.dump(shap_values, fout)
    shap_csv = os.path.join(shap_values_dir,
                            'shap_values_%s.csv' %frmt_str)
    pd.DataFrame(shap_values).to_csv(shap_csv, sep=';', index=False)
    with open(shap_csv, 'a') as fout:
        fout.write('%s;%s\n' % ('Calculcation time', shap_time_end))
    if shap.__version__ >= str(0.37):
        shap.plots.beeswarm(shap_values, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, max_display=10, show=False)
    plt.savefig(os.path.join(shap_values_dir,
                             'Shap values_normal%s.png' %frmt_str),
                dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()

    if shap.__version__ >= str(0.37):
        shap.plots.bar(shap_values, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, plot_type='bar', show=False, max_display=10)
    plt.savefig(
        os.path.join(shap_values_dir, 'Shap values_bar_%s.png' %frmt_str),
        dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    del explainer
    del shap_values


def shap_local(artefacts_dir, cls_method, dataset_name, method_name, encoded_testing_bucket, case_idx, case_id,
               ffeatures, test_bkt_size, test_prfx_len, feat_num, flag):
    """
    algorithm:
    first read shap values and the explainer from the respective pickle files
    then receive the test instance and identify its index
    then plot waterfall_plot and force_plot
    important_notes:
    (1) this function works only on testing data, because it is called in the main program within the testing portion of the event log
    (2) the explainer is fit to the test data, which means that the base value would be the average over the testing portion I have passed to (shap_global)
    """
    shap_values_dir = os.path.join(artefacts_dir, 'shap_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, test_bkt_size, test_prfx_len, feat_num))
    if flag == True:
        typ = 'training'
    elif flag == False:
        typ = 'testing'
    else:
        typ = flag

    frmt_str = '%s_%s_%s_%s_%s_%s_%s' % (
    cls_method, dataset_name, method_name, typ, test_bkt_size, test_prfx_len, feat_num)
    shap_file = os.path.join(shap_values_dir,
                             'shap_values_%s.pickle' %frmt_str)
    with open(shap_file, 'rb') as fin:
        shap_values = pickle.load(fin)
    explainer_file = os.path.join(shap_values_dir,
                                  'shap_explainer_%s.pickle' %frmt_str)
    with open(explainer_file, 'rb') as fin2:
        explainer = pickle.load(fin2)
    expected_value = explainer.expected_value
    # shap.initjs()
    if shap.__version__ >= str(0.37):
        shap.force_plot(expected_value, shap_values[case_idx][:], encoded_testing_bucket[case_idx][:], show=False,
                        feature_names=ffeatures, matplotlib=True)
    else:
        shap.force_plot(expected_value, shap_values[case_idx], show=False, matplotlib=True)
    plt.savefig(os.path.join(shap_values_dir,
                             'force_plot_%s_%s.png' % (case_id, frmt_str)),
                dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    # shap.initjs()
    shap.decision_plot(expected_value, shap_values[case_idx], encoded_testing_bucket[case_idx],
                       feature_names=ffeatures, show=False)
    plt.savefig(os.path.join(shap_values_dir,
                             'decision_%s_%s.png' % (case_id, frmt_str)),
                dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()


def plot_dependence_plots(artefacts_dir, X, dataset_name, cls_method, method_name, ffeatures, flag, bkt_size, prfx_len,
                          feat_num):
    shap_values_dir = os.path.join(artefacts_dir, 'shap_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num))
    EDA_output = 'EDA_output_%s' %method_name
    files = get_corr_files(EDA_output, dataset_name, method_name, cls_method, feat_num)
    if flag == True:
            typ = 'training'
    else:
            typ = 'testing'
    frmt_str = '%s_%s_%s_%s_%s_%s_%s' % (cls_method, dataset_name, method_name, typ, bkt_size, prfx_len, feat_num)
    shap_file = os.path.join(shap_values_dir,
                             'shap_values_%s.pickle' %frmt_str)
    with open(shap_file, 'rb') as fin:
        shap_values = pickle.load(fin)
    # dependence plot based on the highest values of correlations with the target and the highest shap values
    for i in files:
        df = pd.read_csv(i, sep=';')
        df.set_index(df.columns[0], inplace=True)
        interaction_cols = list(df.index[df[df.columns[-1]] >= 0.2].values)
        interaction_cols.append(list(df.index[df[df.columns[-1]] <= -0.2].values))
    interaction_cols.remove('encoded_label')
    interaction_cols = [x for x in interaction_cols if x]
    top_vals = np.argsort(-np.sum(np.abs(shap_values), 0))
    for i in range(5):
        if shap.__version__ >= str(0.37):
            shap.plots.scatter(shap_values[:, top_vals[i]], color=shap_values, show=False)
        else:
            shap.dependence_plot(top_vals[i], shap_values, X, feature_names=ffeatures, show=False)

        plt.savefig(
            os.path.join(shap_values_dir, 'dependence plot_attributes_high_Shap_values_%s_%s.png' %(frmt_str, i)),
            dpi=300, bbox_inches='tight');
        plt.clf()
        plt.close()
    shap.initjs()
    if len(interaction_cols) > 0:
        for j in interaction_cols:
            if shap.__version__ >= str(0.37):
                shap.plots.scatter(shap_values[:, j], color=shap_values, show=False)
            else:
                shap.dependence_plot(j, shap_values, X, feature_names=ffeatures, show=False)
            plt.savefig(os.path.join(shap_values_dir, 'dependence_plot_high_correlation_%s_%s.png' %(frmt_str, j)),
                        dpi=300, bbox_inches='tight');
            plt.clf()
    del shap_values
    del interaction_cols


def logit_plot_coef(artefacts_dir, model, dataset_name, method_name, ffeatures, bkt_size, prfx_len, feat_num):
    frmt_str = '%s_%s_%s_%s_%s' % (dataset_name, method_name, bkt_size, prfx_len, feat_num)
    logit_coef_dir = os.path.join(artefacts_dir, 'logit_coef_%s' %frmt_str)
    if not os.path.exists(logit_coef_dir):
        os.makedirs(logit_coef_dir)
    coefs = model.coef_
    coefs_col = coefs.reshape(-1, 1)
    coefs_df = pd.DataFrame(coefs_col, columns=['coef'])
    coefs_df = coefs_df.astype(float)
    coefs_df['Variable'] = ffeatures
    coefs_df = coefs_df.sort_values(by=['coef'], ascending=False)
    coefficients_csv = 'logit_coefficients_%s.csv' %frmt_str
    coefs_df.to_csv(os.path.join(logit_coef_dir, coefficients_csv), sep=';', index=False)
    plot_data = coefs_df.iloc[:20]
    plot_df = pd.DataFrame(plot_data, columns=['coef', 'Variable'])
    # Set sns plot style back to 'poster'
    # This will make bars wider on plot
    sns.set_context("poster")
    # scatter plot
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_df.plot(x='Variable', y='coef', kind='bar', ax=ax, color='none', fontsize=22, ecolor='steelblue', capsize=0,
                 legend=False)
    fig_title = 'Logit_Coefficients_of_%s_scatter' %frmt_str
    ax.set_ylabel('Coefficients', fontsize=22)
    ax.set_xlabel('feature', fontsize=22)
    ax.scatter(x=plot_df['Variable'], marker='o', s=80, y=plot_df['coef'], color='steelblue')
    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    plt.savefig(os.path.join(logit_coef_dir, fig_title), dpi=300, bbox_inches='tight');
    plt.clf()
    # plotting barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_df.plot(x='Variable', y='coef', kind='bar', ax=ax, color='none', fontsize=22, ecolor='steelblue', capsize=0,
                 legend=False)

    fig_title = 'Logit_Coefficients_of_%s_bar' %frmt_str
    ax.set_ylabel('Coefficients', fontsize=22)
    ax.set_xlabel('feature', fontsize=22)
    xx = plot_df['Variable']
    ax.bar(xx, height=plot_df['coef'], width=0.3, color='steelblue')
    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    plt.savefig(os.path.join(logit_coef_dir, fig_title), dpi=300, bbox_inches='tight');
    plt.clf()



def ALE_Computing(ALE_obj, bkt_size, prfx_len, feat_num):
    """
    Algorithm:
    examine the correlations dataset and compute ALE for features correlated with the target by more than 0.2
    examine the correlations dataset and compute ALE for features correlated together by more than 0.35
    output a file with a list of features highly correlated with the target and a dictionary of features correlated with each other
    """
    ALE_dir, ALE_df, ALE_training_arr, counts_df = ALE_obj.data_processing()
    ALE_features, target_names = ALE_obj.get_ALE_names(ALE_df)
    ALE_cls = ALE_obj.ALE_classifier(ALE_training_arr)
    ALE_df['target'] = ALE_obj.y
    ALE_df['target'].replace({1: ALE_obj.dm.pos_label, 0: ALE_obj.dm.neg_label}, inplace=True)
    count_plot_dir = os.path.join(ALE_dir, 'Count_plots')
    if not os.path.exists(count_plot_dir):
        os.makedirs(count_plot_dir)
    corr_with_others = {}
    frmt_str = '%s_%s_%s_%s' % (ALE_obj.dataset_name, ALE_obj.bkt_enc, bkt_size, prfx_len)
    EDA_output = 'EDA_output_%s' %ALE_obj.bkt_enc
    files = get_corr_files(EDA_output, ALE_obj.dataset_name, ALE_obj.bkt_enc, ALE_obj.cls_method, feat_num)
    pred_exp_start = time.time()
    ale_pred = ALE(ALE_cls.predict_proba, feature_names=ALE_features, target_names=target_names)
    explainer_pred = ale_pred.explain(ALE_training_arr)
    pred_exp_time = time.time() - pred_exp_start
    explainer_pred_data = os.path.join(ALE_dir, 'ALE_pred_explainer_%s_%s.pickle' % (ALE_obj.cls_method, frmt_str))
    with open(explainer_pred_data, 'wb') as output:
        pickle.dump(explainer_pred, output)
    # saving values at which ale values are computed for each feature in csv files:
    cols = ['Feature', 'ALE_vals', 'feature_values_for_ALECalc', 'ale0']
    df_res_exp_pred = pd.DataFrame(
        zip(explainer_pred.feature_names, explainer_pred.ale_values, explainer_pred.feature_values,
            explainer_pred.ale0), columns=cols)
    df_res_exp_pred_file = os.path.join(ALE_dir, 'ALE_pred_explainer_%s_%s.csv' % (ALE_obj.cls_method, frmt_str))
    df_res_exp_pred.to_csv(df_res_exp_pred_file, sep=';', index=False)
    with open(df_res_exp_pred_file, 'a') as fout:
        fout.write('%s;%s\n' % ('Calculcation time', pred_exp_time))
    # get columns with highest correlations with the target:
    for i in files:
        corr_df = pd.read_csv(i, sep=';')
        corr_df.set_index(corr_df.columns[1:], inplace=True)
        # this list represents features with correlation >= 0.2 with the target:
        corr_with_target = corr_df.index[
            abs(corr_df[corr_df.columns[-1]]) >= 0.2].tolist()  # to get features highly correlated with the target
        for col in corr_with_target:
            if 'label' in col:
                corr_with_target.remove(col)
                # plotting ALE of features which have high correlation with the target
        for f in corr_with_target:
            if f in ALE_features:
                idx = list(ALE_features).index(f)
                plot_ale(explainer_pred, features=[idx])
                plt.savefig(os.path.join(ALE_dir, 'ALE_for_feature_%s_%s_%s_corrWithTarget.png' % (
                f, frmt_str, ALE_obj.cls_method)), dpi=300, bbox_inches='tight');
                plt.clf()
                # plotting count plots for categorical attributes
                if (i == cat_csv) or (
                        (i == num_csv) and (counts_df.loc[counts_df['index'] == f, 'values'].item() < 20)):
                    # sns.countplot(x=f, hue=target_names[target], data=ALE_df.iloc[:,idx]);
                    ALE_df_grouped = ALE_df.groupby(['target', f]).size().reset_index().pivot(columns='target', index=f,
                                                                                              values=0)
                    ax = ALE_df_grouped.plot.bar(stacked=True, color=["darkblue", 'mediumvioletred'])
                    for p in ax.patches:
                        width, height = p.get_width(), p.get_height()
                        if height == 0:
                            continue
                        x, y = p.get_xy()
                        ax.text(x + width / 2,
                                y + height / 2,
                                '{:.0f}'.format(height),
                                horizontalalignment='center',
                                verticalalignment='center', fontsize=6)
                    plt_title = 'count_plot_for_feature_%s_corrWithTarget.png' % (f)
                    plt.xlabel(f)
                    plt.title(plt_title)
                    plt.legend()
                    plt.savefig(os.path.join(count_plot_dir, '%s_%s.png' %(plt_title, frmt_str)), dpi=300,
                                bbox_inches='tight');
                    plt.clf()
        # this dictionary represents features with correlation >= 0.35 with the other features:
        try:
            for c in corr_df.columns[1:]:
                l = corr_df.index[(abs(corr_df[c]) >= 0.35)].tolist()  # to get features highly correlated with others
                corr_with_others[c] = l

            tmp = corr_with_others.copy()
            for k, v in tmp.items():
                if k in v:
                    v.remove(k)
                if 'encoded_label' in v:
                    v.remove('encoded_label')
            corr_with_others = tmp.copy()
            for k in tmp.keys():
                if not tmp[k]:
                    corr_with_others.pop(k, None)
            # plotting ALE of features which have high correlation with other features
            for ff in corr_with_others.keys():
                if ff in ALE_features:
                    idx_ff = list(ALE_features).index(ff)
                    plot_ale(explainer_pred, features=[idx_ff])
                    plt.savefig(os.path.join(ALE_dir, 'ALE_for_feature_%s_%s_%s_corrWithOthers.png' % (
                    ff, frmt_str, ALE_obj.cls_method)), dpi=300, bbox_inches='tight');
                    plt.clf()
                    # plotting count plots for categorical attributes
                    if ('Categorical' in i) or (
                            ('Numerical' in i) and (counts_df.loc[counts_df['index'] == ff, 'values'].item() < 20)):
                        ALE_df_grouped = ALE_df.groupby(['target', ff]).size().reset_index().pivot(columns='target',
                                                                                                   index=ff, values=0)
                        ax = ALE_df_grouped.plot.bar(stacked=True, color=["darkblue", 'mediumvioletred'])
                        for p in ax.patches:
                            width, height = p.get_width(), p.get_height()
                            if height == 0:
                                continue
                            x, y = p.get_xy()
                            ax.text(x + width / 2, y + height / 2, '{:.0f}'.format(height),
                                    horizontalalignment='center', verticalalignment='center', fontsize=6)
                        plt_title = 'Count_plt_for_feature_%s_corrWithOthers.png' %ff
                        plt.title(plt_title)
                        plt.xlabel(ff)
                        plt.legend()
                        plt.savefig(os.path.join(count_plot_dir, '%s_%s.png' % (plt_title, frmt_str)), dpi=300,
                                    bbox_inches='tight');
                        plt.clf()
        except ValueError:
            pass

        corr_with_others = {}

