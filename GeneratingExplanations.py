import pandas as pd #using pandas==0.25
import numpy as np
import time
import sys
import os
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
import pickle
from collections import defaultdict, Counter
import io
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb #using xgboost==0.9
import graphviz
import warnings
warnings.simplefilter('ignore')
import csv
from lime_stability.stability import LimeTabularExplainerOvr
import shap #using shap==0.36
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
from helpers.Encoders import get_encoder
from helpers.Bucketers import get_bucketer
from helpers.DatasetManager import DatasetManager
from helpers.ClassifierWrapper import ClassifierWrapper
from helpers.Data_gen import generate_instances_df, get_percentage
import Definitions
from explainers.ExplanationGenerators import xgboost_features_importance, Permutation_importance_analysis, explain_local_instances
from explainers.ExplanationGenerators import shap_global, shap_local, plot_dependence_plots, ALE_Computing, logit_plot_coef
from plotting.CorrelationsEncoded import compute_correlations

#Defining Basic parameters
n_iter = 3
#method_name = 'single_agg'
method_name = 'prefix_index'
#cls_encoding = 'agg'
cls_encoding = 'index'
cls_method = 'xgboost'
#cls_method = 'logit'
#bucket_method = 'single'
bucket_method = 'prefix'
gap = 5
train_ratio = 0.8
n_splits = 3 #number of cv_folds
random_state = 22
min_cases_for_training = 1
encoding_dict = {
    'agg' : ['static', 'agg'],
    'index' : ['static', 'index'],
}
bucket_method , cls_encoding = method_name.split('_')
if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"
methods = encoding_dict[cls_encoding]
dataset_ref_to_datasets = {
    "sepsis_cases": ["sepsis1", "sepsis2", "sepsis3"],
    "bpic2017" :["BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"],
    "Hospital_Billing" :["hospital_billing_%s" %i for i in range(1,3)]
}
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

artefacts_dir = '%s_%s_%s_gap' %(dataset_ref,method_name,cls_method)
if not os.path.exists(artefacts_dir):
  os.makedirs(os.path.join(artefacts_dir))

params_dir = os.path.join(artefacts_dir,'cv_results_revision')
if not os.path.exists(params_dir):
  os.makedirs(os.path.join(params_dir))


# hyperparameter Optimization objective function
def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1

    start = time.time()
    score = 0
    for cv_iter in range(n_splits):

        df_test_prefixes = df_prefixes[cv_iter]
        df_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits):
            if cv_train_iter != cv_iter:
                df_train_prefixes = pd.concat([df_train_prefixes, df_prefixes[cv_train_iter]], axis=0)

        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dataset_manager.case_id_col,
                         'cat_cols': [dataset_manager.activity_col],
                         'num_cols': [],
                         'random_state': random_state}

        bucketer = get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
        bucket_assignments_test = bucketer.predict(df_test_prefixes)

        preds_all = []
        test_y_all = []
        if "prefix" in method_name:
            scores = defaultdict(int)
        for bucket in set(bucket_assignments_test):
            relevant_train_cases_bucket = dataset_manager.get_indexes(df_train_prefixes)[
                bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(df_test_prefixes)[
                bucket_assignments_test == bucket]
            df_test_bucket = dataset_manager.get_relevant_data_by_indexes(df_test_prefixes, relevant_test_cases_bucket)
            test_y = dataset_manager.get_label_numeric(df_test_bucket)
            if len(relevant_train_cases_bucket) == 0:
                preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)
            else:
                df_train_bucket = dataset_manager.get_relevant_data_by_indexes(df_train_prefixes,
                                                                               relevant_train_cases_bucket)  # one row per event
                train_y = dataset_manager.get_label_numeric(df_train_bucket)

                if len(set(train_y)) < 2:
                    preds = [train_y[0]] * len(relevant_test_cases_bucket)
                else:
                    feature_combiner = FeatureUnion(
                        [(method, get_encoder(method, **cls_encoder_args)) for method in methods])

                    if cls_method == "xgboost":
                        cls = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate=args['learning_rate'],
                                                subsample=args['subsample'],
                                                max_depth=int(args['max_depth']),
                                                colsample_bytree=args['colsample_bytree'],
                                                min_child_weight=int(args['min_child_weight']),
                                                seed=random_state)

                    elif cls_method == "logit":
                        cls = LogisticRegression(C=2 ** args['C'],
                                                 random_state=random_state)

                    if cls_method == "logit":
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
                    pipeline.fit(df_train_bucket, train_y)

                    preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                    preds = pipeline.predict_proba(df_test_bucket)[:, preds_pos_label_idx]

            if "prefix" in method_name:
                auc = 0.5
                if len(set(test_y)) == 2:
                    auc = roc_auc_score(test_y, preds)
                scores[bucket] += auc
            preds_all.extend(preds)
            test_y_all.extend(test_y)

        # score += roc_auc_score(test_y_all, preds_all)
        try:
            score += roc_auc_score(test_y_all, preds_all)
        except ValueError:
            pass

    if "prefix" in method_name:
        for k, v in args.items():
            for bucket, bucket_score in scores.items():
                fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, bucket, k, v, bucket_score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
        trial_nr, dataset_name, cls_method, method_name, 0, "processing_time", time.time() - start, 0))
    else:
        for k, v in args.items():
            fout_all.write(
                "%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (
        trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


# code to find the best parameters
for dataset_name in datasets:
    # the folders that contains the folds csv files
    folds_directory = os.path.join(artefacts_dir, 'folds_%s_%s_%s' % (dataset_name, cls_method, method_name))
    if not os.path.exists(os.path.join(folds_directory)):
        os.makedirs(os.path.join(folds_directory))
    dataset_manager = DatasetManager(dataset_name)
    df = dataset_manager.read_dataset()

    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                        'fillna': True}
    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "BPIC2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(df, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(df, 0.90))
    # splitting data into training and testing, then deleting the whole dataframe
    train, _ = dataset_manager.split_data_strict(df, train_ratio, split="temporal")
    del df
    # prepare chunks for cross-validation
    df_prefixes = []
    class_ratios = []
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
        class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
        # generate data where each prefix is a separate instance
        if (method_name == 'prefix_index'):
            df_prefixes.append(
                dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length, gap=5))
        else:
            df_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
    del train
    # search parameters for 'xgboost' algorithm:
    if cls_method == 'xgboost':
        space = {'learning_rate': hp.uniform('learning_rate', 0, 1),
                 'subsample': hp.uniform('subsample', 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
    else:
        space = {'C': hp.uniform('C', -15, 15)}

    trial_nr = 1
    trials = Trials()
    fout_all = open(
        os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
    if "prefix" in method_name:
        fout_all.write(
            "%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))
    else:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
    fout_all.close()
    # write the best parameters
    best_params = hyperopt.space_eval(space, best)
    outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)


# final experiments
saved_artefacts = os.path.join(artefacts_dir, 'model_and_hdf5')
if not os.path.exists(saved_artefacts):
    os.makedirs(os.path.join(saved_artefacts))
results_dir_final = os.path.join(artefacts_dir, 'final_experiments_results')
if not os.path.exists(results_dir_final):
    os.makedirs(os.path.join(results_dir_final))
resexpfile = os.path.join(artefacts_dir, 'results_LIME_%s_%s_%s.csv' % (cls_method, dataset_ref, method_name))
explanationfile = os.path.join(artefacts_dir,
                               'explanationsfile_LIME_%s_%s_%s.csv' % (cls_method, dataset_ref, method_name))
with open(explanationfile, 'w+') as expf:
    header = ['Dataset Name', 'case ID', 'Actual Value', 'Explanation', 'Probability result', 'Predicted class',
              'Class Type', 'Generation Time', 'Coefficient Stability', 'Variable Stability']
    writer = csv.DictWriter(expf, delimiter=';', lineterminator='\n', fieldnames=header)
    writer.writeheader()
with open(resexpfile, 'w+') as resf:
    header2 = ['Case ID', 'Explanation', 'Label']
    writer2 = csv.DictWriter(resf, delimiter=';', lineterminator='\n', fieldnames=header2)
    writer2.writeheader()
for dataset_name in datasets:
    params_file = os.path.join(params_dir, 'optimal_params_%s_%s_%s.pickle' % (cls_method, dataset_name, method_name))
    with open(params_file, 'rb') as fin:
        args = pickle.load(fin)
    current_args = {}
    dm = DatasetManager(dataset_name)
    df = dm.read_dataset()
    cls_encoder_args_final = {'case_id_col': dm.case_id_col,
                              'static_cat_cols': dm.static_cat_cols,
                              'dynamic_cat_cols': dm.dynamic_cat_cols,
                              'static_num_cols': dm.static_num_cols,
                              'dynamic_num_cols': dm.dynamic_num_cols,
                              'fillna': True}
    # determine min and max (truncated) prefix lengths
    min_prefix_length_final = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length_final = 10
    elif "BPIC2017" in dataset_name:
        max_prefix_length_final = min(20, dm.get_pos_case_length_quantile(df, 0.90))
    else:
        max_prefix_length_final = min(40, dm.get_pos_case_length_quantile(df, 0.90))

    train, test = dm.split_data_strict(df, train_ratio=0.8, split='temporal')

    if gap > 1:
        outfile = os.path.join(results_dir_final, "performance_experiments_%s_%s_%s_gap%s.csv" % (
        cls_method, dataset_name, method_name, gap))
    else:
        outfile = os.path.join(results_dir_final,
                               'performance_experiments_%s_%s_%s.csv' % (cls_method, dataset_name, method_name))

    prefix_test_generation_start = time.time()
    if (method_name == 'prefix_index'):
        df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final, gap=5)
    else:
        df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final)
    prefix_test_generation = time.time() - prefix_test_generation_start

    train_prefix_generation_times = []
    offline_times = []
    online_times = []
    for i in range(n_iter):
        train_prefix_generation_start = time.time()
        if method_name == 'prefix_index':
            df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final, gap=5)
        else:
            df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final)
        train_prefix_generation = time.time() - train_prefix_generation_start
        train_prefix_generation_times.append(train_prefix_generation)
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dm.case_id_col,
                         'cat_cols': [dm.activity_col],
                         'num_cols': [],
                         'random_state': random_state}

        bucketer = get_bucketer(bucket_method, **bucketer_args)
        start_offline_bucket = time.time()
        bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
        offline_bucket = time.time() - start_offline_bucket
        bucket_assignment_test = bucketer.predict(df_test_prefixes)
        preds_all = []
        test_y_all = []
        nr_events_all = []
        offline_time_fit = 0
        current_online_times = []
        for bucket in set(bucket_assignment_test):
            if bucket_method == "prefix":
                current_args = args[bucket]
            else:
                current_args = args
            relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
            relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
            df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
            test_prfx_len = dm.get_prefix_lengths(df_test_bucket)[0]
            nr_events_all.extend(list(dm.get_prefix_lengths(df_test_bucket)))
            if len(relevant_train_bucket) == 0:
                preds = [dm.get_class_ratio(train)] * len(relevant_test_bucket)
                current_online_times.extend([0] * len(preds))
            else:
                df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
                train_y_experiment = dm.get_label_numeric(df_train_bucket)
                prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]

                if len(set(train_y_experiment)) < 2:
                    preds = [train_y_experiment[0]] * len(relevant_train_bucket)
                    current_online_times.extend([0] * len(preds))
                    test_y_all.extend(dm.get_label_numeric(df_test_prefixes))
                else:
                    start_offline_time_fit = time.time()
                    featureCombinerExperiment = FeatureUnion(
                        [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])

                    if cls_method == 'xgboost':
                        cls_experiment = xgb.XGBClassifier(objective='binary:logistic',
                                                           n_estimators=500,
                                                           learning_rate=current_args['learning_rate'],
                                                           max_depth=current_args['max_depth'],
                                                           subsample=current_args['subsample'],
                                                           colsample_bytree=current_args['colsample_bytree'],
                                                           min_child_weight=current_args['min_child_weight'],
                                                           seed=random_state)

                        pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('cls', cls_experiment)])
                    else:
                        cls_experiment = LogisticRegression(C=2 ** current_args['C'], random_state=random_state)
                        pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('scaler', StandardScaler()),
                                                   ('cls', cls_experiment)])

                    pipeline_final.fit(df_train_bucket, train_y_experiment)
                    offline_time_train_single_bucket = time.time() - start_offline_time_fit

                    offline_time_fit += time.time() - start_offline_time_fit
                    with open(outfile, 'a') as out:
                        out.write('%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                        'dataset', 'method', 'cls', 'nr_events', 'n_iter', 'prefix_length', 'train_time_bucket','score'))
                        out.write('%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                        dataset_name, method_name, cls_method, -1, i, prfx_len, offline_time_train_single_bucket,-1))
                    if i == 2:
                        ffeatures = pipeline_final.named_steps['encoder'].get_feature_names()
                        encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
                        training_set_df = pd.DataFrame(encoded_training, columns=ffeatures)
                        bkt_size = training_set_df.shape[0]
                        feat_num = training_set_df.shape[1]
                        # save the features names for later use
                        ffeatures_file = os.path.join(saved_artefacts,
                                                      'ffeatures_{0}_{1}_{2}_{3}_{4}.pickle'.format(cls_method,
                                                                                                    dataset_name,
                                                                                                    method_name,
                                                                                                    bkt_size, prfx_len,
                                                                                                     feat_num))
                        with open(ffeatures_file, 'wb') as fout_features:
                            pickle.dump(ffeatures, fout_features)
                        model_saved = pipeline_final.named_steps['cls']
                        # save the model for later use
                        model_file = os.path.join(saved_artefacts, 'model_%s_%s_%s_%s_%s_%s.pickle' % (
                        cls_method, dataset_name, method_name, bkt_size, prfx_len, feat_num))
                        with open(model_file, 'wb') as fout:
                            pickle.dump(model_saved, fout)
                        if (get_percentage(1, train_y_experiment) > 0.1) or (
                                get_percentage(0, train_y_experiment) > 0.1):
                            explain_flag = True
                        if explain_flag:
                            if cls_method == 'xgboost':
                                Importance_score_file = os.path.join(artefacts_dir,
                                                                     'global_Importance_%s_%s_%s_%s_%s_%s.csv' % (
                                                                     cls_method, dataset_ref, method_name, bkt_size,
                                                                     prfx_len, feat_num))
                                with open(Importance_score_file, 'w+') as impf:
                                    header3 = ['Dataset Name', 'Importance Type', 'Feature', 'Score', 'Generation Time']
                                    writer3 = csv.DictWriter(impf, delimiter=';', lineterminator='\n',
                                                             fieldnames=header3)
                                    writer3.writeheader()
                                xgboost_features_importance(artefacts_dir, Importance_score_file, cls_experiment,
                                                            dataset_name, method_name, ffeatures, bkt_size, prfx_len,
                                                            feat_num)
                            else:
                                logit_plot_coef(artefacts_dir, cls_experiment, dataset_name, method_name, ffeatures,
                                                bkt_size, prfx_len, feat_num)
                            # using wrapped module from Lime_stability libarary
                            limeexplainer = LimeTabularExplainerOvr(encoded_training, mode='classification',
                                                                    feature_names=featureCombinerExperiment.get_feature_names(),
                                                                    class_names=[dm.neg_label, dm.pos_label],
                                                                    discretize_continuous=True,
                                                                    verbose=True)

                            encoded_testing_bucket = featureCombinerExperiment.fit_transform(df_test_bucket)
                            testing_set_df = pd.DataFrame(encoded_testing_bucket, columns=ffeatures)
                            test_bkt_size = testing_set_df.shape[0]
                            shap_global(artefacts_dir, cls_experiment, encoded_training,
                                            dataset_name, cls_method, method_name, ffeatures, bkt_size,
                                            prfx_len, feat_num, X_other=encoded_testing_bucket,
                                            flag='training')

                            shap_global(artefacts_dir, cls_experiment, encoded_testing_bucket,
                                            dataset_name, cls_method, method_name, ffeatures, test_bkt_size,
                                            prfx_len, feat_num, flag='testing')
                    preds = []
                    test_y_bucket = []
                    test_buckets_grouped = df_test_bucket.groupby(dm.case_id_col)
                    round_count = 99
                    for idx, grouppred in test_buckets_grouped:
                        round_count += 1
                        # creating counters upon which the local explanation loops will be entered (based on the size of the test bucket)
                        qualifier_dic = {}
                        if method_name == 'prefix_index':
                            for j in range(1, 4):
                                qualifier_dic['qualifier_%s' % (j)] = round_count % (5 ** j)
                        else:
                            for j in range(1, 4):
                                qualifier_dic['qualifier_%s' % (j)] = round_count % (10 ** (j + 1))

                        test_y_all.extend(dm.get_label_numeric(grouppred))
                        if method_name == 'prefix_index':
                            test_y_bucket.extend(dm.get_label_numeric(grouppred))
                        start_prediction = time.time()
                        preds_pos_label_idx = np.where(cls_experiment.classes_ == 1)[0][0]
                        pred = pipeline_final.predict_proba(grouppred)[:, preds_pos_label_idx]
                        pipeline_final_prediction_time = time.time() - start_prediction
                        current_online_times.append(pipeline_final_prediction_time / len(grouppred))
                        preds.extend(pred)
                        if (i == 2) and (explain_flag == True) and (
                                ((dataset_ref == 'sepsis_cases') and (qualifier_dic['qualifier_1'] == 0)) or (
                                (dataset_ref != 'sepsis_cases') and (qualifier_dic['qualifier_2'] == 0))):
                            # explaining the first event of the group
                            encoded_testing_group = featureCombinerExperiment.fit_transform(grouppred)
                            test_instance = np.transpose(encoded_testing_group[0])
                            case_id = dm.get_case_ids(grouppred)

                            explain_local_instances(artefacts_dir, dataset_ref, resexpfile, explanationfile,
                                                    cls_experiment, \
                                                    limeexplainer, test_instance, case_id, grouppred, round_count, dm,
                                                    pipeline_final, method_name)

                            if ((dataset_ref == 'sepsis_cases') and (qualifier_dic['qualifier_2'] == 0)) or (
                                    (dataset_ref != 'sepsis_cases') and (qualifier_dic['qualifier_3'] == 0)):
                                case_id = case_id[0]
                                print(case_id)
                                case_idx = \
                                    [i for i in
                                     np.where(np.all(encoded_testing_bucket == encoded_testing_group, axis=1))][
                                        0][0]

                                shap_local(artefacts_dir, cls_method, dataset_name, method_name,
                                               encoded_testing_bucket,
                                               case_idx, case_id, ffeatures, test_bkt_size, test_prfx_len, feat_num,
                                               False)
                    if (i == 2) and (explain_flag == True):
                        y_real = test_y_bucket if method_name == 'prefix_index' else test_y_all
                        Permutation_importance_analysis(artefacts_dir, pipeline_final.named_steps['cls'], method_name, \
                                                        ffeatures, encoded_training, train_y_experiment, \
                                                        encoded_testing_bucket, y_real, dataset_name, cls_method,
                                                        bkt_size, prfx_len, test_bkt_size, test_prfx_len, feat_num)
                        compute_correlations(cls_method, method_name, ffeatures, encoded_training, train_y_experiment, \
                                             encoded_testing_bucket, y_real, dataset_name, cls_encoder_args_final,
                                             feat_num)
                        ALE_Computing(encoded_training, pipeline_final.named_steps['cls'], \
                                      ffeatures, dataset_name, method_name, cls_method, bkt_size, prfx_len, \
                                      feat_num, [dm.pos_label, dm.neg_label])

                        # plotting dependence plots for the training dataset
                        plot_dependence_plots(artefacts_dir, encoded_training, dataset_name, cls_method,
                                                  method_name, ffeatures, True, bkt_size, prfx_len, feat_num)
                        if (method_name == 'single_agg'):
                                # plotting dependence plots for the testing dataset
                                plot_dependence_plots(artefacts_dir, encoded_testing_bucket, dataset_name, cls_method,
                                                      method_name, ffeatures, False, test_bkt_size, test_prfx_len,
                                                      feat_num)
            preds_all.extend(preds)
        offline_total_time = offline_bucket + offline_time_fit + train_prefix_generation
        offline_times.append(offline_total_time)
        online_times.append(current_online_times)
    with open(outfile, 'w') as out:
        out.write('%s;%s;%s;%s;%s;%s;%s\n' % ('dataset', 'method', 'cls', 'nr_events', 'n_iter', 'metric', 'score'))
        out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
            dataset_name, method_name, cls_method, -1, -1, 'test_prefix_generation_time', prefix_test_generation))
        for j in range(len(offline_times)):
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'train_prefix_generation_time', train_prefix_generation))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'offline_bucket_time', offline_bucket))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'total_offline_time', offline_times[j]))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'online_time_average', np.mean(online_times[j])))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'online_time_standard', np.std(online_times[j])))
        df_results = pd.DataFrame({'actual': test_y_all, 'predicted': preds_all, 'nr_events': nr_events_all})
        for nr_events, group in df_results.groupby('nr_events'):
            if len(set(group.actual)) < 2:
                out.write(
                    "%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan))
            else:
                try:
                    out.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc",
                                                          roc_auc_score(group.actual, group.predicted)))
                except ValueError:
                    pass
        try:
            out.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "auc",
                                                  roc_auc_score(df_results.actual, df_results.predicted)))
        except ValueError:
            pass
        online_event_times_flat = [t for iter_online_event_times in online_times for t in iter_online_event_times]
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat)))
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat)))
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_times)))
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_times)))

