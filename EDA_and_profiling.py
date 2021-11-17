import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
global filename
general_table = []
typeslist = []
EDA_output = 'EDA_output'
if not (os.path.exists(EDA_output)):
  os.makedirs(EDA_output)
Logs = 'logs'


def defineSepsis():
  Sepsis_datasets = ['sepsis%s' %n for n in range(1,4)]
  for d in Sepsis_datasets:
    filename[d] = os.path.join(Logs, '%s.csv'%(d))
    case_id_col[d] = 'case'
    activity_col[d] = 'concept:name'
    timestamp_col[d] = 'time:timestamp'
    resource_col[d] = "org:group"
    label_col[d] = 'label'
    pos_label[d] = 'deviant'
    neg_label[d] = 'regular'

    dynamic_cat_cols[d] = [activity_col[d],resource_col[d]]
    static_cat_cols[d] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                       'SIRSCritTemperature', 'SIRSCriteria2OrMore']
    dynamic_num_cols[d] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "day", "month", "timesincemidnight", "timesincelast", "timesincestart", "OrderOfEvent", "openCases", 'remainingtime']
    static_num_cols[d] = ['Age']
    #print(filename[d])
    if d =='sepsis3':
      pos_label["sepsis3"] = "regular"
      neg_label["sepsis3"] = "deviant"
  return


def defineTraffic():
    traff_datasets = ["traffic_final"]
    for d in traff_datasets:
        filename[d] = os.path.join(Logs, "traffic_final.csv")
        case_id_col[d] = "case:concept:name"
        activity_col[d] = "concept:name"
        resource_col[d] = "org:resource"
        timestamp_col[d] = "time:timestamp"
        label_col[d] = "label"
        pos_label[d] = "deviant"
        neg_label[d] = "regular"

        dynamic_cat_cols[d] = [activity_col[d], resource_col[d], "lastSent", "notificationType", "dismissal"]
        static_cat_cols[d] = ["article", "vehicleClass"]
        dynamic_num_cols[d] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr",
                               "month", "weekday", "hour", "open_cases"]
        static_num_cols[d] = ["amount", "points"]
    return


def defineBPIC2017():
  BPIC2017_datasets = ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused']
  for d in BPIC2017_datasets:
    filename[d] = os.path.join(Logs, '%s.csv'%(d))

    case_id_col[d] = "case:concept:name"
    activity_col[d] = "concept:name"
    resource_col[d] = 'org:resource'
    timestamp_col[d] = 'time:timestamp'
    label_col[d] = "label"
    neg_label[d] = "regular"
    pos_label[d] = "deviant"

    # features for classifier
    dynamic_cat_cols[d] = [activity_col[d], resource_col[d], 'Action', 'CreditScore', 'EventOrigin', 'lifecycle:transition',
                   "Accepted", "Selected"]
    static_cat_cols[d] = ['case:ApplicationType', 'case:LoanGoal']
    dynamic_num_cols[d] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount',
                   "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour",
                    "open_cases"]
    static_num_cols[d] = ['case:RequestedAmount']
  return


def defineHospital():
    hos_datasets = ['hospital_billing_%s' % n for n in range(1, 3)]
    for d in hos_datasets:
        filename[d] = os.path.join(Logs, '%s.csv' % (d))
        case_id_col[d] = 'cID'
        activity_col[d] = 'Activity'
        timestamp_col[d] = 'time:timestamp'
        resource_col[d] = "org:resource"
        label_col[d] = 'label'
        neg_label[d] = "regular"
        pos_label[d] = " deviant"

        dynamic_cat_cols[d] = [activity_col[d], resource_col[d], 'actOrange', 'actRed', 'blocked', 'Type', 'diagnosis',
                               'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version', 'isCancelled', 'isClosed',
                               'closeCode']
        static_cat_cols[d] = ['speciality']
        dynamic_num_cols[d] = ['msgCount', "timesincelast", "timesincestart", "event_nr", "weekday", "hour",
                               "open_cases"]
        static_num_cols[d] = []
        if d == 'hospital_billing_1':
            dynamic_cat_cols[d] = [col for col in dynamic_cat_cols[d] if col != "isClosed"]
    return


#Cramer-V categorical correlation computation
def cramers_V(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# grouping and plotting charts of attributes (value_counts) which are highly correlated to the target
def check_plot_correlations(dataset, corr_df, num):
    for x in corr_df.iloc[:, 0].values:
        if ((x >= 0.5) and (num == False)) or (((-0.5 >= x) or (0.5 <= x)) and (num == True)):
            y_cols = corr_df.index[corr_df[0] == x].to_list()
            count_df = pd.DataFrame()
            for y_col in y_cols:
                if ((y_col in static_num_cols[dataset]) or (y_col in static_cat_cols[dataset])) and (
                        y_col != corr_df[case_id_col[dataset]]):
                    count_df[y_col] = df.groupby(case_id_col[dataset]).first()[y_col]
                    count_df[label_col[dataset]] = df.groupby(case_id_col[dataset]).first()[label_col[dataset]]
                    count_df = count_df.reset_index()
                    count_df = count_df.drop(case_id_col[dataset], axis=1)
                else:
                    count_df[y_col] = df[y_col]
                    count_df[label_col[dataset]] = df[label_col[dataset]]
                count_df['Counts'] = [sum((count_df[y_col] == count_df[y_col][i]) & (
                            count_df[label_col[dataset]] == count_df[label_col[dataset]][i])) for i in
                                      range(len(count_df))]
                count_df = count_df.drop_duplicates().sort_values(by=y_col, ascending=False)
                plt.figure(figsize=(16, 8))
                sns.barplot(x=y_col, y='Counts', hue=label_col[dataset], data=count_df);
                plt.title('Value Counts of %s categorized by label values' % (y_col))
                plt.legend(loc='upper left', title='label', fontsize='large')
                plt.savefig(os.path.join(EDA_output, out,
                                         'plot_valueCounts_highlyCorrelated_%s_withTarget_in_%s.png' % (
                                         y_col, dataset)));
                plt.show()
                plt.close()
    return


def define_datasets(i):
  if i == 'sepsis':
    defineSepsis()
    dss = ['sepsis%s' %n for n in range(1,4)]
  elif i == 'traffic':
    defineTraffic()
    dss = ['traffic_final']
  elif i == 'BPIC2017':
    defineBPIC2017()
    dss = ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused']
  else:
    defineHospital()
    dss = ['hospital_billing_%s' %n for n in range(1,3)]
  return dss


def prepare_dataset(dataset):
    dtypes = {col: 'object' for col in (dynamic_cat_cols[dataset] + static_cat_cols[dataset] + [
        case_id_col[dataset] + label_col[dataset] + timestamp_col[dataset]])}
    for col in dynamic_num_cols[dataset] + static_num_cols[dataset]:
        dtypes[col] = 'float'
    df = pd.read_csv(filename[dataset], sep=';', dtype=dtypes, engine='c', encoding='ISO-8859-1', error_bad_lines=False)
    time_col = timestamp_col[dataset]
    df[time_col] = pd.to_datetime(df[time_col])
    df['encoded_label'] = [1 if label == pos_label[dataset] else 0 for label in df[label_col[dataset]]]
    return df


drefs = ['sepsis', 'traffic', 'BPIC2017', 'hospital']
for i in drefs:
    datasets = define_datasets(i)
    for dataset in datasets:
        out = 'exploration_%s' % (dataset)
        if not (os.path.exists(os.path.join(EDA_output, out))):
            os.makedirs(os.path.join(EDA_output, out))
        df = prepare_dataset(dataset)
        # creating a pandas_profile for the whole dataset while the label is encoded
        profile = ProfileReport(df, title='Pandas Profile _%s' % (dataset), html={'style': {'full_width': True}})
        profile.to_widgets()
        output_html_file = os.path.join(EDA_output, out, 'pandas_profile_%s_encodedLabel.html' % (dataset))
        profile.to_file(output_file=output_html_file)
        # creating a pandas_profile to only static attributes
        df_static = pd.concat(
            [df[static_num_cols[dataset]], df[static_cat_cols[dataset]], df[case_id_col[dataset]], df['encoded_label']],
            axis=1, sort=False)
        static_df = pd.DataFrame(df_static.groupby(case_id_col[dataset], as_index=False).first())
        profile = ProfileReport(static_df, title='Pandas Profile _%s_static_attributes' % (dataset),
                                html={'style': {'full_width': True}})
        # profile.to_widgets()
        output_html_file = os.path.join(EDA_output, out, 'pandas_profile_%s_static.html' % (dataset))
        profile.to_file(output_file=output_html_file)
        # creating a pandas_profile to only dynamic attributes
        df_dynamic = pd.DataFrame(pd.concat(
            [df[dynamic_num_cols[dataset]], df[dynamic_cat_cols[dataset]], df[case_id_col[dataset]],
             df[timestamp_col[dataset]], df['encoded_label']], axis=1, sort=False))
        profile = ProfileReport(df_dynamic, html={'style': {'full_width': True}})
        output_html_file = os.path.join(EDA_output, out, 'pandas_profile_%s_dynamic.html' % (dataset))
        profile.to_file(output_file=output_html_file)

        # i dropped these attributes because they have constant values
        if dataset in ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused']:
            df.drop(['Action', 'EventOrigin', 'lifecycle:transition', 'Selected', 'Accepted', 'case:ApplicationType'],
                    axis=1, inplace=True)
        # plotting scatter plots for numerical attributes
        pd.plotting.scatter_matrix(df, figsize=(15, 15), marker='o',
                                   c=df['encoded_label'], alpha=.8)
        plt.savefig(os.path.join(EDA_output, out, 'scatter matrix_%s_numerical.png' % (dataset)))
        plt.clf()
        # plotting correlations between all numerical attribtues
        num_corr = df.corr()
        num_corr.to_csv(os.path.join(EDA_output, out, 'Numerical_correlations_%s.csv' % (dataset)), sep=';')
        sns.heatmap(num_corr, annot=True, xticklabels=num_corr.columns, yticklabels=num_corr.columns)
        plt.savefig(os.path.join(EDA_output, out, 'correlation matrix_%s_numerical.png' % (dataset)))
        plt.show()
        # to check correlations (numerical) only with the label_col
        correlations = df.corrwith(df['encoded_label']).iloc[:-1].to_frame()
        # computing correlations and plotting of categorical attributes
        if dataset in ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused']:
            cat_df = df[[activity_col[dataset], resource_col[dataset], 'CreditScore'] + ['case:LoanGoal'] + [
                label_col[dataset]]]
        else:
            cat_df = df[dynamic_cat_cols[dataset] + static_cat_cols[dataset] + [label_col[dataset]]]
        rows_camers = []
        for var1 in cat_df:
            col_camers = []
            for var2 in cat_df:
                cramers = cramers_V(pd.crosstab(cat_df[var1], cat_df[var2]).to_numpy())  # Cramer's V test
                col_camers.append(round(cramers, 3))  # Keeping of the rounded value of the Cramer's V
            rows_camers.append(col_camers)
        cramers_results = np.array(rows_camers)
        df_camers = pd.DataFrame(cramers_results, columns=cat_df.columns, index=cat_df.columns)
        df_camers.to_csv(os.path.join(EDA_output, out, 'Categorical_correlations_%s.csv' % (dataset)), sep=';')
        plt.figure(figsize=(20, 10))
        sns.heatmap(df_camers, annot=True, xticklabels=df_camers.columns, yticklabels=df_camers.columns)
        plt.savefig(os.path.join(EDA_output, out, 'correlation matrix_%s_categorical.png' % (dataset)))
        plt.show()
        plt.close()
        # to check correlations only with the label_col
        cat_corr_results = {}
        for x in cat_df:
            if x != label_col[dataset]:
                cat_corr_results[x] = round(cramers_V(pd.crosstab(cat_df[x], cat_df[label_col[dataset]]).to_numpy()), 3)
        cat_corr_results_df = pd.DataFrame(cat_corr_results.values(), index=cat_corr_results.keys())

