import pandas as pd
import numpy as np
import os
from helpers.DatasetManager import DatasetManager


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
logs_dir = 'logs'


def defineSepsis():
  Sepsis_datasets = ['sepsis%s' %n for n in range(1,4)]
  for d in Sepsis_datasets:
    filename[d] = os.path.join(logs_dir, '%s.csv'%(d))
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
    dynamic_num_cols[d] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight", "timesincelast", "timesincestart", "OrderOfEvent", "openCases"]
    static_num_cols[d] = ['Age']
    #print(filename[d])
    if d =='sepsis3':
      pos_label["sepsis3"] = "regular"
      neg_label["sepsis3"] = "deviant"
  return


def defineTraffic():
  traff_datasets = ["traffic_fines"]
  for d in traff_datasets:
    filename[d] = os.path.join(logs_dir, "traffic_fines.csv")
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
    filename[d] = os.path.join(logs_dir, '%s.csv'%(d))
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
    filename[d] = os.path.join(logs_dir, '%s.csv' % (d))
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
    dynamic_num_cols[d] = ['msgCount', "timesincelastevent", "timesincecasestart", "event_nr", "weekday", "hour",
                           "open_cases"]
    static_num_cols[d] = []
    if d == 'hospital_billing_1':
      dynamic_cat_cols[d] = [col for col in dynamic_cat_cols[d] if col != "isClosed"]
  return


drefs = ['sepsis', 'traffic', 'BPIC2017', 'hospital']
dataset_params = {}
for dref in drefs:
  if dref == 'sepsis':
    defineSepsis()
    datasets = ['sepsis%s' % n for n in range(1, 4)]
  elif dref == 'traffic':
    defineTraffic()
    datasets = ['traffic_fines']
  elif dref == 'BPIC2017':
    defineBPIC2017()
    datasets = ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused']
  else:
    defineHospital()
    datasets = ['hospital_billing_%s' % n for n in range(1, 3)]
  for x in datasets:
    dm = DatasetManager(x)
    dataset_params[x] = dm
    df = dm.read_dataset()
    sizes = df.groupby(dm.case_id_col, as_index=False).size()
    class_freqs = df.groupby(dm.case_id_col, as_index=False).first()[dm.label_col].value_counts()
    if "traffic_fines" in x:
      max_prefix_length = 10
    elif "bpic2017" in x:
      max_prefix_length = min(20, dm.get_pos_case_length_quantile(df, 0.90))
    else:
      max_prefix_length = min(40, dm.get_pos_case_length_quantile(df, 0.90))
    df[dm.case_id_col] = df[dm.case_id_col].astype(str)
    df[dm.activity_col] = df[dm.activity_col].astype(str)
    n_trace_variants = len(df.sort_values(dm.timestamp_col, kind="mergesort").groupby(dm.case_id_col, as_index=False) \
                           .head(max_prefix_length).groupby(dm.case_id_col, as_index=False)[dm.activity_col] \
                           .apply(lambda y: "__".join(list(y))).unique())
    n_static_cat_levels = 0
    n_dynamic_cat_levels = 0
    for _, col in enumerate(dm.keys_dynamic_cat_cols):
      n_dynamic_cat_levels += len(df[col].unique())
    for _, col in enumerate(dm.keys_static_cat_cols):
      n_static_cat_levels += len(df[col].unique())
    # to generate files with num of unique values in columns:
    unique_values = df.nunique(dropna=False)
    unique_values.to_csv('unique_vals_counts_%s.csv' % (x), sep=';')
    unique_values_df = unique_values.to_frame()
    with open('counts_file.html', 'a') as c:
      c.write('\n\n' + '%s' % (x))
      c.write(unique_values_df.to_html() + '\n\n')
    c.close()

    general_table.append([x, len(df[dm.case_id_col].unique()), sizes.min(), sizes.quantile(0.50), sizes.max(),
                          max_prefix_length, n_trace_variants,
                          round(class_freqs[dm.pos_label] / len(df[dm.case_id_col].unique()), 5),
                          len(df[dm.activity_col].unique()), len(dm.static_cat_cols) + len(dm.static_num_cols),
                          len(dm.dynamic_cat_cols) + len(dm.dynamic_num_cols), n_static_cat_levels,
                          n_dynamic_cat_levels])
general_table_pd = pd.DataFrame(general_table, columns =['DatasetName', '# of traces', 'shortest trace', 'average trace length',
                                                         'longest trace', 'max prefix length', '# of trace variants', 'pos class percentage',
                                                         '#no of event classes','#of static col', '# dynamic cols','# cat levels in static cols', '#cat levels in dynamic cols'])
stats_csv = general_table_pd.to_csv('statistics.csv', sep=';')
stats_file = 'statistics_file.html'
html_file = general_table_pd.to_html()
stats = open(stats_file, 'w')
stats.write(html_file)
stats.close()