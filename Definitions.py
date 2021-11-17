import os

logs_dir = 'logs'

case_id_col = {}
activity_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
resource_col = {}
global filename
filename = {}
static_cols = {}
dynamic_cols = {}
train_size = {}
test_size = {}

# Sepsis_definitions
Sepsis_datasets = ['sepsis%s' % n for n in range(1, 4)]
for d in Sepsis_datasets:
    filename[d] = os.path.join(logs_dir, '%s.csv' % (d))
    case_id_col[d] = 'case'
    activity_col[d] = 'concept:name'
    timestamp_col[d] = 'time:timestamp'
    resource_col[d] = "org:group"
    label_col[d] = 'label'
    pos_label[d] = 'deviant'
    neg_label[d] = 'regular'

    train_size[d] = 400
    test_size[d] = 100

    dynamic_cat_cols[d] = [activity_col[d], resource_col[d]]
    static_cat_cols[d] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                          'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                          'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                          'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                          'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                          'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                          'SIRSCritTemperature', 'SIRSCriteria2OrMore']
    dynamic_num_cols[d] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "day", "month", "timesincemidnight",
                           "timesincelast", "timesincestart", "OrderOfEvent", "openCases", 'remainingtime']
    static_num_cols[d] = ['Age']
    # print(filename[d])
    if d == 'sepsis3':
        pos_label["sepsis3"] = "regular"
        neg_label["sepsis3"] = "deviant"

# traffic_fines definitions
datasets = ['traffic_fines']
for dataset in datasets:
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % (dataset))
    case_id_col[dataset] = "case:concept:name"
    activity_col[dataset] = "concept:name"
    resource_col[dataset] = "org:resource"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"
    train_size[dataset] = 20000
    test_size[dataset] = 4000

    # features for classifier
    dynamic_cat_cols[dataset] = [activity_col[dataset], resource_col[dataset], "lastSent", "notificationType",
                                 "dismissal"]
    static_cat_cols[dataset] = ["article", "vehicleClass"]
    dynamic_num_cols[dataset] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr",
                                 "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ["amount", "points"]

    static_cols[dataset] = static_cat_cols[dataset] + static_num_cols[dataset] + [case_id_col[dataset]]
    dynamic_cols[dataset] = dynamic_cat_cols[dataset] + dynamic_num_cols[dataset] + [timestamp_col[dataset]]
    cat_cols = dynamic_cat_cols[dataset] + static_cat_cols[dataset]

# BPIC2017_definitions
BPIC2017_datasets = ['BPIC2017_O_Accepted', 'BPIC2017_O_Cancelled', 'BPIC2017_O_Refused']
for d in BPIC2017_datasets:
    filename[d] = os.path.join(logs_dir, '%s.csv' % (d))

    case_id_col[d] = "case:concept:name"
    activity_col[d] = "concept:name"
    resource_col[d] = 'org:resource'
    timestamp_col[d] = 'time:timestamp'
    label_col[d] = "label"
    neg_label[d] = "regular"
    pos_label[d] = "deviant"
    train_size[d] = 10000
    test_size[d] = 2000

    # features for classifier
    dynamic_cat_cols[d] = [activity_col[d], resource_col[d], 'Action', 'CreditScore', 'EventOrigin',
                           'lifecycle:transition',
                           "Accepted", "Selected"]
    static_cat_cols[d] = ['case:ApplicationType', 'case:LoanGoal']
    dynamic_num_cols[d] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount',
                           "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month",
                           "weekday", "hour",
                           "open_cases"]
    static_num_cols[d] = ['case:RequestedAmount']

# Hospital_billing definitions
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
    train_size[d] = 10000
    test_size[d] = 2000

    dynamic_cat_cols[d] = [activity_col[d], resource_col[d], 'actOrange', 'actRed', 'blocked', 'Type', 'diagnosis',
                           'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version', 'isCancelled', 'isClosed',
                           'closeCode']
    static_cat_cols[d] = ['speciality']
    dynamic_num_cols[d] = ['msgCount', "timesincelast", "timesincestart", "event_nr", "weekday", "hour", "open_cases"]
    static_num_cols[d] = []

    if d == 'hospital_billing_1':
        dynamic_cat_cols[d] = [col for col in dynamic_cat_cols[d] if col != "isClosed"]

