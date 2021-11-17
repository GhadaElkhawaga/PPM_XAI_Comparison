import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
import os
import numpy as np

datasets = {'single_agg':["sepsis1", "sepsis2", \
                          "sepsis3", 'traffic_fines', "hospital_billing_1",\
                          "hospital_billing_2","BPIC2017_O_Accepted",\
                          "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"], \
            'prefix_index':["sepsis1", "sepsis2", "sepsis3", 'traffic_fines', "BPIC2017_O_Refused", "BPIC2017_O_Accepted"]}
prfx_length = np.arange(1, 40, 5)
output_dir = os.path.join('discretized_datasets')
if not os.path.exists(output_dir):
  os.makedirs(os.path.join(output_dir))
for method in ['single_agg', 'prefix_index']:
    input_dir = os.path.join('encoded_datasets_%s' %(method))
    #df = pd.read_csv('encoded_training_sepsis3_single_agg_9125_1_176.csv', sep=';')
    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)
        if (filename.endswith('.csv')) and ('training' in filename) and (bool( [ele for ele in datasets[method] if(ele in filename)])):
            if method == 'prefix_index':
                if not (bool([elm for elm in prfx_length if ('_' + str(elm) + '_' in filename)])):
                    continue
            df = pd.read_csv(os.path.join(input_dir,filename), sep=';')
            selected_columns = df[df.columns[:-1]]
            X_df = selected_columns.copy()
            y_series = df[df.columns[-1]].copy()
            for i in X_df:
                if len(X_df[i].value_counts()) >= 10:
                    bins = 4
                else:
                    bins = len(X_df[i].value_counts())
            try:
                discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
                X_df = pd.DataFrame(discretizer.fit_transform(X_df), columns=df.columns[:-1])
                df = pd.concat([X_df, y_series], axis=1, join='inner')
                df.to_csv(os.path.join(output_dir, 'discretized_dataset_%s' % (filename)),
                          sep=',', index=False)
            except:
                continue

