import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import time
import math

from multiprocessing import Pool

import pandas as pd

def evaluate_nan(list_elements):
    list_result = {ki: fill_nan(vi) for ki, vi in list_elements.items()}
    return list_result

def fill_nan(value):
    if math.isnan(value):
        return 0
    else:
        return value

def get_metrics_of_subset(aux_dataset):
    mean=evaluate_nan(aux_dataset.mean(skipna=True))
    q1=evaluate_nan(aux_dataset.quantile(.25))
    q2=evaluate_nan(aux_dataset.quantile(.5))
    q3=evaluate_nan(aux_dataset.quantile(.75))
    max=evaluate_nan(aux_dataset.min())
    min=evaluate_nan(aux_dataset.max())

    return mean,q1,q2,q3,max,min

def get_all_statistics_metrics(list_columns,dataset_sensores,window):
    time_get_all_statistics_metrics=time.time()
    dataset_sensores_aux=dataset_sensores[list_columns]

    elements_rolling=[]

    if len(dataset_sensores_aux)<window:
        for index in range(len(dataset_sensores_aux)+1):
            aux_dataset = dataset_sensores_aux.iloc[0:index]
            if len(aux_dataset)>0:
                elements_rolling.append(aux_dataset)


    else:
        for index in range(window+1):
            aux_dataset = dataset_sensores_aux.iloc[0:index]

            if len(aux_dataset)>0:
                elements_rolling.append(aux_dataset)

        for index in range (len(dataset_sensores)-window):
            aux_dataset = dataset_sensores_aux.iloc[index+1:index + window+1]
            if len(aux_dataset)>0:
                elements_rolling.append(aux_dataset)

    print("init_calc")

    list_data=[]
    for elements in elements_rolling:
        list_data.append(get_metrics_of_subset(elements))

    list_mean_dataset = [item[0] for item in list_data]
    list_q1_dataset = [item[1] for item in list_data]
    list_q2_dataset = [item[2] for item in list_data]
    list_q3_dataset = [item[3] for item in list_data]
    list_max_dataset = [item[4] for item in list_data]
    list_min_dataset = [item[5] for item in list_data]

    print(time.time()-time_get_all_statistics_metrics)

    dataset_mean=pd.DataFrame(list_mean_dataset)
    dataset_mean.rename(columns=lambda x: x+"@"+str(window)+"@avg", inplace=True)
    dataset_mean.reset_index(drop=True, inplace=True)

    dataset_q1 = pd.DataFrame(list_q1_dataset)
    dataset_q1.rename(columns=lambda x: x +"@"+ str(window)+"@q1", inplace=True)
    dataset_q1.reset_index(drop=True, inplace=True)

    dataset_q2 = pd.DataFrame(list_q2_dataset)
    dataset_q2.rename(columns=lambda x: x +"@"+ str(window)+"@q2", inplace=True)
    dataset_q2.reset_index(drop=True, inplace=True)

    dataset_q3 = pd.DataFrame(list_q3_dataset)
    dataset_q3.rename(columns=lambda x: x +"@" + str(window)+"@q3", inplace=True)
    dataset_q3.reset_index(drop=True, inplace=True)

    dataset_max = pd.DataFrame(list_max_dataset)
    dataset_max.rename(columns=lambda x: x +"@" + str(window)+"@max", inplace=True)
    dataset_max.reset_index(drop=True, inplace=True)

    dataset_min = pd.DataFrame(list_min_dataset)
    dataset_min.rename(columns=lambda x: x +"@" + str(window)+"@min", inplace=True)
    dataset_min.reset_index(drop=True, inplace=True)

    result_dataset = pd.concat([dataset_mean,
                                dataset_q1,
                                dataset_q2,
                                dataset_q3,
                                ], axis=1)

    return result_dataset

def calc_allwindows_per_user(user_info):
    dataset=user_info["dataset"]
    dataset = dataset.sort_values(by=['timestamp'], ascending=True)
    dataset.reset_index(drop=True, inplace=True)

    list_columns=user_info["list_columns"]
    window=user_info["window"]
    list_to_merge = [dataset]
    X_window=get_all_statistics_metrics(list_columns,dataset,window)
    list_to_merge.append(X_window)
    dataset_one_player = pd.concat(list_to_merge, axis=1)

    return dataset_one_player


def prepare_dataset_to_rolling(list_params_eval,in_path,out_path,columns_to_drop,window):
    dataset = pd.read_csv(in_path, sep=',')
    dataset = dataset.rename(columns={'session_start': 'timestamp'})
    dataset = dataset.sort_values(by=['timestamp'], ascending=True)
    dataset.reset_index(drop=True, inplace=True)
    print(dataset.shape)

    print(dataset["target"].value_counts())

    list_user_to_calc = []
    for user_id in list(dataset["user_id"].unique()):
        dataset_player = dataset.loc[dataset['user_id'] == user_id]

        list_user_to_calc.append({"dataset": dataset_player, "list_columns": list_params_eval, "window": window})

    p = Pool()
    list_players_updated = p.map(calc_allwindows_per_user, list_user_to_calc, chunksize=10)
    p.close()
    p.join()

    cs_go_rolling_merged = pd.concat(list_players_updated, ignore_index=True)
    cs_go_rolling_merged = cs_go_rolling_merged.sort_values(by=['timestamp'], ascending=True)
    cs_go_rolling_merged.reset_index(drop=True, inplace=True)
    cs_go_rolling_merged = cs_go_rolling_merged[cs_go_rolling_merged.columns.difference(columns_to_drop)]

    cs_go_rolling_merged.to_csv(out_path, index=False, header=True)
    print(cs_go_rolling_merged.shape)
    print(list(cs_go_rolling_merged.columns))
    print(cs_go_rolling_merged["target"].value_counts())
    print(cs_go_rolling_merged.shape)