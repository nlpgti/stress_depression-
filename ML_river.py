import os
import random
import warnings
from multiprocessing import Pool

import pandas as pd
from river import linear_model, dummy, stats
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingAdaptiveTreeClassifier

from utils.utils import print_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from river import stream, cluster
from river import ensemble, feature_selection


def instance_distance(x_original: dict, x_modified: dict):
    """
    Calcula la suma de diferencias absolutas entre features numéricos.
    """
    dist = 0.0
    dist_dic={}
    for feature in x_original:
        dist_aux = abs(x_original[feature] - x_modified[feature])
        if dist_aux > 0.1:
            dist_dic[feature]=x_original[feature] - x_modified[feature]
        dist += dist_aux
    return dist, dist_dic

def find_counterfactual(
        dataset,
        model,
        x_original: dict,
        desired_label,
        n_iter: int = 1000,
        step_size: float = 0.2):
    list_features_to_change = [ 'polaridad', 'interjecciones', 'inseguridad',
       'problemas_salud', 'emociones_positivas', 'emociones_negativas',
       'tristeza', 'angustia', 'soledad', 'negativos', 'adverbios_negativos',
       'términos_catastróficos', 'términos_exagerados', 'conceptos_repetidos']
    list_features_to_change = list(dataset.columns[dataset.columns.str.contains('@')]) + list_features_to_change
    numeric_bounds = {
        col: (dataset[col].min(), dataset[col].max()) for col in dataset.columns
    }

    best_cf = None
    best_distance = float('inf')
    best_dist_dic={}

    # Búsqueda aleatoria
    for _ in range(n_iter):
        x_candidate = x_original.copy()

        # Perturbamos cada feature dentro de sus límites
        for feature in x_candidate:
            if feature in list_features_to_change:
                val = x_candidate[feature]

                # Obtenemos rango del feature
                if feature in numeric_bounds:
                    (min_val, max_val) = numeric_bounds[feature]
                else:
                    min_val=0
                    max_val=1


                # Generamos un factor aleatorio en [-step_size, step_size]
                perturb_factor = random.uniform(-step_size, step_size)
                # Aplicamos perturbación proporcional al rango
                range_ = max_val - min_val
                new_val = val + perturb_factor * range_
                # Clampeamos
                new_val = max(min_val, min(new_val, max_val))
                x_candidate[feature] = new_val

        # Predecimos con la instancia perturbada
        pred = model.predict_one(x_candidate)
        y_pred_proba = model.predict_proba_one(x_candidate)[desired_label]

        # Si la predicción es la que deseamos, calculamos su "distancia"
        if pred == desired_label and y_pred_proba > 0.5:
            dist ,dist_dic= instance_distance(x_original, x_candidate)
            # if dist < best_distance:
            best_dist_dic=dist_dic
            best_distance = dist
            best_cf = x_candidate.copy()
            return y_pred_proba, best_cf, best_distance, best_dist_dic
        return y_pred_proba, best_cf, best_distance, best_dist_dic


def start_river_analysis(data):
    path = data["path"]
    cold_start = data["cold_start"]
    model_name = data["model_name"]
    model_hyper_params = data["model_hyper_params"]
    hyper_params_evaluating=data["hyper_params_evaluating"]
    columns_to_drop = data["columns_to_drop"]
    scenario = data["scenario"]
    target = data["target"]
    labels = data["labels"]
    window_step = data["window_step"]
    step_train = data["step_train"]
    variance = data["variance"]
    only_users=data["only_users"]
    train_slots=data["train_slots"]
    threshold_train_slots=data["threshold_train_slots"]
    correlation_selection=data["correlation_selection"]

    selector_bool = data["selector"]
    explainability = data["explainability"]
    verbose = data["verbose"]
    clustering_boolean = False
    if verbose:
        print("Start river analysis")
    model_to_analyse = None

    if model_name == "baseline":
        model_to_analyse = dummy.PriorClassifier()

    if model_name == "alma":
        model_to_analyse = linear_model.ALMAClassifier( alpha = model_hyper_params["alpha"],
                                                        B = model_hyper_params["B"],
                                                        C = model_hyper_params["C"])

    if model_name == "arfc":
        model_to_analyse = ensemble.AdaptiveRandomForestClassifier(n_models=model_hyper_params["n_models"],
                                                                   max_features=model_hyper_params["max_features"],
                                                                   lambda_value=model_hyper_params["lambda_value"],
                                                                   seed=1)
    elif model_name == "hatc":
        model_to_analyse = HoeffdingAdaptiveTreeClassifier(max_depth=model_hyper_params["max_depth"],
                                                           tie_threshold=model_hyper_params["tie_threshold"],
                                                           max_size=model_hyper_params["max_size"],
                                                           seed=1)

    elif model_name == "gnb":
        model_to_analyse = GaussianNB()
    elif model_name == "clustering":
        model_to_analyse = cluster.KMeans(n_clusters=3,
                                          halflife=model_hyper_params["halflife"],
                                          mu=model_hyper_params["mu"],
                                          sigma=model_hyper_params["sigma"],
                                          p=model_hyper_params["p"], seed=1)
        clustering_boolean = True

    classifier_model = {"model": model_to_analyse,
                        "elements": 0}

    dataset = pd.read_csv(path)


    list_params_eval = [ 'polaridad', 'interjecciones', 'inseguridad',
       'problemas_salud', 'emociones_positivas', 'emociones_negativas',
       'tristeza', 'angustia', 'soledad', 'negativos', 'adverbios_negativos',
       'términos_catastróficos', 'términos_exagerados', 'conceptos_repetidos']+columns_to_drop
    columns_with_at = dataset.columns[dataset.columns.str.contains('@')]
    filtered_columns = [col for col in columns_with_at if not any(x in col.lower() for x in ['ansiedad', 'depresión'])]
    list_params_eval = list(filtered_columns) + list_params_eval

    dataset = dataset[list_params_eval]
    dataset = dataset.sort_values(by=['timestamp'], ascending=True)
    dataset.reset_index(drop=True, inplace=True)

    dataset = dataset[dataset["target"] != "depression"].reset_index(drop=True)

    dataset = dataset.sort_values(by=["user_id", "timestamp"])

    # Aplicar la función y resetear el índice
    dataset = (
        dataset.groupby("user_id", group_keys=False)
        .apply(sample_every_3_include_last)
        .reset_index(drop=True)
    )

    if verbose:
        print(dataset.shape)

    dataset=dataset.round(2)
    if only_users:
        dataset = dataset.drop_duplicates(subset=['user_id'], keep='last')
        dataset.reset_index(drop=True, inplace=True)

    if verbose:
        print(dataset.shape)

    dataset = dataset.fillna(0)
    dataset = dataset.drop_duplicates()
    dataset.reset_index(drop=True, inplace=True)

    if verbose:
        print(dataset.shape)

    dataset = dataset.iloc[:-1:window_step]

    if verbose:
        print(dataset.shape)

    y = dataset[target]

    X = dataset[dataset.columns.difference(columns_to_drop)]

    if correlation_selection:
        selector = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=round(len(X.columns)*0.80))
    else:
        selector = feature_selection.VarianceThreshold(threshold=variance)

    if verbose:
        print(X.shape)
    list_y_pred = []
    list_y = []
    count_train = 0
    list_x_river = []
    list_x_river_window = []
    list_last_items_to_train=[]


    for x_rive_new, y_river in stream.iter_pandas(X, y):
        if selector_bool:
            if correlation_selection:
                x_river = selector.learn_one(x_rive_new,y_river).transform_one(x_rive_new)
            else:
                x_river = selector.learn_one(x_rive_new).transform_one(x_rive_new)

        else:
            x_river = x_rive_new

        if clustering_boolean:
            list_x_river.append(x_river)
            list_x_river_window.append(x_river)

        if classifier_model["elements"] == 0 and clustering_boolean:
            classifier_model["model"] = classifier_model["model"].learn_one(x_river)

        y_pred = classifier_model["model"].predict_one(x_river)
        if y_pred is not None:
            y_pred_proba=classifier_model["model"].predict_proba_one(x_river)[y_pred]

            if model_name == "arfc" and explainability:
                if 0.6 < y_pred_proba < 1.0 and y_river==y_pred:
                    if y_river!="none":
                        desired_label="none"
                        proba_return, best_cf, best_distance, best_dist_dic=find_counterfactual( X,classifier_model["model"], x_rive_new, desired_label, 1000,0.2)

                        print(y_pred_proba, proba_return)
                        for key, value in best_dist_dic.items():
                            print(f"{key}: {value:.2f}")

        if verbose:
            print(classifier_model["elements"])

        if y_pred != None:
            list_y_pred.append(y_pred)
            list_y.append(y_river)

        try:
            if step_train > 0:
                if classifier_model["elements"] % step_train == 0:
                    count_train = count_train + 1
                    if not clustering_boolean:
                        classifier_model["model"] = classifier_model["model"].learn_one(x_river, y_river)
                    else:
                        classifier_model["model"] = classifier_model["model"].learn_one(x_river)
            else:
                count_train = count_train + 1
                if not clustering_boolean:
                    if train_slots:
                        if count_train<threshold_train_slots:
                            classifier_model["model"] = classifier_model["model"].learn_one(x_river, y_river)
                        else:
                            if len(list_last_items_to_train)==threshold_train_slots:
                                for elements_to_train in list_last_items_to_train:
                                    classifier_model["model"] = classifier_model["model"].learn_one(elements_to_train["x_river"], elements_to_train["y_river"])
                                list_last_items_to_train = []
                            else:
                                list_last_items_to_train.append({"x_river":x_river,"y_river":y_river})

                    else:
                        classifier_model["model"] = classifier_model["model"].learn_one(x_river, y_river)
                else:
                    classifier_model["model"] = classifier_model["model"].learn_one(x_river)

            classifier_model["elements"] = classifier_model["elements"] + 1
        except Exception as e:
            print(e)
            selector = feature_selection.VarianceThreshold(threshold=variance)

    if verbose:
        print("Train count: " + str(count_train))

    return print_metrics(data,model_name, list_y, list_y_pred)

def run_paper_experiments(json_experiments):

    target = "target"
    labels = ['both', 'anxiety', 'none']
    only_avg = False
    columns_to_drop = ['timestamp', 'target','user_id']

    window_step = 1
    step_train = json_experiments["step_train"]

    variance=0

    path = "path"

    model_hyper_params = {
                            "n_models": 100, "max_features": 50, "lambda_value": 50,
                            "max_depth": 200, "tie_threshold": 0.0005, "max_size": 200,
                          }

    data = {"path": path,
            "cold_start":json_experiments["cold_start"],
            "model_name": json_experiments["model"],
            "model_hyper_params": model_hyper_params,
            "hyper_params_evaluating": False,
            "columns_to_drop": columns_to_drop,
            "only_avg": only_avg,
            "target": target,
            "labels": labels,
            "scenario": json_experiments["scenario"],
            "window_step": window_step,
            "step_train": step_train,
            "variance": variance,
            "only_users" :json_experiments["only_users"],
            "train_slots":json_experiments["train_slots"],
            "threshold_train_slots":json_experiments["threshold_train_slots"],
            "correlation_selection":json_experiments["correlation_selection"],
            "balanced":json_experiments["balanced"],
            "selector": True, "explainability": False, "verbose": False}

    complete_info_result = start_river_analysis(data)
    print(complete_info_result["latex"])

def run_parallel_paper():
    list_run=[]
    list_models= ["gnb", "alma", "hatc", "arfc"]
    for model in list_models:
        json_experiments = {"cold_start": 0,
                            "correlation_selection": False,
                            "train_slots": False,
                            "threshold_train_slots": 100,
                            "only_users": False,
                            "scenario": 0,
                            "step_train": 0,
                            "model": model,
                            "balanced":False}
        list_run.append(json_experiments)


    p = Pool()
    list_data = p.map(run_paper_experiments,list_run)
    p.close()
    p.join()

    list_scenario= [0]
    for scenario in list_scenario:
        for model in list_models:
           print([d for d in list_data if d['scenario'] == scenario and  d['model'] == model][0]["text"])

def sample_every_3_include_last(group):
    idx = list(range(0, len(group), 3))
    if (len(group) - 1) not in idx:
        idx.append(len(group) - 1)
    return group.iloc[idx]


def assign_predicted(row):
    ansiedad = row['ansiedad@15@q3'] >= 0.5
    depresion = row['depresión@15@q3'] >= 0.5

    if ansiedad and depresion:
        return 'both'
    elif ansiedad:
        return 'anxiety'
    else:
        return 'none'

def evaluate_paper_gpt():
    path = "path"
    dataset = pd.read_csv(path)
    dataset = dataset[dataset["target"] != "depression"].reset_index(drop=True)
    dataset = dataset.sort_values(by=["user_id", "timestamp"])

    dataset = (
        dataset.groupby("user_id", group_keys=False)
        .apply(sample_every_3_include_last)
        .reset_index(drop=True)
    )

    dataset['predicted_gtp'] = dataset.apply(assign_predicted, axis=1)

    complete_info_result=print_metrics(None,"llm", dataset['target'], dataset['predicted_gtp'])
    print(complete_info_result["latex"])


def run_paper():
    evaluate_paper_gpt()
    list_models= ["gnb", "hatc", "arfc"]
    for z in list_models:
        json_experiments = {"cold_start": 0,
                            "correlation_selection": False,
                            "train_slots": False,
                            "threshold_train_slots": 100,
                            "only_users": False,
                            "scenario": 0,
                            "step_train": 0,
                            "model": z,
                            "balanced":False}
        run_paper_experiments(json_experiments)

if __name__ == '__main__':
    run_paper()

