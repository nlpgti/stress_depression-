import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer


def change_target_to_binary(y):
    list_aux=[]
    for z in y:
        if z=="none":
            list_aux.append(["no","no"])
        elif z=="anxiety":
            list_aux.append(["si", "no"])
        elif z=="depression":
            list_aux.append(["no", "si"])
        elif z=="both":
            list_aux.append(["si", "si"])

    return list_aux

def print_metrics(data,model_name, y, y_pred):
    y_binary = change_target_to_binary(y)
    y_pred = change_target_to_binary(y_pred)
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(y_binary)
    y_binary = mlb.transform(y_binary)
    y_pred = mlb.transform(y_pred)

    acc = 0
    for w in range(len(y_pred)):
        acc += sum(np.logical_and(y_pred[w], y_binary[w])) / sum(np.logical_or(y_pred[w], y_binary[w]))

    acc = acc / len(y_pred)
    latex_result = "& \\textsc{" + model_name + "}" + " & {:.2f}".format(acc * 100) \
                   + " & {:.2f}".format(accuracy_score(y_binary, y_pred) * 100) \
                   + " & {:.2f}".format(hamming_loss(y_binary, y_pred) * 100) \
                   + " & {:.2f}".format(
        precision_score(y_true=y_binary, y_pred=y_pred, average='macro', zero_division=0) * 100) \
                   + " & {:.2f}".format(
        precision_score(y_true=y_binary, y_pred=y_pred, average='samples', zero_division=0) * 100) \
                   + " & {:.2f}".format(recall_score(y_true=y_binary, y_pred=y_pred, average='macro') * 100) \
                   + " & {:.2f}".format(recall_score(y_true=y_binary, y_pred=y_pred, average='samples') * 100) \
                   + " & {:.2f}".format(f1_score(y_true=y_binary, y_pred=y_pred, average='macro') * 100) \
                   + " & {:.2f}".format(f1_score(y_true=y_binary, y_pred=y_pred, average='samples') * 100) \
                   + "\\\\"
    if data==None:
        model_hyper_params=""
    else:
        model_hyper_params=data["model_hyper_params"]

    return {"latex":latex_result, "acc":acc, "model_hyper_params":model_hyper_params}
