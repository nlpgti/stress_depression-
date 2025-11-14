import pandas as pd

from gpt_annotation import get_new_annotation_from_dataset


def create_path(x):
    return str(x["user_id"])+"/"+str(x["session_id"])+"/"

def prepare_dataset():
    dataset = pd.read_csv("datasets/celia.csv")
    print(dataset.shape)
    print(dataset.columns)
    print(dataset['target'].value_counts())
    print(dataset['user_id'].value_counts())

    dataset=dataset.drop_duplicates(subset=['session_id'],keep='last')
    dataset.reset_index(drop=True, inplace=True)
    print(dataset.shape)
    print(dataset['target'].value_counts())
    print(dataset['user_id'].value_counts())
    dataset['user_id'].value_counts()
    print(len(set(dataset['user_id'])))
    dataset['messages'] = dataset['messages'].astype(str)
    dataset["human_num_iteration"]=dataset["messages"].apply(lambda x: x.count("Humano: "))
    dataset=dataset.loc[dataset['human_num_iteration'] > 5]
    dataset.reset_index(drop=True, inplace=True)
    print(dataset.shape)
    print(dataset['target'].value_counts())
    dataset.to_csv("datasets/celia_preprocessed.csv", index=False, header=True)

if __name__ == '__main__':
    prepare_dataset()
    get_new_annotation_from_dataset("datasets/celia_preprocessed.csv","datasets/dataset_by_sessions_gptv3.csv")










