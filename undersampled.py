import pandas as pd
import configparser
import matplotlib.pyplot as plt
import json

# read folder path from config file
config = configparser.ConfigParser()
config.read('config.ini')
folder_path = config["dataset"]["folder_path"]


def pie_chart(data, message):
    valcount = data.value_counts()
    print(f"{message} \n({valcount})")
    plt.pie(valcount, labels = ['1', '0'])
    plt.show() 

def get_deleted_keys(df):
    valcount = df['target'].value_counts()
    drop_key = 0 if valcount[0] > valcount[1] else 1
    diff = abs(valcount[0] - valcount[1])
    to_be_deleted = []
    counter = 0
    for i in range(len(df)):
        if df.iloc[i]['target'] == drop_key:
            to_be_deleted.append(df.iloc[i]['id'])
            counter += 1
        if counter == diff:
            break

    return to_be_deleted


def undersample_targets(df, to_be_deleted, new_target_path):
    for i in to_be_deleted:
        df = df.drop(df[df["id"] == i].index)

    if new_target_path:
        df.to_csv(new_target_path)
    return df

def undersample_edges(original_path, to_be_deleted, new_target_path):
    f = open(original_path)
    edges = json.load(f)

    for i in to_be_deleted:
        del edges[str(i)]
    with open(new_target_path, "w") as undersampled_json: 
        json.dump(edges, undersampled_json)

    return edges



"Before Undersampling"

# loading target file as a pandas DataFrame
df_before = pd.read_csv(f"{folder_path}\\raw\\twitch_target.csv", encoding='utf-8')
# isolate target column
targets = df_before['target']
#draw pie chart before undersampling
pie_chart(targets, "before undersampling")

to_be_deleted = get_deleted_keys(df_before)

"After Undersampling"

df_undersampled = undersample_targets(df_before, to_be_deleted, f"{folder_path}\\raw\\twitch_target_undersampled.csv")
edges_undersampled = undersample_edges(f"{folder_path}\\raw\\twitch_edges.json", to_be_deleted, f"{folder_path}\\raw\\twitch_edges_undersampled.json")
# isolate target column
targets_undersampled = df_undersampled['target']
#draw pie chart before undersampling
pie_chart(targets_undersampled, "after undersampling")