import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def read_preprocess_data():
    df = pd.read_csv("../data/tbrain_small.csv")
    final_df = pd.read_csv("../data/需預測的顧客名單及提交檔案範例.csv")
    df = df.drop(columns=['Unnamed: 0'])
    df = df[["chid", "shop_tag", "txn_cnt", "masts"]]

    # simple preprocess
    df = df[df["txn_cnt"]>0]
    df.dropna(inplace=True)
    id_to_num = { id:i for i, id in enumerate(final_df["chid"].unique())}
    df["adj_id"] = df["chid"].map(id_to_num)
    df['masts'] = df['masts'].astype(int)
    train = df.drop(columns=["chid"])

    return train
