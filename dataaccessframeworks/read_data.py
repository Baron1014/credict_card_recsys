import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from dataaccessframeworks.preprocess_datafactory import chunk_preprocess, transfer_type

def read_preprocess_data():
    df = pd.read_csv("../data/tbrain_small.csv")
    final_df = pd.read_csv("../data/需預測的顧客名單及提交檔案範例.csv")
    df = df.drop(columns=['Unnamed: 0'])
    df = df[["dt", "chid", "shop_tag", "txn_amt", "masts"]]

    # simple preprocess
    user_attributes = dict()
    #df = df[df["txn_cnt"]>0]
    df.dropna(inplace=True)
    id_to_num = { id:i for i, id in enumerate(final_df["chid"].unique())}
    df["adj_id"] = df["chid"].map(id_to_num)
    df['masts'] = df['masts'].astype(int)
    df["txn_amt_log"] = df["txn_amt"].apply(np.log)
    # 建立使用者屬性
    train_id = df["chid"].unique()
    for i in final_df["chid"].unique():
        # 若是id有存在於訓練資料中，則取得對應數值，否則為0
        user_attributes[i] = {"masts": df[df["chid"]==i]["masts"][0] if i in train_id else 0}
    train = df.drop(columns=["chid", "txn_amt"])

    return train, user_attributes

def read_raw():
    chunksize = 1000000
    #chunks = pd.read_csv("../data/tbrain_cc_training_48tags_hash_final.csv", chunksize=chunksize, iterator=True)

    #raw_data = list()
    #for chunk in tqdm(chunks, desc="read raw data"):
    #    raw_data.append(chunk[["dt", "chid", "shop_tag", "txn_cnt", "txn_amt", "masts", "educd", "trdtp", "naty", "poscd", "cuorg", "gender_code", "age", "primary_card"]])

    #df = pd.concat(raw_data)
    #print(raw_data[0])

    #test
    chunk = pd.read_csv("../data/tbrain_cc_training_48tags_hash_final.csv", nrows=chunksize)
    chunk = chunk[["dt", "chid", "shop_tag", "txn_cnt", "txn_amt", "masts", "educd", "trdtp", "naty", "poscd", "cuorg", "gender_code", "age", "primary_card"]]
    chunk= chunk_preprocess(chunk)
    print(chunk)
    print(chunk.info())
    chunk = transfer_type(chunk)

    return chunk

if __name__=="__main__":
    df = read_raw()
