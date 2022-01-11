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

def read_raw(read_col):
    chunksize = 1000000
    chunks = pd.read_csv("../data/tbrain_cc_training_48tags_hash_final.csv", chunksize=chunksize, iterator=True)

    raw_data = list()
    for i, chunk in enumerate(chunks):
        print(f"read raw data:{i*chunksize}")
        chunk = chunk[read_col]
        chunk= chunk_preprocess(chunk)
        #chunk = transfer_type(chunk)
        raw_data.append(chunk)
    
    # 資料合併 
    df = pd.concat(raw_data)
    
    # 資料型態轉換
    if "shop_tag" in read_col:
        df["shop_tag"] = df["shop_tag"].astype("category")
    # 需轉換為in欄位
    int_list = ["age", "naty", "cuorg", "masts", "educd",
                "trdtp", "poscd", "gender_code"]
    for col in int_list:
        if col in read_col:
            df[col] = df[col].astype(int)
    # 將價格取log
    if "txn_amt" in read_col:
        df["txn_amt"] = df["txn_amt"].apply(np.log)

    print(df.info())
    print(df.shape)

    return df


def get_user_features(df, user_feature):
    # 建立使用者屬性
    final_df = pd.read_csv("../data/需預測的顧客名單及提交檔案範例.csv")
    user_attributes = dict()
    train_id = df["chid"].unique()
    for i in tqdm(final_df["chid"].unique(), desc="get attributes"):
        # 若是id有存在於訓練資料中，則取得對應數值，否則為0
        if i in train_id:
            filter_col = df[df["chid"]==i].iloc[0]
            user_attributes[i] = {col: filter_col[col].astype("int") for col in user_feature}
        else:
            user_attributes[i] = {col: 0 for col in user_feature}

    return user_attributes

if __name__=="__main__":
    df = read_raw()
