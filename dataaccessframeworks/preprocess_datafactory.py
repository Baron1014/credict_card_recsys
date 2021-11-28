def chunk_preprocess(df):
    # 將移失值排除
    df.dropna(inplace=True)

    # 消費次數為負值不合理，因此進行排除
    df = df[df["txn_cnt"]>0]

    return df

def transfer_type(df):
    df["shop_tag"] = df["shop_tag"].astype("category")
    df["age"] = df["age"].astype(int)
    df["gender_code"] = df["gender_code"].astype(int)

    print(df.info())

    return df
