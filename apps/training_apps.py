import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataaccessframeworks.read_data import read_raw, get_user_features
from sklearn.preprocessing import LabelEncoder
from models.deepfm import get_feature, run_model

def main(save_path):
    user_attribute_col = ["masts", "txn_cnt", "educd", "trdtp", "poscd", "gender_code", "age", "primary_card"]
    sparse_features = ["shop_tag"]+ user_attribute_col
    dense_features = ["dt"]
    target = ["txn_cmt"]
    
    df = read_raw(dense_features + ["chid", "shop_tag", "naty", "cuorg"] + user_attribute_col + target)

    # 取得使用者特徵屬性
    user_features = get_user_features(df, user_attribute_col)

    # 將tag進行label
    lbe = LabelEncoder()
    df["shop_tag"] = df["shop_tag"].astype(str)
    df["shop_tag"] = lbe.fit_transform(df["shop_tag"]) 

    # generate faeture columns
    df, feature_names, linear_feature_columns, dnn_feature_columns = get_feature(df, sparse_features, dense_features)

    # training
    model, history, ndcg = run_model(df, feature_names, linear_feature_columns, dnn_feature_columns, target, save_path, lbe)
    print(history)


if __name__== "__main__":
    save_path = "../model/DeepFM_3.h5"
    main(save_path)
