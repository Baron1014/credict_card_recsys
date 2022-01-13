import sys
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

from dataaccessframeworks.read_data import read_raw, get_user_features
from sklearn.preprocessing import LabelEncoder
from models.deepfm import get_feature, run_model


def main(save_path, user_attribute_col, sparse_features, training):
    dense_features = ["dt"]
    target = ["txn_amt"]
    
    df = read_raw(dense_features + ["chid", "shop_tag"] + user_attribute_col + target)

    # 取得使用者特徵屬性
    #user_features = get_user_features(df, user_attribute_col)

    # 將tag進行label
    lbe = LabelEncoder()
    df["shop_tag"] = df["shop_tag"].astype(str)
    df["shop_tag"] = lbe.fit_transform(df["shop_tag"]) 

    # generate faeture columns
    df, feature_names, linear_feature_columns, dnn_feature_columns = get_feature(df, sparse_features, dense_features)

    # training
    print("start training model..")
    model, ndcg = run_model(df, feature_names, linear_feature_columns, dnn_feature_columns, target, save_path, lbe, training=training)

    return ndcg


if __name__== "__main__":
    print(tf.config.list_physical_devices('GPU'))
    print(device_lib.list_local_devices())
    result = dict()
    # 1. Toy Example
    save_path = "models/model/DeepFM_0.h5"
    user_attribute_col = ["masts"]
    sparse_features = ["shop_tag"]+ user_attribute_col
    score = main(save_path, user_attribute_col, sparse_features, training=False)
    result["Example1"] = score
    # 2. Customer Muti-Feature
    save_path = "models/model/DeepFM_1.h5"
    user_attribute_col = ["masts", "txn_cnt", "educd", "trdtp", "poscd", "gender_code", "age", "primary_card", "naty", "cuorg"]
    sparse_features = ["shop_tag"]+ user_attribute_col
    score = main(save_path, user_attribute_col, sparse_features, training=False)
    result["Example2"] = score
    # 4. Remove naty and cuorg features
    save_path = "models/model/DeepFM_3.h5"
    user_attribute_col = ["masts", "txn_cnt", "educd", "trdtp", "poscd", "gender_code", "age", "primary_card"]
    sparse_features = ["shop_tag"]+ user_attribute_col
    score = main(save_path, user_attribute_col, sparse_features, training=False)
    result["Example3"] = score

    print(result)
