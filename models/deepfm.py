from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from models.eval import ndcg


def run_model(train_data, feature_names,linear_feature_columns, dnn_feature_columns, target, save_path, label_encoder, training=False):
    # traing data & testing data
    train, test = train_test_split(train_data, test_size=0.2, random_state=66)
    
    #training 
    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    model = DeepFM(linear_feature_columns,dnn_feature_columns,task='regression')
    if training is True:
        #model = multi_gpu_model(model, gpus=1)
        model.compile("adam", "mse",
                    metrics=['mse'])
        model.fit(train_model_input, train[target].values.ravel(),
                            batch_size=256, epochs=10, verbose=2, validation_split=0.2)
        # save model 
        model.save_weights(save_path)
    else:
        model.load_weights(save_path)
    
    # ndcg score
    scores = dict()
    scores["training_ndcg"] = ndcg(train, feature_names, model, label_encoder)
    scores["testing_ndcg"] = ndcg(test, feature_names, model, label_encoder)
    print(scores)
    
    return model,  scores

def get_feature(data, sparse_features, dense_features):
    #normalize dense
    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    
    #generate faeture columns 
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    return data, feature_names, linear_feature_columns, dnn_feature_columns
