
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logger
import datetime
from dataaccessframeworks.read_data import read_preprocess_data

def main():
    logger = logger.create_logger(
            'toy_example', 'log/toy_example_log.log')
    start = datetime.datetime.now()

    # get training data
    train = read_preprocess_data()

    sparse_features = ["adj_id", "masts"]
    dense_features = ["txn_cnt"]
    target = ["shop_tag"]

    # normalize dense features
    mms = MinMaxScaler(feature_range=(0,1))
    train[dense_features] = mms.fit_transform(train[dense_features])

    # generate feature columns
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=train[feat].max() + 1,embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # training
    train, test = train_test_split(train, test_size=0.2, random_state=66)

    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}


    model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

    # predict
    pred_ans = model.predict(test_model_input, batch_size=256)
    logger.info(pred_ans)
    end = datetime.datetime.now()
    logger.info(f"Total cost: {end - start}")

def check_and_use_gpus(memory_limit: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
            tf.config.set_visible_devices(gpus[0], "GPU")

            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPU")
        except RuntimeError as error:
            raise(error)

if __name__=="__main__":
    check_and_use_gpus(memory_limit=8192)
    main()
