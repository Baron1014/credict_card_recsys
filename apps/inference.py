import pandas as pd
import numpy as np
from tqdm import tqdm

def make_final(model, tags, feature_names, columns, user_attributes):
    final_df = pd.read_csv("../data/需預測的顧客名單及提交檔案範例.csv")
    output_path = "../upload.csv"
    id_to_num = {id:i for i, id in enumerate(final_df["chid"].unique())}
    output = list()
    for user in tqdm(final_df["chid"].values, desc="output to csv"):
        inference_data = get_inference_data(user, tags, id_to_num, user_attributes)
        user_df = pd.DataFrame(inference_data, columns=columns)
        user_model_input = {name:user_df[name].values for name in feature_names}
        pred_final = model.predict(user_model_input, batch_size=256)
        user_max_tags = tags[np.argsort(pred_final, axis=0)[::-1][:3]]
        output.append(np.insert(user_max_tags[0], 0, user))

    output_csv = pd.DataFrame(output, columns=["chid", "top1", "top2", "top3"])
    print(output_csv.head(10))
    output_csv.to_csv(output_path, index=False)

def get_inference_data(user, tags, id_to_num, user_attributes):
    #generate data
    inference_data = [[id_to_num[user],
                      tag,
                      user_attributes[user]["masts"],
                      25]
                      for tag in tags]

    return inference_data
