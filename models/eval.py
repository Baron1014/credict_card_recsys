import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score

def ndcg(df, feature_names, model, label_encoder, k=3):
    total = 0
    s=0
    for user_id in tqdm(df["chid"].unique(), desc="ndcg score"):
        # 取得使用者數值
        tmp = df[df["chid"]==user_id]
        tmp.sort_values(by=['txn_amt'], ascending=False, inplace=True)
        tmp.drop_duplicates(subset=['shop_tag'], inplace=True)
        tmp = tmp[tmp["shop_tag"] != 0]
        
        # 進行預測
        tmp_input = {name:tmp[name].values for name in feature_names}
        pred_tmp = model.predict(tmp_input, batch_size=256)
        
        # 查找相對應tags
        true = tmp["shop_tag"].values.astype(str)
        if len(true) < 3:
            continue
        tags_label = label_encoder.transform(true.tolist())
        sort_max = tags_label[np.argsort(pred_tmp, axis=0)[::-1]]
        pre = label_encoder.inverse_transform(sort_max)
        score = ndcg_score(np.array(true).reshape(1, -1), np.array(pre).reshape(1, -1), k=k)
        
        #clear_output()
        total += score
        s+=1
    # 返回ndcg平均
    return total / s

# 輸出上傳比賽資料
def make_final(model, tags, feature_names, columns, user_attributes, output_path):
    final_df = pd.read_csv("../../data/需預測的顧客名單及提交檔案範例.csv")

    id_to_num = {id:i for i, id in enumerate(final_df["chid"].unique())}
    output = list()
    for user in tqdm(final_df["chid"].values, desc="output to csv"):
        inference_data = get_inference_data(user, tags, id_to_num, user_attributes)
        user_df = pd.DataFrame(inference_data, columns=columns)
        user_model_input = {name:user_df[name].values for name in feature_names}
        pred_final = model.predict(user_model_input, batch_size=256)
        user_max_tags = tags[np.argsort(pred_final, axis=0)[::-1][:3]]
        output.append(np.insert(user_max_tags, 0, user))
    
    print(output)
    output_csv = pd.DataFrame(output, columns=["chid", "top1", "top2", "top3"])
    print(output_csv.head(10))
    output_csv.to_csv(output_path, index=False)

# 取得預測的輸入資料
def get_inference_data(user, tags, user_attributes):
    #generate data
    inference_data = [[
                      tag,
                      user_attributes[user]["masts"],
                      user_attributes[user]["txn_cnt"],
                      user_attributes[user]["educd"],
                      user_attributes[user]["trdtp"],
                      user_attributes[user]["poscd"],
                      user_attributes[user]["gender_code"],
                      user_attributes[user]["age"],
                      user_attributes[user]["primary_card"],
                      25]
                      for tag in tags]

    return inference_data