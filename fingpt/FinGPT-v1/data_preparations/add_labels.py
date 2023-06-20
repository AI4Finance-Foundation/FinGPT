import sys
sys.path.append("../../FinRL-Meta/")
sys.path.append("D:/python_project/FinRL-Meta/FinRL-Meta")

import os
import datetime
import pandas as pd
from lxml import etree
from tqdm import tqdm

from meta.data_processors.akshare import Akshare

'''
You may install FinRL-Meta package with the following code:

    git clone git@github.com:AI4Finance-Foundation/FinRL-Meta.git
    cd FinRL-Meta
    python setup.py sdist
    cd dist
    pip install finrl-meta-0.3.6.tar.gz
    
'''

from loguru import logger

# The result_path should be the results with only titles which is the IN path
result_path = "D:/python_project/FinRL-Meta/experiment/scrape/results_with_content/"

# The base_path should be the results with labels which is the OUT path
base_path = "D:/python_project/FinRL-Meta/experiment/scrape/content_with_labels/"
file_list = os.listdir(result_path)

def add_label(x, df_price, foward_days = 5, threshold = 0.02, threshold_very = 0.06):
    publish_date = x.publish_date.strftime("%Y-%m-%d")
    last_date = df_price[df_price.time < publish_date].iloc[-1].name
    this_date_index = last_date + 1
    next_date_index = this_date_index + foward_days
    
    if next_date_index >= df_price.shape[0]-1:
        return "No data"
    else:
        this = df_price[df_price.index == this_date_index].open.values[0]
        next_ = df_price[df_price.index == next_date_index].open.values[0]
        change = (next_ - this)/this
        if change > threshold_very:
            return "very positive"
        elif change > threshold:
            return "positive"
        elif change < -threshold_very:
            return "very negative"
        elif change < -threshold:
            return "negative"
        else:
            return "neutral"

@logger.catch()
def process_label(file_name, foward_days = 5, threshold = 0.02, threshold_very = 0.06):
    df = pd.read_csv(os.path.join(result_path, file_name))
    df["post_publish_time"] = pd.to_datetime(df["post_publish_time"])
    df["date"] = df["post_publish_time"].dt.date
    df["time"] = df["post_publish_time"].dt.time
    df["hour"] = df["post_publish_time"].dt.hour

    start_date = df["date"].min() - datetime.timedelta(days = 10)
    end_date = df["date"].max() + datetime.timedelta(days = 25)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    code_list = [df.code_name.unique()[0]]

    as_processor = Akshare("akshare", start_date = start_date, end_date= end_date, time_interval="daily")
    as_processor.download_data(code_list, save_path = './data/dataset.csv')
    df_price = as_processor.dataframe

    df['publish_date'] = df.apply(lambda x:x['date'] if x['hour'] <15 else x['date'] + datetime.timedelta(days = 1), axis=1)
    df["label"] = df.apply(lambda x:add_label(x, df_price = df_price, foward_days = foward_days, threshold = threshold, threshold_very = threshold_very), axis=1)
    out_path = os.path.join(base_path, file_name)
    df.to_csv(out_path, index = False)

for id,file_name in enumerate(file_list):
    print(id, file_name)
    process_label(file_name)