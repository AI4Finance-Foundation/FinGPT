import os
import re
import csv
import math
import time
import json
import finnhub
from tqdm import tqdm
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict
import datasets
from datasets import Dataset
from openai import OpenAI

from indices import *
from prompt import get_all_prompts

finnhub_client = finnhub.Client(api_key=os.environ.get("FINNHUB_KEY"))
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))


# ----------------------------------------------------------------------------------- #
# ---------------------------- RAW FINANCIAL ACQUISITION ---------------------------- #
# ----------------------------------------------------------------------------------- #

def bin_mapping(ret):
    
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    
    return up_down + (str(integer) if integer <= 5 else '5+')


def get_returns(stock_symbol, start_date, end_date):
    # TODO: likely to be merged with get_stock_data
    
    # Download historical stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    weekly_data = stock_data['Adj Close'].resample('W').ffill()
    weekly_returns = weekly_data.pct_change()[1:]
    weekly_start_prices = weekly_data[:-1]
    weekly_end_prices = weekly_data[1:]

    weekly_data = pd.DataFrame({
        'Start Date': weekly_start_prices.index,
        'Start Price': weekly_start_prices.values,
        'End Date': weekly_end_prices.index,
        'End Price': weekly_end_prices.values,
        'Weekly Returns': weekly_returns.values
    })
    
    weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(bin_mapping)

    return weekly_data


def get_news(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        # print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    
    return data


def get_basics(symbol, data, start_date, always=False):
    
    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
            
    for i, row in data.iterrows():
        
        start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = start_date if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
        
        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))
        
    data['Basics'] = final_basics
    
    return data
    

def prepare_data_for_symbol(symbol, data_dir, start_date, end_date, with_basics=True):
    
    data = get_returns(symbol, start_date, end_date)
    data = get_news(symbol, data)
    
    if with_basics:
        data = get_basics(symbol, data, start_date)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}.csv")
    else:
        data['Basics'] = [json.dumps({})] * len(data)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv")
    
    return data


# ----------------------------------------------------------------------------------- #
# ---------------------------------- GPT4 ANALYSIS ---------------------------------- #
# ----------------------------------------------------------------------------------- #


def append_to_csv(filename, input_data, output_data):
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])

        
def initialize_csv(filename):
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])


def query_gpt4(symbol_list, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):

    for symbol in tqdm(symbol_list):
        
        csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_gpt-4.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_gpt-4.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        prompts = get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics)
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        for i, prompt in enumerate(prompts):
            
            if i < pre_done:
                continue

            # print(f"{symbol} - {i}")
            
            cnt = 0
            while cnt < 5:
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                          ]
                    )
                    break    
                except Exception:
                    cnt += 1
                    print(f'retry cnt {cnt}')
            
            answer = completion.choices[0].message.content if cnt < 5 else ""
            append_to_csv(csv_file, prompt, answer)



# ----------------------------------------------------------------------------------- #
# -------------------------- TRANSFORM INTO TRAINING FORMAT ------------------------- #
# ----------------------------------------------------------------------------------- #

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPTS = {
    "company": "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",

    "crypto": "You are a seasoned crypto market analyst. Your task is to list the positive developments and potential concerns for cryptocurrencies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the cryptocurrencies price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",
}

def gpt4_to_llama(symbol, data_dir, start_date, end_date, with_basics=True):

    csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_gpt-4.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_gpt-4.csv'
    
    df = pd.read_csv(csv_file)
    
    prompts, answers, periods, labels = [], [], [], []
    
    for i, row in df.iterrows():
        
        prompt, answer = row['prompt'], row['answer']
        
        res = re.search(r"Then let's assume your prediction for next week \((.*)\) is ((:?up|down) by .*%).", prompt)
        
        period, label = res.group(1), res.group(2)
#         label = label.replace('more than 5', '5+')
        
        prompt = re.sub(
            r"Then let's assume your prediction for next week \((.*)\) is (up|down) by ((:?.*)%). Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.", 
            f"Then make your prediction of the {symbol} cryptocurrency price movement for next week ({period}). Provide a summary analysis to support your prediction.",
            prompt
        )
        try:
            answer = re.sub(
                r"\[Prediction & Analysis\]:\s*",
                f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: ",
                answer
            )
        except Exception:
            print(symbol, i)
            print(label)
            print(answer)
            continue
            
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        new_system_prompt = system_prompt.replace(':\n...', '\nPrediction: ...\nAnalysis: ...')
#         new_system_prompt = SYSTEM_PROMPT.replace(':\n...', '\nPrediction: {Up|Down} by {1-2|2-3|3-4|4-5|5+}%\nAnalysis: ...')
        
        prompt = B_INST + B_SYS + new_system_prompt + E_SYS + prompt + E_INST
        
        prompts.append(prompt)
        answers.append(answer)
        periods.append(period)
        labels.append(label)
        
    return {
        "prompt": prompts,
        "answer": answers,
        "period": periods,
        "label": labels,
    }


def create_dataset(symbol_list, data_dir, start_date, end_date, train_ratio=0.8, with_basics=True):

    train_dataset_list = []
    test_dataset_list = []

    for symbol in symbol_list:

        data_dict = gpt4_to_llama(symbol, data_dir, start_date, end_date,  with_basics)
#         print(data_dict['prompt'][-1])
#         print(data_dict['answer'][-1])
        symbols = [symbol] * len(data_dict['label'])
        data_dict.update({"symbol": symbols})

        dataset = Dataset.from_dict(data_dict)
        train_size = round(train_ratio * len(dataset))

        train_dataset_list.append(dataset.select(range(train_size)))
        if train_size >= len(dataset):
            continue
        test_dataset_list.append(dataset.select(range(train_size, len(dataset))))

    train_dataset = datasets.concatenate_datasets(train_dataset_list)
    test_dataset = datasets.concatenate_datasets(test_dataset_list)

    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset
   