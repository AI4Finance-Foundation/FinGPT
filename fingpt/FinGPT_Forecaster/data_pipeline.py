import os 
import json
from tqdm import tqdm
import argparse

from indices import *
from data import  prepare_data_for_symbol, query_gpt4, create_dataset
from prompt import get_all_prompts
from data_infererence_fetch import get_curday, fetch_all_data, get_all_prompts_online


def main(args):

    index_name = args['index_name']
    start_date = args['start_date']
    end_date = args['end_date']
    min_past_weeks = args['min_past_weeks']
    max_past_weeks = args['max_past_weeks']
    train_ratio = args['train_ratio']

    with_basics = True
    if index_name == "dow":
        index_name = "DOW-30"
        index = DOW_30
    elif index_name == "euro":
        index_name = "EURO-STOXX-50"
        index = EURO_STOXX_50
    elif index_name == "crypto":
        index_name = "CRYPTO"
        index = CRYPTO
        with_basics = False
    else:
        raise ValueError("Invalid index name")
    
    data_dir = f"./data/{index_name}_{start_date}_{end_date}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Acquire data
    print("Acquiring data")
    for symbol in tqdm(index):
        print(f"Processing {symbol}")
        prepare_data_for_symbol(symbol, data_dir, start_date, end_date, with_basics=with_basics)

    # Generate prompt and query GPT-4
    print("Generating prompts and querying GPT-4")
    query_gpt4(index, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics=with_basics)

    # Transform into training format
    print("Transforming into training format")
    dataset = create_dataset(index, data_dir, start_date, end_date, train_ratio, with_basics=with_basics)

    # Save dataset
    dataset.save_to_disk(
        f"./data/fingpt-forecaster-{index_name.lower()}-{start_date.replace('-', '')}-{end_date.replace('-', '')}-{min_past_weeks}-{max_past_weeks}-{str(train_ratio).replace('.', '')}"
    )


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--index_name", default="crypto", choices=["dow", "euro", "crypto"], help="index name")
    ap.add_argument("--start_date", default="2022-12-31", help="start date")
    ap.add_argument("--end_date", default="2023-12-31", help="end date")
    ap.add_argument("--min_past_weeks", default=1, help="min past weeks")
    ap.add_argument("--max_past_weeks", default=4, help="max past weeks")
    ap.add_argument("--train_ratio", default=0.6, help="train ratio")
    args = vars(ap.parse_args())

    main(args)