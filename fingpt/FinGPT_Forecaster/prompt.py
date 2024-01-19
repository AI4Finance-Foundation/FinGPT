import os
import json
import random
import finnhub
import yfinance as yf
import pandas as pd
from openai import OpenAI

from indices import *

finnhub_client = finnhub.Client(api_key=os.environ.get("FINNHUB_KEY"))


def get_company_prompt(symbol):
    
    profile = finnhub_client.company_profile2(symbol=symbol)

    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)
    
    return formatted_str


def get_crypto_prompt(symbol):

    profile = yf.Ticker(symbol).info

    crpyto_template = """[Cryptocurrency Introduction]: {description}. It has a market capilization of {marketCap}."""
    
    formatted_str = crpyto_template.format(**profile)
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. News during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def get_crypto_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. News during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    return head, news, None


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def map_bin_label(bin_lb):
    
    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5+', 'more than 5%')
#         lb = lb.replace('5+', '5+%')
    else:
        lb = lb.replace('5', '4-5%')
    
    return lb

PROMPT_END = {
    "company": "\n\nBased on all the information before {start_date}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        "Then let's assume your prediction for next week ({start_date} to {end_date}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.",

    "crypto": "\n\nBased on all the information before {start_date}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from cryptocurrencies related news. " \
        "Then let's assume your prediction for next week ({start_date} to {end_date}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis."
}

def get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):

    
    if with_basics:
        df = pd.read_csv(f'{data_dir}/{symbol}_{start_date}_{end_date}.csv')
    else:
        df = pd.read_csv(f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv')
    
    if symbol in CRYPTO:
        info_prompt = get_crypto_prompt(symbol)
    else:
        info_prompt = get_company_prompt(symbol)

    prev_rows = []
    all_prompts = []

    for row_idx, row in df.iterrows():

        prompt = ""
        if len(prev_rows) >= min_past_weeks:
            idx = min(random.choice(range(min_past_weeks, max_past_weeks+1)), len(prev_rows))
            for i in range(-idx, 0):
                # Add Price Movement (Head)
                prompt += "\n" + prev_rows[i][0]
                # Add News of previous weeks
                sampled_news = sample_news(
                    prev_rows[i][1],
                    min(5, len(prev_rows[i][1]))
                )
                if sampled_news:
                    prompt += "\n".join(sampled_news)
                else:
                    prompt += "No relative news reported."

        if symbol in CRYPTO:
            head, news, basics = get_crypto_prompt_by_row(symbol, row)
        else:
            head, news, basics = get_prompt_by_row(symbol, row)

        prev_rows.append((head, news, basics))
        if len(prev_rows) > max_past_weeks:
            prev_rows.pop(0)  

        if not prompt:
            continue

        prediction = map_bin_label(row['Bin Label'])
        
        prompt = info_prompt + '\n' + prompt + '\n' + basics

        prompt += PROMPT_END['crypto' if symbol in CRYPTO else 'company'].format(
            start_date=row['Start Date'],
            end_date=row['End Date'],
            prediction=prediction,
            symbol=symbol
        )

        all_prompts.append(prompt.strip())
    
    return all_prompts