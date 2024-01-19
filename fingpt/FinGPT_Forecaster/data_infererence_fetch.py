import os
import finnhub
import yfinance as yf
import pandas as pd
from datetime import date, datetime, timedelta
from collections import defaultdict

from data import get_news
from prompt import get_company_prompt, get_prompt_by_row, sample_news

finnhub_client = finnhub.Client(api_key=os.environ.get("FINNHUB_KEY"))


def get_curday():
    
    return date.today().strftime("%Y-%m-%d")


def n_weeks_before(date_string, n):
    
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    
    return date.strftime("%Y-%m-%d")


def get_stock_data(stock_symbol, steps):

    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    
#     print(stock_data)
    
    dates, prices = [], []
    available_dates = stock_data.index.format()
    
    for date in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date:
                prices.append(stock_data['Close'][i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break

    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'][-1])
    
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })


def get_current_basics(symbol, curday):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
    
    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break
            
    return basic


def fetch_all_data(symbol, curday, n_weeks=3):

    steps = [n_weeks_before(curday, i) for i in range(n_weeks+1)][::-1]

    data = get_stock_data(symbol, steps)
    data = get_news(symbol, data)

    return data
    

def get_all_prompts_online(symbol, data, curday, with_basics=True):

    company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, _ = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, None))
        
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(5, len(prev_rows[i][1]))
        )
        if sampled_news:
            prompt += "\n".join(sampled_news)
        else:
            prompt += "No relative news reported."
        
    period = "{} to {}".format(curday, n_weeks_before(curday, -1))
    
    if with_basics:
        basics = get_current_basics(symbol, curday)
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."

    info = company_prompt + '\n' + prompt + '\n' + basics
    prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."
        
    return info, prompt