import akshare as ak
import pandas as pd
import os
import csv
import re
import time
import math
import json
import random
from datasets import Dataset
import datasets

start_date = "20230201"
end_date = "20240101"

# ------------------------------------------------------------------------------
# Data Aquisition
# ------------------------------------------------------------------------------

# get return
def get_return(symbol, adjust="qfq"):
    """
    Get stock return data.

    Args:
        symbol: str
            A-share market stock symbol
        adjust: str ("qfq", "hfq")
            price ajustment
            default = "qfq" 前复权
    
    Return:
        weekly forward filled return data
    """
    
    # load data
    return_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
    
    # process timestamp
    return_data["日期"] = pd.to_datetime(return_data["日期"])
    return_data.set_index("日期", inplace=True)

    # resample and filled with forward data
    weekly_data = return_data["收盘"].resample("W").ffill()
    weekly_returns = weekly_data.pct_change()[1:]
    weekly_start_prices = weekly_data[:-1]
    weekly_end_prices = weekly_data[1:]
    weekly_data = pd.DataFrame({
        '起始日期': weekly_start_prices.index,
        '起始价': weekly_start_prices.values,
        '结算日期': weekly_end_prices.index,
        '结算价': weekly_end_prices.values,
        '周收益': weekly_returns.values
    })
    weekly_data["简化周收益"] = weekly_data["周收益"].map(return_transform)
    
    return weekly_data
def return_transform(ret):
    
    up_down = '涨' if ret >= 0 else '跌'
    integer = math.ceil(abs(100 * ret))
    if integer == 0:
        return "平"
    
    return up_down + (str(integer) if integer <= 5 else '5+')

# get basics
def get_basic(symbol, data):
    """
    Get and match basic data to news dataframe.

    Args:
        symbol: str
            A-share market stock symbol
        data: DataFrame
            dated news data
    
    Return:
        financial news dataframe with matched basic_financial info
    """
    key_financials = ['报告期', '净利润同比增长率', '营业总收入同比增长率', '流动比率', '速动比率', '资产负债率']
    
    # load quarterly basic data
    basic_quarter_financials = ak.stock_financial_abstract_ths(symbol = symbol, indicator="按单季度")
    basic_fin_dict = basic_quarter_financials.to_dict("index")
    basic_fin_list = [dict([(key, val) for key, val in basic_fin_dict[i].items() if (key in key_financials) and val]) for i in range(len(basic_fin_dict))]

    # match basic financial data to news dataframe
    matched_basic_fin = []
    for i, row in data.iterrows():

        newsweek_enddate = row['结算日期'].strftime("%Y-%m-%d")

        matched_basic = {}
        for basic in basic_fin_list:
            # match the most current financial report
            if basic["报告期"] < newsweek_enddate:
                matched_basic = basic
                break
        matched_basic_fin.append(json.dumps(matched_basic, ensure_ascii=False))

    data['基本面'] = matched_basic_fin

    return data

def raw_financial_data(symbol, with_basics = True):
    
    # get return data from API
    data = get_return(symbol=symbol)
    
    # get news data from local
    file_name = "news_data" + symbol + ".csv"
    news_df = pd.read_csv("HS300_news_data20240118/"+file_name, index_col=0)
    news_df["发布时间"] = pd.to_datetime(news_df["发布时间"], exact=False, format="%Y-%m-%d")
    news_df.sort_values(by=["发布时间"], inplace=True)
    
    # match weekly news for return data
    news_list = []
    for a, row in data.iterrows():
        week_start_date = row['起始日期'].strftime('%Y-%m-%d')
        week_end_date = row['结算日期'].strftime('%Y-%m-%d')
        print(symbol, ': ', week_start_date, ' - ', week_end_date)
        
        weekly_news = news_df.loc[(news_df["发布时间"]>week_start_date) & (news_df["发布时间"]<week_end_date)]

        weekly_news = [
            {
                "发布时间": n["发布时间"].strftime('%Y%m%d'),
                "新闻标题": n['新闻标题'],
                "新闻内容": n['新闻内容'],
            } for a, n in weekly_news.iterrows()
        ]
        news_list.append(json.dumps(weekly_news,ensure_ascii=False))

    data["新闻"] = news_list

    if with_basics:
        data = get_basic(symbol=symbol, data=data)
        # data.to_csv(symbol+start_date+"_"+end_date+".csv")
    else:
        data['新闻'] = [json.dumps({})] * len(data)
        # data.to_csv(symbol+start_date+"_"+end_date+"_nobasics.csv")
    
    return data

# ------------------------------------------------------------------------------
# Prompt Generation
# ------------------------------------------------------------------------------

# SYSTEM_PROMPT = "你是一个经验丰富的股票市场分析师。你的任务是根据过去几周的相关新闻和基本财务状况，列出公司的积极发展和潜在担忧，然后对公司未来一周的股价变化提供分析和预测。" \
#     "你的回答语言应为中文。你的回答格式应该如下：\n\n[积极发展]：\n1. ...\n\n[潜在担忧]：\n1. ...\n\n[预测和分析]：\n...\n"
SYSTEM_PROMPT = "你是一名经验丰富的股票市场分析师。你的任务是根据公司在过去几周内的相关新闻和季度财务状况，列出公司的积极发展和潜在担忧，然后结合你对整体金融经济市场的判断，对公司未来一周的股价变化提供预测和分析。" \
    "你的回答语言应为中文。你的回答格式应该如下：\n\n[积极发展]：\n1. ...\n\n[潜在担忧]：\n1. ...\n\n[预测和分析]：\n...\n"

def get_company_prompt_new(symbol):
    try:
        company_profile = dict(ak.stock_individual_info_em(symbol).values)
    except:
        print("Company Info Request Time Out! Please wait and retry.")
    company_profile["上市时间"] =  pd.to_datetime(str(company_profile["上市时间"])).strftime("%Y年%m月%d日")

    template = "[公司介绍]:\n\n{股票简称}是一家在{行业}行业的领先实体，自{上市时间}成立并公开交易。截止今天，{股票简称}的总市值为{总市值}人民币，总股本数为{总股本}，流通市值为{流通市值}人民币，流通股数为{流通股}。" \
        "\n\n{股票简称}主要在中国运营，以股票代码{股票代码}在交易所进行交易。"
    
    formatted_profile = template.format(**company_profile)
    stockname = company_profile['股票简称']
    return formatted_profile, stockname

def map_return_label(return_lb):
    """
    Map abbrev in the raw data
    Example:
        涨1 -- 上涨1%
        跌2 -- 下跌2%
        平 -- 股价持平
    """

    lb = return_lb.replace('涨', '上涨')
    lb = lb.replace('跌', '下跌')
    lb = lb.replace('平', '股价持平')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5+', '超过5%')
    else:
        lb = lb.replace('5', '4-5%')
    
    return lb

# check news quality
def check_news_quality(n, last_n, week_end_date, repeat_rate = 0.6):
    try:
        # check content avalability
        if not (not(str(n['新闻内容'])[0].isdigit()) and not(str(n['新闻内容'])=='nan') and n['发布时间'][:8] <= week_end_date.replace('-', '')):
            return False
        # check highly duplicated news
        # (assume the duplicated contents happened adjacent)

        elif str(last_n['新闻内容'])=='nan':
            return True
        elif len(set(n['新闻内容'][:20]) & set(last_n['新闻内容'][:20])) >= 20*repeat_rate or len(set(n['新闻标题']) & set(last_n['新闻标题']))/len(last_n['新闻标题']) > repeat_rate:
            return False
        
        else:
            return True
    except TypeError:
        print(n)
        print(last_n)
        raise Exception("Check Error")

def get_prompt_by_row_new(stock, row):
    """
    Generate prompt for each row in the raw data
    Args:
        stock: str
            stock name
        row: pandas.Series
    Return:
        head: heading prompt
        news: news info
        basics: basic financial info
    """

    week_start_date = row['起始日期'] if isinstance(row['起始日期'], str) else row['起始日期'].strftime('%Y-%m-%d')
    week_end_date = row['结算日期'] if isinstance(row['结算日期'], str) else row['结算日期'].strftime('%Y-%m-%d')
    term = '上涨' if row['结算价'] > row['起始价'] else '下跌'
    chg = map_return_label(row['简化周收益'])
    head = "自{}至{}，{}的股票价格由{:.2f}{}至{:.2f}，涨跌幅为：{}。在此期间的公司新闻如下:\n\n".format(
        week_start_date, week_end_date, stock, row['起始价'], term, row['结算价'], chg)

    news = json.loads(row["新闻"])

    left, right = 0, 0
    filtered_news = []
    while left < len(news):
        n = news[left]

        if left == 0:
            # check first news quality
            if (not(str(n['新闻内容'])[0].isdigit()) and not(str(n['新闻内容'])=='nan') and n['发布时间'][:8] <= week_end_date.replace('-', '')):
                filtered_news.append("[新闻标题]：{}\n[新闻内容]：{}\n".format(n['新闻标题'], n['新闻内容']))
            left += 1

        else:
            news_check = check_news_quality(n, last_n = news[right], week_end_date= week_end_date, repeat_rate=0.5)
            if news_check:
                filtered_news.append("[新闻标题]：{}\n[新闻内容]：{}\n".format(n['新闻标题'], n['新闻内容']))
            left += 1
            right += 1


    basics = json.loads(row['基本面'])
    if basics:
        basics = "如下所列为{}近期的一些金融基本面信息，记录时间为{}:\n\n[金融基本面]:\n\n".format(
            stock, basics['报告期']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[金融基本面]:\n\n 无金融基本面记录"

    return head, filtered_news, basics

def sample_news(news, k=5):
    """
    Ramdomly select past news.

    Args:
        news:
            newslist in the timerange
        k: int
            the number of selected news
    """
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]

def get_all_prompts_new(symbol, min_past_week=1, max_past_weeks=2, with_basics=True):
    """
    Generate prompt. The prompt consists of news from past weeks, basics financial information, and weekly return.
    History news in the prompt is chosen from past weeks range from min_past_week to max_past_week, 
    and there is a number constraint on ramdomly selected data (default: up to 5).

    Args:
        symbol: str
            stock ticker
        min_past_week: int
        max_past_week: int
        with_basics: bool
            If true, add basic infomation to the prompt
            
    Return:
        Prompts for the daterange
    """

    # Load Data
    df = raw_financial_data(symbol, with_basics=with_basics)
    
    company_prompt, stock = get_company_prompt_new(symbol)

    prev_rows = []
    all_prompts = []
    
    for row_idx, row in df.iterrows():
        
        prompt = ""

        # judge for available history news 
        if len(prev_rows) >= min_past_week:

            # randomly set retrieve data of past weeks
            # idx = min(random.choice(range(min_past_week, max_past_weeks+1)), len(prev_rows))
            idx = min(max_past_weeks, len(prev_rows))
            for i in range(-idx, 0):
                # Add Head
                prompt += "\n" + prev_rows[i][0]
                # Add History News (with numbers constraint)
                sampled_news = sample_news(
                    prev_rows[i][1],
                    min(3, len(prev_rows[i][1]))
                )
                if sampled_news:
                    prompt += "\n".join(sampled_news)
                else:
                    prompt += "无有关新闻报告"
                    
        head, news, basics = get_prompt_by_row_new(stock, row)
        
        prev_rows.append((head, news, basics))

        if len(prev_rows) > max_past_weeks:
            prev_rows.pop(0)
        
        # set this to make sure there is history news for each considered date
        if not prompt:
            continue
        
        prediction = map_return_label(row['简化周收益'])

        prompt = company_prompt + '\n' + prompt + '\n' + basics

        prompt += f"\n\n基于在{row['起始日期'].strftime('%Y-%m-%d')}之前的所有信息，让我们首先分析{stock}的积极发展和潜在担忧。请简洁地陈述，分别提出2-4个最重要的因素。大部分所提及的因素应该从公司的相关新闻中推断出来。" \
            f"那么让我们假设你对于下一周({row['起始日期'].strftime('%Y-%m-%d')}至{row['结算日期'].strftime('%Y-%m-%d')})的预测是{prediction}。提供一个总结分析来支持你的预测。预测结果需要从你最后的分析中推断出来，因此不作为你分析的基础因素。"

        all_prompts.append(prompt.strip())

    return all_prompts