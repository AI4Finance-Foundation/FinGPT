import os.path
import json

import http.client
import urllib.parse

import pandas as pd
import tushare as ts

import os
import openai


def get_news_from_tushare(api_key: str, data_path: str = 'finance_news_from_tushare.csv') -> str:
    start_date = '2023-02-01'
    end_date = '2023-02-02'
    limit_line = 200
    if_news_or_reports = False

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        pro = ts.pro_api(api_key)

        if if_news_or_reports:
            df = pro.news(**{
                "start_date": start_date,
                "end_date": end_date,
                "src": "sina",
                "limit": limit_line,
                "offset": 0
            }, fields=[
                "datetime",
                "title"
                "content",
            ])
        else:
            df = pro.jinse(**{
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit_line,
                "offset": 0
            }, fields=[
                "datetime",
                "title",
                "content",
            ])

        df.to_csv(data_path)

    max_num_news = 4
    max_len_title = 32
    max_len_content = 0
    data_str = ""
    for i in df.index[:max_num_news]:
        row = df.iloc[i]
        title = row['title'][1:max_len_title]
        content = row['content'][:max_len_content]
        data_str += f"{title}, {content}\n"
    return data_str


def get_news_from_market_aux(api_key: str, data_path: str = 'finance_news_from_market_aux.txt'):
    limit_line = 4

    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        conn = http.client.HTTPSConnection('api.marketaux.com')

        params = urllib.parse.urlencode({
            'api_token': api_key,
            "found": 8,
            "returned": 3,
            "limit": limit_line,
            "page": 1,
            "source_id": "adweek.com-1",
            "domain": "adweek.com",
            "language": "en",
        })

        conn.request('GET', '/v1/news/all?{}'.format(params))

        data = conn.getresponse()
        data = data.read().decode('utf-8')
        data = json.loads(data)
        with open(data_path, 'w') as f:
            f.write(json.dumps(data, indent=2))

    assert isinstance(data, dict)

    '''concert dict to string (Title: ... Content: ...)'''
    max_num_news = 8
    max_len_title = 32 * 4
    max_len_content = 0 * 4

    data = data['data']

    data_str = ""
    for item in data[:max_num_news]:
        title = item['title'][:max_len_title]
        content = item['description'][:max_len_content]
        data_str += f"{title}, {content}\n"
    return data_str


def get_result_from_openai_davinci(api_key: str, prompt_str: str):
    max_tokens = 64

    # openai.api_key =
    openai.api_key = api_key  # os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_str,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


API_KEY_Tushare = ''  # https://www.tushare.pro/user/token
API_KEY_MarketAUX = ''  # https://www.marketaux.com/account/dashboard
API_KEY_OpenAI = ''  # https://platform.openai.com/account/api-keys


def run_news_in_chinese():
    prompt_str = "读下方新闻，列举3个可能受影响的美国股票，分别简短地判断积极或消极:\n\n"
    prompt_str += get_news_from_tushare(api_key=API_KEY_Tushare)
    print(f"\n====\n{prompt_str}")

    result_str = get_result_from_openai_davinci(api_key=API_KEY_OpenAI, prompt_str=prompt_str)
    print(f"\n====\n{result_str}")


def run_news_in_english():
    prompt_str = "Read following news and list 3 stocks that may be affected, " \
                 "briefly judge each one 'positive' or 'negative':\n\n"
    prompt_str += get_news_from_market_aux(api_key=API_KEY_MarketAUX)
    print(f"\n====\n{prompt_str}")

    result_str = get_result_from_openai_davinci(api_key=API_KEY_OpenAI, prompt_str=prompt_str)
    print(f"\n====\n{result_str}")


if __name__ == '__main__':
    run_news_in_chinese()
    run_news_in_english()
