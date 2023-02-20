import os.path

import pandas as pd
import tushare as ts



TuShareAPIKEY = '396edc1958************95e95d816'

data_path = 'tushare_finance_news.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    pro = ts.pro_api(TuShareAPIKEY)

    df = pro.news(**{
        "start_date": "2023-02-14",
        "end_date": "2023-02-15",
        "src": "sina",
        "limit": 32,
        "offset": 0
    }, fields=[
        "datetime",
        "content",
        "title"
    ])
    df.to_csv()
print(df.shape)
print(df)
exit()

import http.client
import urllib.parse

conn = http.client.HTTPSConnection('api.marketaux.com')

params = urllib.parse.urlencode({
    'api_token': 'XeR26870*******Zj6In835C',
    "found": 8,
    "returned": 3,
    "limit": 4,
    "page": 1,
    "source_id": "adweek.com-1",
    "domain": "adweek.com",
    "language": "en",
})

conn.request('GET', '/v1/news/all?{}'.format(params))

res = conn.getresponse()
data = res.read()

data_str = data.decode('utf-8')
print(data_str)
print(';;;')


import json

res = json.loads(data_str)

print(json.dumps(res, indent=2))

"""
https://medium.com/codex/extracting-financial-news-seamlessly-using-python-4dcc732d9ff1
Extracting Financial News seamlessly using Python
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Classify the sentiment :\n\n1. f{prompt},
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
