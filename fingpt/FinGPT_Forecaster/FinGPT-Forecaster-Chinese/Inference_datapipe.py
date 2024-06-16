# Inference Data
# get company news online
from datetime import date, datetime, timedelta
from Ashare_data import *
import akshare as ak
import pandas as pd
import requests

#default symbol
symbol = "600519"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_curday():
    
    return date.today().strftime("%Y%m%d")

def n_weeks_before(date_string, n, format = "%Y%m%d"):
    
    date = datetime.strptime(date_string, "%Y%m%d") - timedelta(days=7*n)
    
    return date.strftime(format=format)

def stock_news_em(symbol: str = "300059", page = 1) -> pd.DataFrame:
    
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    params = {
        "cb": "jQuery3510875346244069884_1668256937995",
        "param": '{"uid":"",'
        + f'"keyword":"{symbol}"'
        + ',"type":["cmsArticleWebOld"],"client":"web","clientType":"web","clientVersion":"curr","param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default",' + f'"pageIndex":{page}'+ ',"pageSize":100,"preTag":"<em>","postTag":"</em>"}}}',
        "_": "1668256937996",
    }
    r = requests.get(url, params=params)
    data_text = r.text
    data_json = json.loads(
        data_text.strip("jQuery3510875346244069884_1668256937995(")[:-1]
    )
    temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])
    temp_df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "code": "-",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
            "image": "-",
        },
        inplace=True,
    )
    temp_df["关键词"] = symbol
    temp_df = temp_df[
        [
            "关键词",
            "新闻标题",
            "新闻内容",
            "发布时间",
            "文章来源",
            "新闻链接",
        ]
    ]
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\u3000", "", regex=True)
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\r\n", " ", regex=True)
    return temp_df


def get_news(symbol, max_page = 3):
    
    df_list = []
    for page in range(1, max_page):
        
        try:
            df_list.append(stock_news_em(symbol, page))
        except KeyError:
            print(str(symbol) + "pages obtained for symbol: " + page)
            break

    news_df = pd.concat(df_list, ignore_index=True)
    return news_df

# get return
def get_cur_return(symbol, start_date, end_date, adjust="qfq"):
    """
    date = "yyyymmdd"
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
    # check enddate
    if weekly_data.iloc[-1, 2] > pd.to_datetime(end_date):
        weekly_data.iloc[-1, 2] = pd.to_datetime(end_date)

    return weekly_data

# get basics
def cur_financial_data(symbol, start_date, end_date, with_basics = True):
    
    # get data
    data = get_cur_return(symbol=symbol, start_date=start_date, end_date=end_date)

    news_df = get_news(symbol=symbol)
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

def get_all_prompts_online(symbol, with_basics=True, max_news_perweek = 3, weeks_before = 2):

    end_date = get_curday()
    start_date = n_weeks_before(end_date, weeks_before)

    company_prompt, stock = get_company_prompt_new(symbol)
    data = cur_financial_data(symbol=symbol, start_date=start_date, end_date=end_date, with_basics=with_basics)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, basics = get_prompt_by_row_new(symbol, row)
        prev_rows.append((head, news, basics))
        
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(max_news_perweek, len(prev_rows[i][1]))
        )
        if sampled_news:
            prompt += "\n".join(sampled_news)
        else:
            prompt += "No relative news reported."
    
    next_date = n_weeks_before(end_date, -1, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    period = "{}至{}".format(end_date, next_date)
    
    if with_basics:
        basics = prev_rows[-1][2]
    else:
        basics = "[金融基本面]:\n\n 无金融基本面记录"
    
    info = company_prompt + '\n' + prompt + '\n' + basics

    new_system_prompt = SYSTEM_PROMPT.replace('：\n...', '：\n预测涨跌幅：...\n总结分析：...')
    prompt = B_INST + B_SYS + new_system_prompt + E_SYS + info + f"\n\n基于在{end_date}之前的所有信息，让我们首先分析{stock}的积极发展和潜在担忧。请简洁地陈述，分别提出2-4个最重要的因素。大部分所提及的因素应该从公司的相关新闻中推断出来。" \
        f"接下来请预测{symbol}下周({period})的股票涨跌幅，并提供一个总结分析来支持你的预测。" + E_INST
        
    return info, prompt

if __name__ == "__main__":
    info, pt = get_all_prompts_online(symbol=symbol)
    print(pt)