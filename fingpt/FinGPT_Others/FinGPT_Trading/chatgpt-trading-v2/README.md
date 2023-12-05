# ChatGPT Trading Bot V2.

[ChatGPT for FinTech](https://github.com/AI4Finance-Foundation/ChatGPT-for-FinTech): a list of resources to use ChatGPT for FinTech

Let's fully use the ChatGPT to create an FinRL agent that trades as smartly as ChatGPT. The codes are available [here](https://github.com/oliverwang15/Alternative-Data/blob/main/demo/chatgpt-trading-v2/trade_with_gpt3.ipynb)


**Disclaimer: We are sharing codes for academic purpose under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

## 1. Data Preparation for Price Data and Tweets 

* First, we fetch price data and Tweets data from [stocknet-dataset](https://github.com/yumoxu/stocknet-dataset)
* Second, we input Tweets data to a GPT model, say "text-curie-001" or "text-davinci-003", and get the corresponding sentiment scores
* Third, we save the sentiment scores to a file under `./data`

## 2. ChatGPT Trading Agent

* We calculate the average sentiment score `S`.
* We implement a simple strategy that buys 100 shares when `S` >= 0.3 and sells 100 shares when `S` <= -0.3
* Parameters of GPT Model are:

  ``` PyThon
  "model_name": "text-davinci-003",  # "text-curie-001","text-davinci-003"
  "source": "local",                 # "local","openai"
  "api_key": OPEN_AI_TOKEN,          # not necessary when the "source" is "local"
  "buy_threshold": 0.3,              # the max positive sentiment is 1, so this should range from 0 to 1 
  "sell_threshold": -0.3             # the min negative sentiment is -1, so this should range from -1 to 0
  ```

## 3. Backtest

* We backtest the agent's performance from '2014-01-01' to '2015-12-30'.
* Parameters are:

  ``` PyThon
  "stock_name" : "AAPL",        # please refer to the stocks provided by stocknet-dataset
  "start_date":"2014-01-01",    # should be later than 2014-01-01
  "end_date":"2015-12-30",      # should be earlier than 2015-12-30
  "init_cash": 100,             # initial available cash
  "init_hold": 0,               # initial available stock holdings
  "cal_on": "Close",            # The column that used to calculate prices
  "trade_volumn": 100,          # Volumns to trade
  ```

## 4. Results

* The result is shown as follows:

  ![image-20230216004801458](https://cdn.jsdelivr.net/gh/oliverwang15/imgbed@main/img/202302181558796.png)

* The performance metrics are as follows
  |        metrics      | result  |
  | :-----------------: | :-----: |
  |    Annual return    | 30.603% |
  | Cumulative returns  | 66.112% |
  |  Annual volatility  | 13.453% |
  |    Sharpe ratio     |  2.06   |
  |    Calmar ratio     |  4.51   |
  |      Stability      |  0.87   |
  |    Max drawdown     | -6.778% |
  |     Omega ratio     |  2.00   |
  |    Sortino ratio    |  4.30   |
  |     Tail ratio      |  1.84   |
  | Daily value at risk | -1.585% |
  |        Alpha        |  0.24   |
  |        Beta         |  0.31   |

## 5. TODOs

1. Combing price features

2. Train an FinRL agent on the sentiment scores given by GPT models
