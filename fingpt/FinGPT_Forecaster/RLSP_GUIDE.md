# Reinforcement Learning for Stock Prices

## Status

The RLSP idea discussed in [issue #36](https://github.com/AI4Finance-Foundation/FinGPT/issues/36)
was theoretical. This repository does not currently include a validated PPO,
REINFORCE, or other reinforcement-learning trainer for stock prices. The
Forecaster is a supervised LoRA model and should not be described as an RL
agent.

## Recommended formulation

An RL experiment needs four explicit pieces:

| Component | Example for a stock environment |
| --- | --- |
| State | Historical returns, volume, sentiment, cash, and current position |
| Action | Target position such as short, flat, or long, with position limits |
| Transition | Advance to the next trading session using data available at that time |
| Reward | Net portfolio return after transaction costs and risk penalties |

The reward should be based on executable portfolio results rather than the
model's generated text. A simple daily reward is:

```text
reward = position * next_return - transaction_cost * turnover - risk_penalty
```

The environment must apply the action after the observation timestamp. News,
financial statements, and market data released later must not appear in the
state for that step.

## Safe experiment sequence

1. Build chronological train, validation, and test periods. Do not randomly
   shuffle market observations across those periods.
2. Start with a buy-and-hold and a no-trade baseline.
3. Add transaction costs, slippage, position limits, and a maximum drawdown
   metric before comparing agents.
4. Train only on the training period and select hyperparameters on validation.
5. Evaluate once on the held-out test period, then use paper trading before
   considering live execution.

Report cumulative return, annualized volatility, Sharpe ratio, maximum
drawdown, turnover, and the number of trades. Accuracy alone is not sufficient
for a trading policy because a small number of high-impact trades can dominate
returns.

## Relationship to FinGPT

FinGPT can provide sentiment or extracted factors as part of an observation,
but the sentiment output is not itself a trading action. A future RLSP
implementation should keep these layers separate:

```text
news and prices -> FinGPT features -> trading environment -> policy -> action
```

Any implementation should include deterministic environment tests, a seeded
replayable experiment, and a comparison with the no-trade baseline. Until
those pieces exist, this guide is an experiment design reference rather than a
claim that RLSP is production-ready or investment advice.