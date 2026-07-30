"""
Configuration for the Forex Intelligence Framework
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from datetime import time

# Global major currencies
MAJOR_CURRENCIES = [
    "USD",  # US Dollar
    "EUR",  # Euro
    "GBP",  # British Pound
    "JPY",  # Japanese Yen
    "CHF",  # Swiss Franc
    "CAD",  # Canadian Dollar
    "AUD",  # Australian Dollar
    "NZD",  # New Zealand Dollar
    "CNY",  # Chinese Yuan
]

# Currency pairs (major crosses)
MAJOR_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "NZD/USD", "USD/CAD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "EUR/CHF", "GBP/CHF",
    "AUD/JPY", "NZD/JPY", "CAD/JPY", "EUR/AUD",
    "GBP/AUD", "AUD/CAD", "NZD/CAD", "EUR/CAD",
    "GBP/CAD", "EUR/NZD", "GBP/NZD", "AUD/NZD",
]

# Trading sessions
@dataclass
class TradingSession:
    name: str
    timezone: str
    start_time: time
    end_time: time
    major_currencies: List[str]
    characteristics: Dict[str, str]

TRADING_SESSIONS = {
    "asia": TradingSession(
        name="Asia",
        timezone="Asia/Tokyo",
        start_time=time(0, 0),  # 00:00 UTC
        end_time=time(8, 0),    # 08:00 UTC
        major_currencies=["JPY", "AUD", "NZD", "CNY"],
        characteristics={
            "liquidity": "low_to_medium",
            "volatility": "low",
            "focus": "Asian economic data, central bank actions",
            "key_pairs": ["USD/JPY", "AUD/USD", "NZD/USD", "USD/CNY"]
        }
    ),
    "london": TradingSession(
        name="London",
        timezone="Europe/London",
        start_time=time(7, 0),  # 07:00 UTC
        end_time=time(16, 0),   # 16:00 UTC
        major_currencies=["EUR", "GBP", "CHF"],
        characteristics={
            "liquidity": "high",
            "volatility": "high",
            "focus": "European economic data, major market moves",
            "key_pairs": ["EUR/USD", "GBP/USD", "EUR/GBP", "EUR/CHF", "GBP/CHF"]
        }
    ),
    "new_york": TradingSession(
        name="New York",
        timezone="America/New_York",
        start_time=time(13, 0),  # 13:00 UTC
        end_time=time(22, 0),    # 22:00 UTC
        major_currencies=["USD", "CAD"],
        characteristics={
            "liquidity": "very_high",
            "volatility": "high",
            "focus": "US economic data, North American flows",
            "key_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "USD/CHF"]
        }
    ),
}

# Economic indicators tier system
@dataclass
class EconomicIndicator:
    name: str
    tier: str  # HIGH, MEDIUM, LOW
    currencies: List[str]
    frequency: str  # daily, weekly, monthly, quarterly
    impact_weight: float
    description: str

ECONOMIC_INDICATORS = {
    # High-tier indicators
    "interest_rate_decision": EconomicIndicator(
        name="Interest Rate Decision",
        tier="HIGH",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=1.0,
        description="Central bank policy rate changes"
    ),
    "cpi_inflation": EconomicIndicator(
        name="Consumer Price Index (CPI)",
        tier="HIGH",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.9,
        description="Primary inflation measure"
    ),
    "gdp": EconomicIndicator(
        name="Gross Domestic Product (GDP)",
        tier="HIGH",
        currencies=MAJOR_CURRENCIES,
        frequency="quarterly",
        impact_weight=0.85,
        description="Economic growth measure"
    ),
    "employment": EconomicIndicator(
        name="Employment Data",
        tier="HIGH",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.85,
        description="Unemployment rate, non-farm payrolls, etc."
    ),
    "pmi": EconomicIndicator(
        name="Purchasing Managers Index (PMI)",
        tier="HIGH",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.8,
        description="Manufacturing and services sector health"
    ),
    
    # Medium-tier indicators
    "retail_sales": EconomicIndicator(
        name="Retail Sales",
        tier="MEDIUM",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.7,
        description="Consumer spending measure"
    ),
    "trade_balance": EconomicIndicator(
        name="Trade Balance",
        tier="MEDIUM",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.7,
        description="Exports vs imports"
    ),
    "industrial_production": EconomicIndicator(
        name="Industrial Production",
        tier="MEDIUM",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.65,
        description="Manufacturing and industrial output"
    ),
    
    # Low-tier indicators
    "consumer_confidence": EconomicIndicator(
        name="Consumer Confidence",
        tier="LOW",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.5,
        description="Consumer sentiment survey"
    ),
    "building_permits": EconomicIndicator(
        name="Building Permits",
        tier="LOW",
        currencies=MAJOR_CURRENCIES,
        frequency="monthly",
        impact_weight=0.4,
        description="Housing market indicator"
    ),
}

# Data source configuration
@dataclass
class DataSource:
    name: str
    base_url: str
    api_key_env: str
    rate_limit: int  # requests per minute
    data_types: List[str]
    reliability_score: float

DATA_SOURCES = {
    "bloomberg": DataSource(
        name="Bloomberg",
        base_url="https://api.bloomberg.com",
        api_key_env="BLOOMBERG_API_KEY",
        rate_limit=10,
        data_types=["market_data", "economic_indicators", "news", "forward_calendar"],
        reliability_score=0.95
    ),
    "reuters": DataSource(
        name="Reuters",
        base_url="https://api.reuters.com",
        api_key_env="REUTERS_API_KEY",
        rate_limit=15,
        data_types=["market_data", "news", "economic_indicators"],
        reliability_score=0.93
    ),
    "tradingeconomics": DataSource(
        name="Trading Economics",
        base_url="https://api.tradingeconomics.com",
        api_key_env="TRADING_ECONOMICS_API_KEY",
        rate_limit=20,
        data_types=["economic_indicators", "historical_data", "forecasts"],
        reliability_score=0.88
    ),
    "fxstreet": DataSource(
        name="FXStreet",
        base_url="https://www.fxstreet.com",
        api_key_env="FXSTREET_API_KEY",
        rate_limit=30,
        data_types=["news", "analysis", "calendar"],
        reliability_score=0.85
    ),
    "yfinance": DataSource(
        name="Yahoo Finance",
        base_url="https://query1.finance.yahoo.com",
        api_key_env="",
        rate_limit=100,
        data_types=["market_data", "historical_data"],
        reliability_score=0.80
    ),
}

# Commodity-currency relationships
COMMODITY_CURRENCY_LINKS = {
    "AUD": {
        "commodities": ["gold", "iron_ore", "coal", "natural_gas"],
        "correlation_strength": "high",
        "primary_commodity": "iron_ore"
    },
    "CAD": {
        "commodities": ["crude_oil", "natural_gas", "wood", "metals"],
        "correlation_strength": "high",
        "primary_commodity": "crude_oil"
    },
    "NZD": {
        "commodities": ["dairy", "meat", "wood", "gold"],
        "correlation_strength": "medium",
        "primary_commodity": "dairy"
    },
    "USD": {
        "commodities": ["gold", "crude_oil"],
        "correlation_strength": "inverse",
        "primary_commodity": "gold"
    },
    "CHF": {
        "commodities": ["gold"],
        "correlation_strength": "high",
        "primary_commodity": "gold"
    },
}

# Analysis weights
ANALYSIS_WEIGHTS = {
    "global_macro": 0.20,
    "capital_flows": 0.15,
    "economic_data": 0.20,
    "central_bank_policy": 0.15,
    "geopolitics": 0.10,
    "commodity_links": 0.08,
    "currency_strength": 0.07,
    "session_analysis": 0.05,
}

# Risk sentiment thresholds
RISK_THRESHOLDS = {
    "risk_on": {
        "vix_threshold": 15.0,
        "bond_yield_spread": 0.5,
        "equity_index_change": 0.01
    },
    "risk_off": {
        "vix_threshold": 25.0,
        "bond_yield_spread": -0.3,
        "equity_index_change": -0.01
    },
}

# Currency strength scoring
STRENGTH_SCORES = {
    "very_strong": 8.0,
    "strong": 6.0,
    "moderate_strong": 4.0,
    "neutral": 0.0,
    "moderate_weak": -4.0,
    "weak": -6.0,
    "very_weak": -8.0,
}

# Event impact levels
EVENT_IMPACT = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.3,
}

# Trade timing rules
@dataclass
class TradeTimingRule:
    condition: str
    action: str
    description: str

TRADE_TIMING_RULES = [
    TradeTimingRule(
        condition="high_impact_event_in_30min",
        action="avoid_entry",
        description="Avoid entering trades 30 minutes before/after high-impact events"
    ),
    TradeTimingRule(
        condition="session_overlap",
        action="prefer_entry",
        description="Prefer entries during session overlaps (London/NY)"
    ),
    TradeTimingRule(
        condition="low_liquidity_session",
        action="reduce_position",
        description="Reduce position sizes during low liquidity sessions"
    ),
    TradeTimingRule(
        condition="strong_trend_aligned",
        action="increase_confidence",
        description="Increase confidence when trend aligns with session bias"
    ),
]