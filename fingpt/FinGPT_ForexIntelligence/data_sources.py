"""
Data Source Management for Forex Intelligence
Handles data ingestion from multiple financial data providers
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle

# Optional yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from fingpt.FinGPT_ForexIntelligence.config import DATA_SOURCES, MAJOR_CURRENCIES, MAJOR_PAIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceException(Exception):
    """Custom exception for data source errors"""
    pass


@dataclass
class MarketData:
    """Container for market data"""
    symbol: str
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    volume: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None


@dataclass
class EconomicEvent:
    """Container for economic calendar events"""
    event_id: str
    name: str
    currency: str
    date: datetime
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    impact: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    tier: str = "MEDIUM"


class BaseDataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unknown")
        self.api_key = os.environ.get(config.get("api_key_env", ""))
        self.base_url = config.get("base_url", "")
        self.rate_limit = config.get("rate_limit", 10)
        self.last_request_time = None
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            min_interval = 60.0 / self.rate_limit
            if elapsed < min_interval:
                import time
                time.sleep(min_interval - elapsed)
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        self._check_rate_limit()
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            self.last_request_time = datetime.now()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {self.name}: {e}")
            raise DataSourceException(f"Request failed: {e}")
    
    @abstractmethod
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[MarketData]:
        """Get historical market data for a symbol"""
        pass
    
    @abstractmethod
    def get_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get economic calendar events"""
        pass
    
    @abstractmethod
    def get_news_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get news sentiment data"""
        pass


class YahooFinanceSource(BaseDataSource):
    """Yahoo Finance data source implementation"""
    
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[MarketData]:
        """Get market data using yfinance"""
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available, returning empty data")
            return []
            
        try:
            # Convert forex pair to yfinance format
            yf_symbol = self._convert_to_yf_symbol(symbol)
            
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
            )
            
            market_data = []
            for date, row in hist.iterrows():
                mid = (row['Open'] + row['Close']) / 2
                change = row['Close'] - row['Open']
                change_percent = (change / row['Open']) * 100 if row['Open'] != 0 else 0
                
                market_data.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=date,
                        mid=mid,
                        volume=row['Volume'] if 'Volume' in row else None,
                        change=change,
                        change_percent=change_percent,
                    )
                )
            
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data from Yahoo: {e}")
            return []
    
    def _convert_to_yf_symbol(self, symbol: str) -> str:
        """Convert forex pair to yfinance format"""
        if "/" in symbol:
            base, quote = symbol.split("/")
            return f"{base}{quote}=X"
        return symbol
    
    def get_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Yahoo Finance doesn't provide economic calendar"""
        return []
    
    def get_news_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get news sentiment using yfinance"""
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available, returning empty sentiment")
            return {"sentiment": 0, "article_count": 0}
            
        try:
            yf_symbol = self._convert_to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            news = ticker.news
            
            if not news:
                return {"sentiment": 0, "article_count": 0}
            
            # Basic sentiment analysis based on news count
            # In production, you'd use NLP for actual sentiment
            return {
                "sentiment": 0,
                "article_count": len(news),
                "articles": news[:10]  # Return first 10 articles
            }
        except Exception as e:
            logger.error(f"Error fetching news from Yahoo: {e}")
            return {"sentiment": 0, "article_count": 0}


class TradingEconomicsSource(BaseDataSource):
    """Trading Economics data source implementation"""
    
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[MarketData]:
        """Get forex market data from Trading Economics"""
        try:
            # Convert symbol to Trading Economics format
            te_symbol = self._convert_to_te_symbol(symbol)
            
            params = {
                "symbol": te_symbol,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            }
            
            data = self._make_request("/historical", params)
            
            market_data = []
            for item in data:
                market_data.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=datetime.strptime(item["Date"], "%Y-%m-%d"),
                        mid=item.get("Close"),
                        change=item.get("Change"),
                        change_percent=item.get("ChangePercent"),
                    )
                )
            
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data from Trading Economics: {e}")
            return []
    
    def _convert_to_te_symbol(self, symbol: str) -> str:
        """Convert forex pair to Trading Economics format"""
        if "/" in symbol:
            return symbol.replace("/", "")
        return symbol
    
    def get_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get economic calendar from Trading Economics"""
        try:
            params = {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            }
            
            data = self._make_request("/calendar", params)
            
            events = []
            for item in data:
                currency = item.get("Country", "")[:3]  # Extract currency code
                if currencies and currency not in currencies:
                    continue
                
                events.append(
                    EconomicEvent(
                        event_id=item.get("Id", ""),
                        name=item.get("Event", ""),
                        currency=currency,
                        timestamp=datetime.strptime(item["Date"], "%Y-%m-%d"),
                        actual=item.get("Actual"),
                        forecast=item.get("Forecast"),
                        previous=item.get("Previous"),
                        impact=self._map_impact(item.get("Importance", "2")),
                        tier=self._map_tier(item.get("Importance", "2")),
                    )
                )
            
            return events
        except Exception as e:
            logger.error(f"Error fetching economic calendar from Trading Economics: {e}")
            return []
    
    def _map_impact(self, importance: str) -> str:
        """Map importance level to impact"""
        mapping = {"3": "HIGH", "2": "MEDIUM", "1": "LOW"}
        return mapping.get(importance, "MEDIUM")
    
    def _map_tier(self, importance: str) -> str:
        """Map importance level to tier"""
        mapping = {"3": "HIGH", "2": "MEDIUM", "1": "LOW"}
        return mapping.get(importance, "MEDIUM")
    
    def get_news_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Trading Economics news sentiment"""
        try:
            params = {
                "symbol": symbol,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            }
            
            data = self._make_request("/news", params)
            
            if not data:
                return {"sentiment": 0, "article_count": 0}
            
            return {
                "sentiment": 0,  # Would need NLP for actual sentiment
                "article_count": len(data),
                "articles": data[:10]
            }
        except Exception as e:
            logger.error(f"Error fetching news from Trading Economics: {e}")
            return {"sentiment": 0, "article_count": 0}


class ForexDataSourceManager:
    """
    Manager for multiple forex data sources with fallback and caching
    """
    
    def __init__(
        self,
        sources: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour cache TTL
    ):
        """
        Initialize data source manager
        
        Args:
            sources: List of source names to use
            cache_dir: Directory for cache storage
            cache_ttl: Cache time-to-live in seconds
        """
        self.sources = sources or ["yfinance"]  # Default to free source
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        
        # Initialize data sources
        self.data_sources = {}
        self._initialize_sources()
        
        # Cache storage
        self.cache = {}
        
        logger.info(f"Data source manager initialized with {len(self.data_sources)} sources")
    
    def _initialize_sources(self):
        """Initialize configured data sources"""
        for source_name in self.sources:
            if source_name in DATA_SOURCES:
                config = DATA_SOURCES[source_name]
                
                if source_name == "yfinance":
                    self.data_sources[source_name] = YahooFinanceSource(config)
                elif source_name == "tradingeconomics":
                    self.data_sources[source_name] = TradingEconomicsSource(config)
                # Add other sources as needed
                else:
                    logger.warning(f"Source {source_name} not yet implemented")
            else:
                logger.warning(f"Unknown data source: {source_name}")
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for request"""
        key_parts = [method]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return ":".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and not expired"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_data
            else:
                del self.cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """Store data in cache"""
        self.cache[cache_key] = (data, datetime.now())
    
    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> List[MarketData]:
        """
        Get market data with fallback and caching
        
        Args:
            symbol: Currency pair symbol (e.g., "EUR/USD")
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cache
            
        Returns:
            List of MarketData objects
        """
        cache_key = self._get_cache_key(
            "market_data",
            symbol=symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )
        
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Try each source in order
        for source_name, source in self.data_sources.items():
            try:
                data = source.get_market_data(symbol, start_date, end_date)
                if data:
                    if use_cache:
                        self._set_cache(cache_key, data)
                    return data
            except Exception as e:
                logger.warning(f"Failed to get data from {source_name}: {e}")
                continue
        
        logger.error(f"Failed to get market data for {symbol} from all sources")
        return []
    
    def get_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[EconomicEvent]:
        """Get economic calendar events"""
        cache_key = self._get_cache_key(
            "economic_calendar",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            currencies=",".join(currencies or []),
        )
        
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        for source_name, source in self.data_sources.items():
            try:
                data = source.get_economic_calendar(start_date, end_date, currencies)
                if data:
                    if use_cache:
                        self._set_cache(cache_key, data)
                    return data
            except Exception as e:
                logger.warning(f"Failed to get calendar from {source_name}: {e}")
                continue
        
        return []
    
    def get_news_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get news sentiment data"""
        cache_key = self._get_cache_key(
            "news_sentiment",
            symbol=symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )
        
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        for source_name, source in self.data_sources.items():
            try:
                data = source.get_news_sentiment(symbol, start_date, end_date)
                if data:
                    if use_cache:
                        self._set_cache(cache_key, data)
                    return data
            except Exception as e:
                logger.warning(f"Failed to get news from {source_name}: {e}")
                continue
        
        return {"sentiment": 0, "article_count": 0}
    
    def get_multiple_currency_data(
        self,
        currencies: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, List[MarketData]]:
        """
        Get market data for multiple currency pairs
        
        Args:
            currencies: List of currency codes
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping currency pairs to market data
        """
        results = {}
        
        # Generate all major pairs from currencies
        pairs = []
        for i, base in enumerate(currencies):
            for quote in currencies[i+1:]:
                pairs.append(f"{base}/{quote}")
        
        for pair in pairs:
            data = self.get_market_data(pair, start_date, end_date)
            if data:
                results[pair] = data
        
        return results