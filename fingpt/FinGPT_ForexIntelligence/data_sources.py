"""
Data Source Management for Forex Intelligence
Handles data ingestion from multiple financial data providers
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, is_dataclass
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

    @property
    def timestamp(self) -> datetime:
        """Alias for date (the event system reads event.timestamp)"""
        return self.date


class BaseDataSource(ABC):
    """Abstract base class for data sources"""

    def __init__(self, config: Dict[str, Any]):
        # DATA_SOURCES entries in config.py are DataSource dataclasses
        if is_dataclass(config):
            config = asdict(config)
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


class FXMacroDataSource(BaseDataSource):
    """
    FXMacroData data source implementation

    Release calendars and announcement values come from official publishers
    (central banks and statistics agencies). The USD calendar and USD
    announcements work without an API key; other currencies and FX rates
    need FXMACRODATA_API_KEY.
    """

    KEYLESS_CURRENCIES = {"USD"}

    def __init__(self, config: Dict[str, Any], include_actuals: bool = True):
        super().__init__(config)
        if not self.base_url.startswith("https://"):
            raise DataSourceException("FXMacroData base_url must use https")
        self.include_actuals = include_actuals

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request, sending the API key in the X-API-Key header"""
        self._check_rate_limit()

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {"Accept": "application/json"}

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            self.last_request_time = datetime.now()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {self.name}: {e}")
            raise DataSourceException(f"Request failed: {e}")

    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[MarketData]:
        """Get daily forex rates from FXMacroData"""
        if not self.api_key:
            logger.warning("FXMACRODATA_API_KEY not set, FXMacroData forex rates need an API key")
            return []

        try:
            base, quote = self._convert_to_fxmd_pair(symbol)

            params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "limit": 100,
                "offset": 0,
            }

            rows = []
            while True:
                data = self._make_request(f"/forex/{base}/{quote}", dict(params))
                page = data.get("data", [])
                rows.extend(page)

                pagination = data.get("pagination") or {}
                next_offset = pagination.get("next_offset")
                if not pagination.get("has_more") or not page:
                    break
                if next_offset is None:
                    next_offset = params["offset"] + len(page)
                if next_offset <= params["offset"]:
                    break
                params["offset"] = next_offset

            # Rows come back most-recent-first
            rows.sort(key=lambda item: item["date"])

            market_data = []
            previous_mid = None
            for item in rows:
                mid = item.get("close")
                if mid is None:
                    mid = item.get("val")
                if mid is None:
                    continue

                change = None
                change_percent = None
                if previous_mid:
                    change = mid - previous_mid
                    change_percent = (change / previous_mid) * 100

                market_data.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=datetime.strptime(item["date"][:10], "%Y-%m-%d"),
                        mid=mid,
                        change=change,
                        change_percent=change_percent,
                    )
                )
                previous_mid = mid

            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data from FXMacroData: {e}")
            return []

    def _convert_to_fxmd_pair(self, symbol: str) -> tuple:
        """Convert forex pair ("EUR/USD" or "EURUSD") to base and quote codes"""
        cleaned = symbol.strip().upper()
        if cleaned.endswith("=X"):
            cleaned = cleaned[:-2]
        if "/" in cleaned:
            base, quote = cleaned.split("/", 1)
        elif len(cleaned) == 6:
            base, quote = cleaned[:3], cleaned[3:]
        else:
            raise DataSourceException(f"Unrecognised forex symbol: {symbol}")

        base, quote = base.strip(), quote.strip()
        if not (len(base) == 3 and len(quote) == 3 and base.isalpha() and quote.isalpha()):
            raise DataSourceException(f"Unrecognised forex symbol: {symbol}")
        return base, quote

    def get_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get economic release calendar from FXMacroData"""
        try:
            params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }

            events = []
            for currency in currencies or MAJOR_CURRENCIES:
                currency = currency.upper()
                if not self.api_key and currency not in self.KEYLESS_CURRENCIES:
                    logger.info(
                        f"Skipping {currency} calendar: FXMacroData needs FXMACRODATA_API_KEY "
                        "for currencies other than USD"
                    )
                    continue

                try:
                    data = self._make_request(f"/calendar/{currency}", params)
                except DataSourceException as e:
                    logger.warning(f"Failed to get {currency} calendar from FXMacroData: {e}")
                    continue

                release_values = {}
                for item in data.get("data", []):
                    release = item.get("release", "")
                    announced_at = item.get("announcement_datetime")
                    if announced_at is None:
                        continue

                    actual, previous = None, None
                    if self.include_actuals and release:
                        actual, previous = self._get_release_values(
                            currency, release, announced_at, release_values
                        )

                    importance = item.get("event_importance") or "medium"
                    events.append(
                        EconomicEvent(
                            event_id=item.get("calendar_event_id") or f"{currency.lower()}_{release}_{announced_at}",
                            name=item.get("name") or release,
                            currency=currency,
                            # Naive UTC datetime
                            date=datetime.fromtimestamp(announced_at, tz=timezone.utc).replace(tzinfo=None),
                            actual=actual,
                            previous=previous,
                            impact=self._map_impact(importance),
                            tier=self._map_tier(importance),
                        )
                    )

            return events
        except Exception as e:
            logger.error(f"Error fetching economic calendar from FXMacroData: {e}")
            return []

    def _get_release_values(
        self,
        currency: str,
        release: str,
        announced_at: int,
        cache: Dict[str, List[Dict[str, Any]]],
    ) -> tuple:
        """
        Get (actual, previous) for a calendar release from the latest announcements.

        A release that has happened is matched on its announcement time. An
        upcoming release gets the latest published value as previous. Forecast
        is left empty because no consensus value is fetched.
        """
        if release not in cache:
            try:
                data = self._make_request(f"/announcements/{currency}/{release}", {"limit": 2})
                cache[release] = data.get("data", [])
            except DataSourceException as e:
                logger.debug(f"No FXMacroData announcements for {currency} {release}: {e}")
                cache[release] = []

        rows = cache[release]
        actual, previous = None, None
        for row in rows:
            if row.get("announcement_datetime") == announced_at:
                actual = row.get("val")
                previous = row.get("previous_value")
                break

        if previous is None:
            earlier = [
                row for row in rows
                if row.get("announcement_datetime") is not None
                and row["announcement_datetime"] < announced_at
                and row.get("val") is not None
            ]
            if earlier:
                previous = max(earlier, key=lambda row: row["announcement_datetime"])["val"]

        return actual, previous

    def _map_impact(self, importance: str) -> str:
        """Map importance level to impact"""
        mapping = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        return mapping.get(str(importance).lower(), "MEDIUM")

    def _map_tier(self, importance: str) -> str:
        """Map importance level to tier"""
        mapping = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        return mapping.get(str(importance).lower(), "MEDIUM")

    def get_news_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """FXMacroData doesn't provide news sentiment"""
        # Empty result lets the manager fall through to the next source
        return {}


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
                elif source_name == "fxmacrodata":
                    self.data_sources[source_name] = FXMacroDataSource(config)
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