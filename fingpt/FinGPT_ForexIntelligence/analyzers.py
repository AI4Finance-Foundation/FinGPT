"""
Analyzer modules for the Forex Intelligence Framework
Implements the multi-layered institutional-style analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from fingpt.FinGPT_ForexIntelligence.config import (
    MAJOR_CURRENCIES,
    TRADING_SESSIONS,
    ECONOMIC_INDICATORS,
    COMMODITY_CURRENCY_LINKS,
    RISK_THRESHOLDS,
    STRENGTH_SCORES,
    ANALYSIS_WEIGHTS,
)
from fingpt.FinGPT_ForexIntelligence.data_sources import ForexDataSourceManager, MarketData, EconomicEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CurrencyStrength:
    """Currency strength analysis result"""
    currency: str
    strength_score: float
    trend: str  # strengthening, weakening, neutral
    momentum: float
    volatility: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


@dataclass
class SessionBias:
    """Trading session bias analysis"""
    session_name: str
    overall_bias: str  # bullish, bearish, neutral
    affected_pairs: List[str]
    confidence: float
    key_factors: List[str]
    liquidity_assessment: str


class GlobalMacroAnalyzer:
    """Analyzes global macroeconomic factors and risk sentiment"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
        self.risk_indices = {
            "VIX": "volatility_index",
            "S&P_500": "equity_index",
            "US_10Y_Yield": "bond_yield",
            "DXY": "dollar_index",
        }
    
    def analyze_risk_sentiment(self, date: datetime) -> Dict[str, Any]:
        """
        Analyze global risk sentiment using multiple indicators
        
        Returns:
            Dict with sentiment analysis and key indicators
        """
        try:
            # Get risk index data
            vix_data = self._get_vix_data(date)
            equity_data = self._get_equity_index_data(date)
            bond_data = self._get_bond_yield_data(date)
            
            # Calculate sentiment scores
            sentiment_score = self._calculate_sentiment_score(
                vix_data,
                equity_data,
                bond_data,
            )
            
            # Determine sentiment label
            if sentiment_score >= 0.6:
                sentiment = "risk_on"
            elif sentiment_score <= -0.6:
                sentiment = "risk_off"
            else:
                sentiment = "neutral"
            
            return {
                "overall_sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "vix_level": vix_data.get("level"),
                "equity_performance": equity_data.get("change_percent"),
                "bond_yield_10y": bond_data.get("yield"),
                "liquidity_conditions": self._assess_liquidity(vix_data, bond_data),
                "key_factors": self._identify_key_factors(
                    vix_data,
                    equity_data,
                    bond_data,
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing risk sentiment: {e}")
            return {"overall_sentiment": "neutral", "sentiment_score": 0}
    
    def _get_vix_data(self, date: datetime) -> Dict[str, Any]:
        """Get VIX volatility index data"""
        try:
            # Try to get VIX data using available sources
            # For now, return mock data
            return {
                "level": 18.5,
                "change": -0.5,
                "change_percent": -2.6,
            }
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return {"level": 20.0, "change": 0, "change_percent": 0}
    
    def _get_equity_index_data(self, date: datetime) -> Dict[str, Any]:
        """Get equity index data (S&P 500, etc.)"""
        try:
            return {
                "level": 4500.0,
                "change": 30.0,
                "change_percent": 0.67,
            }
        except Exception as e:
            logger.error(f"Error getting equity data: {e}")
            return {"level": 4400.0, "change": 0, "change_percent": 0}
    
    def _get_bond_yield_data(self, date: datetime) -> Dict[str, Any]:
        """Get bond yield data"""
        try:
            return {
                "yield": 4.2,
                "change": 0.05,
                "change_percent": 1.2,
            }
        except Exception as e:
            logger.error(f"Error getting bond yield data: {e}")
            return {"yield": 4.0, "change": 0, "change_percent": 0}
    
    def _calculate_sentiment_score(
        self,
        vix_data: Dict[str, Any],
        equity_data: Dict[str, Any],
        bond_data: Dict[str, Any],
    ) -> float:
        """Calculate composite sentiment score"""
        vix_score = -1.0 * (vix_data.get("level", 20) - 20) / 10  # Lower VIX = risk on
        equity_score = min(1.0, max(-1.0, equity_data.get("change_percent", 0) / 2))
        bond_score = min(1.0, max(-1.0, bond_data.get("change_percent", 0) / 2))
        
        # Weighted average
        sentiment_score = (
            0.4 * vix_score +
            0.3 * equity_score +
            0.3 * bond_score
        )
        
        return min(1.0, max(-1.0, sentiment_score))
    
    def _assess_liquidity(
        self,
        vix_data: Dict[str, Any],
        bond_data: Dict[str, Any],
    ) -> str:
        """Assess market liquidity conditions"""
        vix = vix_data.get("level", 20)
        yield_spread = bond_data.get("yield", 4.0)
        
        if vix < 15 and yield_spread > 3.5:
            return "high"
        elif vix < 20 and yield_spread > 3.0:
            return "normal"
        else:
            return "low"
    
    def _identify_key_factors(
        self,
        vix_data: Dict[str, Any],
        equity_data: Dict[str, Any],
        bond_data: Dict[str, Any],
    ) -> List[str]:
        """Identify key factors driving sentiment"""
        factors = []
        
        if abs(vix_data.get("change_percent", 0)) > 2:
            factors.append(f"VIX movement: {vix_data['change_percent']:.1f}%")
        
        if abs(equity_data.get("change_percent", 0)) > 1:
            factors.append(f"Equity market: {equity_data['change_percent']:.1f}%")
        
        if abs(bond_data.get("change_percent", 0)) > 1:
            factors.append(f"Bond yield shift: {bond_data['change_percent']:.1f}%")
        
        return factors if factors else ["No significant factors"]
    
    def analyze_macro_risks(self, date: datetime) -> Dict[str, Any]:
        """Analyze macroeconomic risk factors"""
        return {
            "risk_score": 0.3,
            "key_risks": [
                "Geopolitical tensions",
                "Central bank policy divergence",
                "Economic growth slowdown",
            ],
            "risk_regions": ["Europe", "Asia"],
        }


class CapitalFlowAnalyzer:
    """Analyzes institutional capital flows"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
    
    def analyze_flows(self, date: datetime) -> Dict[str, Any]:
        """
        Analyze institutional capital flows
        
        Returns:
            Dict with flow analysis and currency implications
        """
        try:
            # Get flow data for different asset classes
            equity_flows = self._get_equity_flows(date)
            bond_flows = self._get_bond_flows(date)
            etf_flows = self._get_etf_flows(date)
            
            # Analyze flow patterns
            flow_analysis = {
                "overall_sentiment": self._determine_flow_sentiment(
                    equity_flows,
                    bond_flows,
                    etf_flows,
                ),
                "equity_flows": equity_flows,
                "bond_flows": bond_flows,
                "etf_flows": etf_flows,
                "currency_implications": self._analyze_currency_impacts(
                    equity_flows,
                    bond_flows,
                    etf_flows,
                ),
                "liquidity_demand": self._assess_liquidity_demand(
                    equity_flows,
                    bond_flows,
                ),
            }
            
            return flow_analysis
        except Exception as e:
            logger.error(f"Error analyzing capital flows: {e}")
            return {}
    
    def _get_equity_flows(self, date: datetime) -> Dict[str, Any]:
        """Get equity market flow data"""
        return {
            "direction": "inflow",
            "magnitude": "moderate",
            "key_markets": ["US", "Europe"],
            "change_vs_previous": "+5%",
        }
    
    def _get_bond_flows(self, date: datetime) -> Dict[str, Any]:
        """Get bond market flow data"""
        return {
            "direction": "outflow",
            "magnitude": "small",
            "key_markets": ["US Treasuries"],
            "change_vs_previous": "-2%",
        }
    
    def _get_etf_flows(self, date: datetime) -> Dict[str, Any]:
        """Get ETF flow data"""
        return {
            "direction": "inflow",
            "magnitude": "moderate",
            "key_etfs": ["SPY", "QQQ", "EFA"],
            "change_vs_previous": "+8%",
        }
    
    def _determine_flow_sentiment(
        self,
        equity_flows: Dict[str, Any],
        bond_flows: Dict[str, Any],
        etf_flows: Dict[str, Any],
    ) -> str:
        """Determine overall flow sentiment"""
        score = 0
        
        if equity_flows.get("direction") == "inflow":
            score += 2
        elif equity_flows.get("direction") == "outflow":
            score -= 2
        
        if bond_flows.get("direction") == "inflow":
            score += 1
        elif bond_flows.get("direction") == "outflow":
            score -= 1
        
        if etf_flows.get("direction") == "inflow":
            score += 2
        elif etf_flows.get("direction") == "outflow":
            score -= 2
        
        if score >= 3:
            return "strong_bullish"
        elif score >= 1:
            return "bullish"
        elif score <= -3:
            return "strong_bearish"
        elif score <= -1:
            return "bearish"
        else:
            return "neutral"
    
    def _analyze_currency_impacts(
        self,
        equity_flows: Dict[str, Any],
        bond_flows: Dict[str, Any],
        etf_flows: Dict[str, Any],
    ) -> Dict[str, str]:
        """Analyze capital flow impacts on currencies"""
        impacts = {}
        
        # USD is typically impacted by US equity and bond flows
        if equity_flows.get("direction") == "inflow":
            impacts["USD"] = "positive"
        elif equity_flows.get("direction") == "outflow":
            impacts["USD"] = "negative"
        
        # Risk currencies (AUD, NZD) follow equity flows
        if equity_flows.get("direction") == "inflow":
            impacts["AUD"] = "positive"
            impacts["NZD"] = "positive"
        
        # Safe-haven currencies (CHF, JPY) often move opposite
        if equity_flows.get("direction") == "outflow":
            impacts["CHF"] = "positive"
            impacts["JPY"] = "positive"
        
        return impacts
    
    def _assess_liquidity_demand(
        self,
        equity_flows: Dict[str, Any],
        bond_flows: Dict[str, Any],
    ) -> str:
        """Assess overall liquidity demand"""
        if (equity_flows.get("direction") == "inflow" and
            bond_flows.get("direction") == "inflow"):
            return "high"
        elif (equity_flows.get("direction") == "outflow" and
              bond_flows.get("direction") == "outflow"):
            return "low"
        else:
            return "moderate"


class EconomicDataAnalyzer:
    """Analyzes economic data and indicators"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
    
    def get_currency_score(self, currency: str, date: datetime) -> float:
        """
        Calculate economic score for a currency based on recent data
        
        Returns:
            Score between -1 (weak) and 1 (strong)
        """
        try:
            # Get recent economic data for the currency
            economic_events = self.data_manager.get_economic_calendar(
                date - timedelta(days=30),
                date,
                [currency],
            )
            
            score = 0.0
            event_count = 0
            
            for event in economic_events:
                if event.actual and event.forecast:
                    # Calculate deviation from forecast
                    deviation = (event.actual - event.forecast) / abs(event.forecast)
                    
                    # Weight by impact
                    weight = 1.0 if event.impact == "HIGH" else 0.5
                    
                    # Add to score (positive deviation = positive for currency)
                    score += deviation * weight
                    event_count += 1
            
            if event_count > 0:
                score = score / event_count
                return min(1.0, max(-1.0, score))
            
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating currency score: {e}")
            return 0.0
    
    def analyze_key_indicators(
        self,
        currency: str,
        date: datetime,
    ) -> Dict[str, Any]:
        """Analyze key economic indicators for a currency"""
        return {
            "overall_score": self.get_currency_score(currency, date),
            "key_indicators": [
                "CPI: +0.2% vs forecast",
                "GDP: +2.1% vs forecast",
                "Employment: -0.1% vs forecast",
            ],
            "trend": "improving",
        }


class CentralBankAnalyzer:
    """Analyzes central bank policies and stances"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
        self.central_banks = {
            "USD": "Federal Reserve",
            "EUR": "European Central Bank",
            "GBP": "Bank of England",
            "JPY": "Bank of Japan",
            "CHF": "Swiss National Bank",
            "CAD": "Bank of Canada",
            "AUD": "Reserve Bank of Australia",
            "NZD": "Reserve Bank of New Zealand",
            "CNY": "People's Bank of China",
        }
    
    def get_policy_bias(self, currency: str, date: datetime) -> str:
        """
        Get current policy bias for a central bank
        
        Returns:
            Policy bias: hawkish, dovish, or neutral
        """
        try:
            # In production, this would analyze actual policy statements
            # For now, return a simplified analysis
            central_bank = self.central_banks.get(currency, "Unknown")
            
            # Get recent policy-related events
            policy_events = self.data_manager.get_economic_calendar(
                date - timedelta(days=90),
                date,
                [currency],
            )
            
            # Filter for interest rate decisions
            rate_decisions = [
                e for e in policy_events
                if "interest rate" in e.name.lower()
            ]
            
            if rate_decisions:
                latest_decision = rate_decisions[-1]
                if latest_decision.actual and latest_decision.previous:
                    if latest_decision.actual > latest_decision.previous:
                        return "hawkish"
                    elif latest_decision.actual < latest_decision.previous:
                        return "dovish"
            
            return "neutral"
        except Exception as e:
            logger.error(f"Error getting policy bias: {e}")
            return "neutral"
    
    def analyze_policy_stance(self, currency: str, date: datetime) -> Dict[str, Any]:
        """Comprehensive policy stance analysis"""
        return {
            "central_bank": self.central_banks.get(currency, "Unknown"),
            "current_bias": self.get_policy_bias(currency, date),
            "next_meeting": self._get_next_meeting(currency, date),
            "policy_outlook": self._get_policy_outlook(currency, date),
        }
    
    def _get_next_meeting(self, currency: str, date: datetime) -> Optional[str]:
        """Get next central bank meeting date"""
        # In production, this would come from an actual calendar
        return None
    
    def _get_policy_outlook(self, currency: str, date: datetime) -> str:
        """Get policy outlook"""
        bias = self.get_policy_bias(currency, date)
        if bias == "hawkish":
            return "likely_to_tighten"
        elif bias == "dovish":
            return "likely_to_ease"
        else:
            return "on_hold"


class GeopoliticalAnalyzer:
    """Analyzes geopolitical risks and their currency impacts"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
    
    def analyze_risks(self, date: datetime) -> Dict[str, Any]:
        """
        Analyze current geopolitical risks
        
        Returns:
            Dict with risk analysis and currency impacts
        """
        try:
            # In production, this would analyze actual geopolitical data
            # For now, return a structured template
            return {
                "risk_score": 0.4,
                "risk_level": "moderate",
                "key_risks": [
                    {
                        "type": "trade_tensions",
                        "regions": ["US", "China"],
                        "affected_currencies": ["USD", "CNY"],
                        "impact_level": "medium",
                    },
                    {
                        "type": "geopolitical_conflict",
                        "regions": ["Europe"],
                        "affected_currencies": ["EUR", "CHF"],
                        "impact_level": "low",
                    },
                ],
                "safe_haven_demand": self._assess_safe_haven_demand(),
                "currency_impacts": self._analyze_currency_impacts(),
            }
        except Exception as e:
            logger.error(f"Error analyzing geopolitical risks: {e}")
            return {"risk_score": 0, "risk_level": "low"}
    
    def _assess_safe_haven_demand(self) -> str:
        """Assess demand for safe-haven currencies"""
        return "moderate"
    
    def _analyze_currency_impacts(self) -> Dict[str, str]:
        """Analyze geopolitical impacts on currencies"""
        return {
            "USD": "mixed",
            "CHF": "positive",
            "JPY": "positive",
            "EUR": "negative",
            "GBP": "negative",
        }


class CommodityAnalyzer:
    """Analyzes commodity-currency relationships"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
    
    def analyze_impacts(self, date: datetime) -> Dict[str, Any]:
        """
        Analyze commodity price impacts on currencies
        
        Returns:
            Dict with commodity analysis and currency impacts
        """
        try:
            commodity_prices = self._get_commodity_prices(date)
            
            impacts = {}
            for currency, config in COMMODITY_CURRENCY_LINKS.items():
                impact = self._calculate_commodity_impact(
                    currency,
                    config,
                    commodity_prices,
                )
                impacts[currency] = impact
            
            return {
                "commodity_prices": commodity_prices,
                "currency_impacts": impacts,
                "key_correlations": self._identify_key_correlations(
                    commodity_prices,
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing commodity impacts: {e}")
            return {}
    
    def _get_commodity_prices(self, date: datetime) -> Dict[str, float]:
        """Get current commodity prices"""
        # In production, this would fetch actual prices
        return {
            "gold": 1950.0,
            "crude_oil": 75.0,
            "iron_ore": 120.0,
            "dairy": 3500.0,
        }
    
    def _calculate_commodity_impact(
        self,
        currency: str,
        config: Dict[str, Any],
        commodity_prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate commodity impact on a currency"""
        primary_commodity = config.get("primary_commodity", "")
        correlation = config.get("correlation_strength", "medium")
        
        if primary_commodity in commodity_prices:
            price = commodity_prices[primary_commodity]
            
            # Simplified impact calculation
            if correlation == "high":
                impact_score = 0.6
            elif correlation == "medium":
                impact_score = 0.4
            else:  # inverse
                impact_score = -0.5
            
            return {
                "primary_commodity": primary_commodity,
                "price": price,
                "correlation": correlation,
                "impact_score": impact_score,
                "significance": "high" if abs(impact_score) > 0.5 else "moderate",
            }
        
        return {"significance": "low"}
    
    def _identify_key_correlations(
        self,
        commodity_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Identify key commodity-currency correlations"""
        correlations = []
        
        # Gold - CHF correlation
        if "gold" in commodity_prices:
            correlations.append({
                "commodity": "gold",
                "currencies": ["CHF", "USD"],
                "correlation_type": "positive",
                "strength": "high",
            })
        
        # Oil - CAD correlation
        if "crude_oil" in commodity_prices:
            correlations.append({
                "commodity": "crude_oil",
                "currencies": ["CAD"],
                "correlation_type": "positive",
                "strength": "high",
            })
        
        return correlations


class CurrencyStrengthAnalyzer:
    """Analyzes and ranks currency strength"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
    
    def analyze_strength(self, date: datetime) -> Dict[str, Dict[str, Any]]:
        """
        Analyze strength for all major currencies
        
        Returns:
            Dict mapping currencies to strength analysis
        """
        strength_data = {}
        
        for currency in MAJOR_CURRENCIES:
            try:
                strength = self._analyze_single_currency(currency, date)
                strength_data[currency] = strength
            except Exception as e:
                logger.error(f"Error analyzing {currency}: {e}")
                strength_data[currency] = {"error": str(e)}
        
        # Rank currencies
        ranked = self._rank_currencies(strength_data)
        
        # Add rankings to data
        for i, (currency, data) in enumerate(ranked.items()):
            strength_data[currency]["rank"] = i + 1
            strength_data[currency]["strength_label"] = self._get_strength_label(
                data.get("strength_score", 0),
            )
        
        return strength_data
    
    def _analyze_single_currency(
        self,
        currency: str,
        date: datetime,
    ) -> Dict[str, Any]:
        """Analyze strength for a single currency"""
        # Get recent market data
        market_data = self.data_manager.get_market_data(
            f"{currency}/USD",
            date - timedelta(days=30),
            date,
        )
        
        if not market_data:
            return {"strength_score": 0, "trend": "neutral"}
        
        # Calculate strength metrics
        strength_score = self._calculate_strength_score(market_data)
        trend = self._determine_trend(market_data)
        momentum = self._calculate_momentum(market_data)
        volatility = self._calculate_volatility(market_data)
        
        return {
            "strength_score": strength_score,
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
        }
    
    def _calculate_strength_score(self, market_data: List[MarketData]) -> float:
        """Calculate overall strength score"""
        if not market_data:
            return 0.0
        
        # Calculate return over period
        start_price = market_data[0].mid
        end_price = market_data[-1].mid
        
        if start_price and end_price:
            return (end_price - start_price) / start_price
        
        return 0.0
    
    def _determine_trend(self, market_data: List[MarketData]) -> str:
        """Determine trend direction"""
        if len(market_data) < 5:
            return "neutral"
        
        recent = market_data[-5:]
        changes = [
            d.mid for d in recent if d.mid
        ]
        
        if len(changes) < 2:
            return "neutral"
        
        # Simple trend detection
        if changes[-1] > changes[0]:
            return "strengthening"
        elif changes[-1] < changes[0]:
            return "weakening"
        else:
            return "neutral"
    
    def _calculate_momentum(self, market_data: List[MarketData]) -> float:
        """Calculate momentum indicator"""
        if len(market_data) < 10:
            return 0.0
        
        recent = market_data[-10:]
        changes = []
        
        for i in range(1, len(recent)):
            if recent[i].mid and recent[i-1].mid:
                change = (recent[i].mid - recent[i-1].mid) / recent[i-1].mid
                changes.append(change)
        
        if changes:
            return sum(changes) / len(changes)
        
        return 0.0
    
    def _calculate_volatility(self, market_data: List[MarketData]) -> float:
        """Calculate volatility"""
        if len(market_data) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(market_data)):
            if market_data[i].mid and market_data[i-1].mid:
                change = (market_data[i].mid - market_data[i-1].mid) / market_data[i-1].mid
                changes.append(change)
        
        if changes:
            return np.std(changes) if changes else 0.0
        
        return 0.0
    
    def _rank_currencies(
        self,
        strength_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Rank currencies by strength"""
        # Sort by strength score
        sorted_currencies = sorted(
            strength_data.items(),
            key=lambda x: x[1].get("strength_score", 0),
            reverse=True,
        )
        
        return dict(sorted_currencies)
    
    def _get_strength_label(self, score: float) -> str:
        """Get strength label from score"""
        if score >= 0.03:
            return "very_strong"
        elif score >= 0.015:
            return "strong"
        elif score >= 0.005:
            return "moderate_strong"
        elif score >= -0.005:
            return "neutral"
        elif score >= -0.015:
            return "moderate_weak"
        elif score >= -0.03:
            return "weak"
        else:
            return "very_weak"


class SessionAnalyzer:
    """Analyzes trading session characteristics and biases"""
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
    
    def analyze_session(
        self,
        session_name: str,
        date: datetime,
    ) -> Dict[str, Any]:
        """
        Analyze a specific trading session
        
        Args:
            session_name: Name of session (asia, london, new_york)
            date: Analysis date
            
        Returns:
            Session analysis dict
        """
        try:
            session_config = TRADING_SESSIONS.get(session_name)
            if not session_config:
                return {"error": f"Unknown session: {session_name}"}
            
            # Get session-specific data
            session_pairs = session_config.key_pairs
            market_data = {}
            
            for pair in session_pairs:
                data = self.data_manager.get_market_data(
                    pair,
                    date - timedelta(days=7),  # Week of data
                    date,
                )
                if data:
                    market_data[pair] = data
            
            # Analyze session bias
            bias = self._determine_session_bias(session_name, market_data)
            
            return {
                "session_name": session_name,
                "characteristics": session_config.characteristics,
                "overall_bias": bias["direction"],
                "affected_pairs": bias["affected_pairs"],
                "confidence": bias["confidence"],
                "key_factors": bias["factors"],
                "liquidity_assessment": session_config.characteristics.get("liquidity"),
                "volatility_assessment": session_config.characteristics.get("volatility"),
            }
        except Exception as e:
            logger.error(f"Error analyzing session {session_name}: {e}")
            return {"error": str(e)}
    
    def _determine_session_bias(
        self,
        session_name: str,
        market_data: Dict[str, List[MarketData]],
    ) -> Dict[str, Any]:
        """Determine bias for a session"""
        biases = []
        
        for pair, data in market_data.items():
            if len(data) >= 2:
                change = (data[-1].mid - data[0].mid) / data[0].mid if data[0].mid else 0
                biases.append({
                    "pair": pair,
                    "change": change,
                    "direction": "bullish" if change > 0 else "bearish",
                })
        
        if not biases:
            return {
                "direction": "neutral",
                "affected_pairs": [],
                "confidence": 0.0,
                "factors": ["insufficient_data"],
            }
        
        # Determine overall bias
        positive_changes = sum(1 for b in biases if b["change"] > 0)
        negative_changes = sum(1 for b in biases if b["change"] < 0)
        
        if positive_changes > negative_changes:
            direction = "bullish"
        elif negative_changes > positive_changes:
            direction = "bearish"
        else:
            direction = "neutral"
        
        confidence = abs(positive_changes - negative_changes) / len(biases)
        
        return {
            "direction": direction,
            "affected_pairs": [b["pair"] for b in biases],
            "confidence": confidence,
            "factors": [f"{b['pair']}: {b['direction']}" for b in biases],
        }