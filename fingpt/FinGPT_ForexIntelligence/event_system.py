"""
Event Anticipation System
Monitors and analyzes upcoming economic events and their potential market impact
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from fingpt.FinGPT_ForexIntelligence.config import ECONOMIC_INDICATORS, EVENT_IMPACT, MAJOR_CURRENCIES
from fingpt.FinGPT_ForexIntelligence.data_sources import ForexDataSourceManager, EconomicEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventImpactLevel(Enum):
    """Event impact levels"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class EventImpactAnalysis:
    """Analysis of an economic event's potential impact"""
    event: EconomicEvent
    expected_volatility: str  # high, medium, low
    affected_pairs: List[str]
    expected_direction: Optional[str]  # bullish, bearish, neutral
    confidence: float
    historical_impact: Optional[Dict[str, float]] = None
    risk_assessment: str = "medium"


class EventAnticipationSystem:
    """
    System for anticipating and analyzing upcoming economic events
    """
    
    def __init__(self, data_manager: ForexDataSourceManager):
        self.data_manager = data_manager
        self.impact_history = {}  # Store historical event impacts
    
    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
        min_impact: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get economic events for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            currencies: Filter by currencies
            min_impact: Minimum impact level (HIGH, MEDIUM, LOW)
            
        Returns:
            List of event dicts
        """
        try:
            events = self.data_manager.get_economic_calendar(
                start_date,
                end_date,
                currencies or MAJOR_CURRENCIES,
            )
            
            # Filter by impact level if specified
            if min_impact:
                impact_priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                min_priority = impact_priority.get(min_impact, 0)
                
                events = [
                    event for event in events
                    if impact_priority.get(event.impact, 0) >= min_priority
                ]
            
            # Convert to dicts
            event_dicts = []
            for event in events:
                event_dicts.append({
                    "event_id": event.event_id,
                    "name": event.name,
                    "currency": event.currency,
                    "date": event.timestamp.isoformat(),
                    "actual": event.actual,
                    "forecast": event.forecast,
                    "previous": event.previous,
                    "impact": event.impact,
                    "tier": event.tier,
                })
            
            # Sort by date and impact
            event_dicts.sort(
                key=lambda x: (
                    x["date"],
                    {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x["impact"], 0),
                )
            )
            
            return event_dicts
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def analyze_event_impact(
        self,
        event: EconomicEvent,
    ) -> EventImpactAnalysis:
        """
        Analyze the potential impact of an economic event
        
        Args:
            event: Economic event to analyze
            
        Returns:
            EventImpactAnalysis with impact assessment
        """
        try:
            # Get event indicator configuration
            indicator_key = self._match_indicator_to_config(event.name)
            indicator_config = ECONOMIC_INDICATORS.get(indicator_key)
            
            if not indicator_config:
                return self._default_impact_analysis(event)
            
            # Determine expected volatility
            expected_volatility = self._determine_volatility(
                indicator_config.tier,
                event.impact,
            )
            
            # Determine affected pairs
            affected_pairs = self._determine_affected_pairs(
                event.currency,
                indicator_config,
            )
            
            # Determine expected direction based on forecast vs previous
            expected_direction = self._determine_expected_direction(
                event,
                indicator_config,
            )
            
            # Calculate confidence
            confidence = self._calculate_impact_confidence(
                event,
                indicator_config,
            )
            
            # Get historical impact if available
            historical_impact = self._get_historical_impact(event)
            
            # Risk assessment
            risk_assessment = self._assess_event_risk(
                event,
                indicator_config,
                historical_impact,
            )
            
            return EventImpactAnalysis(
                event=event,
                expected_volatility=expected_volatility,
                affected_pairs=affected_pairs,
                expected_direction=expected_direction,
                confidence=confidence,
                historical_impact=historical_impact,
                risk_assessment=risk_assessment,
            )
        except Exception as e:
            logger.error(f"Error analyzing event impact: {e}")
            return self._default_impact_analysis(event)
    
    def _match_indicator_to_config(self, event_name: str) -> str:
        """Match event name to indicator configuration"""
        event_name_lower = event_name.lower()
        
        if "interest rate" in event_name_lower or "rate decision" in event_name_lower:
            return "interest_rate_decision"
        elif "cpi" in event_name_lower or "inflation" in event_name_lower:
            return "cpi_inflation"
        elif "gdp" in event_name_lower:
            return "gdp"
        elif "employment" in event_name_lower or "non-farm" in event_name_lower or "unemployment" in event_name_lower:
            return "employment"
        elif "pmi" in event_name_lower:
            return "pmi"
        elif "retail sales" in event_name_lower:
            return "retail_sales"
        elif "trade balance" in event_name_lower:
            return "trade_balance"
        elif "industrial production" in event_name_lower:
            return "industrial_production"
        elif "consumer confidence" in event_name_lower:
            return "consumer_confidence"
        elif "building permits" in event_name_lower:
            return "building_permits"
        
        return "unknown"
    
    def _determine_volatility(
        self,
        indicator_tier: str,
        event_impact: str,
    ) -> str:
        """Determine expected volatility from event"""
        tier_priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        max_priority = max(
            tier_priority.get(indicator_tier, 0),
            tier_priority.get(event_impact, 0),
        )
        
        if max_priority >= 3:
            return "high"
        elif max_priority >= 2:
            return "medium"
        else:
            return "low"
    
    def _determine_affected_pairs(
        self,
        currency: str,
        indicator_config: Any,
    ) -> List[str]:
        """Determine which currency pairs are affected"""
        affected_pairs = []
        
        # Major pairs involving the currency
        major_pairs = [
            f"{currency}/USD",
            f"USD/{currency}",
            f"EUR/{currency}",
            f"GBP/{currency}",
            f"JPY/{currency}",
        ]
        
        affected_pairs.extend(major_pairs)
        
        return affected_pairs
    
    def _determine_expected_direction(
        self,
        event: EconomicEvent,
        indicator_config: Any,
    ) -> Optional[str]:
        """Determine expected market direction from event"""
        if not event.forecast or not event.previous:
            return None
        
        # Calculate deviation
        deviation = (event.forecast - event.previous) / abs(event.previous) if event.previous != 0 else 0
        
        # For most economic indicators, higher than expected is bullish for the currency
        if deviation > 0.01:  # 1% better than expected
            return "bullish"
        elif deviation < -0.01:  # 1% worse than expected
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_impact_confidence(
        self,
        event: EconomicEvent,
        indicator_config: Any,
    ) -> float:
        """Calculate confidence in impact analysis"""
        confidence = 0.5
        
        # Higher confidence for high-tier indicators
        if indicator_config.tier == "HIGH":
            confidence += 0.2
        elif indicator_config.tier == "MEDIUM":
            confidence += 0.1
        
        # Higher confidence if forecast is available
        if event.forecast:
            confidence += 0.1
        
        # Higher confidence if previous data is available
        if event.previous:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _get_historical_impact(
        self,
        event: EconomicEvent,
    ) -> Optional[Dict[str, float]]:
        """Get historical impact data for similar events"""
        # In production, this would query historical data
        # For now, return None
        return None
    
    def _assess_event_risk(
        self,
        event: EconomicEvent,
        indicator_config: Any,
        historical_impact: Optional[Dict[str, float]],
    ) -> str:
        """Assess the risk level of an event"""
        if indicator_config.tier == "HIGH" and event.impact == "HIGH":
            return "high"
        elif indicator_config.tier == "HIGH" or event.impact == "HIGH":
            return "medium"
        else:
            return "low"
    
    def _default_impact_analysis(
        self,
        event: EconomicEvent,
    ) -> EventImpactAnalysis:
        """Default impact analysis when config is not available"""
        return EventImpactAnalysis(
            event=event,
            expected_volatility="medium",
            affected_pairs=[f"{event.currency}/USD"],
            expected_direction=None,
            confidence=0.3,
            risk_assessment="medium",
        )
    
    def get_high_impact_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get only high-impact events"""
        return self.get_events(
            start_date,
            end_date,
            currencies,
            min_impact="HIGH",
        )
    
    def get_event_risk_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get events organized by risk level
        
        Returns:
            Dict with keys 'high', 'medium', 'low' containing event lists
        """
        all_events = self.get_events(start_date, end_date)
        
        risk_calendar = {
            "high": [],
            "medium": [],
            "low": [],
        }
        
        for event in all_events:
            impact = event.get("impact", "LOW")
            if impact in risk_calendar:
                risk_calendar[impact].append(event)
        
        return risk_calendar
    
    def analyze_event_cluster(
        self,
        events: List[EconomicEvent],
    ) -> Dict[str, Any]:
        """
        Analyze a cluster of events (multiple events close in time)
        
        Args:
            events: List of events to analyze as a cluster
            
        Returns:
            Cluster analysis dict
        """
        if not events:
            return {"cluster_risk": "low", "total_impact": 0}
        
        # Calculate total impact
        total_impact = sum(
            EVENT_IMPACT.get(event.impact, 0.3)
            for event in events
        )
        
        # Determine cluster risk
        if total_impact >= 2.0:
            cluster_risk = "high"
        elif total_impact >= 1.0:
            cluster_risk = "medium"
        else:
            cluster_risk = "low"
        
        # Identify affected currencies
        affected_currencies = list(set(event.currency for event in events))
        
        return {
            "cluster_risk": cluster_risk,
            "total_impact": total_impact,
            "event_count": len(events),
            "affected_currencies": affected_currencies,
            "time_span": self._calculate_time_span(events),
        }
    
    def _calculate_time_span(
        self,
        events: List[EconomicEvent],
    ) -> str:
        """Calculate the time span of events"""
        if len(events) < 2:
            return "single_event"
        
        timestamps = [event.timestamp for event in events]
        time_diff = max(timestamps) - min(timestamps)
        
        hours = time_diff.total_seconds() / 3600
        
        if hours < 1:
            return "less_than_1_hour"
        elif hours < 6:
            return "1_to_6_hours"
        elif hours < 24:
            return "6_to_24_hours"
        else:
            return "more_than_24_hours"