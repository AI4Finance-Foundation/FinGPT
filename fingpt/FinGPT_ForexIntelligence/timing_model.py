"""
Trade Timing Model
Provides entry/exit timing recommendations based on event risk and session flow
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from fingpt.FinGPT_ForexIntelligence.config import TRADING_SESSIONS, TRADE_TIMING_RULES
from fingpt.FinGPT_ForexIntelligence.event_system import EventAnticipationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimingRecommendation(Enum):
    """Timing recommendations"""
    ENTER_NOW = "enter_now"
    WAIT_FOR_BETTER_ENTRY = "wait_for_better_entry"
    AVOID_ENTRY = "avoid_entry"
    REDUCE_POSITION = "reduce_position"
    EXIT_POSITION = "exit_position"


@dataclass
class TimingAssessment:
    """Complete timing assessment for a trade"""
    recommendation: TimingRecommendation
    confidence: float
    factors: List[str]
    suggested_entry_time: Optional[datetime] = None
    suggested_exit_time: Optional[datetime] = None
    risk_level: str = "medium"
    session_context: str = ""


class TradeTimingModel:
    """
    Model for determining optimal trade timing based on events and sessions
    """
    
    def __init__(self, event_system: EventAnticipationSystem):
        self.event_system = event_system
    
    def assess_timing(
        self,
        trade_opportunity: Dict[str, Any],
        current_time: datetime,
        look_ahead_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Assess timing for a trade opportunity
        
        Args:
            trade_opportunity: Trade opportunity dict
            current_time: Current timestamp
            look_ahead_hours: Hours to look ahead for events
            
        Returns:
            Timing assessment dict
        """
        try:
            pair = trade_opportunity.get("pair", "")
            direction = trade_opportunity.get("direction", "")
            
            # Get upcoming events
            upcoming_events = self.event_system.get_events(
                current_time,
                current_time + timedelta(hours=look_ahead_hours),
            )
            
            # Filter events relevant to the pair
            relevant_events = self._filter_relevant_events(
                pair,
                upcoming_events,
            )
            
            # Assess session context
            session_context = self._assess_session_context(current_time)
            
            # Apply timing rules
            timing_factors = []
            recommendation = "enter_now"
            confidence = 0.7
            risk_level = "medium"
            
            # Check for high-impact events near current time
            event_risk = self._assess_event_risk(
                relevant_events,
                current_time,
            )
            
            if event_risk["avoid_entry"]:
                recommendation = "avoid_entry"
                confidence = 0.9
                risk_level = "high"
                timing_factors.append(event_risk["reason"])
            elif event_risk["reduce_position"]:
                recommendation = "reduce_position"
                confidence = 0.7
                risk_level = "medium_high"
                timing_factors.append(event_risk["reason"])
            
            # Check session alignment
            session_assessment = self._assess_session_timing(
                session_context,
                trade_opportunity,
            )
            
            if session_assessment["recommendation"] != "enter_now":
                if recommendation == "enter_now":
                    recommendation = session_assessment["recommendation"]
                    confidence = session_assessment["confidence"]
                timing_factors.extend(session_assessment["factors"])
            
            # Suggest optimal timing
            suggested_entry = self._suggest_entry_time(
                current_time,
                relevant_events,
                session_context,
            )
            
            suggested_exit = self._suggest_exit_time(
                current_time,
                relevant_events,
                trade_opportunity,
            )
            
            return {
                "recommendation": recommendation,
                "confidence": confidence,
                "factors": timing_factors,
                "suggested_entry_time": suggested_entry,
                "suggested_exit_time": suggested_exit,
                "risk_level": risk_level,
                "session_context": session_context,
                "relevant_events": relevant_events,
            }
        except Exception as e:
            logger.error(f"Error assessing timing: {e}")
            return {
                "recommendation": "enter_now",
                "confidence": 0.5,
                "factors": ["timing_assessment_error"],
                "risk_level": "medium",
            }
    
    def _filter_relevant_events(
        self,
        pair: str,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter events relevant to a currency pair"""
        if not pair:
            return []
        
        # Extract currency codes from pair
        if "/" in pair:
            base, quote = pair.split("/")
            relevant_currencies = [base, quote]
        else:
            relevant_currencies = [pair[:3]]
        
        # Filter events
        relevant_events = [
            event for event in events
            if event.get("currency") in relevant_currencies
            and event.get("impact") in ["HIGH", "MEDIUM"]
        ]
        
        return relevant_events
    
    def _assess_session_context(self, current_time: datetime) -> str:
        """Determine current trading session context"""
        current_utc_time = current_time.time()
        
        for session_name, session_config in TRADING_SESSIONS.items():
            session_start = session_config.start_time
            session_end = session_config.end_time
            
            if session_start <= current_utc_time < session_end:
                return f"in_{session_name}_session"
        
        # Check for overlaps
        if self._is_session_overlap(current_utc_time):
            return "session_overlap"
        
        return "outside_trading_hours"
    
    def _is_session_overlap(self, current_time: time) -> bool:
        """Check if current time is during session overlap"""
        # London/New York overlap (13:00-16:00 UTC)
        london_ny_overlap = time(13, 0) <= current_time < time(16, 0)
        
        # Asia/London overlap (07:00-08:00 UTC)
        asia_london_overlap = time(7, 0) <= current_time < time(8, 0)
        
        return london_ny_overlap or asia_london_overlap
    
    def _assess_event_risk(
        self,
        events: List[Dict[str, Any]],
        current_time: datetime,
    ) -> Dict[str, Any]:
        """Assess risk from upcoming events"""
        if not events:
            return {
                "avoid_entry": False,
                "reduce_position": False,
                "reason": "",
            }
        
        # Check for events within 30 minutes
        near_term_events = [
            event for event in events
            if abs((datetime.fromisoformat(event["date"]) - current_time).total_seconds()) < 1800
        ]
        
        if near_term_events:
            high_impact_near = [
                event for event in near_term_events
                if event.get("impact") == "HIGH"
            ]
            
            if high_impact_near:
                return {
                    "avoid_entry": True,
                    "reduce_position": False,
                    "reason": f"High-impact event within 30 minutes: {high_impact_near[0]['name']}",
                }
            else:
                return {
                    "avoid_entry": False,
                    "reduce_position": True,
                    "reason": f"Medium-impact event within 30 minutes: {near_term_events[0]['name']}",
                }
        
        # Check for events within 2 hours
        short_term_events = [
            event for event in events
            if abs((datetime.fromisoformat(event["date"]) - current_time).total_seconds()) < 7200
        ]
        
        if short_term_events:
            return {
                "avoid_entry": False,
                "reduce_position": True,
                "reason": f"Event within 2 hours: {short_term_events[0]['name']}",
            }
        
        return {
            "avoid_entry": False,
            "reduce_position": False,
            "reason": "",
        }
    
    def _assess_session_timing(
        self,
        session_context: str,
        trade_opportunity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess timing based on session context"""
        factors = []
        recommendation = "enter_now"
        confidence = 0.7
        
        if session_context == "session_overlap":
            recommendation = "enter_now"
            confidence = 0.9
            factors.append("High liquidity during session overlap")
        elif "session" in session_context:
            recommendation = "enter_now"
            confidence = 0.8
            factors.append("Normal trading session conditions")
        elif session_context == "outside_trading_hours":
            recommendation = "wait_for_better_entry"
            confidence = 0.6
            factors.append("Low liquidity outside trading hours")
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "factors": factors,
        }
    
    def _suggest_entry_time(
        self,
        current_time: datetime,
        events: List[Dict[str, Any]],
        session_context: str,
    ) -> Optional[str]:
        """Suggest optimal entry time"""
        if not events:
            return None
        
        # Find next window without high-impact events
        for event in events:
            if event.get("impact") == "HIGH":
                event_time = datetime.fromisoformat(event["date"])
                # Suggest entering 30 minutes after high-impact event
                suggested_time = event_time + timedelta(minutes=30)
                return suggested_time.isoformat()
        
        return None
    
    def _suggest_exit_time(
        self,
        current_time: datetime,
        events: List[Dict[str, Any]],
        trade_opportunity: Dict[str, Any],
    ) -> Optional[str]:
        """Suggest optimal exit time"""
        time_horizon = trade_opportunity.get("time_horizon", "swing")
        
        if time_horizon == "intraday":
            # Suggest end of current session
            return (current_time + timedelta(hours=4)).isoformat()
        elif time_horizon == "swing":
            # Suggest 2-3 days out
            return (current_time + timedelta(days=2)).isoformat()
        else:  # position
            # Suggest 1-2 weeks out
            return (current_time + timedelta(weeks=1)).isoformat()
    
    def assess_entry_conditions(
        self,
        pair: str,
        current_time: datetime,
    ) -> Dict[str, Any]:
        """
        Assess general entry conditions for a pair
        
        Args:
            pair: Currency pair
            current_time: Current timestamp
            
        Returns:
            Entry conditions assessment
        """
        try:
            # Get upcoming events
            upcoming_events = self.event_system.get_events(
                current_time,
                current_time + timedelta(hours=48),
            )
            
            relevant_events = self._filter_relevant_events(pair, upcoming_events)
            
            # Assess session
            session_context = self._assess_session_context(current_time)
            
            # Overall conditions
            conditions = {
                "session_context": session_context,
                "liquidity_assessment": self._assess_liquidity(session_context),
                "volatility_assessment": self._assess_volatility(relevant_events),
                "event_risk": self._assess_overall_event_risk(relevant_events),
                "recommendation": self._generate_entry_recommendation(
                    session_context,
                    relevant_events,
                ),
            }
            
            return conditions
        except Exception as e:
            logger.error(f"Error assessing entry conditions: {e}")
            return {
                "session_context": "unknown",
                "liquidity_assessment": "unknown",
                "volatility_assessment": "unknown",
                "event_risk": "unknown",
                "recommendation": "caution",
            }
    
    def _assess_liquidity(self, session_context: str) -> str:
        """Assess current liquidity conditions"""
        if session_context == "session_overlap":
            return "high"
        elif "session" in session_context:
            return "normal"
        else:
            return "low"
    
    def _assess_volatility(self, events: List[Dict[str, Any]]) -> str:
        """Assess expected volatility based on events"""
        high_impact_count = sum(
            1 for event in events
            if event.get("impact") == "HIGH"
        )
        
        if high_impact_count >= 2:
            return "high"
        elif high_impact_count >= 1:
            return "medium"
        else:
            return "normal"
    
    def _assess_overall_event_risk(self, events: List[Dict[str, Any]]) -> str:
        """Assess overall event risk"""
        if not events:
            return "low"
        
        # Check next 24 hours
        high_impact_24h = sum(
            1 for event in events
            if event.get("impact") == "HIGH"
        )
        
        if high_impact_24h >= 3:
            return "high"
        elif high_impact_24h >= 1:
            return "medium"
        else:
            return "low"
    
    def _generate_entry_recommendation(
        self,
        session_context: str,
        events: List[Dict[str, Any]],
    ) -> str:
        """Generate overall entry recommendation"""
        # Check for immediate event risks
        for event in events:
            if event.get("impact") == "HIGH":
                event_time = datetime.fromisoformat(event["date"])
                time_diff = abs((event_time - datetime.now()).total_seconds())
                
                if time_diff < 3600:  # Within 1 hour
                    return "avoid_entry"
        
        # Check session conditions
        if session_context == "outside_trading_hours":
            return "wait_for_session"
        
        return "favorable"