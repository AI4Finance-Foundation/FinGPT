"""
Trade Opportunity Generation Engine
Generates and ranks forex trade opportunities based on multi-layered analysis
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from fingpt.FinGPT_ForexIntelligence.config import MAJOR_PAIRS, STRENGTH_SCORES, EVENT_IMPACT
from fingpt.FinGPT_ForexIntelligence.analyzers import CurrencyStrengthAnalyzer, SessionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction options"""
    LONG = "long"
    SHORT = "short"


class ConfidenceLevel(Enum):
    """Confidence levels for trade opportunities"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TradeOpportunity:
    """Complete trade opportunity specification"""
    pair: str
    direction: TradeDirection
    confidence: ConfidenceLevel
    entry_zone: Dict[str, float]  # min/max entry prices
    target: float
    stop_loss: float
    risk_reward_ratio: float
    time_horizon: str  # intraday, swing, position
    rationale: List[str]
    supporting_factors: List[str]
    risk_factors: List[str]
    session_alignment: str
    strength_differential: float
    event_risk: str
    generated_at: datetime


class TradeOpportunityGenerator:
    """
    Generates trade opportunities by pairing strong vs weak currencies
    """
    
    def __init__(
        self,
        strength_analyzer: CurrencyStrengthAnalyzer,
        session_analyzer: SessionAnalyzer,
    ):
        self.strength_analyzer = strength_analyzer
        self.session_analyzer = session_analyzer
    
    def generate_opportunities(
        self,
        date: datetime,
        currency_rankings: Dict[str, Dict[str, Any]],
        session_biases: Dict[str, Dict[str, Any]],
        max_opportunities: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate trade opportunities based on analysis
        
        Args:
            date: Current date
            currency_rankings: Currency strength analysis
            session_biases: Session bias analysis
            max_opportunities: Maximum number of opportunities to generate
            
        Returns:
            List of trade opportunity dicts
        """
        try:
            opportunities = []
            
            # Identify strongest and weakest currencies
            strong_currencies = self._identify_strong_currencies(currency_rankings)
            weak_currencies = self._identify_weak_currencies(currency_rankings)
            
            # Generate pairings
            pairings = self._generate_pairings(strong_currencies, weak_currencies)
            
            # Analyze each pairing
            for base, quote in pairings:
                if len(opportunities) >= max_opportunities:
                    break
                
                opportunity = self._analyze_pairing(
                    base,
                    quote,
                    date,
                    currency_rankings,
                    session_biases,
                )
                
                if opportunity:
                    opportunities.append(opportunity)
            
            # Rank opportunities by confidence
            ranked_opportunities = self._rank_opportunities(opportunities)
            
            return ranked_opportunities
        except Exception as e:
            logger.error(f"Error generating trade opportunities: {e}")
            return []
    
    def _identify_strong_currencies(
        self,
        rankings: Dict[str, Dict[str, Any]],
    ) -> List[tuple]:
        """Identify strongest currencies"""
        strong_currencies = []
        
        for currency, data in rankings.items():
            strength_score = data.get("strength_score", 0)
            strength_label = data.get("strength_label", "")
            
            if strength_label in ["very_strong", "strong", "moderate_strong"]:
                strong_currencies.append((currency, strength_score))
        
        # Sort by strength score
        strong_currencies.sort(key=lambda x: x[1], reverse=True)
        
        return strong_currencies
    
    def _identify_weak_currencies(
        self,
        rankings: Dict[str, Dict[str, Any]],
    ) -> List[tuple]:
        """Identify weakest currencies"""
        weak_currencies = []
        
        for currency, data in rankings.items():
            strength_score = data.get("strength_score", 0)
            strength_label = data.get("strength_label", "")
            
            if strength_label in ["very_weak", "weak", "moderate_weak"]:
                weak_currencies.append((currency, strength_score))
        
        # Sort by weakness (most negative first)
        weak_currencies.sort(key=lambda x: x[1])
        
        return weak_currencies
    
    def _generate_pairings(
        self,
        strong_currencies: List[tuple],
        weak_currencies: List[tuple],
    ) -> List[tuple]:
        """Generate currency pairings for trading"""
        pairings = []
        
        # Pair strongest with weakest
        for strong_currency, strong_score in strong_currencies[:3]:
            for weak_currency, weak_score in weak_currencies[:3]:
                if strong_currency != weak_currency:
                    pairings.append((strong_currency, weak_currency))
        
        return pairings
    
    def _analyze_pairing(
        self,
        base: str,
        quote: str,
        date: datetime,
        currency_rankings: Dict[str, Dict[str, Any]],
        session_biases: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Analyze a specific currency pairing"""
        try:
            pair = f"{base}/{quote}"
            
            # Check if pair is valid
            if pair not in MAJOR_PAIRS:
                return None
            
            # Get strength data
            base_data = currency_rankings.get(base, {})
            quote_data = currency_rankings.get(quote, {})
            
            base_strength = base_data.get("strength_score", 0)
            quote_strength = quote_data.get("strength_score", 0)
            
            strength_differential = base_strength - quote_strength
            
            # Determine direction
            if strength_differential > 0:
                direction = "long"
                rationale = [
                    f"{base} showing strength ({base_data.get('strength_label', 'neutral')})",
                    f"{quote} showing weakness ({quote_data.get('strength_label', 'neutral')})",
                    f"Strength differential: {strength_differential:.3f}",
                ]
            else:
                direction = "short"
                rationale = [
                    f"{quote} showing strength ({quote_data.get('strength_label', 'neutral')})",
                    f"{base} showing weakness ({base_data.get('strength_label', 'neutral')})",
                    f"Strength differential: {abs(strength_differential):.3f}",
                ]
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                strength_differential,
                base_data,
                quote_data,
                session_biases,
            )
            
            # Get session alignment
            session_alignment = self._check_session_alignment(
                pair,
                direction,
                session_biases,
            )
            
            # Generate supporting factors
            supporting_factors = self._generate_supporting_factors(
                base,
                quote,
                base_data,
                quote_data,
            )
            
            # Generate risk factors
            risk_factors = self._generate_risk_factors(
                base,
                quote,
                base_data,
                quote_data,
            )
            
            # Calculate entry, target, stop loss
            # In production, this would use actual price data
            entry_zone = {"min": 1.0, "max": 1.02}  # Placeholder
            target = 1.05 if direction == "long" else 0.95
            stop_loss = 0.98 if direction == "long" else 1.02
            risk_reward_ratio = abs(target - 1.0) / abs(stop_loss - 1.0)
            
            return {
                "pair": pair,
                "direction": direction,
                "confidence": confidence,
                "entry_zone": entry_zone,
                "target": target,
                "stop_loss": stop_loss,
                "risk_reward_ratio": risk_reward_ratio,
                "time_horizon": "swing",
                "rationale": rationale,
                "supporting_factors": supporting_factors,
                "risk_factors": risk_factors,
                "session_alignment": session_alignment,
                "strength_differential": strength_differential,
                "event_risk": "medium",
                "generated_at": date.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing pairing {base}/{quote}: {e}")
            return None
    
    def _calculate_confidence(
        self,
        strength_differential: float,
        base_data: Dict[str, Any],
        quote_data: Dict[str, Any],
        session_biases: Dict[str, Dict[str, Any]],
    ) -> str:
        """Calculate confidence level for the trade"""
        confidence_score = 0.0
        
        # Strength differential contributes
        confidence_score += min(0.4, abs(strength_differential) * 10)
        
        # Trend alignment
        base_trend = base_data.get("trend", "neutral")
        quote_trend = quote_data.get("trend", "neutral")
        
        if (strength_differential > 0 and 
            base_trend == "strengthening" and 
            quote_trend == "weakening"):
            confidence_score += 0.3
        elif (strength_differential < 0 and 
              base_trend == "weakening" and 
              quote_trend == "strengthening"):
            confidence_score += 0.3
        
        # Session alignment
        session_bonus = self._calculate_session_bonus(session_biases)
        confidence_score += session_bonus
        
        # Determine confidence level
        if confidence_score >= 0.7:
            return "high"
        elif confidence_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _check_session_alignment(
        self,
        pair: str,
        direction: str,
        session_biases: Dict[str, Dict[str, Any]],
    ) -> str:
        """Check if trade aligns with session biases"""
        aligned_sessions = []
        
        for session_name, session_data in session_biases.items():
            if "error" in session_data:
                continue
            
            session_bias = session_data.get("overall_bias", "neutral")
            affected_pairs = session_data.get("affected_pairs", [])
            
            if pair in affected_pairs:
                if (direction == "long" and session_bias == "bullish") or \
                   (direction == "short" and session_bias == "bearish"):
                    aligned_sessions.append(session_name)
        
        if aligned_sessions:
            return f"aligned_with_{','.join(aligned_sessions)}"
        else:
            return "neutral"
    
    def _calculate_session_bonus(
        self,
        session_biases: Dict[str, Dict[str, Any]],
    ) -> float:
        """Calculate bonus score based on session alignment"""
        bonus = 0.0
        
        for session_data in session_biases.values():
            if "error" in session_data:
                continue
            
            confidence = session_data.get("confidence", 0)
            bonus += confidence * 0.1
        
        return min(0.3, bonus)
    
    def _generate_supporting_factors(
        self,
        base: str,
        quote: str,
        base_data: Dict[str, Any],
        quote_data: Dict[str, Any],
    ) -> List[str]:
        """Generate supporting factors for the trade"""
        factors = []
        
        # Economic data
        if base_data.get("economic_score", 0) > 0.3:
            factors.append(f"{base} positive economic data")
        if quote_data.get("economic_score", 0) < -0.3:
            factors.append(f"{quote} negative economic data")
        
        # Central bank bias
        base_cb = base_data.get("central_bank_bias", "neutral")
        quote_cb = quote_data.get("central_bank_bias", "neutral")
        
        if base_cb == "hawkish" and quote_cb == "dovish":
            factors.append(f"{base} hawkish central bank vs {quote} dovish")
        
        # Momentum
        if base_data.get("momentum", 0) > 0.01:
            factors.append(f"{base} positive momentum")
        if quote_data.get("momentum", 0) < -0.01:
            factors.append(f"{quote} negative momentum")
        
        return factors if factors else ["Strength differential"]
    
    def _generate_risk_factors(
        self,
        base: str,
        quote: str,
        base_data: Dict[str, Any],
        quote_data: Dict[str, Any],
    ) -> List[str]:
        """Generate risk factors for the trade"""
        risks = []
        
        # High volatility
        if base_data.get("volatility", 0) > 0.02:
            risks.append(f"{base} high volatility")
        if quote_data.get("volatility", 0) > 0.02:
            risks.append(f"{quote} high volatility")
        
        # Conflicting trends
        base_trend = base_data.get("trend", "neutral")
        quote_trend = quote_data.get("trend", "neutral")
        
        if base_trend == "neutral" or quote_trend == "neutral":
            risks.append("Unclear trend direction")
        
        # Economic events
        risks.append("Upcoming economic events risk")
        
        return risks if risks else ["Normal market risk"]
    
    def _rank_opportunities(
        self,
        opportunities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank opportunities by confidence score"""
        confidence_order = {"high": 3, "medium": 2, "low": 1}
        
        ranked = sorted(
            opportunities,
            key=lambda x: (
                confidence_order.get(x.get("confidence", "low"), 0),
                abs(x.get("strength_differential", 0)),
                x.get("risk_reward_ratio", 0),
            ),
            reverse=True,
        )
        
        return ranked