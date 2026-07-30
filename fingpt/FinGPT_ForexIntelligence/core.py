"""
Core Forex Intelligence Framework
Main orchestrator for the multi-layered FX analysis system
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd

from fingpt.FinGPT_ForexIntelligence.config import (
    MAJOR_CURRENCIES,
    MAJOR_PAIRS,
    TRADING_SESSIONS,
    ANALYSIS_WEIGHTS,
    DATA_SOURCES,
)
from fingpt.FinGPT_ForexIntelligence.data_sources import ForexDataSourceManager
from fingpt.FinGPT_ForexIntelligence.analyzers import (
    GlobalMacroAnalyzer,
    CapitalFlowAnalyzer,
    EconomicDataAnalyzer,
    CentralBankAnalyzer,
    GeopoliticalAnalyzer,
    CommodityAnalyzer,
    CurrencyStrengthAnalyzer,
    SessionAnalyzer,
)
from fingpt.FinGPT_ForexIntelligence.trade_generator import TradeOpportunityGenerator
from fingpt.FinGPT_ForexIntelligence.event_system import EventAnticipationSystem
from fingpt.FinGPT_ForexIntelligence.timing_model import TradeTimingModel
from fingpt.FinGPT_ForexIntelligence.reporter import ForexAnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForexIntelligenceFramework:
    """
    Main framework orchestrator for institutional-style FX analysis
    """
    
    def __init__(
        self,
        data_sources: Optional[List[str]] = None,
        currencies: Optional[List[str]] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Forex Intelligence Framework
        
        Args:
            data_sources: List of data source names to use (default: all available)
            currencies: List of currencies to analyze (default: major currencies)
            enable_cache: Whether to enable data caching
            cache_dir: Directory for cache storage
        """
        self.data_sources = data_sources or list(DATA_SOURCES.keys())
        self.currencies = currencies or MAJOR_CURRENCIES
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "cache"
        )
        
        # Create cache directory if needed
        if self.enable_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize components
        self.data_manager = ForexDataSourceManager(
            sources=self.data_sources,
            cache_dir=self.cache_dir if enable_cache else None,
        )
        
        # Initialize analyzers
        self.macro_analyzer = GlobalMacroAnalyzer(self.data_manager)
        self.capital_flow_analyzer = CapitalFlowAnalyzer(self.data_manager)
        self.economic_analyzer = EconomicDataAnalyzer(self.data_manager)
        self.central_bank_analyzer = CentralBankAnalyzer(self.data_manager)
        self.geopolitical_analyzer = GeopoliticalAnalyzer(self.data_manager)
        self.commodity_analyzer = CommodityAnalyzer(self.data_manager)
        self.currency_strength_analyzer = CurrencyStrengthAnalyzer(self.data_manager)
        self.session_analyzer = SessionAnalyzer(self.data_manager)
        
        # Initialize systems
        self.trade_generator = TradeOpportunityGenerator(
            self.currency_strength_analyzer,
            self.session_analyzer,
        )
        self.event_system = EventAnticipationSystem(self.data_manager)
        self.timing_model = TradeTimingModel(self.event_system)
        
        logger.info(f"Forex Intelligence Framework initialized with {len(self.currencies)} currencies")
    
    def run_daily_analysis(
        self,
        date: Optional[datetime] = None,
        include_sessions: Optional[List[str]] = None,
    ) -> ForexAnalysisResult:
        """
        Run complete daily FX analysis
        
        Args:
            date: Analysis date (default: current date)
            include_sessions: Sessions to include (default: all)
            
        Returns:
            ForexAnalysisResult with complete analysis
        """
        analysis_date = date or datetime.now()
        sessions = include_sessions or list(TRADING_SESSIONS.keys())
        
        logger.info(f"Starting daily FX analysis for {analysis_date.date()}")
        
        # Run analysis layers
        global_sentiment = self._analyze_global_sentiment(analysis_date)
        currency_rankings = self._analyze_currency_strength(analysis_date)
        session_biases = self._analyze_sessions(analysis_date, sessions)
        trade_opportunities = self._generate_trade_opportunities(
            analysis_date,
            currency_rankings,
            session_biases,
        )
        upcoming_events = self._get_upcoming_events(analysis_date)
        capital_flow_summary = self._analyze_capital_flows(analysis_date)
        risk_analysis = self._analyze_risk_factors(analysis_date)
        commodity_impacts = self._analyze_commodity_impacts(analysis_date)
        
        # Compile results
        result = ForexAnalysisResult(
            timestamp=analysis_date.isoformat(),
            global_sentiment=global_sentiment,
            currency_rankings=currency_rankings,
            session_biases=session_biases,
            trade_opportunities=trade_opportunities,
            upcoming_events=upcoming_events,
            capital_flow_summary=capital_flow_summary,
            risk_analysis=risk_analysis,
            commodity_impacts=commodity_impacts,
        )
        
        logger.info("Daily FX analysis completed")
        return result
    
    def _analyze_global_sentiment(self, date: datetime) -> str:
        """Analyze global risk sentiment"""
        try:
            sentiment_data = self.macro_analyzer.analyze_risk_sentiment(date)
            return sentiment_data.get("overall_sentiment", "neutral")
        except Exception as e:
            logger.error(f"Error analyzing global sentiment: {e}")
            return "neutral"
    
    def _analyze_currency_strength(
        self,
        date: datetime,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze and rank currency strength"""
        try:
            strength_data = self.currency_strength_analyzer.analyze_strength(date)
            
            # Add additional context
            for currency in self.currencies:
                if currency in strength_data:
                    strength_data[currency]["economic_score"] = \
                        self.economic_analyzer.get_currency_score(currency, date)
                    strength_data[currency]["central_bank_bias"] = \
                        self.central_bank_analyzer.get_policy_bias(currency, date)
            
            return strength_data
        except Exception as e:
            logger.error(f"Error analyzing currency strength: {e}")
            return {}
    
    def _analyze_sessions(
        self,
        date: datetime,
        sessions: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze trading session biases"""
        session_results = {}
        
        for session_name in sessions:
            try:
                session_data = self.session_analyzer.analyze_session(
                    session_name,
                    date,
                )
                session_results[session_name] = session_data
            except Exception as e:
                logger.error(f"Error analyzing session {session_name}: {e}")
                session_results[session_name] = {"error": str(e)}
        
        return session_results
    
    def _generate_trade_opportunities(
        self,
        date: datetime,
        currency_rankings: Dict[str, Dict[str, Any]],
        session_biases: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate trade opportunities based on analysis"""
        try:
            opportunities = self.trade_generator.generate_opportunities(
                date,
                currency_rankings,
                session_biases,
            )
            
            # Apply timing model to filter opportunities
            filtered_opportunities = []
            for opp in opportunities:
                timing_assessment = self.timing_model.assess_timing(
                    opp,
                    date,
                )
                opp["timing_assessment"] = timing_assessment
                
                # Filter out trades with poor timing
                if timing_assessment.get("recommend") != "avoid":
                    filtered_opportunities.append(opp)
            
            return filtered_opportunities
        except Exception as e:
            logger.error(f"Error generating trade opportunities: {e}")
            return []
    
    def _get_upcoming_events(
        self,
        date: datetime,
        days_ahead: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get upcoming high-impact events"""
        try:
            events = self.event_system.get_events(
                date,
                date + timedelta(days=days_ahead),
            )
            
            # Filter for high and medium impact events
            significant_events = [
                event for event in events
                if event.get("impact") in ["HIGH", "MEDIUM"]
            ]
            
            return significant_events
        except Exception as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []
    
    def _analyze_capital_flows(
        self,
        date: datetime,
    ) -> Dict[str, Any]:
        """Analyze institutional capital flows"""
        try:
            flow_data = self.capital_flow_analyzer.analyze_flows(date)
            return flow_data
        except Exception as e:
            logger.error(f"Error analyzing capital flows: {e}")
            return {}
    
    def _analyze_risk_factors(
        self,
        date: datetime,
    ) -> Dict[str, Any]:
        """Analyze geopolitical and market risk factors"""
        try:
            geo_risks = self.geopolitical_analyzer.analyze_risks(date)
            macro_risks = self.macro_analyzer.analyze_macro_risks(date)
            
            return {
                "geopolitical": geo_risks,
                "macro": macro_risks,
                "overall_risk_level": self._calculate_overall_risk(
                    geo_risks,
                    macro_risks,
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {e}")
            return {}
    
    def _analyze_commodity_impacts(
        self,
        date: datetime,
    ) -> Dict[str, Any]:
        """Analyze commodity price impacts on currencies"""
        try:
            commodity_data = self.commodity_analyzer.analyze_impacts(date)
            return commodity_data
        except Exception as e:
            logger.error(f"Error analyzing commodity impacts: {e}")
            return {}
    
    def _calculate_overall_risk(
        self,
        geo_risks: Dict[str, Any],
        macro_risks: Dict[str, Any],
    ) -> str:
        """Calculate overall risk level"""
        geo_score = geo_risks.get("risk_score", 0)
        macro_score = macro_risks.get("risk_score", 0)
        
        combined_score = (geo_score + macro_score) / 2
        
        if combined_score >= 0.7:
            return "high"
        elif combined_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def generate_report(
        self,
        analysis_result: ForexAnalysisResult,
        format: str = "json",
    ) -> str:
        """
        Generate human-readable report from analysis results
        
        Args:
            analysis_result: Result from run_daily_analysis
            format: Output format (json, text, html)
            
        Returns:
            Formatted report string
        """
        # Import here to avoid circular dependency
        from fingpt.FinGPT_ForexIntelligence.reporter import ForexReportGenerator
        reporter = ForexReportGenerator()
        return reporter.generate_report(
            analysis_result,
            format,
        )
    
    def save_analysis(
        self,
        analysis_result: ForexAnalysisResult,
        filepath: str,
    ) -> None:
        """
        Save analysis results to file
        
        Args:
            analysis_result: Result from run_daily_analysis
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            f.write(analysis_result.to_json())
        
        logger.info(f"Analysis saved to {filepath}")