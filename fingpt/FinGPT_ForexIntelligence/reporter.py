"""
Report Generation System
Creates human-readable reports from forex analysis results
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class ForexAnalysisResult:
    """Container for complete forex analysis results"""
    timestamp: str
    global_sentiment: str
    currency_rankings: Dict[str, Dict[str, Any]]
    session_biases: Dict[str, Dict[str, Any]]
    trade_opportunities: List[Dict[str, Any]]
    upcoming_events: List[Dict[str, Any]]
    capital_flow_summary: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    commodity_impacts: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForexReportGenerator:
    """
    Generates comprehensive forex intelligence reports
    """
    
    def __init__(self):
        self.template_sections = [
            "executive_summary",
            "global_sentiment",
            "currency_rankings",
            "session_analysis",
            "trade_opportunities",
            "upcoming_events",
            "capital_flows",
            "risk_analysis",
            "commodity_impacts",
        ]
    
    def generate_report(
        self,
        analysis_result: ForexAnalysisResult,
        format: str = "json",
    ) -> str:
        """
        Generate report in specified format
        
        Args:
            analysis_result: Analysis result from framework
            format: Output format (json, text, html)
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report(analysis_result)
        elif format == "text":
            return self._generate_text_report(analysis_result)
        elif format == "html":
            return self._generate_html_report(analysis_result)
        else:
            logger.warning(f"Unknown format {format}, defaulting to json")
            return self._generate_json_report(analysis_result)
    
    def _generate_json_report(
        self,
        analysis_result: ForexAnalysisResult,
    ) -> str:
        """Generate JSON format report"""
        return analysis_result.to_json()
    
    def _generate_text_report(
        self,
        analysis_result: ForexAnalysisResult,
    ) -> str:
        """Generate human-readable text report"""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("FOREX GLOBAL INTELLIGENCE REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {analysis_result.timestamp}")
        lines.append("")
        
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Global Sentiment: {analysis_result.global_sentiment.upper()}")
        lines.append(f"Total Trade Opportunities: {len(analysis_result.trade_opportunities)}")
        lines.append(f"High-Impact Events (7 days): {len(analysis_result.upcoming_events)}")
        lines.append("")
        
        # Global Sentiment
        lines.append("GLOBAL MARKET SENTIMENT")
        lines.append("-" * 40)
        lines.append(f"Overall: {analysis_result.global_sentiment.upper()}")
        lines.append("")
        
        # Currency Rankings
        lines.append("CURRENCY STRENGTH RANKINGS")
        lines.append("-" * 40)
        sorted_currencies = sorted(
            analysis_result.currency_rankings.items(),
            key=lambda x: x[1].get("rank", 999),
        )
        
        for currency, data in sorted_currencies:
            rank = data.get("rank", "-")
            strength_label = data.get("strength_label", "unknown")
            strength_score = data.get("strength_score", 0)
            trend = data.get("trend", "neutral")
            
            lines.append(f"{rank:2d}. {currency:3s} - {strength_label:15s} ({strength_score:+.3f}) - {trend}")
        
        lines.append("")
        
        # Session Analysis
        lines.append("TRADING SESSION ANALYSIS")
        lines.append("-" * 40)
        for session_name, session_data in analysis_result.session_biases.items():
            if "error" in session_data:
                continue
            
            bias = session_data.get("overall_bias", "neutral")
            confidence = session_data.get("confidence", 0)
            liquidity = session_data.get("liquidity_assessment", "unknown")
            
            lines.append(f"{session_name.upper()}:")
            lines.append(f"  Bias: {bias}")
            lines.append(f"  Confidence: {confidence:.2f}")
            lines.append(f"  Liquidity: {liquidity}")
            lines.append("")
        
        # Trade Opportunities
        lines.append("TRADE OPPORTUNITIES")
        lines.append("-" * 40)
        if analysis_result.trade_opportunities:
            for i, opp in enumerate(analysis_result.trade_opportunities, 1):
                lines.append(f"{i}. {opp['pair']} - {opp['direction'].upper()}")
                lines.append(f"   Confidence: {opp['confidence']}")
                lines.append(f"   Risk/Reward: {opp['risk_reward_ratio']:.2f}")
                lines.append(f"   Entry Zone: {opp['entry_zone']}")
                lines.append(f"   Target: {opp['target']}")
                lines.append(f"   Stop Loss: {opp['stop_loss']}")
                lines.append(f"   Time Horizon: {opp['time_horizon']}")
                lines.append(f"   Rationale: {', '.join(opp['rationale'][:2])}")
                lines.append("")
        else:
            lines.append("No trade opportunities generated")
            lines.append("")
        
        # Upcoming Events
        lines.append("UPCOMING HIGH-IMPACT EVENTS")
        lines.append("-" * 40)
        if analysis_result.upcoming_events:
            for event in analysis_result.upcoming_events[:10]:  # Top 10
                lines.append(f"{event['date']} - {event['currency']} {event['name']}")
                lines.append(f"  Impact: {event['impact']}")
                lines.append(f"  Forecast: {event['forecast']}, Previous: {event['previous']}")
                lines.append("")
        else:
            lines.append("No upcoming high-impact events")
            lines.append("")
        
        # Capital Flows
        lines.append("INSTITUTIONAL CAPITAL FLOWS")
        lines.append("-" * 40)
        if analysis_result.capital_flow_summary:
            sentiment = analysis_result.capital_flow_summary.get("overall_sentiment", "unknown")
            lines.append(f"Overall Flow Sentiment: {sentiment}")
            lines.append("")
        
        # Risk Analysis
        lines.append("RISK ANALYSIS")
        lines.append("-" * 40)
        if analysis_result.risk_analysis:
            overall_risk = analysis_result.risk_analysis.get("overall_risk_level", "unknown")
            lines.append(f"Overall Risk Level: {overall_risk}")
            lines.append("")
        
        # Commodity Impacts
        lines.append("COMMODITY CURRENCY IMPACTS")
        lines.append("-" * 40)
        if analysis_result.commodity_impacts:
            currency_impacts = analysis_result.commodity_impacts.get("currency_impacts", {})
            for currency, impact in currency_impacts.items():
                significance = impact.get("significance", "low")
                lines.append(f"{currency}: {significance} impact")
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_html_report(
        self,
        analysis_result: ForexAnalysisResult,
    ) -> str:
        """Generate HTML format report"""
        html = []
        
        html.append("<html>")
        html.append("<head>")
        html.append("<title>Forex Intelligence Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #2c3e50; }")
        html.append("h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append(".bullish { color: green; }")
        html.append(".bearish { color: red; }")
        html.append(".neutral { color: gray; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append("<h1>FOREX GLOBAL INTELLIGENCE REPORT</h1>")
        html.append(f"<p>Generated: {analysis_result.timestamp}</p>")
        
        # Executive Summary
        html.append("<h2>Executive Summary</h2>")
        html.append("<ul>")
        html.append(f"<li>Global Sentiment: <strong>{analysis_result.global_sentiment.upper()}</strong></li>")
        html.append(f"<li>Total Trade Opportunities: {len(analysis_result.trade_opportunities)}</li>")
        html.append(f"<li>High-Impact Events (7 days): {len(analysis_result.upcoming_events)}</li>")
        html.append("</ul>")
        
        # Currency Rankings
        html.append("<h2>Currency Strength Rankings</h2>")
        html.append("<table>")
        html.append("<tr><th>Rank</th><th>Currency</th><th>Strength</th><th>Score</th><th>Trend</th></tr>")
        
        sorted_currencies = sorted(
            analysis_result.currency_rankings.items(),
            key=lambda x: x[1].get("rank", 999),
        )
        
        for currency, data in sorted_currencies:
            rank = data.get("rank", "-")
            strength_label = data.get("strength_label", "unknown")
            strength_score = data.get("strength_score", 0)
            trend = data.get("trend", "neutral")
            
            row_class = f"{strength_label}" if strength_label in ["bullish", "bearish", "neutral"] else ""
            
            html.append(f"<tr class='{row_class}'>")
            html.append(f"<td>{rank}</td>")
            html.append(f"<td>{currency}</td>")
            html.append(f"<td>{strength_label}</td>")
            html.append(f"<td>{strength_score:+.3f}</td>")
            html.append(f"<td>{trend}</td>")
            html.append("</tr>")
        
        html.append("</table>")
        
        # Trade Opportunities
        html.append("<h2>Trade Opportunities</h2>")
        if analysis_result.trade_opportunities:
            html.append("<table>")
            html.append("<tr><th>Pair</th><th>Direction</th><th>Confidence</th><th>R/R Ratio</th><th>Time Horizon</th></tr>")
            
            for opp in analysis_result.trade_opportunities:
                direction_class = opp['direction']
                html.append(f"<tr class='{direction_class}'>")
                html.append(f"<td>{opp['pair']}</td>")
                html.append(f"<td>{opp['direction'].upper()}</td>")
                html.append(f"<td>{opp['confidence']}</td>")
                html.append(f"<td>{opp['risk_reward_ratio']:.2f}</td>")
                html.append(f"<td>{opp['time_horizon']}</td>")
                html.append("</tr>")
            
            html.append("</table>")
        else:
            html.append("<p>No trade opportunities generated</p>")
        
        # Upcoming Events
        html.append("<h2>Upcoming High-Impact Events</h2>")
        if analysis_result.upcoming_events:
            html.append("<table>")
            html.append("<tr><th>Date</th><th>Currency</th><th>Event</th><th>Impact</th><th>Forecast</th><th>Previous</th></tr>")
            
            for event in analysis_result.upcoming_events[:10]:
                html.append(f"<tr>")
                html.append(f"<td>{event['date']}</td>")
                html.append(f"<td>{event['currency']}</td>")
                html.append(f"<td>{event['name']}</td>")
                html.append(f"<td>{event['impact']}</td>")
                html.append(f"<td>{event['forecast']}</td>")
                html.append(f"<td>{event['previous']}</td>")
                html.append("</tr>")
            
            html.append("</table>")
        else:
            html.append("<p>No upcoming high-impact events</p>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def generate_summary_report(
        self,
        analysis_result: ForexAnalysisResult,
    ) -> str:
        """Generate a brief summary report"""
        lines = []
        
        lines.append("FOREX INTELLIGENCE SUMMARY")
        lines.append(f"Time: {analysis_result.timestamp}")
        lines.append(f"Sentiment: {analysis_result.global_sentiment.upper()}")
        
        # Top 3 strongest currencies
        sorted_currencies = sorted(
            analysis_result.currency_rankings.items(),
            key=lambda x: x[1].get("rank", 999),
        )
        
        strong_currencies = [c[0] for c in sorted_currencies[:3]]
        weak_currencies = [c[0] for c in sorted_currencies[-3:]]
        
        lines.append(f"Strongest: {', '.join(strong_currencies)}")
        lines.append(f"Weakest: {', '.join(weak_currencies)}")
        lines.append(f"Trade Opportunities: {len(analysis_result.trade_opportunities)}")
        
        if analysis_result.trade_opportunities:
            top_opportunity = analysis_result.trade_opportunities[0]
            lines.append(f"Top Trade: {top_opportunity['pair']} {top_opportunity['direction']}")
        
        return "\n".join(lines)