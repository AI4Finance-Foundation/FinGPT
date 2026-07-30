"""
Example usage of the Forex Intelligence Framework
Demonstrates basic functionality and usage patterns
"""

from datetime import datetime, timedelta
from fingpt.FinGPT_ForexIntelligence import ForexIntelligenceFramework

def main():
    """Main example function"""
    
    print("=" * 80)
    print("FinGPT Forex Intelligence Framework - Example Usage")
    print("=" * 80)
    print()
    
    # Initialize the framework with free data source
    print("Initializing Forex Intelligence Framework...")
    framework = ForexIntelligenceFramework(
        data_sources=["yfinance"],  # Use free Yahoo Finance
        currencies=["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD"],  # Major currencies
        enable_cache=True,
    )
    print("Framework initialized successfully!")
    print()
    
    # Run daily analysis
    print("Running daily FX analysis...")
    analysis_date = datetime.now()
    analysis_result = framework.run_daily_analysis(
        date=analysis_date,
        include_sessions=["asia", "london", "new_york"],
    )
    print("Analysis completed!")
    print()
    
    # Display summary
    print("ANALYSIS SUMMARY")
    print("-" * 40)
    print(f"Analysis Timestamp: {analysis_result.timestamp}")
    print(f"Global Sentiment: {analysis_result.global_sentiment.upper()}")
    print(f"Total Trade Opportunities: {len(analysis_result.trade_opportunities)}")
    print(f"Upcoming High-Impact Events: {len(analysis_result.upcoming_events)}")
    print()
    
    # Display currency rankings
    print("CURRENCY STRENGTH RANKINGS")
    print("-" * 40)
    sorted_currencies = sorted(
        analysis_result.currency_rankings.items(),
        key=lambda x: x[1].get("rank", 999),
    )
    
    for currency, data in sorted_currencies:
        rank = data.get("rank", "-")
        strength_label = data.get("strength_label", "unknown")
        strength_score = data.get("strength_score", 0)
        trend = data.get("trend", "neutral")
        
        print(f"{rank:2d}. {currency:3s} - {strength_label:15s} ({strength_score:+.3f}) - {trend}")
    print()
    
    # Display session analysis
    print("TRADING SESSION ANALYSIS")
    print("-" * 40)
    for session_name, session_data in analysis_result.session_biases.items():
        if "error" in session_data:
            continue
        
        bias = session_data.get("overall_bias", "neutral")
        confidence = session_data.get("confidence", 0)
        liquidity = session_data.get("liquidity_assessment", "unknown")
        
        print(f"{session_name.upper()}:")
        print(f"  Bias: {bias}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Liquidity: {liquidity}")
        print()
    
    # Display trade opportunities
    print("TRADE OPPORTUNITIES")
    print("-" * 40)
    if analysis_result.trade_opportunities:
        for i, opp in enumerate(analysis_result.trade_opportunities[:5], 1):  # Top 5
            print(f"{i}. {opp['pair']} - {opp['direction'].upper()}")
            print(f"   Confidence: {opp['confidence']}")
            print(f"   Risk/Reward: {opp['risk_reward_ratio']:.2f}")
            print(f"   Time Horizon: {opp['time_horizon']}")
            print(f"   Rationale: {', '.join(opp['rationale'][:2])}")
            print()
    else:
        print("No trade opportunities generated")
        print()
    
    # Display upcoming events
    print("UPCOMING HIGH-IMPACT EVENTS")
    print("-" * 40)
    if analysis_result.upcoming_events:
        for event in analysis_result.upcoming_events[:5]:  # Top 5
            print(f"{event['date']} - {event['currency']} {event['name']}")
            print(f"  Impact: {event['impact']}")
            if event['forecast']:
                print(f"  Forecast: {event['forecast']}, Previous: {event['previous']}")
            print()
    else:
        print("No upcoming high-impact events")
        print()
    
    # Generate and save reports
    print("Generating reports...")
    
    # Text report
    text_report = framework.generate_report(analysis_result, format="text")
    print("Text report generated (sample):")
    print(text_report[:500] + "...")
    print()
    
    # Save analysis to JSON
    output_file = "forex_analysis_example.json"
    framework.save_analysis(analysis_result, output_file)
    print(f"Analysis saved to {output_file}")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


def individual_components_example():
    """Example of using individual components"""
    
    print("\n" + "=" * 80)
    print("Individual Components Example")
    print("=" * 80)
    print()
    
    from fingpt.FinGPT_ForexIntelligence.data_sources import ForexDataSourceManager
    from fingpt.FinGPT_ForexIntelligence.analyzers import CurrencyStrengthAnalyzer
    from fingpt.FinGPT_ForexIntelligence.event_system import EventAnticipationSystem
    
    # Initialize data manager
    print("Initializing data manager...")
    data_manager = ForexDataSourceManager(sources=["yfinance"])
    print("Data manager initialized!")
    print()
    
    # Analyze currency strength
    print("Analyzing currency strength...")
    strength_analyzer = CurrencyStrengthAnalyzer(data_manager)
    strength_data = strength_analyzer.analyze_strength(datetime.now())
    
    print("Currency Strength Results:")
    for currency, data in sorted(strength_data.items(), 
                                key=lambda x: x[1].get("strength_score", 0), 
                                reverse=True):
        print(f"{currency}: {data.get('strength_score', 0):.3f} ({data.get('trend', 'neutral')})")
    print()
    
    # Get economic calendar
    print("Getting economic calendar...")
    event_system = EventAnticipationSystem(data_manager)
    events = event_system.get_events(
        datetime.now(),
        datetime.now() + timedelta(days=7),
    )
    
    print(f"Found {len(events)} events in the next 7 days")
    if events:
        print("Sample events:")
        for event in events[:3]:
            print(f"  {event['date']} - {event['currency']} {event['name']} ({event['impact']})")
    print()


def custom_analysis_example():
    """Example of custom analysis workflow"""
    
    print("\n" + "=" * 80)
    print("Custom Analysis Example")
    print("=" * 80)
    print()
    
    from fingpt.FinGPT_ForexIntelligence.data_sources import ForexDataSourceManager
    from fingpt.FinGPT_ForexIntelligence.analyzers import (
        GlobalMacroAnalyzer,
        SessionAnalyzer,
    )
    
    # Initialize components
    data_manager = ForexDataSourceManager(sources=["yfinance"])
    macro_analyzer = GlobalMacroAnalyzer(data_manager)
    session_analyzer = SessionAnalyzer(data_manager)
    
    # Custom analysis: Focus on risk sentiment and London session
    print("Analyzing global risk sentiment...")
    sentiment_data = macro_analyzer.analyze_risk_sentiment(datetime.now())
    print(f"Overall Sentiment: {sentiment_data.get('overall_sentiment')}")
    print(f"VIX Level: {sentiment_data.get('vix_level')}")
    print(f"Liquidity Conditions: {sentiment_data.get('liquidity_conditions')}")
    print()
    
    print("Analyzing London session...")
    london_data = session_analyzer.analyze_session("london", datetime.now())
    print(f"London Bias: {london_data.get('overall_bias')}")
    print(f"Confidence: {london_data.get('confidence', 0):.2f}")
    print()


if __name__ == "__main__":
    try:
        # Run main example
        main()
        
        # Run individual components example
        individual_components_example()
        
        # Run custom analysis example
        custom_analysis_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()