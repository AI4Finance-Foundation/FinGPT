# FinGPT Forex Intelligence Framework

An institutional-style multi-currency monitoring and trade opportunity detection system for the global forex market.

## Overview

The FOREX GLOBAL INTELLIGENCE FRAMEWORK provides daily FX analysis aligned with institutional macro/hedge fund practices. It implements a comprehensive multi-layered analytical structure for monitoring 9 global major currencies (USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, CNY) and generating actionable trade opportunities.

## Features

### Multi-Layered Analysis

1. **Global Macro Analysis**
   - Risk sentiment assessment
   - Market indices monitoring
   - Liquidity conditions
   - Global economic factors

2. **Institutional Capital Flow Analysis**
   - Treasury yield tracking
   - Equity/bond/ETF flow monitoring
   - Liquidity demand measures
   - Currency impact assessment

3. **Economic Data Analysis**
   - Tiered indicator system (HIGH/MEDIUM/LOW)
   - Interest rate, inflation, GDP, employment tracking
   - PMI, trade balance, and other key indicators
   - Deviation from forecast analysis

4. **Central Bank Policy Monitoring**
   - Rate change tracking
   - Policy stance assessment
   - Liquidity program monitoring
   - Meeting calendar analysis

5. **Geopolitical Risk Analysis**
   - Conflict monitoring
   - Sanctions tracking
   - Trade/energy tension assessment
   - Safe-haven demand analysis

6. **Commodity-Currency Linkage Analysis**
   - Oil, metals, dairy price impacts
   - Correlation strength assessment
   - Commodity flow analysis

7. **Currency Strength Determination**
   - Multi-factor strength scoring
   - Trend and momentum analysis
   - Volatility assessment
   - Support/resistance level identification

8. **Session Analysis**
   - Asia/London/US session-specific analysis
   - Session bias determination
   - Liquidity and volatility assessment
   - Key pair identification per session

9. **Trade Opportunity Generation**
   - Strong vs weak currency pairing
   - Event-aware opportunity filtering
   - Risk/reward ratio calculation
   - Confidence scoring

10. **Forward Event Anticipation**
    - Economic calendar integration
    - Impact assessment (HIGH/MEDIUM/LOW)
    - Event cluster analysis
    - Risk level determination

11. **Trade Timing Model**
    - Entry/exit rule generation
    - Event risk consideration
    - Session flow analysis
    - Optimal timing suggestions

### Outputs

- Daily market sentiment assessment
- Ranked currencies (strong/weak)
- Session-specific biases
- Best trade pairs with entry/exit levels
- Upcoming event lists with impact assessment
- Capital flow summary
- Risk analysis
- Commodity impact assessment

## Installation

### Requirements

```bash
pip install pandas numpy requests yfinance
```

### Setup

1. Ensure you have the required API keys for data sources:
   - Set environment variables for paid services (optional):
     - `BLOOMBERG_API_KEY`
     - `REUTERS_API_KEY`
     - `TRADING_ECONOMICS_API_KEY`
     - `FXSTREET_API_KEY`

2. The framework will use Yahoo Finance (free) by default if no API keys are provided.

## Usage

### Basic Usage

```python
from datetime import datetime
from fingpt.FinGPT_ForexIntelligence import ForexIntelligenceFramework

# Initialize the framework
framework = ForexIntelligenceFramework(
    data_sources=["yfinance"],  # Use free data source
    currencies=["USD", "EUR", "GBP", "JPY"],  # Analyze specific currencies
)

# Run daily analysis
analysis_result = framework.run_daily_analysis(
    date=datetime.now(),
)

# Generate report
report = framework.generate_report(analysis_result, format="text")
print(report)

# Save analysis
framework.save_analysis(analysis_result, "forex_analysis.json")
```

### Advanced Usage

```python
from fingpt.FinGPT_ForexIntelligence import ForexIntelligenceFramework

# Initialize with multiple data sources
framework = ForexIntelligenceFramework(
    data_sources=["yfinance", "tradingeconomics"],
    currencies=None,  # Use all major currencies
    enable_cache=True,
    cache_dir="./cache",
)

# Run analysis for specific sessions
analysis_result = framework.run_daily_analysis(
    date=datetime.now(),
    include_sessions=["london", "new_york"],
)

# Generate different report formats
json_report = framework.generate_report(analysis_result, format="json")
text_report = framework.generate_report(analysis_result, format="text")
html_report = framework.generate_report(analysis_result, format="html")

# Access specific components
currency_rankings = analysis_result.currency_rankings
trade_opportunities = analysis_result.trade_opportunities
upcoming_events = analysis_result.upcoming_events
```

### Individual Component Usage

```python
from fingpt.FinGPT_ForexIntelligence.data_sources import ForexDataSourceManager
from fingpt.FinGPT_ForexIntelligence.analyzers import CurrencyStrengthAnalyzer
from datetime import datetime, timedelta

# Initialize data manager
data_manager = ForexDataSourceManager(sources=["yfinance"])

# Analyze currency strength
strength_analyzer = CurrencyStrengthAnalyzer(data_manager)
strength_data = strength_analyzer.analyze_strength(datetime.now())

# Get economic calendar
from fingpt.FinGPT_ForexIntelligence.event_system import EventAnticipationSystem
event_system = EventAnticipationSystem(data_manager)
events = event_system.get_events(
    datetime.now(),
    datetime.now() + timedelta(days=7),
)
```

## Architecture

### Core Components

1. **ForexIntelligenceFramework**: Main orchestrator
2. **ForexDataSourceManager**: Data ingestion with fallback
3. **Analyzers**: Multi-layered analysis modules
4. **TradeOpportunityGenerator**: Trade pairing logic
5. **EventAnticipationSystem**: Event monitoring
6. **TradeTimingModel**: Entry/exit timing
7. **ForexReportGenerator**: Report generation

### Data Flow

```
Data Sources → Data Manager → Analyzers → Trade Generator → Report Generator
                    ↓              ↓
               Event System → Timing Model
```

### Currency Pairs

The framework analyzes all major crosses from the 9 major currencies:
- USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, CNY

### Trading Sessions

- **Asia Session**: 00:00-08:00 UTC (JPY, AUD, NZD, CNY focus)
- **London Session**: 07:00-16:00 UTC (EUR, GBP, CHF focus)
- **New York Session**: 13:00-22:00 UTC (USD, CAD focus)

## Configuration

### Data Sources

The framework supports multiple data sources with automatic fallback:

1. **Yahoo Finance** (free, default)
2. **Trading Economics** (API key required)
3. **Bloomberg** (API key required)
4. **Reuters** (API key required)
5. **FXStreet** (API key required)

### Economic Indicators

Indicators are tiered by importance:

- **HIGH Tier**: Interest rates, CPI, GDP, Employment, PMI
- **MEDIUM Tier**: Retail sales, Trade balance, Industrial production
- **LOW Tier**: Consumer confidence, Building permits

### Analysis Weights

Default weights for different analysis layers:
- Global Macro: 20%
- Capital Flows: 15%
- Economic Data: 20%
- Central Bank Policy: 15%
- Geopolitics: 10%
- Commodity Links: 8%
- Currency Strength: 7%
- Session Analysis: 5%

## Output Formats

### JSON Format

Structured data format suitable for programmatic access:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "global_sentiment": "risk_on",
  "currency_rankings": {...},
  "trade_opportunities": [...],
  "upcoming_events": [...]
}
```

### Text Format

Human-readable report with sections:
- Executive Summary
- Global Market Sentiment
- Currency Strength Rankings
- Trading Session Analysis
- Trade Opportunities
- Upcoming High-Impact Events
- Institutional Capital Flows
- Risk Analysis
- Commodity Currency Impacts

### HTML Format

Interactive web report with:
- Color-coded indicators
- Sortable tables
- Responsive design

## Use Cases

### Robo-Advisory Services

Integrate the framework into automated trading systems:
```python
# Get daily signals
analysis = framework.run_daily_analysis()
best_trades = analysis.trade_opportunities[:3]

# Execute trades programmatically
for trade in best_trades:
    if trade['confidence'] == 'high':
        execute_trade(trade)
```

### Educational Tools

Use for teaching institutional FX analysis:
```python
# Generate educational reports
report = framework.generate_report(analysis, format="html")
save_report(report, "daily_fx_analysis.html")
```

### Research and Analysis

Conduct market research:
```python
# Historical analysis
historical_data = []
for date in date_range:
    analysis = framework.run_daily_analysis(date)
    historical_data.append(analysis)
```

## Limitations

1. **Data Availability**: Free data sources may have limitations in accuracy and completeness
2. **Real-time Constraints**: Not designed for high-frequency trading
3. **API Rate Limits**: Subject to rate limits of data providers
4. **Market Conditions**: Extreme market conditions may affect accuracy

## Future Enhancements

1. Machine learning integration for pattern recognition
2. Additional data source integrations
3. Real-time streaming data support
4. Backtesting capabilities
5. Performance analytics
6. Custom alert system
7. Multi-language support

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests are included for new features
- Documentation is updated
- API keys are not committed

## License

This framework is part of FinGPT and follows the same license terms.

## Disclaimer

This software is for educational and research purposes only. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.

## Support

For issues and questions:
- GitHub Issues: [FinGPT Repository](https://github.com/AI4Finance-Foundation/FinGPT)
- Documentation: [FinGPT Docs](https://ai4finance.org/)