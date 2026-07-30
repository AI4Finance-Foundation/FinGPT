"""
FinGPT Forex Intelligence Framework
An institutional-style multi-currency monitoring and trade opportunity detection system
"""

__version__ = "0.1.0"
__author__ = "FinGPT Team"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "ForexIntelligenceFramework":
        from .core import ForexIntelligenceFramework
        return ForexIntelligenceFramework
    elif name == "ForexDataSourceManager":
        from .data_sources import ForexDataSourceManager
        return ForexDataSourceManager
    elif name == "GlobalMacroAnalyzer":
        from .analyzers import GlobalMacroAnalyzer
        return GlobalMacroAnalyzer
    elif name == "CapitalFlowAnalyzer":
        from .analyzers import CapitalFlowAnalyzer
        return CapitalFlowAnalyzer
    elif name == "EconomicDataAnalyzer":
        from .analyzers import EconomicDataAnalyzer
        return EconomicDataAnalyzer
    elif name == "CentralBankAnalyzer":
        from .analyzers import CentralBankAnalyzer
        return CentralBankAnalyzer
    elif name == "GeopoliticalAnalyzer":
        from .analyzers import GeopoliticalAnalyzer
        return GeopoliticalAnalyzer
    elif name == "CommodityAnalyzer":
        from .analyzers import CommodityAnalyzer
        return CommodityAnalyzer
    elif name == "CurrencyStrengthAnalyzer":
        from .analyzers import CurrencyStrengthAnalyzer
        return CurrencyStrengthAnalyzer
    elif name == "SessionAnalyzer":
        from .analyzers import SessionAnalyzer
        return SessionAnalyzer
    elif name == "TradeOpportunityGenerator":
        from .trade_generator import TradeOpportunityGenerator
        return TradeOpportunityGenerator
    elif name == "EventAnticipationSystem":
        from .event_system import EventAnticipationSystem
        return EventAnticipationSystem
    elif name == "TradeTimingModel":
        from .timing_model import TradeTimingModel
        return TradeTimingModel
    elif name == "ForexReportGenerator":
        from .reporter import ForexReportGenerator
        return ForexReportGenerator
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "ForexIntelligenceFramework",
    "ForexDataSourceManager",
    "GlobalMacroAnalyzer",
    "CapitalFlowAnalyzer",
    "EconomicDataAnalyzer",
    "CentralBankAnalyzer",
    "GeopoliticalAnalyzer",
    "CommodityAnalyzer",
    "CurrencyStrengthAnalyzer",
    "SessionAnalyzer",
    "TradeOpportunityGenerator",
    "EventAnticipationSystem",
    "TradeTimingModel",
    "ForexReportGenerator",
]