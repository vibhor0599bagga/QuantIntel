# quantintel/agents/states.py
from typing import Annotated
from langgraph.graph.message import MessagesState


class QuantIntelState(MessagesState):
    ticker:               Annotated[str,  "Stock ticker"]
    trade_date:           Annotated[str,  "Analysis date yyyy-mm-dd"]
    portfolio_context:    Annotated[dict, "User portfolio context (optional)"]
    fundamentals_report:  Annotated[str,  "Fundamentals agent output"]
    sentiment_report:     Annotated[str,  "Sentiment agent output"]
    technical_report:     Annotated[str,  "Technical agent output"]
    risk_report:          Annotated[str,  "Risk agent output"]
    macro_report:         Annotated[str,  "Macro/Regime agent output"]
    final_recommendation: Annotated[str,  "Supervisor final BUY/HOLD/SELL + rationale"]
    sender:               Annotated[str,  "Last agent that wrote to state"]
