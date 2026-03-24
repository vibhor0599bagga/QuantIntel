"""
quantintel/agents/agent_6_supervisor.py
SUPERVISOR AGENT — "Given everything, what should THIS investor do?"
Reads all 5 agent reports + optional portfolio context. Makes the final call.
"""

from quantintel.memory import FinancialSituationMemory


def create_supervisor_agent(llm, memory: FinancialSituationMemory = None):
    """
    No tools — pure LLM reasoning over all 5 agent reports.
    Uses deep_think_llm for best synthesis quality.
    """

    def node(state):
        ticker     = state["ticker"]
        trade_date = state["trade_date"]

        fundamentals_report = state.get("fundamentals_report", "Not available.")
        sentiment_report    = state.get("sentiment_report",    "Not available.")
        technical_report    = state.get("technical_report",    "Not available.")
        risk_report         = state.get("risk_report",         "Not available.")
        macro_report        = state.get("macro_report",        "Not available.")

        # ── portfolio context ──────────────────────────────────────────────────
        portfolio_context = state.get("portfolio_context") or {}
        if portfolio_context:
            portfolio_section = f"""
USER PORTFOLIO CONTEXT (personalise for this investor):
  Sector exposure   : {portfolio_context.get('sector_exposure',  'not provided')}
  Investment horizon: {portfolio_context.get('horizon',          'not provided')}
  Risk tolerance    : {portfolio_context.get('risk_tolerance',   'not provided')}
  Key holdings      : {portfolio_context.get('holdings',         'not provided')}

Assess concentration risk, horizon alignment, risk suitability, holding overlap.
"""
        else:
            portfolio_section = "No portfolio context — give a general recommendation."

        # ── memory ─────────────────────────────────────────────────────────────
        past_memory_str = "No past decisions on file."
        if memory:
            summary = f"{fundamentals_report[:500]}\n{sentiment_report[:300]}\n{technical_report[:300]}"
            memories = memory.get_memories(summary, n_matches=2)
            if memories:
                past_memory_str = "\n\n".join(r["recommendation"] for r in memories)

        # ── build prompt ───────────────────────────────────────────────────────
        prompt = f"""
You are the Supervisor Agent in QuantIntel — the final decision-maker.
Synthesise all 5 agent signals, resolve conflicts, and deliver one clear recommendation.

============================================================
TICKER: {ticker}   |   DATE: {trade_date}
============================================================

━━━ AGENT 1 — FUNDAMENTALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{fundamentals_report}

━━━ AGENT 2 — SENTIMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sentiment_report}

━━━ AGENT 3 — TECHNICAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{technical_report}

━━━ AGENT 4 — RISK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{risk_report}

━━━ AGENT 5 — MACRO / REGIME ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{macro_report}

━━━ PORTFOLIO CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{portfolio_section}

━━━ PAST DECISION MEMORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{past_memory_str}

━━━ REASONING INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Read each agent's structured signal block at the end of their report.

STEP 2 — Score signals (only use explicit labels from reports; if absent, infer cautiously):
  Fundamentals: undervalued +2, fair +1, overvalued -2
  Sentiment:    bullish +1, neutral 0, bearish -1
  Technical:    up +1, sideways/flat 0, down -1
  Risk:         low +1, moderate 0, high -1, extreme -2
  Macro:        risk_on +1, recession or risk_off -1, crisis -2, otherwise 0
  Macro trend:  bullish +0.5, bearish -0.5 (small adjustment, optional)
  Total score = sum of the above.

STEP 3 — Apply overrides and decide:
  • If Risk is EXTREME or Macro is CRISIS → SELL (capital preservation override).
  • If Fundamentals UNDERVALUED AND Technical UP → BUY unless Risk is extreme or Macro is crisis.
  • Otherwise use the score:
      Score ≥ 1  → BUY (provided no extreme risk/crisis)
      Score ≤ -2 → SELL
      Else       → HOLD
  Tie-break: if score is between -1 and 1 but ≥3 signals align bullish → BUY; if ≥3 align bearish → SELL.
  Do NOT default to SELL because of missing/uncertain data — pick HOLD when uncertain.

STEP 4 — Apply portfolio context:
  • If risk_tolerance is conservative: only downgrade a BUY when Risk is high/extreme. Moderate risk with strong signals (score ≥1 or fundamentals undervalued + technical up) can stay BUY.
  • If risk_tolerance is aggressive and score is near zero but ≥3 bullish signals → allow BUY.

STEP 5 — Output your recommendation in this EXACT format:

FINAL RECOMMENDATION
====================
TICKER: {ticker}
DATE: {trade_date}
RECOMMENDATION: BUY | HOLD | SELL
CONFIDENCE: <float 0.0-1.0>

SIGNAL SUMMARY:
  Fundamentals : <valuation signal + key reason>
  Sentiment    : <polarity + key catalyst>
  Technical    : <trend + key level>
  Risk         : <risk level + primary risk>
  Macro        : <regime type + key macro driver>

CONFLICT RESOLUTION:
  <Explain any signal disagreements and how you resolved them.
   If all agreed, state that explicitly.>

PORTFOLIO SUITABILITY:
  <How this fits the investor's profile. If no context: "General recommendation.">

INVESTMENT THESIS:
  <2-4 sentences grounded in specific evidence from agent reports.>
"""

        result = llm.invoke(prompt)

        # store in memory for future learning
        if memory:
            summary = f"{fundamentals_report[:400]}\n{sentiment_report[:200]}\n{technical_report[:200]}"
            memory.add_situations([(summary, result.content)])

        return {
            "messages":            [result],
            "final_recommendation": result.content,
            "sender":               "supervisor_agent",
        }

    return node
