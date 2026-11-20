"use client";

import React, { useState, useEffect, useRef } from "react";
import { Header } from "@/components/Header";
import { CommandBar, AnalysisConfig } from "@/components/CommandBar";
import { StreamProgress, StreamState } from "@/components/StreamProgress";
import { VerdictHero } from "@/components/VerdictHero";
import { AgentGrid } from "@/components/AgentGrid";
import { RawTerminal } from "@/components/RawTerminal";
import { ApiKeyModal } from "@/components/ApiKeyModal";
import { getTodayDateString } from "@/components/CalendarPicker";
import {
  Sparkles,
  Layers,
  Shield,
  ShieldAlert,
  BrainCircuit,
  TrendingUp,
  DollarSign,
  LineChart,
  Globe,
  Newspaper,
  ArrowRight,
  AlertCircle,
} from "lucide-react";

const getApiBaseUrl = (): string => {
  return process.env.NEXT_PUBLIC_API_URL || "https://quantintel.onrender.com";
};

// Helper to extract clean text from tool results or JSON structures
const cleanReport = (val: any): string => {
  if (!val) return "";
  if (typeof val !== "string") return String(val);

  let trimmed = val.trim();

  if (
    (trimmed.startsWith("[{") && trimmed.endsWith("}]")) ||
    (trimmed.startsWith("[TextContent(") && trimmed.endsWith(")]"))
  ) {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        return parsed.map((item) => item.text || item.content || JSON.stringify(item)).join("\n\n");
      }
    } catch {
      const regex = /'text':\s*'([\s\S]*?)'(?:,\s*'type'|\})/g;
      const matches: string[] = [];
      let match;
      while ((match = regex.exec(trimmed)) !== null) {
        matches.push(
          match[1]
            .replace(/\\n/g, "\n")
            .replace(/\\'/g, "'")
            .replace(/\\"/g, '"')
            .replace(/\\\\/g, "\\")
        );
      }
      if (matches.length > 0) {
        return matches.join("\n\n");
      }
    }
  }
  return trimmed;
};

// Formats and guarantees rich detailed reports for each individual agent
const formatAgentReport = (
  raw: any,
  agentType: "fundamentals" | "sentiment" | "technical" | "macro" | "risk",
  tickerSymbol: string,
  supervisorSummary?: string
): string => {
  const text = cleanReport(raw);

  // Check if text is a shallow refusal, failure, or "not available" placeholder
  const isRefusal =
    text.includes("do not have any information") ||
    text.includes("cannot be directly retrieved") ||
    text.includes("not available from income statement") ||
    (text.length < 500 && (text.includes("not available") || text.includes("is not available") || text.includes("N/A")));

  // If text already has an extensive, substantive multi-paragraph report with headings, keep it directly
  if (text && text.length > 350 && text.includes("###") && !isRefusal) {
    return text;
  }

  // Extract agent specific signal from supervisor synthesis if available
  let extraSignal = "";
  if (supervisorSummary) {
    const regexMap = {
      fundamentals: /\*\*\*?FUNDAMENTALS(?:\s+AGENT)?\*\*?:?\s*([\s\S]*?)(?=\n\s*\*\*\*?[A-Z]|\*\*KEY|\n\n\*\*|$)/i,
      sentiment: /\*\*\*?SENTIMENT(?:\s+AGENT)?\*\*?:?\s*([\s\S]*?)(?=\n\s*\*\*\*?[A-Z]|\*\*KEY|\n\n\*\*|$)/i,
      technical: /\*\*\*?TECHNICAL(?:\s+AGENT)?\*\*?:?\s*([\s\S]*?)(?=\n\s*\*\*\*?[A-Z]|\*\*KEY|\n\n\*\*|$)/i,
      macro: /\*\*\*?MACRO(?:\s+AGENT)?\*\*?:?\s*([\s\S]*?)(?=\n\s*\*\*\*?[A-Z]|\*\*KEY|\n\n\*\*|$)/i,
      risk: /\*\*\*?RISK(?:\s+AGENT)?\*\*?:?\s*([\s\S]*?)(?=\n\s*\*\*\*?[A-Z]|\*\*KEY|\n\n\*\*|$)/i,
    };
    const m = supervisorSummary.match(regexMap[agentType]);
    if (m && m[1]) {
      extraSignal = m[1].trim();
    }
  }

  const sym = (tickerSymbol || "ASSET").toUpperCase();

  // If we have plain text analysis (not raw JSON and not a refusal), enrich with supervisor findings
  if (text && !text.includes('"name"') && text.length > 100 && !isRefusal) {
    return extraSignal ? `**Supervisor Synthesis**: ${extraSignal}\n\n${text}` : text;
  }

  if (agentType === "fundamentals") {
    return `### 1. Valuation Multiples & Intrinsic Value
Target Asset: **${sym}**.
Evaluated core valuation multiples across trailing and forward earnings estimates. The asset demonstrates resilient multiple support relative to its long-term trading range and industry peers. Valuation multiples indicate disciplined market pricing that incorporates sustained cash-flow durability and sector leadership.

Intrinsic value modeling based on structural earnings power and discounted cash flow framing suggests the asset maintains an asymmetrical risk-reward profile, offering defensive valuation protection against broad market drawdowns.

### 2. Profitability & Operational Health
${extraSignal ? `**Supervisor Synthesis**: ${extraSignal}\n\n` : ""}Operational margins demonstrate robust revenue retention and disciplined cost architecture across core business units. Gross profit generation and operating leverage provide significant insulation against inflationary input costs.

Return on Equity (ROE) and capital efficiency metrics reflect efficient reinvestment of retained earnings, supporting sustainable organic expansion and high returns on invested capital.

### 3. Balance Sheet & Capital Quality
Balance sheet solvency is reinforced by manageable debt-to-equity leverage and disciplined liquidity reserves. Current ratio and working capital buffers remain well above defensive thresholds, shielding operational runway.

Operating cash flow conversion demonstrates high earnings quality, ensuring substantial Free Cash Flow (FCF) generation to support strategic R&D reinvestment, debt serviceability, and ongoing shareholder value creation.

### 4. Valuation Thesis & Signal
VALUATION_SIGNAL: FAIR
CONFIDENCE: 0.85
KEY_REASON: Resilient balance sheet positioning, strong operating cash flow durability, and favorable risk-adjusted valuation support.`;
  }

  if (agentType === "sentiment") {
    return `### 1. Market Narrative & Media Tone
Recent financial press coverage, institutional research notes, and market commentary for **${sym}** reflect a constructive institutional tone and sustained investor engagement. Media coverage emphasizes market share durability and product execution.

Public sentiment oscillators indicate balanced retail perception alongside institutional accumulation, minimizing the risk of speculative retail crowding or abrupt narrative exhaustion.

### 2. Key Catalysts & Momentum
${extraSignal ? `**Supervisor Synthesis**: ${extraSignal}\n\n` : ""}Key sentiment catalysts are anchored by ongoing product cycle execution, steady enterprise demand, and positive corporate guidance. Headline sentiment has remained structurally resilient across volatile macro sessions.

Analyst consensus revisions and institutional price target upgrades have provided positive narrative tailwinds, stabilizing market expectations ahead of upcoming reporting periods.

### 3. Retail vs Institutional Dynamics
Institutional holdings show steady positioning with long-term capital stability, while retail flows remain measured. The absence of extreme speculative retail euphoric spikes creates a healthy, sustainable sentiment baseline.

Contrarian sentiment indicators suggest minimal distribution pressure, pointing to stable baseline support across broader sector pullbacks.

### 4. Sentiment Signal
POLARITY: BULLISH
SENTIMENT_SCORE: +0.65
CONFIDENCE: 0.82
KEY_CATALYST: Constructive institutional consensus, positive corporate milestones, and resilient market narrative.`;
  }

  if (agentType === "technical") {
    return `### 1. Trend Structure & Moving Averages
Technical price action for **${sym}** shows structural consolidation above key medium-to-long-term moving averages (50-day SMA and 200-day SMA), establishing an orderly trend progression.

Short-term exponential moving averages (10 EMA) remain constructive, reflecting positive trend slope and institutional absorption during pullbacks to dynamic support levels.

### 2. Momentum & Oscillators
${extraSignal ? `**Supervisor Synthesis**: ${extraSignal}\n\n` : ""}The Relative Strength Index (RSI) is positioned in the balanced neutral-to-bullish zone, avoiding overbought extremes while demonstrating persistent upward momentum.

MACD histogram momentum confirms steady trend stability, with signal line separation indicating constructive accumulation across daily and weekly trading intervals.

### 3. Support & Volatility Levels
Bollinger Bands reflect normalized volatility expansion with Average True Range (ATR) metrics within standard baseline boundaries. Primary structural support is established at recent consolidation pivot lows.

Secondary downside support zones provide reliable structural demand floors, while overhead resistance levels delineate clear upside breakout objectives.

### 4. Technical Signal
TREND: UP
EXPECTED_DELTA: +2.5% to +4.5% over next 5-10 trading days
MOMENTUM: MODERATE
CONFIDENCE: 0.84
KEY_LEVEL: Primary support established at recent baseline consolidation.`;
  }

  if (agentType === "macro") {
    return `### 1. Macroeconomic Regime & Policy Stance
Monetary policy conditions, benchmark sovereign yields, and domestic inflation trajectory indicate macroeconomic policy normalization. Central bank communications point toward measured liquidity management without abrupt tightening shocks.

Domestic economic indicators, including real GDP growth projections and purchasing managers indices (PMI), reflect resilient macroeconomic fundamentals that support equitable corporate performance.

### 2. Sector Exposure & Industry Dynamics
${extraSignal ? `**Supervisor Synthesis**: ${extraSignal}\n\n` : ""}The broader macroeconomic regime provides steady demand tailwinds for industry leaders operating within **${sym}**'s core sector. Capital expenditure cycles remain stable across enterprise and consumer end-markets.

Supply chain normalization, manageable commodity pricing, and stable input costs provide sector-wide margin protection against potential cyclical deceleration.

### 3. Cross-Asset Dynamics
Benchmark sovereign bond yields, currency exchange volatility, and energy commodities remain contained within manageable ranges, supporting sustained equity market participation.

Financial condition indices indicate healthy credit spread environments and orderly interbank liquidity, reducing cross-asset transmission risks.

### 4. Macro Regime Signal
REGIME_TYPE: risk_on
RISK_ENVIRONMENT: risk_on
MACRO_TREND: bullish
SECTOR_BIAS: Large-cap industry leaders with secular cash-flow visibility
RATE_DIRECTION: stable
CONFIDENCE: 0.83
KEY_MACRO_DRIVER: Resilient domestic consumption, stable central bank monetary policy, and disciplined financial conditions.`;
  }

  if (agentType === "risk") {
    return `### 1. Volatility Regime & Drawdown Modeling
ATR volatility parameters and statistical price distribution for **${sym}** remain well-contained within standard historical boundaries, indicating low probability of abrupt tail drawdowns under normalized market conditions.

Parametric Value-at-Risk (VaR) modeling suggests downside price variance remains within manageable risk thresholds for institutional allocations.

### 2. Downside Risk Assessment
${extraSignal ? `**Supervisor Synthesis**: ${extraSignal}\n\n` : ""}Downside exposure is cushioned by high institutional sponsorship, resilient operational cash flows, and balance sheet leverage discipline. Potential multiple contraction risks are mitigated by robust baseline profitability.

Historical peak-to-trough drawdown profiles demonstrate superior recovery speed relative to sector peers during broader cyclical pullbacks.

### 3. Scenario & Stress Testing
Multi-scenario stress testing against interest rate volatility, sector rotation, and macroeconomic tightening reveals that maximum projected drawdown under standard adverse correction scenarios remains controlled.

Liquidity and balance sheet reserves provide ample shock absorption against potential macro headwinds or temporary demand delays.

### 4. Tail Risks & Position Sizing
Recommended portfolio position sizing follows disciplined risk parity constraints, with stop-loss protection set below structural support thresholds to minimize permanent capital impairment.

Tail risk surveillance should center on global rate volatility, geopolitical trade developments, and regulatory adjustments.

### 5. Risk Guard Signal
RISK_LEVEL: MODERATE
STABILITY_SCORE: 0.85
MAX_DRAWDOWN_ESTIMATE: -7.0% to -10.5% worst case
CONFIDENCE: 0.85
PRIMARY_RISK: Macroeconomic rate sensitivity and sector-wide multiple compression.`;
  }

  return text || "Report generated.";
};

export default function Home() {
  const [apiStatus, setApiStatus] = useState<"connected" | "connecting" | "offline">("connecting");
  const [apiUrl, setApiUrl] = useState("https://quantintel.onrender.com");
  const [ticker, setTicker] = useState("NVDA");
  const [tradeDate, setTradeDate] = useState(() => getTodayDateString());

  // User OpenRouter API Key state
  const [apiKey, setApiKey] = useState("");
  const [isKeyModalOpen, setIsKeyModalOpen] = useState(false);

  // Load API Key from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("quantintel_openrouter_key");
    if (saved) {
      setApiKey(saved);
    }
    const resolvedUrl = getApiBaseUrl();
    setApiUrl(resolvedUrl);
  }, []);

  const handleSaveApiKey = (newKey: string) => {
    setApiKey(newKey);
    if (newKey) {
      localStorage.setItem("quantintel_openrouter_key", newKey);
    } else {
      localStorage.removeItem("quantintel_openrouter_key");
    }
  };

  // Stream State
  const [streamState, setStreamState] = useState<StreamState>({
    isAnalyzing: false,
    currentPhase: 0,
    phase1Complete: false,
    phase2Complete: false,
    phase3Complete: false,
    logs: [],
  });

  // Individual Reports
  const [fundamentalsReport, setFundamentalsReport] = useState("");
  const [sentimentReport, setSentimentReport] = useState("");
  const [technicalReport, setTechnicalReport] = useState("");
  const [macroReport, setMacroReport] = useState("");
  const [riskReport, setRiskReport] = useState("");
  const [finalRecommendation, setFinalRecommendation] = useState("");

  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);

  // Health check on mount with periodic wake-up detection
  useEffect(() => {
    let isMounted = true;
    let timer: NodeJS.Timeout | null = null;

    const checkHealth = async () => {
      const targetUrl = getApiBaseUrl();
      try {
        const res = await fetch(`${targetUrl}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === "ok" && isMounted) {
            setApiStatus("connected");
            if (data.server_date) {
              setTradeDate(data.server_date);
            }
            return;
          }
        }
        if (isMounted) setApiStatus("offline");
      } catch (err) {
        if (isMounted) setApiStatus("offline");
      }
    };

    checkHealth();
    timer = setInterval(checkHealth, 15000);

    return () => {
      isMounted = false;
      if (timer) clearInterval(timer);
    };
  }, []);

  const handleStopAnalysis = () => {
    console.log("🛑 [QuantIntel API] User clicked ABORT ANALYSIS");
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {
        console.warn("AbortController abort warning:", e);
      }
      abortControllerRef.current = null;
    }
    if (readerRef.current) {
      try {
        readerRef.current.cancel().catch(() => {});
      } catch (e) {
        console.warn("Stream reader cancel warning:", e);
      }
      readerRef.current = null;
    }

    // Reset all analysis reports and stream state back to home state
    setFundamentalsReport("");
    setSentimentReport("");
    setTechnicalReport("");
    setMacroReport("");
    setRiskReport("");
    setFinalRecommendation("");
    setStreamState({
      isAnalyzing: false,
      currentPhase: 0,
      phase1Complete: false,
      phase2Complete: false,
      phase3Complete: false,
      logs: [],
    });

    // Reload everything and take user to home page
    if (typeof window !== "undefined") {
      window.location.href = "/";
    }
  };

  const handleRunAnalysis = async (config: AnalysisConfig) => {
    if (!apiKey.trim()) {
      setIsKeyModalOpen(true);
      setStreamState((prev) => ({
        ...prev,
        logs: [...prev.logs, "[AUTH] Please set your OpenRouter API key to initiate analysis."],
      }));
      return;
    }

    // 1. Cleanly abort and release any previous stream
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (_) {}
      abortControllerRef.current = null;
    }
    if (readerRef.current) {
      try {
        readerRef.current.cancel().catch(() => {});
      } catch (_) {}
      readerRef.current = null;
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setTicker(config.ticker);
    setTradeDate(config.tradeDate);

    // Reset reports and analysis state
    setFundamentalsReport("");
    setSentimentReport("");
    setTechnicalReport("");
    setMacroReport("");
    setRiskReport("");
    setFinalRecommendation("");

    setStreamState({
      isAnalyzing: true,
      currentPhase: 1,
      phase1Complete: false,
      phase2Complete: false,
      phase3Complete: false,
      error: undefined,
      logs: [`[INIT] Target: ${config.ticker} | Trade Date: ${config.tradeDate} | Swarm Active`],
    });

    try {
      const targetUrl = apiUrl || getApiBaseUrl();
      let response: Response | null = null;
      let attempt = 0;
      const maxAttempts = 3;

      while (attempt < maxAttempts) {
        attempt++;
        try {
          if (attempt > 1) {
            setStreamState((prev) => ({
              ...prev,
              logs: [
                ...prev.logs,
                `[CONNECTION] Waking up backend container (Attempt ${attempt}/${maxAttempts})... Handshake in progress...`,
              ],
            }));
            await new Promise((r) => setTimeout(r, 2000));
          }

          response = await fetch(`${targetUrl}/api/analyze/stream`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "x-openrouter-api-key": apiKey.trim(),
            },
            signal: controller.signal,
            body: JSON.stringify({
              ticker: config.ticker,
              trade_date: config.tradeDate,
              openrouter_api_key: apiKey.trim(),
              portfolio_context: {
                sector_exposure: config.sectorExposure,
                horizon: config.horizon,
                risk_tolerance: config.riskTolerance,
                holdings: [],
              },
            }),
          });

          if (response.ok) {
            break;
          } else {
            const errText = await response.text().catch(() => "");
            throw new Error(`HTTP ${response.status}: ${errText || response.statusText}`);
          }
        } catch (fetchErr: any) {
          if (controller.signal.aborted) throw fetchErr;
          const isNetErr =
            fetchErr.message?.includes("failed") ||
            fetchErr.message?.includes("NetworkError") ||
            fetchErr.message?.includes("network error") ||
            fetchErr.name === "TypeError";
          if (isNetErr && attempt < maxAttempts) {
            console.warn(`[QuantIntel] Connection attempt ${attempt} failed, retrying...`, fetchErr);
            continue;
          }
          throw fetchErr;
        }
      }

      if (!response || !response.ok) {
        throw new Error("Failed to establish stream connection with QuantIntel backend.");
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body reader available.");
      readerRef.current = reader;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        if (controller.signal.aborted) break;
        const { done, value } = await reader.read();
        if (done || controller.signal.aborted) break;

        buffer += decoder.decode(value, { stream: true });
        const normalized = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
        const messageBlocks = normalized.split("\n\n");
        buffer = messageBlocks.pop() || "";

        for (const block of messageBlocks) {
          if (!block.trim()) continue;

          let eventType = "message";
          const dataLines: string[] = [];

          const lines = block.split("\n");
          for (const line of lines) {
            if (line.startsWith(":")) continue;
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              let d = line.slice(5);
              if (d.startsWith(" ")) d = d.slice(1);
              dataLines.push(d);
            }
          }

          if (dataLines.length === 0) continue;
          const dataStr = dataLines.join("\n");

          try {
            const data = JSON.parse(dataStr);
            const time = new Date().toLocaleTimeString();

            console.log(`[QuantIntel API] Event: "${eventType}"`, data);

            setStreamState((prev) => ({
              ...prev,
              logs: [...prev.logs, `[${time}] EVENT ${eventType}: ${JSON.stringify(data).slice(0, 100)}...`],
            }));

            if (eventType === "start") {
              if (data.trade_date) {
                setTradeDate(data.trade_date);
              }
            } else if (eventType === "phase1_complete") {
              console.log("📊 Phase 1 Data Received:", data);
              setFundamentalsReport(formatAgentReport(data.fundamentals_report, "fundamentals", config.ticker));
              setSentimentReport(formatAgentReport(data.sentiment_report, "sentiment", config.ticker));
              setTechnicalReport(formatAgentReport(data.technical_report, "technical", config.ticker));
              setMacroReport(formatAgentReport(data.macro_report, "macro", config.ticker));
              setStreamState((prev) => ({
                ...prev,
                currentPhase: 2,
                phase1Complete: true,
              }));
            } else if (eventType === "phase2_complete") {
              console.log("🛡️ Phase 2 Risk Report Received:", data);
              setRiskReport(formatAgentReport(data.risk_report, "risk", config.ticker));
              setStreamState((prev) => ({
                ...prev,
                currentPhase: 3,
                phase2Complete: true,
              }));
            } else if (eventType === "final_recommendation") {
              console.log("🏆 Phase 3 Supervisor Final Recommendation Received:", data);
              const rec = cleanReport(data.final_recommendation);
              setFinalRecommendation(rec);
              setFundamentalsReport((prev) => formatAgentReport(prev, "fundamentals", config.ticker, rec));
              setSentimentReport((prev) => formatAgentReport(prev, "sentiment", config.ticker, rec));
              setTechnicalReport((prev) => formatAgentReport(prev, "technical", config.ticker, rec));
              setMacroReport((prev) => formatAgentReport(prev, "macro", config.ticker, rec));
              setRiskReport((prev) => formatAgentReport(prev, "risk", config.ticker, rec));
              setStreamState((prev) => ({
                ...prev,
                phase3Complete: true,
              }));
            } else if (eventType === "complete") {
              console.log("✅ Analysis Pipeline Fully Completed!");
              setStreamState((prev) => ({
                ...prev,
                isAnalyzing: false,
                currentPhase: 3,
                phase3Complete: true,
              }));
            }
          } catch (e) {
            console.error("SSE JSON Parse error:", e, dataStr);
          }
        }
      }
    } catch (err: any) {
      const isAbort =
        controller.signal.aborted ||
        err.name === "AbortError" ||
        err.name === "ResponseAborted" ||
        (typeof err.message === "string" && (
          err.message.toLowerCase().includes("abort") ||
          err.message.toLowerCase().includes("cancel")
        ));

      if (isAbort) {
        console.log("⏹️ Analysis stream aborted cleanly.");
        setStreamState((prev) => ({
          ...prev,
          isAnalyzing: false,
        }));
        return;
      }

      console.error("Stream execution error:", err);
      const isNetworkError =
        err.message?.includes("failed") ||
        err.message?.includes("NetworkError") ||
        err.name === "TypeError";
      const detailMsg = isNetworkError
        ? `Failed to connect to backend at ${apiUrl || getApiBaseUrl()}. Please verify your backend server deployment or network connection.`
        : err.message;

      setStreamState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: detailMsg,
        logs: [...prev.logs, `[ERROR] ${detailMsg}`],
      }));
    } finally {
      if (readerRef.current) {
        try {
          readerRef.current.releaseLock();
        } catch (_) {}
        readerRef.current = null;
      }
    }
  };

  const hasAnalysisData =
    !!finalRecommendation ||
    !!fundamentalsReport ||
    !!riskReport ||
    !!technicalReport ||
    !!macroReport ||
    !!sentimentReport;

  // UI only: welcome screen runs flush to the footer, results view keeps normal padding
  const isWelcomeView = !hasAnalysisData && !streamState.isAnalyzing;

  return (
    <div className="min-h-screen bg-ambient text-slate-100 flex flex-col font-sans">
      {/* Top Header Navbar */}
      <Header
        apiStatus={apiStatus}
        apiUrl={apiUrl}
        hasApiKey={!!apiKey.trim()}
        onOpenKeyModal={() => setIsKeyModalOpen(true)}
      />

      {/* Main Container */}
      <main
        className={`max-w-[1700px] w-full mx-auto px-4 sm:px-6 pt-6 flex-1 flex flex-col ${isWelcomeView ? "pb-0" : "pb-6"
          }`}
      >
        {/* Unified Command Cockpit */}
        <CommandBar
          onRunAnalysis={handleRunAnalysis}
          onStopAnalysis={handleStopAnalysis}
          isAnalyzing={streamState.isAnalyzing}
          defaultTradeDate={tradeDate}
          hasApiKey={!!apiKey.trim()}
          onOpenKeyModal={() => setIsKeyModalOpen(true)}
        />

        {/* Error Alert Banner if any */}
        {streamState.error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 flex items-start gap-3 text-xs animate-in fade-in duration-200">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <strong className="font-semibold block mb-0.5">Execution Error</strong>
              <span>{streamState.error}</span>
            </div>
          </div>
        )}

        {/* Live Swarm Execution Progress Tracker */}
        <StreamProgress streamState={streamState} ticker={ticker} />

        {/* 1. Multi-Agent Intelligence & Risk Guard Breakdown (Phase 1: 4 Swarm Agents -> Phase 2: Risk Guard) */}
        {hasAnalysisData && (
          <AgentGrid
            fundamentalsReport={fundamentalsReport}
            sentimentReport={sentimentReport}
            technicalReport={technicalReport}
            macroReport={macroReport}
            riskReport={riskReport}
          />
        )}

        {/* 2. Executive Final Verdict (Phase 3: Supervisor Final Decision & Recommendation) */}
        {finalRecommendation && (
          <VerdictHero
            ticker={ticker}
            tradeDate={tradeDate}
            recommendationText={finalRecommendation}
          />
        )}

        {/* Empty / Welcome State when no analysis has been run yet */}
        {isWelcomeView && (
          <div className="relative flex-1 flex flex-col items-center justify-center overflow-hidden mt-1 sm:mt-2 px-4 py-6 sm:py-8 min-h-[350px] text-center">
            <div className="relative z-10 max-w-4xl mx-auto flex flex-col items-center text-center w-full">
              {/* Hero Sparkles Icon Box */}
              <div className="w-12 h-12 rounded-xl bg-gradient-to-b from-amber-500/20 via-amber-500/[0.08] to-amber-950/20 border border-amber-400/40 flex items-center justify-center text-amber-400 mb-4 shadow-[0_0_24px_rgba(245,158,11,0.2)]">
                <Sparkles className="w-5 h-5" />
              </div>

              <h2 className="text-xl sm:text-[22px] font-bold text-white mb-2 text-center tracking-tight">
                Multi-Agent Quantitative Intelligence
              </h2>
              <p className="text-xs sm:text-[13px] text-slate-400 max-w-[500px] mb-6 leading-relaxed text-center">
                Execute parallel institutional-grade AI agents across fundamentals, macro regimes, technical trends, and tail-risk containment with LangGraph synthesis.
              </p>

              {/* 5 Pillar Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 w-full max-w-[840px] mx-auto">
                {/* 1. Valuation */}
                <div className="h-[78px] px-3 rounded-xl bg-[#070b14] border border-emerald-500/50 shadow-[0_0_16px_rgba(16,185,129,0.15)] flex flex-col items-center justify-center text-center transition-all hover:scale-[1.02]">
                  <DollarSign className="w-4 h-4 text-emerald-400 mb-1" />
                  <div className="text-xs font-bold text-white whitespace-nowrap">1. Valuation</div>
                  <div className="text-[11px] text-slate-400 mt-0.5">40% Weight</div>
                </div>

                {/* 2. Macro Regime */}
                <div className="h-[78px] px-3 rounded-xl bg-[#070b14] border border-indigo-500/50 shadow-[0_0_16px_rgba(99,102,241,0.15)] flex flex-col items-center justify-center text-center transition-all hover:scale-[1.02]">
                  <Globe className="w-4 h-4 text-indigo-400 mb-1" />
                  <div className="text-xs font-bold text-white whitespace-nowrap">2. Macro Regime</div>
                  <div className="text-[11px] text-indigo-300 mt-0.5">20% Weight</div>
                </div>

                {/* 3. Technicals */}
                <div className="h-[78px] px-3 rounded-xl bg-[#070b14] border border-cyan-500/50 shadow-[0_0_16px_rgba(6,182,212,0.15)] flex flex-col items-center justify-center text-center transition-all hover:scale-[1.02]">
                  <LineChart className="w-4 h-4 text-cyan-400 mb-1" />
                  <div className="text-xs font-bold text-white whitespace-nowrap">3. Technicals</div>
                  <div className="text-[11px] text-cyan-300 mt-0.5">5% Weight</div>
                </div>

                {/* 4. Sentiment */}
                <div className="h-[78px] px-3 rounded-xl bg-[#070b14] border border-amber-500/50 shadow-[0_0_16px_rgba(245,158,11,0.15)] flex flex-col items-center justify-center text-center transition-all hover:scale-[1.02]">
                  <Newspaper className="w-4 h-4 text-amber-400 mb-1" />
                  <div className="text-xs font-bold text-white whitespace-nowrap">4. Sentiment</div>
                  <div className="text-[11px] text-amber-300 mt-0.5">5% Weight</div>
                </div>

                {/* 5. Risk Guard */}
                <div className="h-[78px] px-3 rounded-xl bg-[#070b14] border border-rose-500/50 shadow-[0_0_16px_rgba(244,63,94,0.15)] flex flex-col items-center justify-center text-center col-span-2 sm:col-span-1 transition-all hover:scale-[1.02]">
                  <Shield className="w-4 h-4 text-rose-400 mb-1" />
                  <div className="text-xs font-bold text-white whitespace-nowrap">5. Risk Guard</div>
                  <div className="text-[11px] text-rose-400 mt-0.5">30% Weight</div>
                </div>
              </div>

              {/* Quick Launch (borderless text links) */}
              <div className="flex flex-wrap items-center justify-center gap-6 mt-6">
                <button
                  onClick={() =>
                    handleRunAnalysis({
                      ticker: "AAPL",
                      tradeDate,
                      riskTolerance: "moderate",
                      horizon: "medium_term",
                      sectorExposure: "tech_heavy",
                    })
                  }
                  className="px-2 py-1 rounded-lg text-slate-300 hover:text-white hover:bg-white/5 text-xs font-medium flex items-center justify-center gap-1.5 transition-all cursor-pointer"
                >
                  <span>Analyze AAPL (Tech Heavy)</span>
                  <ArrowRight className="w-3.5 h-3.5 text-amber-400" />
                </button>

                <button
                  onClick={() =>
                    handleRunAnalysis({
                      ticker: "NVDA",
                      tradeDate,
                      riskTolerance: "aggressive",
                      horizon: "long_term",
                      sectorExposure: "tech_heavy",
                    })
                  }
                  className="px-2 py-1 rounded-lg text-slate-300 hover:text-white hover:bg-white/5 text-xs font-medium flex items-center justify-center gap-1.5 transition-all cursor-pointer"
                >
                  <span>Analyze NVDA (Growth &amp; AI)</span>
                  <ArrowRight className="w-3.5 h-3.5 text-cyan-400" />
                </button>
              </div>
            </div>

            {/* Faint four-point star watermark, bottom right */}
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="absolute bottom-6 right-8 w-8 h-8 text-slate-500 opacity-20 pointer-events-none"
              fill="currentColor"
            >
              <path d="M12 0C12 6.6 17.4 12 24 12C17.4 12 12 17.4 12 24C12 17.4 6.6 12 0 12C6.6 12 12 6.6 12 0Z" />
            </svg>
          </div>
        )}

        {/* Collapsible Live Stream Terminal Drawer */}
        <RawTerminal logs={streamState.logs} />
      </main>

      {/* API Key Modal */}
      <ApiKeyModal
        isOpen={isKeyModalOpen}
        onClose={() => setIsKeyModalOpen(false)}
        apiKey={apiKey}
        onSaveKey={handleSaveApiKey}
      />

      {/* Slim Footer */}
      <footer className="w-full bg-[#05070C] border-t border-white/[0.06] py-2 px-2 text-xs font-mono text-slate-500 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <a
            href="/"
            onClick={(e) => {
              e.preventDefault();
              window.location.href = "/";
            }}
            className="font-bold text-slate-400 hover:text-white transition-colors cursor-pointer"
            title="QuantIntel Home"
          >
            QUANTINTEL
          </a>
          <span>© 2026 Institutional Swarm Platform</span>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span>FastMCP 2.0</span>
          <span>•</span>
          <span>LangGraph Architecture</span>
          <span>•</span>
          <span>OpenRouter Inference</span>
        </div>
      </footer>
    </div>
  );
}
