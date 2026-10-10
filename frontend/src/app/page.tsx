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

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      const targetUrl = getApiBaseUrl();
      try {
        const res = await fetch(`${targetUrl}/health`);
        if (res.ok) {
          const data = await res.json();
          console.log("🌐 [QuantIntel API] Live Health Status:", data);
          if (data.status === "ok") {
            setApiStatus("connected");
            if (data.server_date) {
              setTradeDate(data.server_date);
            }
          } else {
            setApiStatus("offline");
          }
        } else {
          setApiStatus("offline");
        }
      } catch (err) {
        console.warn("API Health Check Warning:", err);
        setApiStatus("offline");
      }
    };
    checkHealth();
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
      const response = await fetch(`${targetUrl}/api/analyze/stream`, {
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

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
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
              setFundamentalsReport(cleanReport(data.fundamentals_report));
              setSentimentReport(cleanReport(data.sentiment_report));
              setTechnicalReport(cleanReport(data.technical_report));
              setMacroReport(cleanReport(data.macro_report));
              setStreamState((prev) => ({
                ...prev,
                currentPhase: 2,
                phase1Complete: true,
              }));
            } else if (eventType === "phase2_complete") {
              console.log("🛡️ Phase 2 Risk Report Received:", data);
              setRiskReport(cleanReport(data.risk_report));
              setStreamState((prev) => ({
                ...prev,
                currentPhase: 3,
                phase2Complete: true,
              }));
            } else if (eventType === "final_recommendation") {
              console.log("🏆 Phase 3 Supervisor Final Recommendation Received:", data);
              setFinalRecommendation(cleanReport(data.final_recommendation));
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
