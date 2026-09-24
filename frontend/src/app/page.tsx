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

const getApiBaseUrl = (): string => {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined") {
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }
  }
  return "https://quantintel.onrender.com";
};

// Helper to extract clean Markdown text from tool results or JSON structures
const cleanReport = (val: any): string => {
  if (!val) return "";
  if (typeof val !== "string") return String(val);

  let trimmed = val.trim();
  // Handle stringified Python AST or list of TextContent/dicts like "[{'type': 'text', 'text': '...'}]"
  if ((trimmed.startsWith("[{") && trimmed.endsWith("}]")) || (trimmed.startsWith("[TextContent(") && trimmed.endsWith(")]"))) {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        return parsed.map((item) => item.text || item.content || JSON.stringify(item)).join("\n\n");
      }
    } catch {
      // Python dict syntax with single quotes
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
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [ticker, setTicker] = useState("AAPL");
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
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (readerRef.current) {
      try {
        readerRef.current.cancel();
        readerRef.current = null;
      } catch (e) {
        console.warn("Stream reader cancel warning:", e);
      }
    }
    setStreamState((prev) => ({
      ...prev,
      isAnalyzing: false,
      currentPhase: 0,
      logs: [...prev.logs, "[ABORT] Analysis aborted by user."],
    }));
  };

  const handleRunAnalysis = async (config: AnalysisConfig) => {
    // If no key is set, prompt user to set it
    if (!apiKey.trim()) {
      setIsKeyModalOpen(true);
      setStreamState((prev) => ({
        ...prev,
        logs: [...prev.logs, "[AUTH] Please set your OpenRouter API key to initiate the analysis."],
      }));
      return;
    }

    setTicker(config.ticker);
    setTradeDate(config.tradeDate);

    // Reset reports
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
      logs: [`[INIT] Target Ticker: ${config.ticker} | Date: ${config.tradeDate} | Using User OpenRouter Key`],
    });

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const targetUrl = apiUrl || getApiBaseUrl();
      const response = await fetch(`${targetUrl}/api/analyze/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-openrouter-api-key": apiKey.trim(),
        },
        signal: abortControllerRef.current.signal,
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
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // Split by standard SSE double-newline message delimiter
        const normalized = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
        const messageBlocks = normalized.split("\n\n");
        buffer = messageBlocks.pop() || "";

        for (const block of messageBlocks) {
          if (!block.trim()) continue;

          let eventType = "message";
          const dataLines: string[] = [];

          const lines = block.split("\n");
          for (const line of lines) {
            if (line.startsWith(":")) {
              // Ignore SSE ping/comment lines
              continue;
            }
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
      if (err.name === "AbortError") return;
      console.error("Stream execution error:", err);
      const isNetworkError = err.message?.includes("failed") || err.message?.includes("NetworkError") || err.name === "TypeError";
      const detailMsg = isNetworkError
        ? `Failed to connect to backend at ${apiUrl || getApiBaseUrl()}. Please make sure the FastAPI server is running (uvicorn quantintel.api.app:app --port 8000).`
        : err.message;

      setStreamState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: detailMsg,
        logs: [...prev.logs, `[ERROR] ${detailMsg}`],
      }));
    }
  };

  return (
    <div className="min-h-screen bg-[#06090e] text-[#e2e8f0] flex flex-col font-mono">
      {/* Header Bar */}
      <Header
        apiStatus={apiStatus}
        apiUrl={apiUrl}
        hasApiKey={!!apiKey.trim()}
        onOpenKeyModal={() => setIsKeyModalOpen(true)}
      />

      {/* Main Content Area */}
      <main className="max-w-[1700px] w-full mx-auto px-4 py-6 flex-1">
        {/* Command Bar Input */}
        <CommandBar
          onRunAnalysis={handleRunAnalysis}
          onStopAnalysis={handleStopAnalysis}
          isAnalyzing={streamState.isAnalyzing}
          defaultTradeDate={tradeDate}
          hasApiKey={!!apiKey.trim()}
          onOpenKeyModal={() => setIsKeyModalOpen(true)}
        />

        {/* Live Swarm Execution Progress Tracker */}
        <StreamProgress streamState={streamState} ticker={ticker} />

        {/* Supervisor Final Verdict */}
        <VerdictHero
          ticker={ticker}
          tradeDate={tradeDate}
          recommendationText={finalRecommendation}
        />

        {/* 5-Agent Detailed Grid Breakdown */}
        <AgentGrid
          fundamentalsReport={fundamentalsReport}
          sentimentReport={sentimentReport}
          technicalReport={technicalReport}
          macroReport={macroReport}
          riskReport={riskReport}
        />

        {/* Collapsible Developer Stream Terminal */}
        <RawTerminal logs={streamState.logs} />
      </main>

      {/* API Key Modal */}
      <ApiKeyModal
        isOpen={isKeyModalOpen}
        onClose={() => setIsKeyModalOpen(false)}
        apiKey={apiKey}
        onSaveKey={handleSaveApiKey}
      />

      {/* Footer */}
      <footer className="w-full bg-[#040609] border-t border-[#121824] py-3 text-center text-xs text-[#64748b]">
        QUANTINTEL TERMINAL &copy; 2026 — MULTI-AGENT SWARM ENGINE POWERED BY FASTMCP &amp; LANGGRAPH
      </footer>
    </div>
  );
}
