"use client";

import React, { useState, useEffect, useRef } from "react";
import { Header } from "@/components/Header";
import { CommandBar, AnalysisConfig } from "@/components/CommandBar";
import { StreamProgress, StreamState } from "@/components/StreamProgress";
import { VerdictHero } from "@/components/VerdictHero";
import { AgentGrid } from "@/components/AgentGrid";
import { RawTerminal } from "@/components/RawTerminal";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://quantintel.onrender.com";

export default function Home() {
  const [apiStatus, setApiStatus] = useState<"connected" | "connecting" | "offline">("connecting");
  const [ticker, setTicker] = useState("AAPL");
  const [tradeDate, setTradeDate] = useState("2026-09-14");

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

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === "ok") {
            setApiStatus("connected");
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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setStreamState((prev) => ({
      ...prev,
      isAnalyzing: false,
      logs: [...prev.logs, "[ABORT] Analysis aborted by user."],
    }));
  };

  const handleRunAnalysis = async (config: AnalysisConfig) => {
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
      logs: [`[INIT] Target Ticker: ${config.ticker} | Date: ${config.tradeDate}`],
    });

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          ticker: config.ticker,
          trade_date: config.tradeDate,
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

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          let eventType = "message";
          let dataStr = "";

          const lineParts = line.split("\n");
          for (const part of lineParts) {
            if (part.startsWith("event:")) {
              eventType = part.replace("event:", "").trim();
            } else if (part.startsWith("data:")) {
              dataStr += part.replace("data:", "").trim();
            }
          }

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            const time = new Date().toLocaleTimeString();

            setStreamState((prev) => ({
              ...prev,
              logs: [...prev.logs, `[${time}] EVENT ${eventType}: ${JSON.stringify(data).slice(0, 100)}...`],
            }));

            if (eventType === "phase1_complete") {
              setFundamentalsReport(data.fundamentals_report || "");
              setSentimentReport(data.sentiment_report || "");
              setTechnicalReport(data.technical_report || "");
              setMacroReport(data.macro_report || "");
              setStreamState((prev) => ({
                ...prev,
                currentPhase: 2,
                phase1Complete: true,
              }));
            } else if (eventType === "phase2_complete") {
              setRiskReport(data.risk_report || "");
              setStreamState((prev) => ({
                ...prev,
                currentPhase: 3,
                phase2Complete: true,
              }));
            } else if (eventType === "final_recommendation") {
              setFinalRecommendation(data.final_recommendation || "");
              setStreamState((prev) => ({
                ...prev,
                phase3Complete: true,
              }));
            } else if (eventType === "complete") {
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
      setStreamState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: err.message,
        logs: [...prev.logs, `[ERROR] Stream failed: ${err.message}`],
      }));
    }
  };

  return (
    <div className="min-h-screen bg-[#06090e] text-[#e2e8f0] flex flex-col font-mono">
      {/* Header Bar */}
      <Header apiStatus={apiStatus} apiUrl={API_BASE_URL} />

      {/* Main Content Area */}
      <main className="max-w-[1700px] w-full mx-auto px-4 py-6 flex-1">
        {/* Command Bar Input */}
        <CommandBar
          onRunAnalysis={handleRunAnalysis}
          onStopAnalysis={handleStopAnalysis}
          isAnalyzing={streamState.isAnalyzing}
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

      {/* Footer */}
      <footer className="w-full bg-[#040609] border-t border-[#121824] py-3 text-center text-xs text-[#64748b]">
        QUANTINTEL TERMINAL &copy; 2026 — MULTI-AGENT SWARM ENGINE POWERED BY FASTMCP &amp; LANGGRAPH
      </footer>
    </div>
  );
}
