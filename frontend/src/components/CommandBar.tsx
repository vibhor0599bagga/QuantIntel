"use client";

import React, { useState } from "react";
import { Play, Square, Sliders, Calendar, Shield, Compass, Sparkles, Key } from "lucide-react";
import { CalendarPicker, getTodayDateString } from "@/components/CalendarPicker";

export interface AnalysisConfig {
  ticker: string;
  tradeDate: string;
  riskTolerance: "conservative" | "moderate" | "aggressive";
  horizon: "short_term" | "medium_term" | "long_term";
  sectorExposure: "tech_heavy" | "banking_focused" | "diversified";
}

interface CommandBarProps {
  onRunAnalysis: (config: AnalysisConfig) => void;
  onStopAnalysis: () => void;
  isAnalyzing: boolean;
  defaultTradeDate?: string;
  hasApiKey: boolean;
  onOpenKeyModal: () => void;
}

const PRESET_TICKERS = ["AAPL", "MSFT", "GOOGL", "HDFCBANK", "NVDA", "TSLA", "RELIANCE"];

export const CommandBar: React.FC<CommandBarProps> = ({
  onRunAnalysis,
  onStopAnalysis,
  isAnalyzing,
  defaultTradeDate,
  hasApiKey,
  onOpenKeyModal,
}) => {
  const [ticker, setTicker] = useState("AAPL");
  const [tradeDate, setTradeDate] = useState(
    defaultTradeDate || getTodayDateString()
  );

  React.useEffect(() => {
    if (defaultTradeDate) {
      setTradeDate(defaultTradeDate);
    }
  }, [defaultTradeDate]);

  const [riskTolerance, setRiskTolerance] = useState<"conservative" | "moderate" | "aggressive">("moderate");
  const [horizon, setHorizon] = useState<"short_term" | "medium_term" | "long_term">("medium_term");
  const [sectorExposure, setSectorExposure] = useState<"tech_heavy" | "banking_focused" | "diversified">("tech_heavy");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker.trim()) return;
    onRunAnalysis({
      ticker: ticker.trim().toUpperCase(),
      tradeDate,
      riskTolerance,
      horizon,
      sectorExposure,
    });
  };

  return (
    <div className="w-full bt-panel p-6 mb-8 shadow-xl">
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        {/* Bloomberg Command Input Line */}
        <div className="flex flex-wrap lg:flex-nowrap items-center gap-4">
          {/* Main Input Box */}
          <div className="flex-1 min-w-[280px] h-13 flex items-center bg-[#05080f] border border-[#ff9d00]/50 rounded-lg px-4 focus-within:border-[#ff9d00] focus-within:shadow-[0_0_16px_rgba(255,157,0,0.35)] transition-all">
            <span className="text-[#ff9d00] font-extrabold text-sm sm:text-base mr-3 select-none tracking-wide shrink-0">
              QUANTINTEL&gt;
            </span>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="ENTER TICKER (e.g. AAPL, MSFT, GOOGL, HDFCBANK, NVDA)"
              disabled={isAnalyzing}
              className="w-full bg-transparent text-[#e2e8f0] font-mono font-bold text-sm sm:text-base focus:outline-none uppercase placeholder:text-[#475569] placeholder:font-normal"
            />
            <span className="bt-cursor"></span>
          </div>

          {/* Quick Date Selector in Command Line */}
          <div className="w-full sm:w-56 shrink-0">
            <CalendarPicker
              selectedDate={tradeDate}
              onSelectDate={setTradeDate}
              disabled={isAnalyzing}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3 w-full sm:w-auto shrink-0">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={`h-13 px-4 rounded-lg border text-xs font-mono font-bold flex items-center gap-2 transition-all cursor-pointer ${
                showAdvanced
                  ? "bg-[#ff9d00]/20 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_10px_rgba(255,157,0,0.2)]"
                  : "bg-[#0d121d] border-[#1e293b] text-[#94a3b8] hover:border-[#ff9d00]/50 hover:text-[#e2e8f0]"
              }`}
            >
              <Sliders className="w-4 h-4" />
              <span>SETTINGS</span>
            </button>

            {isAnalyzing ? (
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onStopAnalysis();
                }}
                className="h-13 px-6 bg-[#ff3333]/20 border border-[#ff3333] text-[#ff3333] hover:bg-[#ff3333]/30 rounded-lg font-mono font-bold text-xs sm:text-sm flex items-center gap-2 transition-all shadow-[0_0_12px_rgba(255,51,51,0.25)] cursor-pointer"
              >
                <Square className="w-4 h-4 fill-current" />
                <span>ABORT ANALYSIS</span>
              </button>
            ) : (
              <button
                type="submit"
                className="h-13 px-7 bg-[#ff9d00] text-[#06090e] hover:bg-[#ffb033] rounded-lg font-mono font-black text-xs sm:text-sm flex items-center gap-2 transition-all shadow-[0_0_20px_rgba(255,157,0,0.45)] cursor-pointer tracking-wider"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>RUN SWARM ANALYSIS</span>
              </button>
            )}
          </div>
        </div>

        {/* Quick Preset Ticker Buttons Row */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[#141b27]">
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="text-xs font-mono font-bold text-[#8492a6] mr-1 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#ff9d00]" /> QUICK PRESETS:
            </span>
            {PRESET_TICKERS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => {
                  setTicker(t);
                  onRunAnalysis({
                    ticker: t,
                    tradeDate,
                    riskTolerance,
                    horizon,
                    sectorExposure,
                  });
                }}
                disabled={isAnalyzing}
                className={`px-3 py-1.5 text-xs font-mono font-bold rounded-md border transition-all cursor-pointer ${
                  ticker === t
                    ? "bg-[#ff9d00]/25 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_10px_rgba(255,157,0,0.25)]"
                    : "bg-[#0d121d] border-[#1e293b] text-[#94a3b8] hover:border-[#ff9d00]/50 hover:text-[#e2e8f0]"
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={onOpenKeyModal}
            className="text-xs font-mono font-bold flex items-center gap-1.5 text-[#ff9d00] hover:text-[#ffb033] hover:underline cursor-pointer"
          >
            <Key className="w-3.5 h-3.5" />
            <span>{hasApiKey ? "Edit OpenRouter Key" : "Configure API Key"}</span>
          </button>
        </div>

        {/* Advanced Parameters Panel */}
        {showAdvanced && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 pt-4 border-t border-[#1a2333] bg-[#070b12] p-5 rounded-lg border border-[#1e293b]">
            <div>
              <label className="text-xs text-[#8492a6] font-mono font-bold flex items-center gap-1.5 mb-2">
                <Calendar className="w-3.5 h-3.5 text-[#00e5ff]" /> ANALYSIS DATE
              </label>
              <CalendarPicker
                selectedDate={tradeDate}
                onSelectDate={setTradeDate}
                disabled={isAnalyzing}
              />
            </div>

            <div>
              <label className="text-xs text-[#8492a6] font-mono font-bold flex items-center gap-1.5 mb-2">
                <Shield className="w-3.5 h-3.5 text-[#00ff66]" /> RISK TOLERANCE
              </label>
              <select
                value={riskTolerance}
                onChange={(e) => setRiskTolerance(e.target.value as any)}
                disabled={isAnalyzing}
                className="w-full bg-[#0d121d] border border-[#1e293b] text-[#e2e8f0] font-mono text-xs px-3 py-2.5 rounded-md focus:border-[#00ff66] focus:outline-none"
              >
                <option value="conservative">CONSERVATIVE (Capital Preservation)</option>
                <option value="moderate">MODERATE (Balanced Risk/Return)</option>
                <option value="aggressive">AGGRESSIVE (High Beta / Growth)</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-[#8492a6] font-mono font-bold flex items-center gap-1.5 mb-2">
                <Compass className="w-3.5 h-3.5 text-[#ff9d00]" /> INVESTMENT HORIZON
              </label>
              <select
                value={horizon}
                onChange={(e) => setHorizon(e.target.value as any)}
                disabled={isAnalyzing}
                className="w-full bg-[#0d121d] border border-[#1e293b] text-[#e2e8f0] font-mono text-xs px-3 py-2.5 rounded-md focus:border-[#ff9d00] focus:outline-none"
              >
                <option value="short_term">SHORT TERM (1-4 Weeks)</option>
                <option value="medium_term">MEDIUM TERM (1-6 Months)</option>
                <option value="long_term">LONG TERM (1+ Years)</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-[#8492a6] font-mono font-bold flex items-center gap-1.5 mb-2">
                <Key className="w-3.5 h-3.5 text-[#ff9d00]" /> OPENROUTER KEY
              </label>
              <button
                type="button"
                onClick={onOpenKeyModal}
                className="w-full h-[38px] text-left bg-[#0d121d] border border-[#1e293b] hover:border-[#ff9d00]/60 text-[#e2e8f0] font-mono text-xs px-3 rounded-md flex items-center justify-between cursor-pointer"
              >
                <span className={hasApiKey ? "text-[#00ff66] font-bold" : "text-[#ff9d00]"}>
                  {hasApiKey ? "••••••••••••••••" : "Set API Key"}
                </span>
                <span className="text-[11px] bg-[#1e293b] px-2 py-0.5 rounded text-[#94a3b8] font-bold">
                  Edit
                </span>
              </button>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};
