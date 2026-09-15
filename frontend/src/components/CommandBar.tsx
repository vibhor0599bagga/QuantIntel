"use client";

import React, { useState } from "react";
import { Play, Square, Sliders, Calendar, Shield, Compass, Sparkles } from "lucide-react";

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
}

const PRESET_TICKERS = ["AAPL", "MSFT", "GOOGL", "HDFCBANK", "NVDA", "TSLA", "RELIANCE"];

export const CommandBar: React.FC<CommandBarProps> = ({
  onRunAnalysis,
  onStopAnalysis,
  isAnalyzing,
}) => {
  const [ticker, setTicker] = useState("AAPL");
  const [tradeDate, setTradeDate] = useState("2026-09-14");
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
    <div className="w-full bt-panel p-4 mb-6">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {/* Bloomberg Command Input Line */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex-1 flex items-center bg-[#05080f] border border-[#ff9d00]/50 rounded px-3 py-2.5 focus-within:border-[#ff9d00] focus-within:shadow-[0_0_12px_rgba(255,157,0,0.3)] transition-all">
            <span className="text-[#ff9d00] font-bold text-sm mr-2 select-none">QUANTINTEL&gt;</span>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="ENTER TICKER (e.g. AAPL, MSFT, HDFCBANK)"
              disabled={isAnalyzing}
              className="w-full bg-transparent text-[#e2e8f0] font-mono font-bold text-sm focus:outline-none uppercase placeholder:text-[#475569]"
            />
            <span className="bt-cursor"></span>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={`px-3 py-2.5 rounded border text-xs font-mono font-bold flex items-center gap-1.5 transition-colors ${
                showAdvanced
                  ? "bg-[#ff9d00]/20 border-[#ff9d00] text-[#ff9d00]"
                  : "bg-[#0f172a] border-[#1e293b] text-[#94a3b8] hover:border-[#475569]"
              }`}
            >
              <Sliders className="w-3.5 h-3.5" />
              SETTINGS
            </button>

            {isAnalyzing ? (
              <button
                type="button"
                onClick={onStopAnalysis}
                className="px-5 py-2.5 bg-[#ff3333]/20 border border-[#ff3333] text-[#ff3333] hover:bg-[#ff3333]/30 rounded font-mono font-bold text-xs flex items-center gap-2 transition-all shadow-[0_0_10px_rgba(255,51,51,0.2)]"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                ABORT SWARM
              </button>
            ) : (
              <button
                type="submit"
                className="px-6 py-2.5 bg-[#ff9d00] text-[#06090e] hover:bg-[#ffb033] rounded font-mono font-extrabold text-xs flex items-center gap-2 transition-all shadow-[0_0_15px_rgba(255,157,0,0.4)]"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                RUN SWARM ANALYSIS
              </button>
            )}
          </div>
        </div>

        {/* Quick Preset Ticker Buttons */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-[#121824]">
          <span className="text-[11px] font-mono text-[#64748b] mr-1 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-[#ff9d00]" /> QUICK PRESETS:
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
              className={`px-2.5 py-1 text-xs font-mono rounded border transition-all ${
                ticker === t
                  ? "bg-[#ff9d00]/20 border-[#ff9d00] text-[#ff9d00] font-bold"
                  : "bg-[#0f172a] border-[#1e293b] text-[#94a3b8] hover:border-[#ff9d00]/40 hover:text-[#e2e8f0]"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Advanced Parameters Panel */}
        {showAdvanced && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-[#1a2333] bg-[#080c14] p-3 rounded border border-[#1e293b]">
            <div>
              <label className="text-[11px] text-[#64748b] font-mono flex items-center gap-1 mb-1.5">
                <Calendar className="w-3 h-3 text-[#00e5ff]" /> ANALYSIS DATE
              </label>
              <input
                type="date"
                value={tradeDate}
                onChange={(e) => setTradeDate(e.target.value)}
                disabled={isAnalyzing}
                className="w-full bg-[#0d121d] border border-[#1e293b] text-[#e2e8f0] font-mono text-xs px-2.5 py-1.5 rounded focus:border-[#00e5ff] focus:outline-none"
              />
            </div>

            <div>
              <label className="text-[11px] text-[#64748b] font-mono flex items-center gap-1 mb-1.5">
                <Shield className="w-3 h-3 text-[#00ff66]" /> RISK TOLERANCE
              </label>
              <select
                value={riskTolerance}
                onChange={(e) => setRiskTolerance(e.target.value as any)}
                disabled={isAnalyzing}
                className="w-full bg-[#0d121d] border border-[#1e293b] text-[#e2e8f0] font-mono text-xs px-2.5 py-1.5 rounded focus:border-[#00ff66] focus:outline-none"
              >
                <option value="conservative">CONSERVATIVE (Capital Preservation)</option>
                <option value="moderate">MODERATE (Balanced Risk/Return)</option>
                <option value="aggressive">AGGRESSIVE (High Beta / Growth)</option>
              </select>
            </div>

            <div>
              <label className="text-[11px] text-[#64748b] font-mono flex items-center gap-1 mb-1.5">
                <Compass className="w-3 h-3 text-[#ff9d00]" /> INVESTMENT HORIZON
              </label>
              <select
                value={horizon}
                onChange={(e) => setHorizon(e.target.value as any)}
                disabled={isAnalyzing}
                className="w-full bg-[#0d121d] border border-[#1e293b] text-[#e2e8f0] font-mono text-xs px-2.5 py-1.5 rounded focus:border-[#ff9d00] focus:outline-none"
              >
                <option value="short_term">SHORT TERM (1-4 Weeks)</option>
                <option value="medium_term">MEDIUM TERM (1-6 Months)</option>
                <option value="long_term">LONG TERM (1+ Years)</option>
              </select>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};
