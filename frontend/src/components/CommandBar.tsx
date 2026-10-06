"use client";

import React, { useState } from "react";
import { Play, Square, Sliders, Sparkles, ChevronDown, ChevronUp, Layers, Target, Shield, Search } from "lucide-react";
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

const PRESET_TICKERS = [
  { symbol: "NVDA", name: "Nvidia" },
  { symbol: "AAPL", name: "Apple" },
  { symbol: "MSFT", name: "Microsoft" },
  { symbol: "GOOGL", name: "Alphabet" },
  { symbol: "TSLA", name: "Tesla" },
  { symbol: "HDFCBANK", name: "HDFC Bank" },
  { symbol: "RELIANCE", name: "Reliance" },
];

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
    <div className="w-full glass-panel p-4 sm:p-5 mb-8">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3.5">
        {/* Main Search & Controls Row (Unified h-11 / 44px Controls) */}
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center gap-2.5 sm:gap-3">
          {/* Ticker Search Box (Clean prefix & hero ticker typography) */}
          <div className="flex-1 min-w-[220px] h-11 flex items-center bg-[#090D16] border border-slate-800/90 rounded-full px-4 sm:px-5 focus-within:border-amber-500/50 focus-within:ring-1 focus-within:ring-amber-500/30 transition-all">
            <div className="flex items-center shrink-0 select-none mr-3">
              <Search className="w-3.5 h-3.5 text-slate-500 mr-2 shrink-0" />
              <span className="text-[10px] font-mono font-semibold tracking-wider text-slate-400 uppercase">
                TICKER
              </span>
              <span className="w-px h-3.5 bg-slate-800/90 ml-3" />
            </div>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL, NVDA, HDFCBANK"
              disabled={isAnalyzing}
              className="w-full bg-transparent text-white font-mono font-bold text-sm tracking-wide focus:outline-none uppercase placeholder:text-slate-600 placeholder:font-normal placeholder:tracking-normal placeholder:text-xs"
            />
          </div>

          {/* Quick Date Selector (Compact, unified height and styling) */}
          <div className="w-full sm:w-[210px] shrink-0">
            <CalendarPicker
              selectedDate={tradeDate}
              onSelectDate={setTradeDate}
              disabled={isAnalyzing}
            />
          </div>

          {/* Action Buttons Group */}
          <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
            {/* Strategy Context Toggle */}
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={`h-11 px-4.5 sm:px-5 rounded-full border text-xs font-medium flex items-center justify-between gap-2.5 transition-all duration-150 cursor-pointer shrink-0 whitespace-nowrap ${
                showAdvanced
                  ? "bg-slate-800/90 border-slate-700 text-white ring-1 ring-white/10"
                  : "bg-[#090D16] border-slate-800/90 text-slate-300 hover:border-slate-700 hover:text-white"
              }`}
              title="Portfolio context & risk parameters"
            >
              <div className="flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                <span className="text-xs">Strategy Context</span>
              </div>
              {showAdvanced ? (
                <ChevronUp className="w-3.5 h-3.5 text-slate-400 shrink-0 ml-1" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5 text-slate-500 shrink-0 ml-1" />
              )}
            </button>

            {/* Execute / Abort Button */}
            {isAnalyzing ? (
              <button
                type="button"
                onClick={onStopAnalysis}
                className="h-11 px-6 rounded-full bg-rose-500/20 border border-rose-500/50 hover:bg-rose-500/30 text-rose-400 text-xs font-mono font-bold flex items-center justify-center gap-2.5 transition-all cursor-pointer shadow-sm shadow-rose-500/10 shrink-0 whitespace-nowrap"
              >
                <Square className="w-3.5 h-3.5 fill-current animate-pulse shrink-0" />
                <span>ABORT</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!ticker.trim()}
                className="h-11 px-6 sm:px-7 rounded-full bg-amber-500 hover:bg-amber-400 active:bg-amber-500 text-slate-950 font-bold text-xs font-sans tracking-wider flex items-center justify-center gap-2.5 transition-all duration-150 shadow-sm shadow-amber-500/20 active:scale-[0.99] cursor-pointer shrink-0 whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Sparkles className="w-3.5 h-3.5 fill-slate-950/20 shrink-0" />
                <span>RUN SWARM ANALYSIS</span>
              </button>
            )}
          </div>
        </div>

        {/* Quick Ticker Chips */}
        <div className="flex items-center gap-2 overflow-x-auto pt-1 pb-1 text-xs no-scrollbar">
          <span className="text-[11px] font-medium text-slate-500 shrink-0 select-none mr-1">
            Presets:
          </span>
          {PRESET_TICKERS.map((preset) => (
            <button
              key={preset.symbol}
              type="button"
              disabled={isAnalyzing}
              onClick={() => setTicker(preset.symbol)}
              className={`px-4 py-1.5 rounded-full text-xs font-mono tracking-wide transition-all duration-150 shrink-0 cursor-pointer border flex items-center justify-center ${
                ticker === preset.symbol
                  ? "bg-amber-500/15 border-amber-500/50 text-amber-300 font-bold shadow-sm shadow-amber-500/10 ring-1 ring-amber-500/20"
                  : "bg-slate-900/70 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 hover:bg-slate-800/50"
              }`}
            >
              {preset.symbol}
            </button>
          ))}
        </div>

        {/* Collapsible Strategy & Context Drawer */}
        {showAdvanced && (
          <div className="pt-3.5 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-3 gap-3 animate-in fade-in slide-in-from-top-1.5 duration-150">
            {/* Risk Tolerance */}
            <div className="p-4 rounded-xl bg-[#090D16] border border-slate-800/90 flex flex-col gap-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Shield className="w-3.5 h-3.5 text-rose-400" /> Risk Tolerance
                </span>
                <span className="font-mono text-[10px] text-slate-500 uppercase">
                  {riskTolerance}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-0.5">
                {(["conservative", "moderate", "aggressive"] as const).map((level) => (
                  <button
                    key={level}
                    type="button"
                    onClick={() => setRiskTolerance(level)}
                    className={`py-1.5 px-3 rounded-full text-[11px] font-medium capitalize transition-all cursor-pointer border flex items-center justify-center ${
                      riskTolerance === level
                        ? "bg-rose-500/15 border-rose-500/40 text-rose-300 font-semibold shadow-xs"
                        : "bg-slate-900 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                    }`}
                  >
                    {level}
                  </button>
                ))}
              </div>
            </div>

            {/* Time Horizon */}
            <div className="p-4 rounded-xl bg-[#090D16] border border-slate-800/90 flex flex-col gap-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Target className="w-3.5 h-3.5 text-cyan-400" /> Target Horizon
                </span>
                <span className="font-mono text-[10px] text-slate-500 uppercase">
                  {horizon.replace("_", " ")}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-0.5">
                {(
                  [
                    { id: "short_term", label: "Short" },
                    { id: "medium_term", label: "Medium" },
                    { id: "long_term", label: "Long" },
                  ] as const
                ).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setHorizon(item.id)}
                    className={`py-1.5 px-3 rounded-full text-[11px] font-medium transition-all cursor-pointer border flex items-center justify-center ${
                      horizon === item.id
                        ? "bg-cyan-500/15 border-cyan-500/40 text-cyan-300 font-semibold shadow-xs"
                        : "bg-slate-900 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sector Exposure */}
            <div className="p-4 rounded-xl bg-[#090D16] border border-slate-800/90 flex flex-col gap-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" /> Sector Exposure
                </span>
                <span className="font-mono text-[10px] text-slate-500 uppercase">
                  {sectorExposure.replace("_", " ")}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-0.5">
                {(
                  [
                    { id: "tech_heavy", label: "Tech" },
                    { id: "banking_focused", label: "Banking" },
                    { id: "diversified", label: "Diversified" },
                  ] as const
                ).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSectorExposure(item.id)}
                    className={`py-1.5 px-3 rounded-full text-[11px] font-medium transition-all cursor-pointer border flex items-center justify-center ${
                      sectorExposure === item.id
                        ? "bg-indigo-500/15 border-indigo-500/40 text-indigo-300 font-semibold shadow-xs"
                        : "bg-slate-900 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};
