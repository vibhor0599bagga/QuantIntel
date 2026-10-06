"use client";

import React, { useState, useEffect } from "react";
import { Activity, Radio, Key, Zap, Clock, ShieldCheck, ChevronRight } from "lucide-react";

interface HeaderProps {
  apiStatus: "connected" | "connecting" | "offline";
  apiUrl: string;
  hasApiKey: boolean;
  onOpenKeyModal: () => void;
}

const TICKER_DATA = [
  { symbol: "NIFTY 50", price: "24,852.10", change: "+142.30", pct: "+0.58%", up: true },
  { symbol: "S&P 500", price: "5,648.40", change: "+32.10", pct: "+0.57%", up: true },
  { symbol: "NASDAQ", price: "17,713.78", change: "+114.30", pct: "+0.65%", up: true },
  { symbol: "NVDA", price: "119.37", change: "+3.42", pct: "+2.95%", up: true },
  { symbol: "AAPL", price: "224.23", change: "+2.85", pct: "+1.29%", up: true },
  { symbol: "MSFT", price: "448.90", change: "-1.20", pct: "-0.27%", up: false },
  { symbol: "GOOGL", price: "178.35", change: "+1.45", pct: "+0.82%", up: true },
  { symbol: "TSLA", price: "254.80", change: "+8.12", pct: "+3.29%", up: true },
  { symbol: "BTC/USD", price: "64,280.00", change: "+1,420.00", pct: "+2.26%", up: true },
  { symbol: "GOLD", price: "2,504.20", change: "+12.80", pct: "+0.51%", up: true },
];

export const Header: React.FC<HeaderProps> = ({ apiStatus, apiUrl, hasApiKey, onOpenKeyModal }) => {
  const [timeStr, setTimeStr] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleTimeString("en-US", {
          timeZone: "UTC",
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }) + " UTC"
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="w-full bg-[#080C14]/90 backdrop-blur-xl border-b border-white/[0.07] sticky top-0 z-40 transition-all">
      {/* Top Navbar */}
      <div className="max-w-[1700px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Left: Branding & Tagline */}
        <div className="flex items-center gap-3.5">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
              <Zap className="w-3.5 h-3.5 fill-amber-400/20" />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm sm:text-base tracking-tight text-white font-sans">
                QUANT<span className="text-amber-400">INTEL</span>
              </span>
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-slate-800/80 border border-slate-700/60 text-slate-400 tracking-wide">
                v1.0
              </span>
            </div>
          </div>

          <div className="hidden xl:flex items-center gap-2 pl-3.5 border-l border-slate-800 text-xs text-slate-400">
            <span>Institutional Multi-Agent Intelligence</span>
          </div>
        </div>

        {/* Right: Status Badges, API Key, Time */}
        <div className="flex items-center gap-2.5 sm:gap-3 text-xs">
          {/* OpenRouter API Key Button */}
          <button
            onClick={onOpenKeyModal}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-medium transition-all duration-200 cursor-pointer ${
              hasApiKey
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
                : "bg-amber-500/10 border-amber-500/40 text-amber-400 hover:bg-amber-500/20 animate-pulse shadow-[0_0_12px_rgba(245,158,11,0.2)]"
            }`}
            title="Configure OpenRouter API Key"
          >
            <Key className="w-3.5 h-3.5" />
            <span className="font-mono text-[11px] font-semibold">
              {hasApiKey ? "API KEY: ACTIVE" : "SET API KEY"}
            </span>
          </button>

          {/* Backend API Live Status */}
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/80 border border-slate-800">
            <span className="relative flex h-2 w-2">
              {apiStatus === "connected" && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  apiStatus === "connected"
                    ? "bg-emerald-400"
                    : apiStatus === "connecting"
                    ? "bg-amber-400"
                    : "bg-rose-500"
                }`}
              ></span>
            </span>
            <span className="text-[11px] font-mono text-slate-400 hidden sm:inline">BACKEND:</span>
            <span
              className={`text-[11px] font-mono font-bold uppercase ${
                apiStatus === "connected"
                  ? "text-emerald-400"
                  : apiStatus === "connecting"
                  ? "text-amber-400"
                  : "text-rose-400"
              }`}
            >
              {apiStatus}
            </span>
          </div>

          {/* UTC Clock */}
          <div className="hidden md:flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-slate-900/50 border border-slate-800/80 text-slate-400 font-mono text-[11px]">
            <Clock className="w-3 h-3 text-slate-500" />
            <span>{timeStr || "00:00:00 UTC"}</span>
          </div>
        </div>
      </div>

      {/* Slim Live Market Ticker Tape */}
      <div className="w-full bg-[#05070C] border-t border-b border-white/[0.04] py-1.5 px-4 overflow-hidden relative flex items-center">
        <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-900 border border-slate-800 text-slate-400 text-[10px] font-mono font-bold tracking-wider shrink-0 z-10 mr-4 rounded-full">
          <Activity className="w-3 h-3 text-emerald-400" /> MARKETS
        </div>

        <div className="animate-marquee flex items-center gap-8 text-xs font-mono select-none">
          {TICKER_DATA.concat(TICKER_DATA).map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 shrink-0">
              <span className="text-slate-300 font-semibold">{item.symbol}</span>
              <span className="text-slate-500 text-[11px]">{item.price}</span>
              <span
                className={`text-[11px] font-bold ${
                  item.up ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {item.pct}
              </span>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
};
