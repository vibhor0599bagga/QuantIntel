"use client";

import React, { useState, useEffect } from "react";
import { Activity, Radio, Key, Terminal as TerminalIcon } from "lucide-react";

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
  { symbol: "AAPL", price: "224.23", change: "+2.85", pct: "+1.29%", up: true },
  { symbol: "MSFT", price: "448.90", change: "-1.20", pct: "-0.27%", up: false },
  { symbol: "GOOGL", price: "178.35", change: "+1.45", pct: "+0.82%", up: true },
  { symbol: "HDFCBANK", price: "1,642.50", change: "+18.90", pct: "+1.16%", up: true },
  { symbol: "NVDA", price: "119.37", change: "+3.42", pct: "+2.95%", up: true },
  { symbol: "BTC/USD", price: "58,420.00", change: "-410.00", pct: "-0.70%", up: false },
  { symbol: "GOLD", price: "2,504.20", change: "+12.80", pct: "+0.51%", up: true },
];

export const Header: React.FC<HeaderProps> = ({ apiStatus, apiUrl, hasApiKey, onOpenKeyModal }) => {
  const [timeStr, setTimeStr] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toUTCString().replace("GMT", "UTC"));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="w-full bg-[#080c14] border-b border-[#1a2333] sticky top-0 z-50">
      {/* Top Banner */}
      <div className="max-w-[1700px] mx-auto px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 border-b border-[#141b27]">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1 bg-[#ff9d00]/10 border border-[#ff9d00]/40 rounded text-[#ff9d00]">
            <TerminalIcon className="w-4 h-4 animate-pulse" />
            <span className="font-bold tracking-wider text-xs">QUANTINTEL TERMINAL v1.0</span>
          </div>
          <span className="hidden md:inline text-xs text-[#94a3b8] font-medium border-l border-[#1e293b] pl-3">
            GEN AI-POWERED MULTI-AGENT PLATFORM FOR FINANCIAL MARKETS
          </span>
        </div>

        {/* Action / Server Connection Indicator */}
        <div className="flex items-center gap-3 text-xs font-mono">
          {/* User OpenRouter Key Trigger */}
          <button
            onClick={onOpenKeyModal}
            className={`flex items-center gap-2 px-3 py-1 rounded border transition-all cursor-pointer ${
              hasApiKey
                ? "bg-[#00ff66]/10 border-[#00ff66]/40 text-[#00ff66] hover:bg-[#00ff66]/20"
                : "bg-[#ff9d00]/15 border-[#ff9d00]/50 text-[#ff9d00] hover:bg-[#ff9d00]/25 animate-pulse"
            }`}
            title="Configure OpenRouter API Key"
          >
            <Key className="w-3.5 h-3.5" />
            <span className="font-bold">
              {hasApiKey ? "KEY: CONFIGURED" : "🔑 SET OPENROUTER KEY"}
            </span>
          </button>

          <div className="flex items-center gap-2 px-2.5 py-1 rounded bg-[#0f172a] border border-[#1e293b]">
            <Radio className={`w-3.5 h-3.5 ${apiStatus === "connected" ? "text-[#00ff66] animate-pulse" : apiStatus === "connecting" ? "text-[#ffd700] animate-spin" : "text-[#ff3333]"}`} />
            <span className="text-[#94a3b8]">API:</span>
            <span className={apiStatus === "connected" ? "text-[#00ff66] font-bold" : apiStatus === "connecting" ? "text-[#ffd700]" : "text-[#ff3333]"}>
              {apiStatus.toUpperCase()}
            </span>
          </div>
          <span className="text-[#64748b] hidden sm:inline">{timeStr}</span>
        </div>
      </div>

      {/* Marquee Ticker Tape */}
      <div className="overflow-hidden bg-[#040609] py-1.5 border-b border-[#121824] flex items-center">
        <div className="flex items-center gap-1 px-3 py-0.5 bg-[#ff9d00]/15 text-[#ff9d00] text-[10px] font-bold tracking-widest shrink-0 border-r border-[#ff9d00]/30 z-10">
          <Activity className="w-3 h-3" /> LIVE MARKETS
        </div>
        <div className="animate-marquee flex items-center gap-8 text-xs font-mono">
          {TICKER_DATA.concat(TICKER_DATA).map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 shrink-0">
              <span className="text-[#e2e8f0] font-bold">{item.symbol}</span>
              <span className="text-[#94a3b8]">{item.price}</span>
              <span className={`font-bold ${item.up ? "text-[#00ff66]" : "text-[#ff3333]"}`}>
                {item.pct}
              </span>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
};
