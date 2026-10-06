"use client";

import React, { useEffect, useState } from "react";
import { Award, TrendingUp, TrendingDown, MinusCircle, Download, Copy, Check, Sparkles, ShieldCheck } from "lucide-react";
import confetti from "canvas-confetti";

interface VerdictHeroProps {
  ticker: string;
  tradeDate: string;
  recommendationText: string;
}

export const VerdictHero: React.FC<VerdictHeroProps> = ({
  ticker,
  tradeDate,
  recommendationText,
}) => {
  const [copied, setCopied] = useState(false);

  // Parsing logic for stance & conviction
  const upperText = (recommendationText || "").toUpperCase();
  let stance: "BUY" | "SELL" | "HOLD" = "HOLD";
  if (upperText.includes("BUY") && !upperText.includes("DO NOT BUY")) {
    stance = "BUY";
  } else if (upperText.includes("SELL")) {
    stance = "SELL";
  }

  // Extract conviction if available
  const convictionMatch = upperText.match(/CONVICTION\s*(?:LEVEL)?:\s*([0-9\.]+)/i);
  const conviction = convictionMatch ? convictionMatch[1] : null;

  useEffect(() => {
    if (recommendationText && stance === "BUY") {
      try {
        confetti({
          particleCount: 60,
          spread: 60,
          origin: { y: 0.6 },
          colors: ["#10B981", "#06B6D4", "#F59E0B"],
        });
      } catch (err) {
        console.warn("Confetti error:", err);
      }
    }
  }, [stance, recommendationText]);

  if (!recommendationText) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(recommendationText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob(
      [`QUANTINTEL SUPERVISOR REPORT FOR ${ticker} (${tradeDate})\n\n` + recommendationText],
      { type: "text/plain" }
    );
    element.href = URL.createObjectURL(file);
    element.download = `QUANTINTEL_${ticker}_REPORT_${tradeDate}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="w-full glass-panel p-6 sm:p-7 mb-6 relative overflow-hidden border-white/[0.1] shadow-2xl animate-in fade-in duration-300">
      {/* Background Subtle Radial Glow */}
      <div
        className={`absolute -right-16 -top-16 w-72 h-72 rounded-full blur-3xl opacity-15 pointer-events-none ${
          stance === "BUY"
            ? "bg-emerald-500"
            : stance === "SELL"
            ? "bg-rose-500"
            : "bg-amber-500"
        }`}
      />

      {/* Top Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-white/[0.08]">
        {/* Left: Ticker & Title */}
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 shadow-md shadow-amber-500/10">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-amber-400" /> Executive Verdict Synthesis
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white font-sans flex items-center gap-2">
              <span>{ticker}</span>
              <span className="text-xs font-mono font-medium px-2.5 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-slate-400">
                {tradeDate}
              </span>
            </h2>
          </div>
        </div>

        {/* Right: Recommendation Badge & Actions */}
        <div className="flex flex-wrap items-center gap-3">
          {conviction && (
            <div className="px-4 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-mono font-medium text-slate-300">
              Conviction: <strong className="text-cyan-400">{conviction}</strong>
            </div>
          )}

          <div
            className={`px-5 py-2 rounded-full text-xs sm:text-sm font-bold flex items-center justify-center gap-2.5 tracking-wide shadow-lg ${
              stance === "BUY"
                ? "bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 shadow-emerald-500/10"
                : stance === "SELL"
                ? "bg-rose-500/15 border border-rose-500/40 text-rose-300 shadow-rose-500/10"
                : "bg-amber-500/15 border border-amber-500/40 text-amber-300 shadow-amber-500/10"
            }`}
          >
            {stance === "BUY" ? (
              <TrendingUp className="w-4 h-4" />
            ) : stance === "SELL" ? (
              <TrendingDown className="w-4 h-4" />
            ) : (
              <MinusCircle className="w-4 h-4" />
            )}
            <span>VERDICT: {stance}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="h-9 px-4.5 bg-slate-900/90 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-full text-xs font-medium transition-all flex items-center justify-center gap-2 cursor-pointer"
              title="Copy markdown report"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>

            <button
              onClick={handleDownload}
              className="h-9 px-4.5 bg-slate-900/90 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-full text-xs font-medium transition-all flex items-center justify-center gap-2 cursor-pointer"
              title="Download text report"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </button>
          </div>
        </div>
      </div>

      {/* Rationale Content Body with Clean Readable Typography */}
      <div className="mt-5 p-5 rounded-xl bg-slate-950/70 border border-white/[0.06] text-slate-300 text-xs sm:text-sm leading-relaxed font-sans max-h-[460px] overflow-y-auto whitespace-pre-line selection:bg-amber-500/20 selection:text-white">
        {recommendationText}
      </div>

      {/* Footer Weights Bar */}
      <div className="mt-4 pt-3 border-t border-white/[0.06] flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-500">
        <span>Allocation Weights: Fundamentals (40%) • Risk (30%) • Macro (20%) • Tech+Sentiment (10%)</span>
        <span className="flex items-center gap-1 text-slate-400">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> Risk-Balanced Synthesis
        </span>
      </div>
    </div>
  );
};
