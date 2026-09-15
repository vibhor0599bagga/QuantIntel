"use client";

import React, { useEffect } from "react";
import { Award, AlertTriangle, CheckCircle, TrendingUp, TrendingDown, MinusCircle, Download, Share2 } from "lucide-react";
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
  if (!recommendationText) return null;

  // Simple parsing logic for stance & conviction
  const upperText = recommendationText.toUpperCase();
  let stance: "BUY" | "SELL" | "HOLD" = "HOLD";
  if (upperText.includes("BUY") && !upperText.includes("DO NOT BUY")) {
    stance = "BUY";
  } else if (upperText.includes("SELL")) {
    stance = "SELL";
  }

  // Trigger confetti on BUY
  useEffect(() => {
    if (stance === "BUY") {
      confetti({
        particleCount: 70,
        spread: 60,
        origin: { y: 0.6 },
        colors: ["#00ff66", "#00e5ff", "#ff9d00"],
      });
    }
  }, [stance]);

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([`QUANTINTEL SUPERVISOR REPORT FOR ${ticker} (${tradeDate})\n\n` + recommendationText], {
      type: "text/plain",
    });
    element.href = URL.createObjectURL(file);
    element.download = `QUANTINTEL_${ticker}_REPORT_${tradeDate}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="w-full bt-panel p-6 mb-6 relative overflow-hidden border-2 border-[#ff9d00]/40 shadow-[0_0_30px_rgba(255,157,0,0.15)]">
      {/* Background Accent Glow */}
      <div
        className={`absolute -right-20 -top-20 w-64 h-64 rounded-full blur-3xl opacity-20 pointer-events-none ${
          stance === "BUY" ? "bg-[#00ff66]" : stance === "SELL" ? "bg-[#ff3333]" : "bg-[#ffd700]"
        }`}
      />

      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1a2333] pb-4 mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#ff9d00]/10 border border-[#ff9d00]/40 rounded text-[#ff9d00]">
            <Award className="w-6 h-6" />
          </div>
          <div>
            <div className="text-[11px] font-mono text-[#64748b] tracking-widest uppercase">
              SUPERVISOR DECISION SYNTHESIS
            </div>
            <h2 className="text-xl font-mono font-bold text-[#e2e8f0]">
              TARGET TICKER: <span className="text-[#00e5ff]">{ticker}</span>
            </h2>
          </div>
        </div>

        {/* Stance Badge & Download */}
        <div className="flex items-center gap-3">
          <div
            className={`px-5 py-2 rounded text-base font-mono font-extrabold flex items-center gap-2 ${
              stance === "BUY"
                ? "bt-badge-buy text-lg"
                : stance === "SELL"
                ? "bt-badge-sell text-lg"
                : "bt-badge-hold text-lg"
            }`}
          >
            {stance === "BUY" ? (
              <TrendingUp className="w-5 h-5" />
            ) : stance === "SELL" ? (
              <TrendingDown className="w-5 h-5" />
            ) : (
              <MinusCircle className="w-5 h-5" />
            )}
            RECOMMENDATION: {stance}
          </div>

          <button
            onClick={handleDownload}
            className="p-2 bg-[#0f172a] border border-[#1e293b] hover:border-[#ff9d00] text-[#94a3b8] hover:text-[#ff9d00] rounded font-mono text-xs transition-colors flex items-center gap-1.5"
            title="Download Full Report"
          >
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">EXPORT</span>
          </button>
        </div>
      </div>

      {/* Rationale Body */}
      <div className="font-mono text-xs leading-relaxed text-[#cbd5e1] whitespace-pre-line bg-[#080c14] p-5 rounded border border-[#1a2333] max-h-[400px] overflow-y-auto">
        {recommendationText}
      </div>

      {/* Signal Weights Legend */}
      <div className="mt-4 pt-3 border-t border-[#1a2333] flex flex-wrap items-center justify-between text-[11px] font-mono text-[#64748b]">
        <span>WEIGHTING: Fundamentals (40%) | Risk (30%) | Macro (20%) | Tech+Sent (10%)</span>
        <span className="text-[#00e5ff]">DATE: {tradeDate}</span>
      </div>
    </div>
  );
};
