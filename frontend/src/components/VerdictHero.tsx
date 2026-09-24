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
  const [copied, setCopied] = React.useState(false);

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

  // Trigger confetti on BUY unconditionally at the top level
  useEffect(() => {
    if (recommendationText && stance === "BUY") {
      try {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 },
          colors: ["#00ff66", "#00e5ff", "#ff9d00"],
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
    <div className="w-full bt-panel p-6 sm:p-8 mb-8 relative overflow-hidden border-2 border-[#ff9d00]/50 shadow-[0_0_36px_rgba(255,157,0,0.18)]">
      {/* Background Accent Glow */}
      <div
        className={`absolute -right-24 -top-24 w-80 h-80 rounded-full blur-3xl opacity-20 pointer-events-none ${
          stance === "BUY" ? "bg-[#00ff66]" : stance === "SELL" ? "bg-[#ff3333]" : "bg-[#ffd700]"
        }`}
      />

      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-5 border-b border-[#1a2333] pb-6 mb-6">
        <div className="flex items-center gap-4">
          <div className="p-3.5 bg-[#ff9d00]/15 border border-[#ff9d00]/50 rounded-lg text-[#ff9d00] shadow-[0_0_15px_rgba(255,157,0,0.2)]">
            <Award className="w-7 h-7" />
          </div>
          <div>
            <div className="text-xs font-mono text-[#8492a6] tracking-widest uppercase font-bold mb-1">
              SUPERVISOR DECISION SYNTHESIS
            </div>
            <h2 className="text-xl sm:text-2xl font-mono font-black text-[#e2e8f0] tracking-wide">
              TARGET TICKER: <span className="text-[#00e5ff]">{ticker}</span>
            </h2>
          </div>
        </div>

        {/* Stance Badge & Actions */}
        <div className="flex flex-wrap items-center gap-3.5">
          {conviction && (
            <div className="px-3.5 py-2 rounded-lg bg-[#0d121d] border border-[#1e293b] text-xs font-mono font-bold text-[#8492a6]">
              CONVICTION: <span className="text-[#00e5ff] font-extrabold">{conviction}</span>
            </div>
          )}

          <div
            className={`px-6 py-2.5 rounded-lg text-sm sm:text-base font-mono font-black flex items-center gap-2.5 tracking-wider shadow-lg ${
              stance === "BUY"
                ? "bt-badge-buy"
                : stance === "SELL"
                ? "bt-badge-sell"
                : "bt-badge-hold"
            }`}
          >
            {stance === "BUY" ? (
              <TrendingUp className="w-5 h-5" />
            ) : stance === "SELL" ? (
              <TrendingDown className="w-5 h-5" />
            ) : (
              <MinusCircle className="w-5 h-5" />
            )}
            <span>RECOMMENDATION: {stance}</span>
          </div>

          <button
            onClick={handleCopy}
            className="h-10 px-3.5 bg-[#0f172a] border border-[#1e293b] hover:border-[#ff9d00] text-[#94a3b8] hover:text-[#ff9d00] rounded-lg font-mono text-xs font-bold transition-all flex items-center gap-2 cursor-pointer"
            title="Copy Report"
          >
            {copied ? "COPIED!" : "COPY"}
          </button>

          <button
            onClick={handleDownload}
            className="h-10 px-4 bg-[#0f172a] border border-[#1e293b] hover:border-[#ff9d00] text-[#94a3b8] hover:text-[#ff9d00] rounded-lg font-mono text-xs font-bold transition-all flex items-center gap-2 cursor-pointer shadow-sm"
            title="Download Full Report"
          >
            <Download className="w-4 h-4" />
            <span>EXPORT</span>
          </button>
        </div>
      </div>

      {/* Rationale Body */}
      <div className="font-mono text-xs sm:text-sm leading-relaxed text-[#cbd5e1] whitespace-pre-line bg-[#070b12] p-6 rounded-lg border border-[#1a2333] max-h-[500px] overflow-y-auto space-y-3 selection:bg-[#ff9d00]/30 selection:text-white">
        {recommendationText}
      </div>

      {/* Signal Weights Legend */}
      <div className="mt-5 pt-4 border-t border-[#1a2333] flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-[#8492a6]">
        <span className="font-semibold">WEIGHTING: Fundamentals (40%) | Risk (30%) | Macro (20%) | Tech+Sent (10%)</span>
        <span className="text-[#00e5ff] font-bold">DATE: {tradeDate}</span>
      </div>
    </div>
  );
};
