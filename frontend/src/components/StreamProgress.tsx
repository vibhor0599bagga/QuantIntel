"use client";

import React from "react";
import { CheckCircle2, Loader2, Cpu, ShieldAlert, BrainCircuit, Terminal } from "lucide-react";

export interface StreamState {
  isAnalyzing: boolean;
  currentPhase: 0 | 1 | 2 | 3; // 0 = idle, 1 = phase1, 2 = phase2, 3 = phase3/complete
  phase1Complete: boolean;
  phase2Complete: boolean;
  phase3Complete: boolean;
  error?: string;
  logs: string[];
}

interface StreamProgressProps {
  streamState: StreamState;
  ticker: string;
}

export const StreamProgress: React.FC<StreamProgressProps> = ({ streamState, ticker }) => {
  if (!streamState.isAnalyzing && streamState.currentPhase === 0) return null;

  const getProgressPercentage = () => {
    if (streamState.phase3Complete) return 100;
    if (streamState.phase2Complete) return 85;
    if (streamState.phase1Complete) return 60;
    if (streamState.isAnalyzing) return 25;
    return 0;
  };

  return (
    <div className="w-full bt-panel p-6 mb-8 border-[#ff9d00]/30 shadow-[0_0_24px_rgba(255,157,0,0.12)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 border-b border-[#1a2333] pb-3">
        <div className="flex items-center gap-2.5">
          <Terminal className="w-5 h-5 text-[#ff9d00] animate-pulse" />
          <span className="text-sm font-mono font-extrabold text-[#ff9d00] uppercase tracking-wider">
            SWARM EXECUTION ENGINE — {ticker}
          </span>
        </div>
        <span className="text-sm font-mono font-black text-[#00e5ff] tracking-wide">
          {getProgressPercentage()}% COMPLETE
        </span>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-3 bg-[#05080f] rounded-full overflow-hidden mb-6 border border-[#1e293b]">
        <div
          className="h-full bg-gradient-to-r from-[#ff9d00] via-[#00e5ff] to-[#00ff66] transition-all duration-500 shadow-[0_0_14px_rgba(0,229,255,0.6)]"
          style={{ width: `${getProgressPercentage()}%` }}
        />
      </div>

      {/* Phase Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Phase 1 */}
        <div
          className={`p-4 rounded-lg border font-mono text-xs transition-all ${
            streamState.phase1Complete
              ? "bg-[#00ff66]/10 border-[#00ff66]/50 text-[#e2e8f0] shadow-[0_0_12px_rgba(0,255,102,0.15)]"
              : streamState.currentPhase === 1
              ? "bg-[#ff9d00]/15 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_15px_rgba(255,157,0,0.25)] animate-pulse"
              : "bg-[#080c14] border-[#1e293b] text-[#64748b]"
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="font-extrabold text-xs sm:text-sm flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#ff9d00]" /> 01 DATA SWARM
            </span>
            {streamState.phase1Complete ? (
              <CheckCircle2 className="w-5 h-5 text-[#00ff66]" />
            ) : streamState.currentPhase === 1 ? (
              <Loader2 className="w-5 h-5 text-[#ff9d00] animate-spin" />
            ) : (
              <span className="text-[11px] text-[#475569] font-bold">QUEUED</span>
            )}
          </div>
          <p className="text-xs text-[#94a3b8] leading-relaxed">
            Fundamentals, Sentiment, Technicals &amp; Macro agents running in parallel.
          </p>
        </div>

        {/* Phase 2 */}
        <div
          className={`p-4 rounded-lg border font-mono text-xs transition-all ${
            streamState.phase2Complete
              ? "bg-[#00ff66]/10 border-[#00ff66]/50 text-[#e2e8f0] shadow-[0_0_12px_rgba(0,255,102,0.15)]"
              : streamState.currentPhase === 2
              ? "bg-[#ff9d00]/15 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_15px_rgba(255,157,0,0.25)] animate-pulse"
              : "bg-[#080c14] border-[#1e293b] text-[#64748b]"
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="font-extrabold text-xs sm:text-sm flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-[#ff3333]" /> 02 RISK SYNTHESIS
            </span>
            {streamState.phase2Complete ? (
              <CheckCircle2 className="w-5 h-5 text-[#00ff66]" />
            ) : streamState.currentPhase === 2 ? (
              <Loader2 className="w-5 h-5 text-[#ff9d00] animate-spin" />
            ) : (
              <span className="text-[11px] text-[#475569] font-bold">QUEUED</span>
            )}
          </div>
          <p className="text-xs text-[#94a3b8] leading-relaxed">
            Quantifying ATR volatility, drawdowns &amp; multi-perspective risk profile.
          </p>
        </div>

        {/* Phase 3 */}
        <div
          className={`p-4 rounded-lg border font-mono text-xs transition-all ${
            streamState.phase3Complete
              ? "bg-[#00ff66]/10 border-[#00ff66]/50 text-[#e2e8f0] shadow-[0_0_12px_rgba(0,255,102,0.15)]"
              : streamState.currentPhase === 3
              ? "bg-[#ff9d00]/15 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_15px_rgba(255,157,0,0.25)] animate-pulse"
              : "bg-[#080c14] border-[#1e293b] text-[#64748b]"
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="font-extrabold text-xs sm:text-sm flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-[#00e5ff]" /> 03 SUPERVISOR VERDICT
            </span>
            {streamState.phase3Complete ? (
              <CheckCircle2 className="w-5 h-5 text-[#00ff66]" />
            ) : streamState.currentPhase === 3 ? (
              <Loader2 className="w-5 h-5 text-[#ff9d00] animate-spin" />
            ) : (
              <span className="text-[11px] text-[#475569] font-bold">QUEUED</span>
            )}
          </div>
          <p className="text-xs text-[#94a3b8] leading-relaxed">
            Deep LLM synthesis, weighted signal resolution &amp; final recommendation.
          </p>
        </div>
      </div>
    </div>
  );
};
