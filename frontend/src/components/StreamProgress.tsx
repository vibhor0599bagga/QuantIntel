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
    <div className="w-full bt-panel p-4 mb-6 border-[#ff9d00]/30 shadow-[0_0_20px_rgba(255,157,0,0.1)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 border-b border-[#1a2333] pb-2">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#ff9d00] animate-pulse" />
          <span className="text-xs font-mono font-bold text-[#ff9d00] uppercase tracking-wider">
            SWARM EXECUTION ENGINE — {ticker}
          </span>
        </div>
        <span className="text-xs font-mono font-bold text-[#00e5ff]">
          {getProgressPercentage()}% COMPLETE
        </span>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-2 bg-[#05080f] rounded overflow-hidden mb-4 border border-[#1e293b]">
        <div
          className="h-full bg-gradient-to-r from-[#ff9d00] via-[#00e5ff] to-[#00ff66] transition-all duration-500 shadow-[0_0_12px_rgba(0,229,255,0.5)]"
          style={{ width: `${getProgressPercentage()}%` }}
        />
      </div>

      {/* Phase Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Phase 1 */}
        <div
          className={`p-3 rounded border font-mono text-xs transition-all ${
            streamState.phase1Complete
              ? "bg-[#00ff66]/10 border-[#00ff66]/40 text-[#e2e8f0]"
              : streamState.currentPhase === 1
              ? "bg-[#ff9d00]/10 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_10px_rgba(255,157,0,0.2)]"
              : "bg-[#080c14] border-[#1e293b] text-[#64748b]"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" /> PHASE 1: DATA SWARM
            </span>
            {streamState.phase1Complete ? (
              <CheckCircle2 className="w-4 h-4 text-[#00ff66]" />
            ) : streamState.currentPhase === 1 ? (
              <Loader2 className="w-4 h-4 text-[#ff9d00] animate-spin" />
            ) : (
              <span className="text-[10px] text-[#475569]">PENDING</span>
            )}
          </div>
          <p className="text-[11px] text-[#94a3b8]">
            Fundamentals, Sentiment, Technicals &amp; Macro agents running in parallel.
          </p>
        </div>

        {/* Phase 2 */}
        <div
          className={`p-3 rounded border font-mono text-xs transition-all ${
            streamState.phase2Complete
              ? "bg-[#00ff66]/10 border-[#00ff66]/40 text-[#e2e8f0]"
              : streamState.currentPhase === 2
              ? "bg-[#ff9d00]/10 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_10px_rgba(255,157,0,0.2)]"
              : "bg-[#080c14] border-[#1e293b] text-[#64748b]"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5" /> PHASE 2: RISK SYNTHESIS
            </span>
            {streamState.phase2Complete ? (
              <CheckCircle2 className="w-4 h-4 text-[#00ff66]" />
            ) : streamState.currentPhase === 2 ? (
              <Loader2 className="w-4 h-4 text-[#ff9d00] animate-spin" />
            ) : (
              <span className="text-[10px] text-[#475569]">PENDING</span>
            )}
          </div>
          <p className="text-[11px] text-[#94a3b8]">
            Quantifying ATR volatility, drawdowns &amp; multi-perspective risk profile.
          </p>
        </div>

        {/* Phase 3 */}
        <div
          className={`p-3 rounded border font-mono text-xs transition-all ${
            streamState.phase3Complete
              ? "bg-[#00ff66]/10 border-[#00ff66]/40 text-[#e2e8f0]"
              : streamState.currentPhase === 3
              ? "bg-[#ff9d00]/10 border-[#ff9d00] text-[#ff9d00] shadow-[0_0_10px_rgba(255,157,0,0.2)]"
              : "bg-[#080c14] border-[#1e293b] text-[#64748b]"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold flex items-center gap-1.5">
              <BrainCircuit className="w-3.5 h-3.5" /> PHASE 3: SUPERVISOR VERDICT
            </span>
            {streamState.phase3Complete ? (
              <CheckCircle2 className="w-4 h-4 text-[#00ff66]" />
            ) : streamState.currentPhase === 3 ? (
              <Loader2 className="w-4 h-4 text-[#ff9d00] animate-spin" />
            ) : (
              <span className="text-[10px] text-[#475569]">PENDING</span>
            )}
          </div>
          <p className="text-[11px] text-[#94a3b8]">
            Deep LLM synthesis, weighted signal resolution &amp; final recommendation.
          </p>
        </div>
      </div>
    </div>
  );
};
