"use client";

import React from "react";
import { CheckCircle2, Loader2, Cpu, ShieldAlert, BrainCircuit, Activity } from "lucide-react";

export interface StreamState {
  isAnalyzing: boolean;
  currentPhase: 0 | 1 | 2 | 3;
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

  const pct = getProgressPercentage();

  return (
    <div className="w-full glass-panel p-5 mb-6 border-amber-500/20 shadow-xl animate-in fade-in duration-300">
      {/* Top Header & Percentage */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-amber-400">
            <Activity className={`w-3.5 h-3.5 ${streamState.isAnalyzing ? "animate-pulse" : ""}`} />
          </div>
          <div>
            <span className="text-xs font-mono font-bold text-amber-400 tracking-wide uppercase">
              Swarm Execution Pipeline
            </span>
            <span className="text-xs text-slate-400 ml-2 font-mono">
              Target: <strong className="text-white">{ticker}</strong>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {streamState.isAnalyzing && (
            <span className="text-xs font-mono text-slate-400 animate-pulse hidden sm:inline">
              Synthesizing Signals...
            </span>
          )}
          <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-cyan-400">
            {pct}%
          </span>
        </div>
      </div>

      {/* Sleek Smooth Progress Bar */}
      <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden mb-4 border border-slate-800/80">
        <div
          className="h-full bg-gradient-to-r from-amber-500 via-cyan-500 to-emerald-400 transition-all duration-500 ease-out rounded-full shadow-[0_0_12px_rgba(6,182,212,0.4)]"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* 3 Pipeline Steppers */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Phase 1 */}
        <div
          className={`p-3.5 rounded-xl border text-xs transition-all duration-200 ${
            streamState.phase1Complete
              ? "bg-emerald-500/10 border-emerald-500/30 text-slate-200"
              : streamState.currentPhase === 1
              ? "bg-amber-500/10 border-amber-500/40 text-amber-300 ring-1 ring-amber-500/30 shadow-lg shadow-amber-500/5"
              : "bg-slate-950/50 border-slate-800/60 text-slate-500"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-amber-400" />
              <span>Phase 1: Parallel Swarm</span>
            </span>
            {streamState.phase1Complete ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : streamState.currentPhase === 1 ? (
              <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
            ) : (
              <span className="text-[10px] font-mono text-slate-600 font-semibold">WAITING</span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            Parallel extraction: Fundamentals, Technicals, Sentiment &amp; Macro.
          </p>
        </div>

        {/* Phase 2 */}
        <div
          className={`p-3.5 rounded-xl border text-xs transition-all duration-200 ${
            streamState.phase2Complete
              ? "bg-emerald-500/10 border-emerald-500/30 text-slate-200"
              : streamState.currentPhase === 2
              ? "bg-amber-500/10 border-amber-500/40 text-amber-300 ring-1 ring-amber-500/30 shadow-lg shadow-amber-500/5"
              : "bg-slate-950/50 border-slate-800/60 text-slate-500"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>Phase 2: Risk Synthesis</span>
            </span>
            {streamState.phase2Complete ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : streamState.currentPhase === 2 ? (
              <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
            ) : (
              <span className="text-[10px] font-mono text-slate-600 font-semibold">WAITING</span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            Quantifying max drawdown, ATR volatility &amp; scenario exposure.
          </p>
        </div>

        {/* Phase 3 */}
        <div
          className={`p-3.5 rounded-xl border text-xs transition-all duration-200 ${
            streamState.phase3Complete
              ? "bg-emerald-500/10 border-emerald-500/30 text-slate-200"
              : streamState.currentPhase === 3
              ? "bg-amber-500/10 border-amber-500/40 text-amber-300 ring-1 ring-amber-500/30 shadow-lg shadow-amber-500/5"
              : "bg-slate-950/50 border-slate-800/60 text-slate-500"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-bold flex items-center gap-1.5">
              <BrainCircuit className="w-3.5 h-3.5 text-cyan-400" />
              <span>Phase 3: Decision Engine</span>
            </span>
            {streamState.phase3Complete ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : streamState.currentPhase === 3 ? (
              <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
            ) : (
              <span className="text-[10px] font-mono text-slate-600 font-semibold">WAITING</span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            Supervisor weighted resolution &amp; final strategic recommendation.
          </p>
        </div>
      </div>
    </div>
  );
};
