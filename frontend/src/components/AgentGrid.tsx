"use client";

import React, { useState } from "react";
import {
  DollarSign,
  Newspaper,
  LineChart,
  Globe,
  ShieldAlert,
  Maximize2,
  Minimize2,
  Copy,
  Check,
  Sparkles,
  LayoutGrid,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Info,
  X,
  ShieldCheck,
  Cpu,
} from "lucide-react";

interface AgentGridProps {
  fundamentalsReport: string;
  sentimentReport: string;
  technicalReport: string;
  macroReport: string;
  riskReport: string;
}

interface AgentMeta {
  id: "fundamentals" | "macro" | "technical" | "sentiment" | "risk";
  name: string;
  role: string;
  phase: "Phase 1: Research Swarm" | "Phase 2: Risk Guard";
  weight: string;
  icon: React.ElementType;
  accentColor: string;
  badgeBg: string;
  badgeText: string;
  report: string;
  description: string;
}

// Extract key signal badge from report
const extractSignal = (report: string, agentId: string) => {
  if (!report) return null;

  const lines = report.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (
      trimmed.startsWith("VALUATION_SIGNAL:") ||
      trimmed.startsWith("OVERALL_SENTIMENT:") ||
      trimmed.startsWith("SIGNAL:") ||
      trimmed.startsWith("RECOMMENDED_ACTION:") ||
      trimmed.startsWith("REGIME_SIGNAL:") ||
      trimmed.startsWith("MACRO_STANCE:") ||
      trimmed.startsWith("RISK_ASSESSMENT:")
    ) {
      const parts = trimmed.split(":");
      return {
        label: parts[0].replace(/_/g, " "),
        value: parts.slice(1).join(":").trim(),
      };
    }
  }

  const lower = report.toLowerCase();
  if (agentId === "fundamentals") {
    if (lower.includes("undervalued")) return { label: "VALUATION", value: "UNDERVALUED" };
    if (lower.includes("overvalued")) return { label: "VALUATION", value: "OVERVALUED" };
    if (lower.includes("fair")) return { label: "VALUATION", value: "FAIR VALUE" };
  } else if (agentId === "macro") {
    if (lower.includes("favorable") || lower.includes("tailwinds")) return { label: "REGIME", value: "TAILWINDS" };
    if (lower.includes("headwinds") || lower.includes("restrictive")) return { label: "REGIME", value: "HEADWINDS" };
    if (lower.includes("neutral")) return { label: "REGIME", value: "NEUTRAL" };
  } else if (agentId === "technical") {
    if (lower.includes("bullish")) return { label: "MOMENTUM", value: "BULLISH" };
    if (lower.includes("bearish")) return { label: "MOMENTUM", value: "BEARISH" };
    if (lower.includes("neutral")) return { label: "MOMENTUM", value: "NEUTRAL" };
  } else if (agentId === "sentiment") {
    if (lower.includes("positive") || lower.includes("optimistic")) return { label: "SENTIMENT", value: "BULLISH" };
    if (lower.includes("negative") || lower.includes("pessimistic")) return { label: "SENTIMENT", value: "BEARISH" };
  } else if (agentId === "risk") {
    if (lower.includes("high risk")) return { label: "RISK LEVEL", value: "HIGH RISK" };
    if (lower.includes("moderate risk") || lower.includes("medium risk")) return { label: "RISK LEVEL", value: "MODERATE RISK" };
    if (lower.includes("low risk")) return { label: "RISK LEVEL", value: "LOW RISK" };
  }

  return null;
};

export const AgentGrid: React.FC<AgentGridProps> = ({
  fundamentalsReport,
  sentimentReport,
  technicalReport,
  macroReport,
  riskReport,
}) => {
  const [activeTab, setActiveTab] = useState<"matrix" | "fundamentals" | "macro" | "technical" | "sentiment" | "risk">("matrix");
  const [modalAgent, setModalAgent] = useState<AgentMeta | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // EXACT ORDER: First 4 Data Swarm Agents -> Then Risk Synthesis Agent
  const AGENTS: AgentMeta[] = [
    {
      id: "fundamentals",
      name: "Fundamentals Agent",
      role: "Intrinsic Valuation & Financials",
      phase: "Phase 1: Research Swarm",
      weight: "40% Weight",
      icon: DollarSign,
      accentColor: "emerald",
      badgeBg: "bg-emerald-500/10 border-emerald-500/30",
      badgeText: "text-emerald-400",
      report: fundamentalsReport,
      description: "DCF modeling, P/E multiples, balance sheet liquidity & earnings growth quality.",
    },
    {
      id: "macro",
      name: "Macro Regime Agent",
      role: "Monetary Policy & Yields",
      phase: "Phase 1: Research Swarm",
      weight: "20% Weight",
      icon: Globe,
      accentColor: "indigo",
      badgeBg: "bg-indigo-500/10 border-indigo-500/30",
      badgeText: "text-indigo-400",
      report: macroReport,
      description: "Federal Reserve rates, treasury yield curves, CPI inflation & credit spreads.",
    },
    {
      id: "technical",
      name: "Technicals Agent",
      role: "Trend, Momentum & Support/Resistance",
      phase: "Phase 1: Research Swarm",
      weight: "5% Weight",
      icon: LineChart,
      accentColor: "cyan",
      badgeBg: "bg-cyan-500/10 border-cyan-500/30",
      badgeText: "text-cyan-400",
      report: technicalReport,
      description: "SMA 50/200 breakouts, RSI oscillator overbought/oversold levels & MACD momentum.",
    },
    {
      id: "sentiment",
      name: "Sentiment Agent",
      role: "News Tone & Social Momentum",
      phase: "Phase 1: Research Swarm",
      weight: "5% Weight",
      icon: Newspaper,
      accentColor: "amber",
      badgeBg: "bg-amber-500/10 border-amber-500/30",
      badgeText: "text-amber-400",
      report: sentimentReport,
      description: "Financial press tone analysis, earnings call sentiment & market narrative indexing.",
    },
    {
      id: "risk",
      name: "Risk Synthesis Agent",
      role: "Volatility & Tail Drawdown Guard",
      phase: "Phase 2: Risk Guard",
      weight: "30% Weight",
      icon: ShieldAlert,
      accentColor: "rose",
      badgeBg: "bg-rose-500/10 border-rose-500/30",
      badgeText: "text-rose-400",
      report: riskReport,
      description: "ATR volatility, historical drawdowns, beta sensitivity & loss containment parameters.",
    },
  ];

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const hasAnyReport = AGENTS.some((a) => !!a.report);
  if (!hasAnyReport) return null;

  return (
    <div className="w-full mb-8">
      {/* Section Header */}
      <div className="flex items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400">
            1. Multi-Agent Swarm Intelligence &amp; Risk Guard
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-slate-500">
            5 Agents
          </span>
        </div>
      </div>

      {/* Tab Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2 p-1.5 bg-slate-950/80 border border-slate-800 rounded-full overflow-x-auto no-scrollbar">
          <button
            onClick={() => setActiveTab("matrix")}
            className={`px-4 py-1.5 rounded-full text-xs font-medium flex items-center gap-2 transition-all cursor-pointer ${
              activeTab === "matrix"
                ? "bg-slate-800 text-white font-semibold shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <LayoutGrid className="w-3.5 h-3.5 text-amber-400" />
            <span>Matrix Overview</span>
          </button>

          {AGENTS.map((agent, idx) => {
            const Icon = agent.icon;
            const isSelected = activeTab === agent.id;
            const isReady = !!agent.report;

            return (
              <button
                key={agent.id}
                onClick={() => setActiveTab(agent.id)}
                className={`px-4 py-1.5 rounded-full text-xs font-medium flex items-center gap-2 transition-all cursor-pointer ${
                  isSelected
                    ? "bg-slate-800 text-white font-semibold shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                } ${agent.id === "risk" ? "border-l border-slate-800 pl-4.5 ml-1" : ""}`}
              >
                <Icon className={`w-3.5 h-3.5 ${agent.badgeText}`} />
                <span>
                  {idx + 1}. {agent.name.replace(" Agent", "")}
                </span>
                {isReady && (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 ml-0.5" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* MATRIX VIEW (Ordered: 4 Swarm Agents first, then Risk Synthesis) */}
      {activeTab === "matrix" ? (
        <div className="space-y-4">
          {/* Phase 1: 4 Research Swarm Agents */}
          <div>
            <div className="text-[11px] font-mono text-slate-400 font-semibold mb-2.5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Phase 1 — Parallel Research Swarm (4 Agents)</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3.5">
              {AGENTS.slice(0, 4).map((agent, idx) => {
                const Icon = agent.icon;
                const signal = extractSignal(agent.report, agent.id);
                const hasData = !!agent.report;

                return (
                  <div
                    key={agent.id}
                    className="glass-panel p-4 flex flex-col justify-between hover:border-slate-700 transition-all duration-200"
                  >
                    <div>
                      {/* Card Header */}
                      <div className="flex items-start justify-between gap-2 mb-2.5">
                        <div className="flex items-center gap-2">
                          <div className={`w-8 h-8 rounded-lg ${agent.badgeBg} border flex items-center justify-center ${agent.badgeText}`}>
                            <Icon className="w-3.5 h-3.5" />
                          </div>
                          <div>
                            <h3 className="font-bold text-xs text-white font-sans">
                              {idx + 1}. {agent.name}
                            </h3>
                            <p className="text-[10px] text-slate-400 truncate max-w-[130px]">{agent.role}</p>
                          </div>
                        </div>

                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                          {agent.weight}
                        </span>
                      </div>

                      {/* Signal Badge */}
                      {signal ? (
                        <div className="mb-2.5 p-2 rounded-lg bg-slate-950/70 border border-slate-800/80 flex items-center justify-between">
                          <span className="text-[9px] font-mono uppercase text-slate-500">
                            {signal.label}:
                          </span>
                          <span className={`text-[11px] font-mono font-bold ${agent.badgeText}`}>
                            {signal.value}
                          </span>
                        </div>
                      ) : hasData ? (
                        <div className="mb-2.5 p-1.5 rounded-lg bg-slate-950/50 text-[10px] font-mono text-slate-500">
                          Report Generated
                        </div>
                      ) : (
                        <div className="mb-2.5 p-1.5 rounded-lg bg-slate-950/30 text-[10px] font-mono text-slate-600">
                          Awaiting stream...
                        </div>
                      )}

                      {/* Snippet preview */}
                      <p className="text-[11px] text-slate-400 line-clamp-3 leading-relaxed mb-3">
                        {hasData ? agent.report : agent.description}
                      </p>
                    </div>

                    {/* Bottom Actions */}
                    <div className="pt-2.5 border-t border-slate-800/80 flex items-center justify-between">
                      <button
                        type="button"
                        onClick={() => setActiveTab(agent.id)}
                        className="text-[11px] font-medium text-amber-400 hover:text-amber-300 flex items-center gap-1 transition-colors cursor-pointer"
                      >
                        <span>Deep-Dive</span>
                        <ChevronRight className="w-3 h-3" />
                      </button>

                      {hasData && (
                        <button
                          type="button"
                          onClick={() => setModalAgent(agent)}
                          className="p-1 rounded-md bg-slate-900 border border-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer"
                          title="Fullscreen modal"
                        >
                          <Maximize2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Phase 2: Risk Synthesis Agent Card */}
          {(() => {
            const riskAgent = AGENTS[4];
            const Icon = riskAgent.icon;
            const signal = extractSignal(riskAgent.report, riskAgent.id);
            const hasData = !!riskAgent.report;

            return (
              <div>
                <div className="text-[11px] font-mono text-rose-400 font-semibold mb-2.5 flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                  <span>Phase 2 — Risk Synthesis &amp; Drawdown Containment Guard</span>
                </div>
                <div className="glass-panel p-5 border-rose-500/20 bg-rose-950/10 hover:border-rose-500/40 transition-all duration-200">
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-rose-500/15 border border-rose-500/40 flex items-center justify-center text-rose-400 shadow-md shadow-rose-500/10">
                        <Icon className="w-5 h-5" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-bold text-sm text-white font-sans">
                            5. {riskAgent.name}
                          </h3>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/15 border border-rose-500/30 text-rose-300 font-semibold">
                            {riskAgent.weight}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400">{riskAgent.role}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      {signal && (
                        <div className="px-3.5 py-1.5 rounded-full bg-slate-950/80 border border-rose-500/30 text-xs font-mono font-bold text-rose-300">
                          {signal.label}: {signal.value}
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => setActiveTab(riskAgent.id)}
                        className="px-4 py-1.5 rounded-full bg-slate-900 border border-slate-800 hover:border-rose-500/40 text-xs font-semibold text-rose-300 hover:text-white transition-colors cursor-pointer flex items-center justify-center gap-2"
                      >
                        <span>Full Risk Breakdown</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed font-sans line-clamp-3 bg-slate-950/60 p-3.5 rounded-xl border border-white/[0.04]">
                    {hasData ? riskAgent.report : riskAgent.description}
                  </p>
                </div>
              </div>
            );
          })()}
        </div>
      ) : (
        /* INDIVIDUAL AGENT DEEP DIVE VIEW */
        (() => {
          const agent = AGENTS.find((a) => a.id === activeTab);
          if (!agent) return null;
          const Icon = agent.icon;
          const signal = extractSignal(agent.report, agent.id);

          return (
            <div className="glass-panel p-6 sm:p-7 border-slate-800 animate-in fade-in duration-200">
              {/* Top Banner */}
              <div className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-white/[0.08]">
                <div className="flex items-center gap-3.5">
                  <div className={`w-11 h-11 rounded-xl ${agent.badgeBg} border flex items-center justify-center ${agent.badgeText}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono uppercase px-2.5 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-slate-400">
                        {agent.phase}
                      </span>
                      <h2 className="text-lg font-bold text-white font-sans">
                        {agent.name}
                      </h2>
                      <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-slate-400">
                        {agent.weight}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">{agent.role}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2.5">
                  {signal && (
                    <div className={`px-3.5 py-1.5 rounded-full ${agent.badgeBg} border text-xs font-mono font-bold ${agent.badgeText}`}>
                      {signal.label}: {signal.value}
                    </div>
                  )}

                  {agent.report && (
                    <>
                      <button
                        onClick={() => handleCopy(agent.id, agent.report)}
                        className="h-9 px-4 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-full text-xs font-medium transition-all flex items-center justify-center gap-2 cursor-pointer"
                      >
                        {copiedId === agent.id ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                        <span>{copiedId === agent.id ? "Copied" : "Copy"}</span>
                      </button>

                      <button
                        onClick={() => setModalAgent(agent)}
                        className="h-9 px-4 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-full text-xs font-medium transition-all flex items-center justify-center gap-2 cursor-pointer"
                        title="Fullscreen view"
                      >
                        <Maximize2 className="w-3.5 h-3.5" />
                        <span>Fullscreen</span>
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Report Body */}
              <div className="mt-5 p-5 rounded-xl bg-slate-950/70 border border-white/[0.06] text-slate-300 text-xs sm:text-sm leading-relaxed font-sans max-h-[500px] overflow-y-auto whitespace-pre-line selection:bg-amber-500/20 selection:text-white">
                {agent.report || (
                  <div className="py-12 text-center text-slate-500 font-mono text-xs">
                    No report received yet for {agent.name}. Execute an analysis to view findings.
                  </div>
                )}
              </div>
            </div>
          );
        })()
      )}

      {/* FULLSCREEN AGENT MODAL */}
      {modalAgent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
          <div className="bg-[#0B0F19] border border-slate-800 rounded-2xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="p-5 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-9 h-9 rounded-lg ${modalAgent.badgeBg} border flex items-center justify-center ${modalAgent.badgeText}`}>
                  {React.createElement(modalAgent.icon, { className: "w-4 h-4" })}
                </div>
                <div>
                  <h3 className="font-bold text-white text-base font-sans">{modalAgent.name}</h3>
                  <p className="text-xs text-slate-400">{modalAgent.role} ({modalAgent.weight})</p>
                </div>
              </div>

              <button
                onClick={() => setModalAgent(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto text-slate-200 text-xs sm:text-sm leading-relaxed font-sans whitespace-pre-line selection:bg-amber-500/20 selection:text-white">
              {modalAgent.report}
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between">
              <span className="text-xs font-mono text-slate-500">QuantIntel Swarm Agent Output</span>
              <button
                onClick={() => handleCopy(modalAgent.id, modalAgent.report)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 cursor-pointer"
              >
                {copiedId === modalAgent.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copiedId === modalAgent.id ? "Copied" : "Copy Report"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
