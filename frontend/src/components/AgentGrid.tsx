"use client";

import React, { useState } from "react";
import { DollarSign, Newspaper, LineChart, Globe, ShieldAlert, Maximize2, Minimize2, Copy, Check, Sparkles, AlertTriangle } from "lucide-react";

interface AgentGridProps {
  fundamentalsReport: string;
  sentimentReport: string;
  technicalReport: string;
  macroReport: string;
  riskReport: string;
}

interface AgentMeta {
  id: string;
  name: string;
  role: string;
  weight: string;
  icon: React.ElementType;
  accentColor: string;
  borderColor: string;
  bgLight: string;
  report: string;
  description: string;
}

// Helper to extract key signal line from agent report
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

  // Fallback keyword scanning
  const lower = report.toLowerCase();
  if (agentId === "fundamentals") {
    if (lower.includes("undervalued")) return { label: "VALUATION", value: "UNDERVALUED" };
    if (lower.includes("overvalued")) return { label: "VALUATION", value: "OVERVALUED" };
    if (lower.includes("fair")) return { label: "VALUATION", value: "FAIR VALUE" };
  } else if (agentId === "technical") {
    if (lower.includes("bullish")) return { label: "TECHNICAL", value: "BULLISH" };
    if (lower.includes("bearish")) return { label: "TECHNICAL", value: "BEARISH" };
    if (lower.includes("neutral")) return { label: "TECHNICAL", value: "NEUTRAL" };
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
  const [activeTab, setActiveTab] = useState<"all" | "fundamentals" | "sentiment" | "technical" | "macro" | "risk">("all");
  const [modalAgent, setModalAgent] = useState<AgentMeta | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const AGENTS: AgentMeta[] = [
    {
      id: "fundamentals",
      name: "FUNDAMENTALS AGENT",
      role: "Valuation & Financial Health",
      weight: "40% WEIGHT",
      icon: DollarSign,
      accentColor: "#00ff66",
      borderColor: "border-[#00ff66]/30",
      bgLight: "bg-[#00ff66]/10",
      report: fundamentalsReport,
      description: "DCF intrinsic valuation, P/E multiples, operating cash flows & balance sheet health.",
    },
    {
      id: "risk",
      name: "RISK SYNTHESIS AGENT",
      role: "Drawdown & Volatility Guard",
      weight: "30% WEIGHT",
      icon: ShieldAlert,
      accentColor: "#ff3333",
      borderColor: "border-[#ff3333]/30",
      bgLight: "bg-[#ff3333]/10",
      report: riskReport,
      description: "ATR volatility modeling, maximum drawdown stress-tests & aggressive vs conservative risk limits.",
    },
    {
      id: "macro",
      name: "MACRO / REGIME AGENT",
      role: "Economic & Sector Tailwinds",
      weight: "20% WEIGHT",
      icon: Globe,
      accentColor: "#00e5ff",
      borderColor: "border-[#00e5ff]/30",
      bgLight: "bg-[#00e5ff]/10",
      report: macroReport,
      description: "Fed interest rate trajectories, CPI inflation trends & global Risk-On/Risk-Off regime shifts.",
    },
    {
      id: "technical",
      name: "TECHNICAL AGENT",
      role: "Momentum & Price Action",
      weight: "5% WEIGHT",
      icon: LineChart,
      accentColor: "#ff9d00",
      borderColor: "border-[#ff9d00]/30",
      bgLight: "bg-[#ff9d00]/10",
      report: technicalReport,
      description: "Exponential Moving Averages, RSI divergence, MACD histogram crossovers & key pivot levels.",
    },
    {
      id: "sentiment",
      name: "SENTIMENT AGENT",
      role: "Media & Narrative Polarity",
      weight: "5% WEIGHT",
      icon: Newspaper,
      accentColor: "#ffd700",
      borderColor: "border-[#ffd700]/30",
      bgLight: "bg-[#ffd700]/10",
      report: sentimentReport,
      description: "30-day headline sentiment analysis, corporate event catalysts & institutional sentiment tracking.",
    },
  ];

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const filteredAgents = activeTab === "all" ? AGENTS : AGENTS.filter((a) => a.id === activeTab);

  return (
    <div className="w-full mb-10">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Sparkles className="w-4 h-4 text-[#ff9d00]" />
            <h2 className="text-base font-bold tracking-wider text-[#e2e8f0]">
              AUTONOMOUS AGENT INTELLIGENCE DECK
            </h2>
          </div>
          <p className="text-xs text-[#94a3b8]">
            Inspect individual reasoning, telemetry data, and specialized outputs from each sub-agent.
          </p>
        </div>

        {/* Tab Filter Bar */}
        <div className="flex flex-wrap items-center gap-1.5 bg-[#0a0f18] p-1.5 rounded-lg border border-[#1e293b]">
          <button
            onClick={() => setActiveTab("all")}
            className={`px-3 py-1.5 text-xs font-mono font-bold rounded transition-all ${
              activeTab === "all"
                ? "bg-[#ff9d00] text-[#06090e] shadow-md shadow-[#ff9d00]/20"
                : "text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#141d2c]"
            }`}
          >
            ALL AGENTS (5)
          </button>
          {AGENTS.map((agent) => {
            const Icon = agent.icon;
            const isSelected = activeTab === agent.id;
            const hasData = Boolean(agent.report && agent.report.trim());

            return (
              <button
                key={agent.id}
                onClick={() => setActiveTab(agent.id as any)}
                className={`px-3 py-1.5 text-xs font-mono font-bold rounded flex items-center gap-1.5 transition-all ${
                  isSelected
                    ? "bg-[#141d2c] border border-[#ff9d00] text-[#ff9d00]"
                    : "text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#141d2c]"
                }`}
              >
                <Icon className="w-3.5 h-3.5" style={{ color: agent.accentColor }} />
                <span>{agent.name.replace(" AGENT", "")}</span>
                {hasData && (
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00ff66] animate-pulse" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Grid of Agent Cards */}
      <div
        className={`grid gap-6 ${
          activeTab === "all" ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3" : "grid-cols-1"
        }`}
      >
        {filteredAgents.map((agent) => {
          const Icon = agent.icon;
          const hasReport = Boolean(agent.report && agent.report.trim());
          const signal = extractSignal(agent.report, agent.id);

          return (
            <div
              key={agent.id}
              className={`bg-[#0b0f19] border ${
                hasReport ? agent.borderColor : "border-[#1e293b]"
              } rounded-xl flex flex-col transition-all hover:border-[#ff9d00]/50 shadow-lg relative overflow-hidden`}
            >
              {/* Top Accent Line */}
              <div
                className="h-1 w-full"
                style={{
                  backgroundColor: hasReport ? agent.accentColor : "#1e293b",
                }}
              />

              {/* Card Header */}
              <div className="p-5 border-b border-[#162032] flex items-start justify-between gap-4 bg-[#0e1422]">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center border"
                    style={{
                      backgroundColor: `${agent.accentColor}15`,
                      borderColor: `${agent.accentColor}40`,
                    }}
                  >
                    <Icon className="w-5 h-5" style={{ color: agent.accentColor }} />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[#e2e8f0] tracking-wide flex items-center gap-2">
                      {agent.name}
                    </h3>
                    <p className="text-xs text-[#94a3b8] font-mono">{agent.role}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[11px] font-mono font-semibold px-2.5 py-1 bg-[#06090e] text-[#94a3b8] rounded-md border border-[#1e293b]">
                    {agent.weight}
                  </span>
                  {hasReport && (
                    <button
                      onClick={() => setModalAgent(agent)}
                      className="p-1.5 text-[#64748b] hover:text-[#ff9d00] hover:bg-[#162032] rounded-md transition-colors"
                      title="Expand Report"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>

              {/* Description Strip */}
              <div className="px-5 py-2.5 bg-[#080c14] border-b border-[#141b27] text-xs font-mono text-[#64748b]">
                {agent.description}
              </div>

              {/* Key Signal Highlight Strip if available */}
              {signal && (
                <div className="px-5 py-2.5 bg-[#101726] border-b border-[#1a2538] flex items-center justify-between">
                  <span className="text-[11px] font-mono text-[#94a3b8] tracking-wider uppercase">
                    {signal.label}
                  </span>
                  <span
                    className="text-xs font-mono font-bold px-2.5 py-0.5 rounded border"
                    style={{
                      backgroundColor: `${agent.accentColor}20`,
                      color: agent.accentColor,
                      borderColor: `${agent.accentColor}50`,
                    }}
                  >
                    {signal.value}
                  </span>
                </div>
              )}

              {/* Report Body */}
              <div className="p-5 flex-1 flex flex-col justify-between">
                {hasReport ? (
                  <div className="relative">
                    <div className="font-mono text-xs text-[#cbd5e1] leading-relaxed max-h-[260px] overflow-y-auto pr-2 custom-scrollbar whitespace-pre-line">
                      {agent.report}
                    </div>
                  </div>
                ) : (
                  <div className="py-14 text-center text-[#475569] flex flex-col items-center justify-center gap-3">
                    <div className="w-8 h-8 rounded-full border border-dashed border-[#334155] flex items-center justify-center animate-spin">
                      <div className="w-2 h-2 rounded-full bg-[#475569]" />
                    </div>
                    <span className="text-xs font-mono animate-pulse">Awaiting swarm telemetry...</span>
                  </div>
                )}

                {/* Footer Action */}
                {hasReport && (
                  <div className="mt-4 pt-3 border-t border-[#162032] flex items-center justify-between">
                    <button
                      onClick={() => handleCopy(agent.id, agent.report)}
                      className="text-xs font-mono text-[#94a3b8] hover:text-[#ff9d00] flex items-center gap-1.5 transition-colors"
                    >
                      {copiedId === agent.id ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-[#00ff66]" />
                          <span className="text-[#00ff66]">COPIED</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>COPY REPORT</span>
                        </>
                      )}
                    </button>

                    <button
                      onClick={() => setModalAgent(agent)}
                      className="text-xs font-mono text-[#ff9d00] hover:underline flex items-center gap-1"
                    >
                      READ FULL ANALYSIS &rarr;
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Expanded Modal for In-Depth Reading */}
      {modalAgent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-[#0b0f19] border border-[#ff9d00] rounded-xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="p-5 bg-[#0e1422] border-b border-[#1e293b] flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center border"
                  style={{
                    backgroundColor: `${modalAgent.accentColor}15`,
                    borderColor: `${modalAgent.accentColor}40`,
                  }}
                >
                  <modalAgent.icon className="w-5 h-5" style={{ color: modalAgent.accentColor }} />
                </div>
                <div>
                  <h3 className="text-base font-bold text-[#e2e8f0] tracking-wide">
                    {modalAgent.name}
                  </h3>
                  <p className="text-xs text-[#94a3b8] font-mono">{modalAgent.role} &bull; {modalAgent.weight}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleCopy("modal", modalAgent.report)}
                  className="px-3 py-1.5 text-xs font-mono bg-[#162032] hover:bg-[#1e2d44] text-[#e2e8f0] rounded border border-[#2d3f58] flex items-center gap-1.5 transition-colors"
                >
                  {copiedId === "modal" ? <Check className="w-3.5 h-3.5 text-[#00ff66]" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedId === "modal" ? "COPIED" : "COPY REPORT"}
                </button>
                <button
                  onClick={() => setModalAgent(null)}
                  className="p-1.5 text-[#94a3b8] hover:text-[#ff3333] hover:bg-[#162032] rounded-md transition-colors"
                >
                  <Minimize2 className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6 flex-1 overflow-y-auto custom-scrollbar font-mono text-sm text-[#cbd5e1] leading-relaxed whitespace-pre-line bg-[#06090e]">
              {modalAgent.report}
            </div>

            {/* Modal Footer */}
            <div className="p-4 bg-[#080c14] border-t border-[#141b27] flex items-center justify-between text-xs text-[#64748b]">
              <span>QUANTINTEL SUB-AGENT TELEMETRY REPORT</span>
              <button
                onClick={() => setModalAgent(null)}
                className="px-4 py-1.5 bg-[#ff9d00] text-[#06090e] font-bold font-mono rounded hover:bg-[#e08b00] transition-colors"
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
