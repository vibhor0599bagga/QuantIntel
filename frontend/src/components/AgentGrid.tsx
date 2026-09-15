"use client";

import React, { useState } from "react";
import { DollarSign, Newspaper, LineChart, Globe, ShieldAlert, Maximize2, Minimize2 } from "lucide-react";

interface AgentGridProps {
  fundamentalsReport: string;
  sentimentReport: string;
  technicalReport: string;
  macroReport: string;
  riskReport: string;
}

export const AgentGrid: React.FC<AgentGridProps> = ({
  fundamentalsReport,
  sentimentReport,
  technicalReport,
  macroReport,
  riskReport,
}) => {
  const [activeTab, setActiveTab] = useState<"all" | "fundamentals" | "sentiment" | "technical" | "macro" | "risk">("all");
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

  const AGENTS = [
    {
      id: "fundamentals",
      name: "FUNDAMENTALS AGENT",
      weight: "40% WEIGHT",
      icon: DollarSign,
      color: "#00ff66",
      report: fundamentalsReport,
      description: "Intrinsic valuation, P/E ratios, cash flows & balance sheet health.",
    },
    {
      id: "risk",
      name: "RISK SYNTHESIS AGENT",
      weight: "30% WEIGHT",
      icon: ShieldAlert,
      color: "#ff3333",
      report: riskReport,
      description: "ATR volatility, drawdown limits & aggressive/conservative perspectives.",
    },
    {
      id: "macro",
      name: "MACRO / REGIME AGENT",
      weight: "20% WEIGHT",
      icon: Globe,
      color: "#00e5ff",
      report: macroReport,
      description: "Federal Reserve policy, inflation dynamics & Risk-On/Off environment.",
    },
    {
      id: "technical",
      name: "TECHNICAL AGENT",
      weight: "5% WEIGHT",
      icon: LineChart,
      color: "#ff9d00",
      report: technicalReport,
      description: "Moving averages, RSI momentum, support/resistance price levels.",
    },
    {
      id: "sentiment",
      name: "SENTIMENT AGENT",
      weight: "5% WEIGHT",
      icon: Newspaper,
      color: "#ffd700",
      report: sentimentReport,
      description: "30-day headline sentiment, catalysts & market narrative polarity.",
    },
  ];

  const toggleExpand = (id: string) => {
    setExpandedCard(expandedCard === id ? null : id);
  };

  const filteredAgents = activeTab === "all" ? AGENTS : AGENTS.filter((a) => a.id === activeTab);

  return (
    <div className="w-full mb-8">
      {/* Tab Filter Bar */}
      <div className="flex flex-wrap items-center gap-2 mb-4 bg-[#080c14] p-1.5 rounded border border-[#1a2333]">
        <button
          onClick={() => setActiveTab("all")}
          className={`px-3 py-1.5 text-xs font-mono font-bold rounded transition-all ${
            activeTab === "all"
              ? "bg-[#ff9d00] text-[#06090e]"
              : "text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#0f172a]"
          }`}
        >
          ALL AGENTS (5)
        </button>
        {AGENTS.map((agent) => {
          const Icon = agent.icon;
          return (
            <button
              key={agent.id}
              onClick={() => setActiveTab(agent.id as any)}
              className={`px-3 py-1.5 text-xs font-mono font-bold rounded flex items-center gap-1.5 transition-all ${
                activeTab === agent.id
                  ? "bg-[#0f172a] border border-[#ff9d00] text-[#ff9d00]"
                  : "text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#0f172a]"
              }`}
            >
              <Icon className="w-3.5 h-3.5" style={{ color: agent.color }} />
              {agent.name.replace(" AGENT", "")}
            </button>
          );
        })}
      </div>

      {/* Grid of Agent Cards */}
      <div
        className={`grid gap-4 ${
          activeTab === "all" ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3" : "grid-cols-1"
        }`}
      >
        {filteredAgents.map((agent) => {
          const Icon = agent.icon;
          const isExpanded = expandedCard === agent.id;
          const hasReport = Boolean(agent.report && agent.report.trim());

          return (
            <div
              key={agent.id}
              className={`bt-panel flex flex-col transition-all ${
                isExpanded ? "md:col-span-2 lg:col-span-3 border-[#ff9d00]" : ""
              }`}
            >
              {/* Header */}
              <div className="bt-header">
                <div className="flex items-center gap-2">
                  <Icon className="w-4 h-4" style={{ color: agent.color }} />
                  <span className="text-[#e2e8f0] font-bold">{agent.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[#64748b] font-mono px-2 py-0.5 bg-[#080c14] rounded border border-[#1e293b]">
                    {agent.weight}
                  </span>
                  <button
                    onClick={() => toggleExpand(agent.id)}
                    className="p-1 text-[#64748b] hover:text-[#ff9d00] transition-colors"
                    title={isExpanded ? "Collapse" : "Expand"}
                  >
                    {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* Sub-description */}
              <div className="px-4 py-2 bg-[#080c14] border-b border-[#141b27] text-[11px] font-mono text-[#64748b]">
                {agent.description}
              </div>

              {/* Content Body */}
              <div className="p-4 flex-1 font-mono text-xs text-[#cbd5e1] leading-relaxed">
                {hasReport ? (
                  <div className={`whitespace-pre-line ${isExpanded ? "" : "max-h-[250px] overflow-y-auto"}`}>
                    {agent.report}
                  </div>
                ) : (
                  <div className="py-8 text-center text-[#475569] italic flex flex-col items-center gap-2">
                    <span className="animate-pulse">Awaiting swarm data...</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
