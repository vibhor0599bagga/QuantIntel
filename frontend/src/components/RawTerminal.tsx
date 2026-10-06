"use client";

import React, { useState, useEffect, useRef } from "react";
import { Terminal, ChevronDown, ChevronUp, Copy, Check, Trash2, Shield, CircleDot } from "lucide-react";

interface RawTerminalProps {
  logs: string[];
}

export const RawTerminal: React.FC<RawTerminalProps> = ({ logs }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (isOpen && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, isOpen]);

  const handleCopy = () => {
    navigator.clipboard.writeText(logs.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (logs.length === 0) return null;

  return (
    <div className="w-full glass-panel overflow-hidden mb-8 transition-all">
      {/* Drawer Header */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="px-5 py-3.5 bg-slate-950/80 cursor-pointer hover:bg-slate-900/60 transition-colors flex items-center justify-between"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Terminal className="w-3.5 h-3.5" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold text-slate-200">
              Live Swarm Event Stream
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-cyan-400">
              {logs.length} events
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            className="px-2.5 py-1 text-[11px] font-mono bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? "Copied" : "Copy Logs"}</span>
          </button>

          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setIsOpen(!isOpen);
            }}
            className="p-1 text-slate-400 hover:text-white rounded-lg transition-colors"
          >
            {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Logs Window */}
      {isOpen && (
        <div className="p-4 bg-[#05070C] font-mono text-xs max-h-[280px] overflow-y-auto leading-relaxed border-t border-slate-800/80 space-y-1">
          {logs.map((log, idx) => {
            const isError = log.includes("[ERROR]") || log.includes("Error") || log.includes("failed");
            const isInit = log.includes("[INIT]") || log.includes("Target Ticker");
            const isEvent = log.includes("EVENT") || log.includes("Agent");

            let textColor = "text-slate-300";
            if (isError) textColor = "text-rose-400 font-semibold";
            else if (isInit) textColor = "text-amber-400 font-medium";
            else if (isEvent) textColor = "text-cyan-400";

            return (
              <div key={idx} className={`flex items-start gap-2.5 py-0.5 ${textColor}`}>
                <span className="text-slate-600 select-none text-[10px] w-5 text-right shrink-0">
                  {idx + 1}
                </span>
                <span className="break-all">{log}</span>
              </div>
            );
          })}
          <div ref={logsEndRef} />
        </div>
      )}
    </div>
  );
};
