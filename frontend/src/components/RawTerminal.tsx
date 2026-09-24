"use client";

import React, { useState, useEffect, useRef } from "react";
import { Terminal, ChevronDown, ChevronUp, Copy, Check, Trash2, Radio } from "lucide-react";

interface RawTerminalProps {
  logs: string[];
}

export const RawTerminal: React.FC<RawTerminalProps> = ({ logs }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (isOpen && autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, isOpen, autoScroll]);

  const handleCopy = () => {
    navigator.clipboard.writeText(logs.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (logs.length === 0) return null;

  return (
    <div className="w-full bg-[#080c14] border border-[#1e293b] rounded-xl overflow-hidden mb-10 shadow-lg">
      {/* Header */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="px-5 py-4 bg-[#0e1422] border-b border-[#1a2538] cursor-pointer hover:bg-[#121a2c] transition-colors flex items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-[#00e5ff]/10 border border-[#00e5ff]/30 flex items-center justify-center">
            <Terminal className="w-4 h-4 text-[#00e5ff]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-[#e2e8f0]">LIVE STREAMING TERMINAL CONSOLE</span>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/30">
                {logs.length} EVENTS
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            className="px-3 py-1.5 text-xs bg-[#080c14] border border-[#1e293b] hover:border-[#00e5ff] text-[#94a3b8] hover:text-[#00e5ff] rounded-md flex items-center gap-1.5 font-mono transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-[#00ff66]" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? "COPIED" : "COPY LOGS"}
          </button>
          
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsOpen(!isOpen);
            }}
            className="p-1.5 text-[#94a3b8] hover:text-[#e2e8f0] rounded-md hover:bg-[#1a2538]"
          >
            {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Logs Window */}
      {isOpen && (
        <div className="p-5 bg-[#040609] font-mono text-xs text-[#00ff66] max-h-[320px] overflow-y-auto leading-relaxed border-t border-[#121824] custom-scrollbar">
          {logs.map((log, idx) => {
            const isError = log.includes("[ERROR]") || log.includes("Error") || log.includes("failed");
            const isInit = log.includes("[INIT]") || log.includes("Target Ticker");
            const isEvent = log.includes("EVENT");

            let textColor = "text-[#00ff66]";
            if (isError) textColor = "text-[#ff3333]";
            else if (isInit) textColor = "text-[#ff9d00]";
            else if (isEvent) textColor = "text-[#00e5ff]";

            return (
              <div key={idx} className={`mb-1.5 flex items-start gap-3 ${textColor}`}>
                <span className="text-[#475569] shrink-0 select-none font-semibold">
                  {String(idx + 1).padStart(2, "0")}
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
