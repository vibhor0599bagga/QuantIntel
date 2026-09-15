"use client";

import React, { useState } from "react";
import { Terminal, ChevronDown, ChevronUp, Copy, Check } from "lucide-react";

interface RawTerminalProps {
  logs: string[];
}

export const RawTerminal: React.FC<RawTerminalProps> = ({ logs }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(logs.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (logs.length === 0) return null;

  return (
    <div className="w-full bt-panel mb-8 border-[#1e293b]">
      {/* Header */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="bt-header cursor-pointer hover:bg-[#161f2e] transition-colors"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#00e5ff]" />
          <span className="text-[#00e5ff] font-bold">LIVE STREAMING TERMINAL LOGS ({logs.length})</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            className="px-2 py-0.5 text-[10px] bg-[#080c14] border border-[#1e293b] hover:border-[#00e5ff] text-[#94a3b8] hover:text-[#00e5ff] rounded flex items-center gap-1 font-mono"
          >
            {copied ? <Check className="w-3 h-3 text-[#00ff66]" /> : <Copy className="w-3 h-3" />}
            {copied ? "COPIED" : "COPY LOGS"}
          </button>
          {isOpen ? <ChevronUp className="w-4 h-4 text-[#64748b]" /> : <ChevronDown className="w-4 h-4 text-[#64748b]" />}
        </div>
      </div>

      {/* Logs Window */}
      {isOpen && (
        <div className="p-4 bg-[#040609] font-mono text-xs text-[#00ff66] max-h-[300px] overflow-y-auto leading-relaxed border-t border-[#121824]">
          {logs.map((log, idx) => (
            <div key={idx} className="mb-1 flex items-start gap-2">
              <span className="text-[#64748b] shrink-0 select-none">[{idx + 1}]</span>
              <span className="break-all">{log}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
