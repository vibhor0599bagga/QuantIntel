"use client";

import React, { useState, useEffect } from "react";
import { Key, Eye, EyeOff, Check, X, Shield, ExternalLink } from "lucide-react";

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiKey: string;
  onSaveKey: (key: string) => void;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({
  isOpen,
  onClose,
  apiKey,
  onSaveKey,
}) => {
  const [inputKey, setInputKey] = useState(apiKey);
  const [showKey, setShowKey] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    setInputKey(apiKey);
  }, [apiKey, isOpen]);

  if (!isOpen) return null;

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveKey(inputKey.trim());
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 600);
  };

  const handleClear = () => {
    setInputKey("");
    onSaveKey("");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-[#080c14] border border-[#ff9d00]/40 rounded-lg max-w-lg w-full p-6 shadow-[0_0_30px_rgba(255,157,0,0.15)] relative font-mono">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-[#64748b] hover:text-[#e2e8f0] transition-colors p-1"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2.5 bg-[#ff9d00]/15 border border-[#ff9d00]/40 rounded text-[#ff9d00]">
            <Key className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-[#e2e8f0] tracking-wide">
              OPENROUTER API KEY CONFIGURATION
            </h3>
            <p className="text-xs text-[#94a3b8]">
              Supply your personal API key for per-user execution
            </p>
          </div>
        </div>

        {/* Security Notice */}
        <div className="mb-5 p-3 bg-[#0d121d] border border-[#1e293b] rounded flex items-start gap-2.5 text-xs text-[#94a3b8]">
          <Shield className="w-4 h-4 text-[#00ff66] shrink-0 mt-0.5" />
          <span>
            Your API key is stored securely in your browser's local storage and passed directly with your analysis queries. It is never logged on any server.
          </span>
        </div>

        {/* Form */}
        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="block text-xs text-[#e2e8f0] font-bold mb-2">
              OPENROUTER API KEY (<code className="text-[#ff9d00]">sk-or-v1-...</code>)
            </label>
            <div className="relative flex items-center">
              <input
                type={showKey ? "text" : "password"}
                value={inputKey}
                onChange={(e) => setInputKey(e.target.value)}
                placeholder="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                className="w-full bg-[#05080f] border border-[#ff9d00]/40 rounded px-3.5 py-2.5 pr-10 text-[#e2e8f0] text-xs font-mono focus:border-[#ff9d00] focus:outline-none focus:shadow-[0_0_10px_rgba(255,157,0,0.3)] transition-all"
                autoFocus
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 text-[#64748b] hover:text-[#e2e8f0] transition-colors"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-[#64748b] pt-1">
            <a
              href="https://openrouter.ai/keys"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[#00e5ff] hover:underline"
            >
              Get an OpenRouter key <ExternalLink className="w-3 h-3" />
            </a>
            {apiKey && (
              <button
                type="button"
                onClick={handleClear}
                className="text-[#ff3333] hover:underline"
              >
                Remove key
              </button>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#1a2333]">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-[#0f172a] border border-[#1e293b] text-[#94a3b8] hover:text-[#e2e8f0] rounded text-xs font-bold transition-all"
            >
              CANCEL
            </button>
            <button
              type="submit"
              className={`px-5 py-2 rounded text-xs font-extrabold flex items-center gap-1.5 transition-all ${
                savedSuccess
                  ? "bg-[#00ff66] text-black shadow-[0_0_15px_rgba(0,255,102,0.4)]"
                  : "bg-[#ff9d00] text-[#06090e] hover:bg-[#ffb033] shadow-[0_0_15px_rgba(255,157,0,0.4)]"
              }`}
            >
              {savedSuccess ? (
                <>
                  <Check className="w-4 h-4" /> SAVED
                </>
              ) : (
                "SAVE KEY"
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
