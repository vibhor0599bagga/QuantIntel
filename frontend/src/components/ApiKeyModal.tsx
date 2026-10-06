"use client";

import React, { useState, useEffect } from "react";
import { Key, Eye, EyeOff, Check, X, Shield, ExternalLink, Sparkles } from "lucide-react";

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="bg-[#0B0F19] border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl relative font-sans">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
            <Key className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">OpenRouter API Key</h3>
            <p className="text-xs text-slate-400">Configure your personal key for LLM inference</p>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSave} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-300">
              API Key (sk-or-v1-...)
            </label>
            <div className="relative flex items-center">
              <input
                type={showKey ? "text" : "password"}
                value={inputKey}
                onChange={(e) => setInputKey(e.target.value)}
                placeholder="sk-or-v1-..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 pr-10 text-xs font-mono text-slate-200 focus:outline-none focus:border-amber-500/60 focus:ring-2 focus:ring-amber-500/15"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 text-slate-500 hover:text-slate-300"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
            <p className="flex items-center gap-1.5 text-slate-300 font-medium">
              <Shield className="w-3.5 h-3.5 text-emerald-400" /> Client-Side Storage
            </p>
            <p>
              Your key is saved locally in your browser and used securely to run the multi-agent swarm without storing on external servers.
            </p>
          </div>

          <div className="flex items-center justify-between pt-2">
            <a
              href="https://openrouter.ai/keys"
              target="_blank"
              rel="noreferrer"
              className="text-xs text-amber-400 hover:underline flex items-center gap-1 font-medium"
            >
              <span>Get OpenRouter Key</span>
              <ExternalLink className="w-3 h-3" />
            </a>

            <div className="flex items-center gap-2.5">
              {apiKey && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-4 py-2 text-xs font-medium text-rose-400 hover:bg-rose-500/10 rounded-full transition-colors cursor-pointer"
                >
                  Clear Key
                </button>
              )}
              <button
                type="submit"
                className="px-6 py-2 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs rounded-full transition-all shadow-md shadow-amber-500/20 flex items-center justify-center gap-2 cursor-pointer"
              >
                {savedSuccess ? <Check className="w-4 h-4" /> : null}
                <span>{savedSuccess ? "Saved!" : "Save Key"}</span>
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
