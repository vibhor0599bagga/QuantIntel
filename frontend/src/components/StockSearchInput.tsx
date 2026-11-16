"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import { Search, X, TrendingUp, Sparkles, Building2, Globe2, ArrowRight, Loader2 } from "lucide-react";
import { StockItem, searchStocks, getStockBySymbol, POPULAR_STOCKS } from "@/data/stockDatabase";

interface StockSearchInputProps {
  value: string;
  onChange: (ticker: string) => void;
  onSubmit?: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export const StockSearchInput: React.FC<StockSearchInputProps> = ({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = "Search ticker or company (e.g. AAPL, NVDA, RELIANCE, Apple)...",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const [isLoadingApi, setIsLoadingApi] = useState(false);
  const [apiResults, setApiResults] = useState<StockItem[]>([]);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Load recent searches on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem("quantintel_recent_stocks");
      if (stored) {
        setRecentSearches(JSON.parse(stored).slice(0, 5));
      }
    } catch {
      // ignore
    }
  }, []);

  const saveToRecent = (symbol: string) => {
    try {
      const s = symbol.trim().toUpperCase();
      const next = [s, ...recentSearches.filter((item) => item !== s)].slice(0, 5);
      setRecentSearches(next);
      localStorage.setItem("quantintel_recent_stocks", JSON.stringify(next));
    } catch {
      // ignore
    }
  };

  // 1. Get instant local matches
  const localResults = useMemo(() => {
    return searchStocks(value, 8);
  }, [value]);

  // 2. Fetch external recommendations debounced if query is 2+ characters
  useEffect(() => {
    const trimmed = value.trim();
    if (trimmed.length < 2) {
      setApiResults([]);
      setIsLoadingApi(false);
      return;
    }

    let active = true;
    const timeoutId = setTimeout(async () => {
      setIsLoadingApi(true);
      try {
        const res = await fetch(`/api/tickers/search?q=${encodeURIComponent(trimmed)}`);
        if (res.ok) {
          const data = await res.json();
          if (active && data.results && Array.isArray(data.results)) {
            setApiResults(data.results);
          }
        }
      } catch {
        // graceful fallback to local results
      } finally {
        if (active) setIsLoadingApi(false);
      }
    }, 250);

    return () => {
      active = false;
      clearTimeout(timeoutId);
    };
  }, [value]);

  // Merge local & API results (local prioritised, duplicates removed)
  const recommendations = useMemo(() => {
    const map = new Map<string, StockItem>();
    for (const item of localResults) {
      map.set(item.symbol.toUpperCase(), item);
    }
    for (const item of apiResults) {
      const key = item.symbol.toUpperCase();
      if (!map.has(key)) {
        map.set(key, item);
      }
    }
    return Array.from(map.values()).slice(0, 8);
  }, [localResults, apiResults]);

  // Reset highlight index when recommendations change
  useEffect(() => {
    setHighlightedIndex(0);
  }, [value, recommendations.length]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Scroll active item into view
  useEffect(() => {
    if (!isOpen || !listRef.current) return;
    const element = listRef.current.children[highlightedIndex] as HTMLElement;
    if (element) {
      element.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [highlightedIndex, isOpen]);

  const selectStock = (item: StockItem) => {
    onChange(item.symbol.toUpperCase());
    saveToRecent(item.symbol.toUpperCase());
    setIsOpen(false);
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        setIsOpen(true);
        e.preventDefault();
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev + 1) % Math.max(1, recommendations.length));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev - 1 + recommendations.length) % Math.max(1, recommendations.length));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (recommendations.length > 0 && recommendations[highlightedIndex]) {
        selectStock(recommendations[highlightedIndex]);
      } else if (value.trim()) {
        saveToRecent(value.trim().toUpperCase());
        setIsOpen(false);
        if (onSubmit) onSubmit();
      }
    } else if (e.key === "Escape") {
      setIsOpen(false);
    } else if (e.key === "Tab" && recommendations.length > 0 && isOpen) {
      selectStock(recommendations[highlightedIndex]);
    }
  };

  const selectedStockDetails = useMemo(() => {
    return getStockBySymbol(value);
  }, [value]);

  // Highlight matched substrings in symbol or name
  const renderHighlighted = (text: string, query: string) => {
    const q = query.trim();
    if (!q || !text) return text;
    const idx = text.toLowerCase().indexOf(q.toLowerCase());
    if (idx === -1) return text;
    const before = text.substring(0, idx);
    const match = text.substring(idx, idx + q.length);
    const after = text.substring(idx + q.length);
    return (
      <>
        {before}
        <span className="text-amber-400 font-bold bg-amber-500/15 px-0.5 rounded">{match}</span>
        {after}
      </>
    );
  };

  return (
    <div ref={containerRef} className="relative flex-1 min-w-[200px]">
      {/* Ticker Search Box Input */}
      <div
        className={`w-full h-9 sm:h-[38px] flex items-center bg-[#090D16] border rounded-full px-3.5 sm:px-4 transition-all ${
          isOpen
            ? "border-amber-500/70 ring-1 ring-amber-500/40 shadow-[0_0_15px_rgba(245,158,11,0.15)]"
            : "border-slate-800/90 hover:border-slate-700 focus-within:border-amber-500/50 focus-within:ring-1 focus-within:ring-amber-500/30"
        }`}
      >
        <div className="flex items-center shrink-0 select-none mr-2">
          <Search className="w-3 h-3 text-slate-500 mr-1.5 shrink-0" />
          <span className="text-[10px] font-mono font-bold tracking-wider text-slate-400 uppercase">
            TICKER
          </span>
        </div>

        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value.toUpperCase());
            if (!isOpen) setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          spellCheck={false}
          className="w-full bg-transparent text-white font-mono font-bold text-xs sm:text-[13px] tracking-wide focus:outline-none uppercase placeholder:text-slate-600 placeholder:font-normal placeholder:tracking-normal placeholder:text-xs"
        />

        {/* Selected Stock Company Preview or Clear Button */}
        <div className="flex items-center gap-1.5 shrink-0 ml-1">
          {isLoadingApi && (
            <Loader2 className="w-3 h-3 text-amber-400 animate-spin shrink-0" />
          )}

          {selectedStockDetails && !isOpen && (
            <span className="hidden xl:inline-flex text-[11px] font-sans font-medium text-slate-400 truncate max-w-[130px] border-l border-slate-800 pl-2">
              {selectedStockDetails.name}
            </span>
          )}

          {value && !disabled && (
            <button
              type="button"
              onClick={() => {
                onChange("");
                inputRef.current?.focus();
                setIsOpen(true);
              }}
              className="p-1 text-slate-500 hover:text-slate-200 hover:bg-slate-800/60 rounded-full transition-colors cursor-pointer"
              title="Clear input"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Autocomplete / Recommendations Dropdown (Weather App Style) */}
      {isOpen && (
        <div className="absolute top-[calc(100%+6px)] left-0 w-full sm:min-w-[420px] sm:max-w-[480px] z-50 bg-[#090D16]/98 backdrop-blur-xl border border-slate-700/80 rounded-2xl shadow-2xl shadow-black/90 ring-1 ring-white/10 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150">
          {/* Dropdown Header */}
          <div className="px-3.5 py-2 border-b border-slate-800/80 flex items-center justify-between text-[11px] font-medium bg-slate-900/40">
            <div className="flex items-center gap-1.5 text-slate-400">
              {value.trim() ? (
                <>
                  <Search className="w-3 h-3 text-amber-400" />
                  <span>
                    Recommendations for <span className="text-white font-mono font-bold">"{value.trim()}"</span>
                  </span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3 h-3 text-amber-400" />
                  <span>Popular & Trending Equities</span>
                </>
              )}
            </div>

            <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
              {recommendations.length > 0 && (
                <span>{recommendations.length} {recommendations.length === 1 ? "match" : "matches"}</span>
              )}
            </div>
          </div>

          {/* Recent Searches Pills (if any and query is empty) */}
          {!value.trim() && recentSearches.length > 0 && (
            <div className="px-3.5 py-2 border-b border-slate-800/60 bg-slate-950/40 flex items-center gap-1.5 overflow-x-auto no-scrollbar">
              <span className="text-[10px] text-slate-500 shrink-0">Recent:</span>
              {recentSearches.map((rec) => (
                <button
                  key={rec}
                  type="button"
                  onClick={() => {
                    onChange(rec);
                    setIsOpen(false);
                  }}
                  className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:border-amber-500/40 hover:bg-slate-800 transition-colors cursor-pointer"
                >
                  {rec}
                </button>
              ))}
            </div>
          )}

          {/* Recommendations List */}
          <div ref={listRef} className="max-h-[300px] sm:max-h-[340px] overflow-y-auto py-1 divide-y divide-slate-800/40">
            {recommendations.length > 0 ? (
              recommendations.map((stock, index) => {
                const isSelected = index === highlightedIndex;
                const isCurrentTicker = stock.symbol.toUpperCase() === value.trim().toUpperCase();

                // Exchange badge colors
                const exchangeBadgeClass =
                  stock.exchange === "NSE" || stock.exchange === "BSE"
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    : stock.exchange === "CRYPTO"
                    ? "bg-purple-500/10 border-purple-500/30 text-purple-400"
                    : stock.exchange === "ETF" || stock.exchange === "INDEX"
                    ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-400"
                    : "bg-cyan-500/10 border-cyan-500/30 text-cyan-400";

                const countryBadge =
                  stock.country === "IN" ? "🇮🇳 IN" : stock.country === "US" ? "🇺🇸 US" : "🌐";

                return (
                  <div
                    key={`${stock.symbol}-${index}`}
                    onClick={() => selectStock(stock)}
                    onMouseEnter={() => setHighlightedIndex(index)}
                    className={`px-3.5 py-2.5 flex items-center justify-between gap-3 cursor-pointer transition-all ${
                      isSelected
                        ? "bg-amber-500/15 border-l-2 border-amber-500 pl-3"
                        : "hover:bg-slate-800/40 pl-3.5"
                    }`}
                  >
                    {/* Left: Ticker & Company Name */}
                    <div className="flex items-center gap-2.5 min-w-0">
                      {/* Ticker Symbol Pill */}
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-mono font-bold tracking-wide border shrink-0 ${
                          isCurrentTicker
                            ? "bg-amber-500/25 border-amber-500 text-amber-300"
                            : "bg-slate-900 border-slate-700/80 text-white"
                        }`}
                      >
                        {renderHighlighted(stock.symbol, value)}
                      </span>

                      {/* Company Name & Sector */}
                      <div className="flex flex-col min-w-0">
                        <span className="text-xs font-medium text-slate-200 truncate">
                          {renderHighlighted(stock.name, value)}
                        </span>
                        <span className="text-[10px] text-slate-400 truncate flex items-center gap-1">
                          <span>{stock.sector}</span>
                        </span>
                      </div>
                    </div>

                    {/* Right: Exchange Badge & Country */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold border ${exchangeBadgeClass}`}>
                        {stock.exchange}
                      </span>
                      <span className="text-[10px] text-slate-400" title={stock.country}>
                        {countryBadge}
                      </span>
                    </div>
                  </div>
                );
              })
            ) : (
              /* Empty state if no recommendations matched */
              <div className="px-4 py-6 text-center">
                <p className="text-xs text-slate-400 mb-1">
                  No predefined stock matches for <span className="text-white font-mono font-bold">"{value.trim()}"</span>
                </p>
                <p className="text-[11px] text-slate-500">
                  Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-amber-400 font-mono text-[10px]">Enter</kbd> to analyze this ticker directly.
                </p>
              </div>
            )}
          </div>

          {/* Keyboard Navigation Footer Helper */}
          <div className="px-3.5 py-1.5 bg-slate-950/80 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-500">
            <div className="flex items-center gap-2">
              <span>
                <kbd className="px-1 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 font-mono">↑↓</kbd> navigate
              </span>
              <span>
                <kbd className="px-1 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 font-mono">↵</kbd> select
              </span>
              <span>
                <kbd className="px-1 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 font-mono">esc</kbd> close
              </span>
            </div>
            <span className="text-amber-500/80 font-mono text-[9px]">QuantIntel Smart Search</span>
          </div>
        </div>
      )}
    </div>
  );
};

