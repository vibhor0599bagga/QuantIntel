"use client";

import React, { useState, useRef, useEffect } from "react";
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Check, History, Sparkles } from "lucide-react";

export const getTodayDateString = (): string => {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

interface CalendarPickerProps {
  selectedDate: string;
  onSelectDate: (date: string) => void;
  maxDate?: string; // defaults to today (YYYY-MM-DD)
  disabled?: boolean;
}

export const CalendarPicker: React.FC<CalendarPickerProps> = ({
  selectedDate,
  onSelectDate,
  maxDate,
  disabled = false,
}) => {
  const todayStr = getTodayDateString();
  const maxAllowedDate = maxDate || todayStr;

  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Parse current viewing month/year from selectedDate or today
  const initialDate = selectedDate ? new Date(selectedDate + "T00:00:00") : new Date();
  const [viewYear, setViewYear] = useState(initialDate.getFullYear());
  const [viewMonth, setViewMonth] = useState(initialDate.getMonth()); // 0-indexed

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  // Sync view when selectedDate changes externally
  useEffect(() => {
    if (selectedDate) {
      const d = new Date(selectedDate + "T00:00:00");
      if (!isNaN(d.getTime())) {
        setViewYear(d.getFullYear());
        setViewMonth(d.getMonth());
      }
    }
  }, [selectedDate]);

  const monthNames = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
  ];

  const daysOfWeek = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"];

  // Helper to format Date to YYYY-MM-DD
  const formatDateStr = (year: number, month: number, day: number) => {
    const m = String(month + 1).padStart(2, "0");
    const d = String(day).padStart(2, "0");
    return `${year}-${m}-${d}`;
  };

  // Navigate months
  const handlePrevMonth = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (viewMonth === 0) {
      setViewMonth(11);
      setViewYear(viewYear - 1);
    } else {
      setViewMonth(viewMonth - 1);
    }
  };

  const maxDateObj = new Date(maxAllowedDate + "T00:00:00");
  const isNextMonthDisabled =
    viewYear > maxDateObj.getFullYear() ||
    (viewYear === maxDateObj.getFullYear() && viewMonth >= maxDateObj.getMonth());

  const handleNextMonth = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isNextMonthDisabled) return;
    if (viewMonth === 11) {
      setViewMonth(0);
      setViewYear(viewYear + 1);
    } else {
      setViewMonth(viewMonth + 1);
    }
  };

  // Generate calendar grid days
  const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
  // Sunday is 0, Monday is 1... Adjust to Monday as 0:
  const firstDayIndex = (new Date(viewYear, viewMonth, 1).getDay() + 6) % 7;

  const days = [];
  for (let i = 0; i < firstDayIndex; i++) {
    days.push(null);
  }
  for (let day = 1; day <= daysInMonth; day++) {
    days.push(day);
  }

  // Quick preset handlers
  const handleSelectOffset = (daysOffset: number) => {
    const d = new Date();
    d.setDate(d.getDate() - daysOffset);
    const dateStr = d.toISOString().split("T")[0];
    onSelectDate(dateStr);
    setIsOpen(false);
  };

  const isToday = selectedDate === todayStr;

  return (
    <div className="relative font-mono" ref={containerRef}>
      {/* Trigger Button / Display */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between px-3 py-2 bg-[#0d121d] border rounded text-xs transition-all ${
          isOpen
            ? "border-[#00e5ff] shadow-[0_0_12px_rgba(0,229,255,0.25)] text-[#e2e8f0]"
            : "border-[#1e293b] hover:border-[#00e5ff]/60 text-[#cbd5e1]"
        }`}
      >
        <div className="flex items-center gap-2">
          <CalendarIcon className="w-3.5 h-3.5 text-[#00e5ff]" />
          <span className="font-bold tracking-wider">{selectedDate || todayStr}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {isToday ? (
            <span className="text-[10px] bg-[#00ff66]/15 border border-[#00ff66]/40 text-[#00ff66] px-1.5 py-0.5 rounded font-bold">
              TODAY
            </span>
          ) : (
            <span className="text-[10px] bg-[#ff9d00]/15 border border-[#ff9d00]/40 text-[#ff9d00] px-1.5 py-0.5 rounded font-bold">
              BACKTEST
            </span>
          )}
        </div>
      </button>

      {/* Popover Calendar Modal */}
      {isOpen && (
        <div className="absolute left-0 top-full mt-2 z-50 w-72 bg-[#090d16] border border-[#00e5ff]/40 rounded-lg shadow-[0_8px_30px_rgba(0,0,0,0.8)] p-3 backdrop-blur-md">
          {/* Quick Presets */}
          <div className="grid grid-cols-3 gap-1 mb-3 pb-2 border-b border-[#1a2333]">
            <button
              type="button"
              onClick={() => handleSelectOffset(0)}
              className={`px-1.5 py-1 text-[10px] rounded border transition-colors ${
                isToday
                  ? "bg-[#00e5ff]/20 border-[#00e5ff] text-[#00e5ff] font-bold"
                  : "bg-[#0f172a] border-[#1e293b] text-[#94a3b8] hover:text-[#e2e8f0]"
              }`}
            >
              TODAY
            </button>
            <button
              type="button"
              onClick={() => handleSelectOffset(1)}
              className="px-1.5 py-1 text-[10px] rounded border bg-[#0f172a] border-[#1e293b] text-[#94a3b8] hover:text-[#e2e8f0] transition-colors"
            >
              YESTERDAY
            </button>
            <button
              type="button"
              onClick={() => handleSelectOffset(7)}
              className="px-1.5 py-1 text-[10px] rounded border bg-[#0f172a] border-[#1e293b] text-[#94a3b8] hover:text-[#e2e8f0] transition-colors"
            >
              -7 DAYS
            </button>
          </div>

          {/* Month / Year Header Navigation */}
          <div className="flex items-center justify-between mb-2">
            <button
              type="button"
              onClick={handlePrevMonth}
              className="p-1 hover:bg-[#1a2333] rounded text-[#94a3b8] hover:text-[#00e5ff] transition-colors"
              title="Previous Month"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            <span className="text-xs font-bold text-[#e2e8f0] tracking-wider">
              {monthNames[viewMonth]} {viewYear}
            </span>

            <button
              type="button"
              onClick={handleNextMonth}
              disabled={isNextMonthDisabled}
              className={`p-1 rounded transition-colors ${
                isNextMonthDisabled
                  ? "opacity-20 cursor-not-allowed text-[#475569]"
                  : "hover:bg-[#1a2333] text-[#94a3b8] hover:text-[#00e5ff]"
              }`}
              title="Next Month"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Weekday headers */}
          <div className="grid grid-cols-7 gap-1 text-center mb-1">
            {daysOfWeek.map((day) => (
              <span key={day} className="text-[10px] font-bold text-[#64748b]">
                {day}
              </span>
            ))}
          </div>

          {/* Days Grid */}
          <div className="grid grid-cols-7 gap-1 text-center">
            {days.map((day, idx) => {
              if (day === null) {
                return <div key={`empty-${idx}`} className="h-7 w-7" />;
              }

              const dateStr = formatDateStr(viewYear, viewMonth, day);
              const isFuture = dateStr > maxAllowedDate;
              const isSelected = dateStr === selectedDate;
              const isCurrentDay = dateStr === todayStr;

              return (
                <button
                  key={dateStr}
                  type="button"
                  disabled={isFuture}
                  onClick={() => {
                    onSelectDate(dateStr);
                    setIsOpen(false);
                  }}
                  className={`h-7 w-7 rounded text-[11px] font-mono flex items-center justify-center relative transition-all ${
                    isFuture
                      ? "opacity-20 cursor-not-allowed text-[#475569] bg-transparent"
                      : isSelected
                      ? "bg-[#00e5ff] text-[#06090e] font-extrabold shadow-[0_0_10px_rgba(0,229,255,0.6)]"
                      : isCurrentDay
                      ? "border border-[#00ff66] text-[#00ff66] hover:bg-[#00ff66]/20"
                      : "text-[#cbd5e1] hover:bg-[#1a2333] hover:text-[#00e5ff]"
                  }`}
                >
                  {day}
                  {isCurrentDay && !isSelected && (
                    <span className="absolute bottom-0.5 w-1 h-1 bg-[#00ff66] rounded-full" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Footer note */}
          <div className="mt-3 pt-2 border-t border-[#1a2333] flex items-center justify-between text-[9px] text-[#64748b]">
            <span>* Future dates restricted</span>
            <button
              type="button"
              onClick={() => {
                onSelectDate(todayStr);
                setIsOpen(false);
              }}
              className="text-[#00e5ff] hover:underline"
            >
              Reset to Today
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
