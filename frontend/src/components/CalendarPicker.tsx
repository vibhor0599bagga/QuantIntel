"use client";

import React, { useState, useRef, useEffect } from "react";
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";

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
  maxDate?: string;
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

  const initialDate = selectedDate ? new Date(selectedDate + "T00:00:00") : new Date();
  const [viewYear, setViewYear] = useState(initialDate.getFullYear());
  const [viewMonth, setViewMonth] = useState(initialDate.getMonth());

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
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ];

  const daysOfWeek = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

  const formatDateStr = (year: number, month: number, day: number) => {
    const m = String(month + 1).padStart(2, "0");
    const d = String(day).padStart(2, "0");
    return `${year}-${m}-${d}`;
  };

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

  const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
  const firstDayIndex = (new Date(viewYear, viewMonth, 1).getDay() + 6) % 7;

  const days = [];
  for (let i = 0; i < firstDayIndex; i++) {
    days.push(null);
  }
  for (let day = 1; day <= daysInMonth; day++) {
    days.push(day);
  }

  const handleSelectOffset = (daysOffset: number) => {
    const d = new Date();
    d.setDate(d.getDate() - daysOffset);
    const dateStr = d.toISOString().split("T")[0];
    onSelectDate(dateStr);
    setIsOpen(false);
  };

  const isToday = selectedDate === todayStr;

  return (
    <div className="relative font-sans" ref={containerRef}>
      {/* Trigger Button */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full h-11 flex items-center justify-between px-4 sm:px-5 bg-[#090D16] border rounded-full text-xs transition-all duration-150 cursor-pointer ${
          isOpen
            ? "border-amber-500/50 ring-1 ring-amber-500/30 text-white"
            : "border-slate-800/90 hover:border-slate-700 text-slate-300 hover:text-white"
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <CalendarIcon className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span className="font-mono font-medium text-slate-200 text-xs tracking-tight truncate">
            {selectedDate || todayStr}
          </span>
        </div>
        <span
          className={`text-[10px] font-mono font-medium px-2.5 py-0.5 rounded-full shrink-0 ml-2 tracking-wide ${
            isToday
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
              : "bg-amber-500/10 text-amber-400/90 border border-amber-500/20"
          }`}
        >
          {isToday ? "LIVE" : "BACKTEST"}
        </span>
      </button>

      {/* Popover Calendar */}
      {isOpen && (
        <div className="absolute right-0 sm:left-0 top-full mt-2.5 z-50 w-72 bg-[#0B0F19] border border-slate-800 rounded-2xl shadow-2xl p-4 backdrop-blur-2xl animate-in fade-in zoom-in-95 duration-150">
          {/* Quick Presets */}
          <div className="grid grid-cols-3 gap-1.5 mb-3 pb-2.5 border-b border-slate-800/80">
            <button
              type="button"
              onClick={() => handleSelectOffset(0)}
              className={`px-3 py-1.5 text-[11px] font-mono rounded-full border transition-colors flex items-center justify-center ${
                isToday
                  ? "bg-amber-500/20 border-amber-500/40 text-amber-300 font-bold"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
              }`}
            >
              Today
            </button>
            <button
              type="button"
              onClick={() => handleSelectOffset(1)}
              className="px-3 py-1.5 text-[11px] font-mono rounded-full border bg-slate-900 border-slate-800 text-slate-400 hover:text-white transition-colors flex items-center justify-center"
            >
              -1 Day
            </button>
            <button
              type="button"
              onClick={() => handleSelectOffset(7)}
              className="px-3 py-1.5 text-[11px] font-mono rounded-full border bg-slate-900 border-slate-800 text-slate-400 hover:text-white transition-colors flex items-center justify-center"
            >
              -7 Days
            </button>
          </div>

          {/* Month / Year Header */}
          <div className="flex items-center justify-between mb-2 px-1">
            <button
              type="button"
              onClick={handlePrevMonth}
              className="p-1 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            <span className="text-xs font-semibold text-slate-200">
              {monthNames[viewMonth]} {viewYear}
            </span>

            <button
              type="button"
              onClick={handleNextMonth}
              disabled={isNextMonthDisabled}
              className={`p-1 rounded-lg transition-colors ${
                isNextMonthDisabled
                  ? "opacity-20 cursor-not-allowed text-slate-600"
                  : "hover:bg-slate-800 text-slate-400 hover:text-white"
              }`}
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Weekdays */}
          <div className="grid grid-cols-7 gap-1 text-center mb-1 text-[10px] font-mono font-medium text-slate-500">
            {daysOfWeek.map((day) => (
              <span key={day}>{day}</span>
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
                  className={`h-7 w-7 rounded-lg text-xs font-mono flex items-center justify-center transition-all ${
                    isFuture
                      ? "opacity-20 cursor-not-allowed text-slate-600"
                      : isSelected
                      ? "bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/30"
                      : isCurrentDay
                      ? "border border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  {day}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
