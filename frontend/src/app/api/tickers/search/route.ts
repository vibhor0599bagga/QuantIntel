import { NextRequest, NextResponse } from "next/server";
import { searchStocks } from "@/data/stockDatabase";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q") || "";

  // 1. Get instant local results first
  const localResults = searchStocks(q, 10);

  // If query is very short or empty, return local recommendations immediately
  if (!q.trim() || q.trim().length < 2) {
    return NextResponse.json({
      status: "ok",
      source: "local",
      results: localResults,
    });
  }

  // 2. Concurrently attempt Yahoo Finance search if available
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000); // 2-second timeout

    const yfUrl = `https://query2.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(
      q.trim()
    )}&quotesCount=8&newsCount=0&enableFuzzyQuery=true`;

    const yfRes = await fetch(yfUrl, {
      signal: controller.signal,
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        Accept: "application/json",
      },
      next: { revalidate: 3600 },
    });

    clearTimeout(timeout);

    if (yfRes.ok) {
      const data = await yfRes.json();
      const quotes = data.quotes || [];

      const externalResults = quotes
        .filter((quote: any) => quote.symbol && !quote.symbol.includes("="))
        .map((quote: any) => ({
          symbol: quote.symbol,
          name: quote.shortname || quote.longname || quote.symbol,
          exchange: quote.exchDisp || quote.exchange || "GLOBAL",
          country: quote.symbol.endsWith(".NS") || quote.symbol.endsWith(".BO") ? "IN" : "US",
          sector: quote.sector || quote.quoteType || "Equity",
        }));

      // Merge results, prioritizing local high-confidence matches while eliminating duplicates
      const merged: any[] = [...localResults];
      const seen = new Set(merged.map((m) => m.symbol.toUpperCase()));

      for (const ext of externalResults) {
        if (!seen.has(ext.symbol.toUpperCase())) {
          seen.add(ext.symbol.toUpperCase());
          merged.push(ext);
        }
      }

      return NextResponse.json({
        status: "ok",
        source: "hybrid",
        results: merged.slice(0, 10),
      });
    }
  } catch {
    // If external fetch fails, gracefully return local matches
  }

  return NextResponse.json({
    status: "ok",
    source: "local",
    results: localResults,
  });
}

