import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QUANTINTEL – Gen AI-Powered Multi-Agent Platform",
  description: "Bloomberg Terminal styled Multi-Agent Quantitative Analysis Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-full flex flex-col" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
