export interface StockItem {
  symbol: string;
  name: string;
  exchange: "NASDAQ" | "NYSE" | "NSE" | "BSE" | "CRYPTO" | "ETF" | "INDEX";
  country: "US" | "IN" | "GLOBAL";
  sector: string;
  aliases?: string[];
  popular?: boolean;
}

export const POPULAR_STOCKS: StockItem[] = [
  { symbol: "NVDA", name: "NVIDIA Corporation", exchange: "NASDAQ", country: "US", sector: "Semiconductors", aliases: ["NVIDIA", "GPU", "AI", "CHIPS"], popular: true },
  { symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", country: "US", sector: "Consumer Electronics", aliases: ["APPL", "APPLE", "IPHONE", "MAC", "IOS"], popular: true },
  { symbol: "MSFT", name: "Microsoft Corporation", exchange: "NASDAQ", country: "US", sector: "Software & Cloud", aliases: ["MICROSOFT", "WINDOWS", "AZURE", "XBOX"], popular: true },
  { symbol: "GOOGL", name: "Alphabet Inc. (Class A)", exchange: "NASDAQ", country: "US", sector: "Internet & Search", aliases: ["GOOGLE", "ALPHABET", "YOUTUBE", "GOOG"], popular: true },
  { symbol: "TSLA", name: "Tesla, Inc.", exchange: "NASDAQ", country: "US", sector: "Automotive & EV", aliases: ["TESLA", "ELON", "EV", "CYBERTRUCK"], popular: true },
  { symbol: "HDFCBANK", name: "HDFC Bank Limited", exchange: "NSE", country: "IN", sector: "Banking & Finance", aliases: ["HDFC", "HDFC BANK", "HDFCBANK.NS"], popular: true },
  { symbol: "RELIANCE", name: "Reliance Industries Limited", exchange: "NSE", country: "IN", sector: "Energy & Telecom", aliases: ["RELIANCE", "JIO", "MUKESH AMBANI", "RIL", "RELIANCE.NS"], popular: true },
  { symbol: "AMZN", name: "Amazon.com, Inc.", exchange: "NASDAQ", country: "US", sector: "E-Commerce & Cloud", aliases: ["AMAZON", "AWS", "PRIME"], popular: true },
  { symbol: "META", name: "Meta Platforms, Inc.", exchange: "NASDAQ", country: "US", sector: "Social Media", aliases: ["FACEBOOK", "META", "INSTAGRAM", "WHATSAPP", "FB"], popular: true },
  { symbol: "TCS", name: "Tata Consultancy Services Limited", exchange: "NSE", country: "IN", sector: "IT Services", aliases: ["TCS", "TATA", "TCS.NS"], popular: true },
  { symbol: "BTC-USD", name: "Bitcoin USD", exchange: "CRYPTO", country: "GLOBAL", sector: "Cryptocurrency", aliases: ["BITCOIN", "BTC", "CRYPTO"], popular: true },
];

export const STOCK_DATABASE: StockItem[] = [
  ...POPULAR_STOCKS,

  // US Mega Caps & Tech
  { symbol: "GOOG", name: "Alphabet Inc. (Class C)", exchange: "NASDAQ", country: "US", sector: "Internet & Search", aliases: ["GOOGLE", "ALPHABET", "YOUTUBE"] },
  { symbol: "AVGO", name: "Broadcom Inc.", exchange: "NASDAQ", country: "US", sector: "Semiconductors", aliases: ["BROADCOM", "VMWARE"] },
  { symbol: "LLY", name: "Eli Lilly and Company", exchange: "NYSE", country: "US", sector: "Pharmaceuticals", aliases: ["ELI LILLY", "MOUNJARO", "ZEPBOUND"] },
  { symbol: "BRK.B", name: "Berkshire Hathaway Inc.", exchange: "NYSE", country: "US", sector: "Conglomerate", aliases: ["BERKSHIRE", "WARREN BUFFETT", "BUFFETT", "BRK"] },
  { symbol: "JPM", name: "JPMorgan Chase & Co.", exchange: "NYSE", country: "US", sector: "Banking & Finance", aliases: ["JPMORGAN", "CHASE", "JPM", "BANK"] },
  { symbol: "V", name: "Visa Inc.", exchange: "NYSE", country: "US", sector: "Payment Technology", aliases: ["VISA", "PAYMENTS", "CREDIT CARD"] },
  { symbol: "MA", name: "Mastercard Incorporated", exchange: "NYSE", country: "US", sector: "Payment Technology", aliases: ["MASTERCARD", "PAYMENTS"] },
  { symbol: "WMT", name: "Walmart Inc.", exchange: "NYSE", country: "US", sector: "Retail & Consumer", aliases: ["WALMART", "RETAIL"] },
  { symbol: "XOM", name: "Exxon Mobil Corporation", exchange: "NYSE", country: "US", sector: "Oil & Gas", aliases: ["EXXON", "MOBIL", "OIL"] },
  { symbol: "UNH", name: "UnitedHealth Group Inc.", exchange: "NYSE", country: "US", sector: "Healthcare & Insurance", aliases: ["UNITEDHEALTH", "UNH"] },
  { symbol: "COST", name: "Costco Wholesale Corporation", exchange: "NASDAQ", country: "US", sector: "Retail Wholesale", aliases: ["COSTCO", "KIRKLAND"] },
  { symbol: "ORCL", name: "Oracle Corporation", exchange: "NYSE", country: "US", sector: "Enterprise Software", aliases: ["ORACLE", "DATABASE", "CLOUD"] },
  { symbol: "HD", name: "The Home Depot, Inc.", exchange: "NYSE", country: "US", sector: "Home Improvement", aliases: ["HOME DEPOT"] },
  { symbol: "PG", name: "The Procter & Gamble Company", exchange: "NYSE", country: "US", sector: "Consumer Goods", aliases: ["P&G", "PROCTER AND GAMBLE", "GILLETTE"] },
  { symbol: "BAC", name: "Bank of America Corporation", exchange: "NYSE", country: "US", sector: "Banking", aliases: ["BANK OF AMERICA", "BOFA", "BAML"] },
  { symbol: "NFLX", name: "Netflix, Inc.", exchange: "NASDAQ", country: "US", sector: "Entertainment & Streaming", aliases: ["NETFLIX", "STREAMING"] },
  { symbol: "CRM", name: "Salesforce, Inc.", exchange: "NYSE", country: "US", sector: "CRM & Cloud", aliases: ["SALESFORCE", "CRM", "SLACK"] },
  { symbol: "AMD", name: "Advanced Micro Devices, Inc.", exchange: "NASDAQ", country: "US", sector: "Semiconductors", aliases: ["AMD", "RYZEN", "RADEON", "EPYC"] },
  { symbol: "QCOM", name: "QUALCOMM Incorporated", exchange: "NASDAQ", country: "US", sector: "Semiconductors & 5G", aliases: ["QUALCOMM", "SNAPDRAGON"] },
  { symbol: "INTC", name: "Intel Corporation", exchange: "NASDAQ", country: "US", sector: "Semiconductors", aliases: ["INTEL", "CHIPS", "CORE"] },
  { symbol: "CSCO", name: "Cisco Systems, Inc.", exchange: "NASDAQ", country: "US", sector: "Networking Hardware", aliases: ["CISCO", "NETWORKING"] },
  { symbol: "IBM", name: "International Business Machines", exchange: "NYSE", country: "US", sector: "IT & Hybrid Cloud", aliases: ["IBM", "RED HAT", "MAINFRAME"] },
  { symbol: "ADBE", name: "Adobe Inc.", exchange: "NASDAQ", country: "US", sector: "Creative Software", aliases: ["ADOBE", "PHOTOSHOP", "PDF", "ILLUSTRATOR"] },
  { symbol: "TXN", name: "Texas Instruments Incorporated", exchange: "NASDAQ", country: "US", sector: "Analog Semiconductors", aliases: ["TEXAS INSTRUMENTS", "TI"] },
  { symbol: "PLTR", name: "Palantir Technologies Inc.", exchange: "NYSE", country: "US", sector: "AI & Data Analytics", aliases: ["PALANTIR", "AIP", "GOTHAM", "FOUNDRY"] },
  { symbol: "UBER", name: "Uber Technologies, Inc.", exchange: "NYSE", country: "US", sector: "Mobility & Delivery", aliases: ["UBER", "RIDESHARE", "UBER EATS"] },
  { symbol: "ABNB", name: "Airbnb, Inc.", exchange: "NASDAQ", country: "US", sector: "Travel & Hospitality", aliases: ["AIRBNB", "TRAVEL"] },
  { symbol: "COIN", name: "Coinbase Global, Inc.", exchange: "NASDAQ", country: "US", sector: "Crypto Exchange", aliases: ["COINBASE", "CRYPTO"] },
  { symbol: "HOOD", name: "Robinhood Markets, Inc.", exchange: "NASDAQ", country: "US", sector: "Fintech & Brokerage", aliases: ["ROBINHOOD", "FINTECH"] },
  { symbol: "SNOW", name: "Snowflake Inc.", exchange: "NYSE", country: "US", sector: "Cloud Data Platform", aliases: ["SNOWFLAKE", "DATA WAREHOUSE"] },
  { symbol: "NOW", name: "ServiceNow, Inc.", exchange: "NYSE", country: "US", sector: "Workflow Automation", aliases: ["SERVICENOW"] },
  { symbol: "SHOP", name: "Shopify Inc.", exchange: "NYSE", country: "US", sector: "E-Commerce Platform", aliases: ["SHOPIFY", "COMMERCE"] },
  { symbol: "SPOT", name: "Spotify Technology S.A.", exchange: "NYSE", country: "US", sector: "Audio Streaming", aliases: ["SPOTIFY", "MUSIC", "PODCASTS"] },
  { symbol: "SQ", name: "Block, Inc.", exchange: "NYSE", country: "US", sector: "Fintech & Payments", aliases: ["SQUARE", "BLOCK", "CASH APP", "JACK DORSEY"] },
  { symbol: "PYPL", name: "PayPal Holdings, Inc.", exchange: "NASDAQ", country: "US", sector: "Digital Payments", aliases: ["PAYPAL", "VENMO"] },
  { symbol: "DIS", name: "The Walt Disney Company", exchange: "NYSE", country: "US", sector: "Entertainment & Media", aliases: ["DISNEY", "MARVEL", "STAR WARS", "DISNEY+"] },
  { symbol: "NKE", name: "NIKE, Inc.", exchange: "NYSE", country: "US", sector: "Apparel & Footwear", aliases: ["NIKE", "JORDAN", "SNEAKERS"] },
  { symbol: "MCD", name: "McDonald's Corporation", exchange: "NYSE", country: "US", sector: "Fast Food", aliases: ["MCDONALDS", "BURGER", "FAST FOOD"] },
  { symbol: "SBUX", name: "Starbucks Corporation", exchange: "NASDAQ", country: "US", sector: "Coffee & Beverage", aliases: ["STARBUCKS", "COFFEE"] },
  { symbol: "KO", name: "The Coca-Cola Company", exchange: "NYSE", country: "US", sector: "Beverages", aliases: ["COCA COLA", "COKE", "SODA"] },
  { symbol: "PEP", name: "PepsiCo, Inc.", exchange: "NASDAQ", country: "US", sector: "Snacks & Beverages", aliases: ["PEPSI", "FRITO LAY", "LAYS"] },
  { symbol: "BA", name: "The Boeing Company", exchange: "NYSE", country: "US", sector: "Aerospace & Defense", aliases: ["BOEING", "AEROSPACE", "AIRPLANES"] },
  { symbol: "CAT", name: "Caterpillar Inc.", exchange: "NYSE", country: "US", sector: "Heavy Machinery", aliases: ["CATERPILLAR", "CONSTRUCTION"] },
  { symbol: "GE", name: "GE Aerospace", exchange: "NYSE", country: "US", sector: "Aerospace Engines", aliases: ["GENERAL ELECTRIC", "GE"] },
  { symbol: "SMCI", name: "Super Micro Computer, Inc.", exchange: "NASDAQ", country: "US", sector: "AI Servers & Hardware", aliases: ["SUPERMICRO", "SMCI", "SERVER"] },
  { symbol: "ARM", name: "Arm Holdings plc", exchange: "NASDAQ", country: "US", sector: "Semiconductor Architecture", aliases: ["ARM", "ARM CHIPS", "CPU"] },
  { symbol: "MU", name: "Micron Technology, Inc.", exchange: "NASDAQ", country: "US", sector: "Memory Semiconductors", aliases: ["MICRON", "DRAM", "NAND", "HBM"] },
  { symbol: "AMAT", name: "Applied Materials, Inc.", exchange: "NASDAQ", country: "US", sector: "Semiconductor Equipment", aliases: ["APPLIED MATERIALS"] },
  { symbol: "LRCX", name: "Lam Research Corporation", exchange: "NASDAQ", country: "US", sector: "Semiconductor Equipment", aliases: ["LAM RESEARCH"] },
  { symbol: "PANW", name: "Palo Alto Networks, Inc.", exchange: "NASDAQ", country: "US", sector: "Cybersecurity", aliases: ["PALO ALTO", "FIREWALL"] },
  { symbol: "CRWD", name: "CrowdStrike Holdings, Inc.", exchange: "NASDAQ", country: "US", sector: "Endpoint Cybersecurity", aliases: ["CROWDSTRIKE", "FALCON"] },
  { symbol: "FTNT", name: "Fortinet, Inc.", exchange: "NASDAQ", country: "US", sector: "Network Cybersecurity", aliases: ["FORTINET"] },
  { symbol: "DDOG", name: "Datadog, Inc.", exchange: "NASDAQ", country: "US", sector: "Cloud Monitoring", aliases: ["DATADOG", "OBSERVABILITY"] },
  { symbol: "MDB", name: "MongoDB, Inc.", exchange: "NASDAQ", country: "US", sector: "NoSQL Database", aliases: ["MONGODB", "NOSQL", "DATABASE"] },
  { symbol: "ROKU", name: "Roku, Inc.", exchange: "NASDAQ", country: "US", sector: "Streaming Hardware", aliases: ["ROKU"] },
  { symbol: "INCY", name: "Incyte Corporation", exchange: "NASDAQ", country: "US", sector: "Biopharmaceuticals", aliases: ["INCYTE"] },
  { symbol: "ATLN", name: "Atlantica Sustainable Infrastructure", exchange: "NASDAQ", country: "US", sector: "Renewable Energy", aliases: ["ATLANTICA"] },
  { symbol: "CVX", name: "Chevron Corporation", exchange: "NYSE", country: "US", sector: "Oil & Gas", aliases: ["CHEVRON", "OIL"] },
  { symbol: "MRK", name: "Merck & Co., Inc.", exchange: "NYSE", country: "US", sector: "Pharmaceuticals", aliases: ["MERCK", "KEYTRUDA"] },
  { symbol: "ABBV", name: "AbbVie Inc.", exchange: "NYSE", country: "US", sector: "Biotechnology", aliases: ["ABBVIE", "HUMIRA"] },
  { symbol: "PFE", name: "Pfizer Inc.", exchange: "NYSE", country: "US", sector: "Pharmaceuticals", aliases: ["PFIZER", "VACCINE"] },
  { symbol: "JNJ", name: "Johnson & Johnson", exchange: "NYSE", country: "US", sector: "Healthcare & MedTech", aliases: ["JOHNSON & JOHNSON", "JNJ"] },

  // Indian Equities (NSE / BSE)
  { symbol: "INFY", name: "Infosys Limited", exchange: "NSE", country: "IN", sector: "IT Services", aliases: ["INFOSYS", "NARAYANA MURTHY", "INFY.NS"] },
  { symbol: "ICICIBANK", name: "ICICI Bank Limited", exchange: "NSE", country: "IN", sector: "Banking & Finance", aliases: ["ICICI", "ICICI BANK", "ICICIBANK.NS"] },
  { symbol: "SBIN", name: "State Bank of India", exchange: "NSE", country: "IN", sector: "Public Sector Banking", aliases: ["SBI", "STATE BANK", "SBIN.NS"] },
  { symbol: "TATAMOTORS", name: "Tata Motors Limited", exchange: "NSE", country: "IN", sector: "Automotive & EV", aliases: ["TATA MOTORS", "JLR", "JAGUAR", "TATAMOTORS.NS"] },
  { symbol: "BHARTIARTL", name: "Bharti Airtel Limited", exchange: "NSE", country: "IN", sector: "Telecommunications", aliases: ["AIRTEL", "BHARTI", "BHARTIARTL.NS"] },
  { symbol: "ITC", name: "ITC Limited", exchange: "NSE", country: "IN", sector: "FMCG & Hotels", aliases: ["ITC", "CIGARETTES", "AASHIRVAAD", "ITC.NS"] },
  { symbol: "LT", name: "Larsen & Toubro Limited", exchange: "NSE", country: "IN", sector: "Infrastructure & EPC", aliases: ["L&T", "LARSEN", "INFRASTRUCTURE", "LT.NS"] },
  { symbol: "HINDUNILVR", name: "Hindustan Unilever Limited", exchange: "NSE", country: "IN", sector: "Consumer Staples", aliases: ["HUL", "UNILEVER", "HINDUNILVR.NS"] },
  { symbol: "BAJFINANCE", name: "Bajaj Finance Limited", exchange: "NSE", country: "IN", sector: "Consumer Lending & NBFC", aliases: ["BAJAJ FINANCE", "BAJAJ", "BAJFINANCE.NS"] },
  { symbol: "KOTAKBANK", name: "Kotak Mahindra Bank Limited", exchange: "NSE", country: "IN", sector: "Private Banking", aliases: ["KOTAK", "UDAY KOTAK", "KOTAKBANK.NS"] },
  { symbol: "AXISBANK", name: "Axis Bank Limited", exchange: "NSE", country: "IN", sector: "Private Banking", aliases: ["AXIS", "AXIS BANK", "AXISBANK.NS"] },
  { symbol: "SUNPHARMA", name: "Sun Pharmaceutical Industries", exchange: "NSE", country: "IN", sector: "Pharmaceuticals", aliases: ["SUN PHARMA", "SUNPHARMA.NS"] },
  { symbol: "TITAN", name: "Titan Company Limited", exchange: "NSE", country: "IN", sector: "Jewelry & Watches", aliases: ["TITAN", "TANISHQ", "FASTTRACK", "TITAN.NS"] },
  { symbol: "MARUTI", name: "Maruti Suzuki India Limited", exchange: "NSE", country: "IN", sector: "Automobiles", aliases: ["MARUTI", "SUZUKI", "SWIFT", "MARUTI.NS"] },
  { symbol: "ASIANPAINT", name: "Asian Paints Limited", exchange: "NSE", country: "IN", sector: "Paints & Coatings", aliases: ["ASIAN PAINTS", "ASIANPAINT.NS"] },
  { symbol: "ZOMATO", name: "Zomato Limited", exchange: "NSE", country: "IN", sector: "Quick Commerce & Delivery", aliases: ["ZOMATO", "BLINKIT", "FOOD DELIVERY", "ZOMATO.NS"] },
  { symbol: "TATASTEEL", name: "Tata Steel Limited", exchange: "NSE", country: "IN", sector: "Metals & Steel", aliases: ["TATA STEEL", "STEEL", "TATASTEEL.NS"] },
  { symbol: "ADANIENT", name: "Adani Enterprises Limited", exchange: "NSE", country: "IN", sector: "Conglomerate & Mining", aliases: ["ADANI", "GAUTAM ADANI", "ADANIENT.NS"] },
  { symbol: "ADANIPORTS", name: "Adani Ports and Special Economic Zone", exchange: "NSE", country: "IN", sector: "Ports & Logistics", aliases: ["ADANI PORTS", "ADANIPORTS.NS"] },
  { symbol: "WIPRO", name: "Wipro Limited", exchange: "NSE", country: "IN", sector: "IT Services", aliases: ["WIPRO", "AZIM PREMJI", "WIPRO.NS"] },
  { symbol: "HCLTECH", name: "HCL Technologies Limited", exchange: "NSE", country: "IN", sector: "IT Services", aliases: ["HCL", "HCLTECH.NS"] },
  { symbol: "NTPC", name: "NTPC Limited", exchange: "NSE", country: "IN", sector: "Power Generation", aliases: ["NTPC", "POWER", "NTPC.NS"] },
  { symbol: "POWERGRID", name: "Power Grid Corporation of India", exchange: "NSE", country: "IN", sector: "Electric Utilities", aliases: ["POWER GRID", "POWERGRID.NS"] },
  { symbol: "COALINDIA", name: "Coal India Limited", exchange: "NSE", country: "IN", sector: "Coal & Energy", aliases: ["COAL INDIA", "COALINDIA.NS"] },
  { symbol: "ONGC", name: "Oil and Natural Gas Corporation", exchange: "NSE", country: "IN", sector: "Oil Exploration", aliases: ["ONGC", "ONGC.NS"] },
  { symbol: "M&M", name: "Mahindra & Mahindra Limited", exchange: "NSE", country: "IN", sector: "Automotive & Tractors", aliases: ["MAHINDRA", "THAR", "SCORPIO", "M&M.NS"] },
  { symbol: "BAJAJ-AUTO", name: "Bajaj Auto Limited", exchange: "NSE", country: "IN", sector: "Motorcycles & 2-Wheelers", aliases: ["BAJAJ AUTO", "PULSAR", "CHETAK"] },
  { symbol: "NESTLEIND", name: "Nestle India Limited", exchange: "NSE", country: "IN", sector: "Packaged Food", aliases: ["NESTLE", "MAGGI", "KITKAT", "NESTLEIND.NS"] },
  { symbol: "ULTRACEMCO", name: "UltraTech Cement Limited", exchange: "NSE", country: "IN", sector: "Cement & Materials", aliases: ["ULTRATECH", "CEMENT", "ULTRACEMCO.NS"] },
  { symbol: "JSWSTEEL", name: "JSW Steel Limited", exchange: "NSE", country: "IN", sector: "Steel & Metallurgy", aliases: ["JSW", "JSW STEEL", "JSWSTEEL.NS"] },
  { symbol: "CIPLA", name: "Cipla Limited", exchange: "NSE", country: "IN", sector: "Pharmaceuticals", aliases: ["CIPLA", "INHALERS", "CIPLA.NS"] },
  { symbol: "DRREDDY", name: "Dr. Reddy's Laboratories", exchange: "NSE", country: "IN", sector: "Generics & Pharma", aliases: ["DR REDDY", "DRREDDY.NS"] },
  { symbol: "TRENT", name: "Trent Limited", exchange: "NSE", country: "IN", sector: "Retail Fashion", aliases: ["TRENT", "ZUDIO", "WESTSIDE", "TATA", "TRENT.NS"] },
  { symbol: "DMART", name: "Avenue Supermarts Limited", exchange: "NSE", country: "IN", sector: "Supermarkets & Retail", aliases: ["DMART", "RADHAKISHAN DAMANI", "DMART.NS"] },
  { symbol: "PAYTM", name: "One 97 Communications Limited", exchange: "NSE", country: "IN", sector: "Digital Payments", aliases: ["PAYTM", "VIJAY SHEKHAR SHARMA"] },
  { symbol: "JIOFIN", name: "Jio Financial Services Limited", exchange: "NSE", country: "IN", sector: "Financial Services", aliases: ["JIO FINANCIAL", "JFS", "JIOFIN.NS"] },
  { symbol: "HAL", name: "Hindustan Aeronautics Limited", exchange: "NSE", country: "IN", sector: "Defense & Aerospace", aliases: ["HAL", "TEJAS", "DEFENSE"] },
  { symbol: "BEL", name: "Bharat Electronics Limited", exchange: "NSE", country: "IN", sector: "Defense Electronics", aliases: ["BEL", "RADAR", "DEFENSE"] },

  // Top Global ETFs & Market Indices
  { symbol: "SPY", name: "SPDR S&P 500 ETF Trust", exchange: "ETF", country: "US", sector: "US Large Cap Equity", aliases: ["S&P 500", "SP500", "INDEX ETF"] },
  { symbol: "QQQ", name: "Invesco QQQ Trust (NASDAQ 100)", exchange: "ETF", country: "US", sector: "Tech & Growth ETF", aliases: ["QQQ", "NASDAQ 100", "TECH ETF"] },
  { symbol: "VOO", name: "Vanguard S&P 500 ETF", exchange: "ETF", country: "US", sector: "US Large Cap Equity", aliases: ["VANGUARD S&P 500", "VOO"] },
  { symbol: "IWM", name: "iShares Russell 2000 ETF", exchange: "ETF", country: "US", sector: "Small Cap US Equity", aliases: ["RUSSELL 2000", "SMALL CAP"] },
  { symbol: "DIA", name: "SPDR Dow Jones Industrial Average ETF", exchange: "ETF", country: "US", sector: "Dow Jones ETF", aliases: ["DOW JONES", "DOW 30"] },
  { symbol: "SMH", name: "VanEck Semiconductor ETF", exchange: "ETF", country: "US", sector: "Semiconductor Sector ETF", aliases: ["SEMICONDUCTOR ETF", "CHIP ETF"] },
  { symbol: "XLK", name: "Technology Select Sector SPDR Fund", exchange: "ETF", country: "US", sector: "Tech Sector ETF", aliases: ["TECH SPDR", "XLK"] },
  { symbol: "XLF", name: "Financial Select Sector SPDR Fund", exchange: "ETF", country: "US", sector: "Financials ETF", aliases: ["FINANCIAL ETF", "XLF"] },
  { symbol: "^GSPC", name: "S&P 500 Index", exchange: "INDEX", country: "US", sector: "Market Benchmark", aliases: ["S&P", "SPX", "INDEX"] },
  { symbol: "^IXIC", name: "NASDAQ Composite Index", exchange: "INDEX", country: "US", sector: "Tech Index Benchmark", aliases: ["NASDAQ", "COMPOSITE"] },
  { symbol: "^NSEI", name: "NIFTY 50 Index", exchange: "INDEX", country: "IN", sector: "Indian Market Benchmark", aliases: ["NIFTY", "NIFTY 50", "NSE"] },
  { symbol: "^BSESN", name: "S&P BSE SENSEX Index", exchange: "INDEX", country: "IN", sector: "Indian Market Benchmark", aliases: ["SENSEX", "BSE"] },

  // Leading Cryptocurrencies
  { symbol: "ETH-USD", name: "Ethereum USD", exchange: "CRYPTO", country: "GLOBAL", sector: "Smart Contracts & DeFi", aliases: ["ETHEREUM", "ETH", "VITALIK"] },
  { symbol: "SOL-USD", name: "Solana USD", exchange: "CRYPTO", country: "GLOBAL", sector: "Layer 1 Blockchain", aliases: ["SOLANA", "SOL"] },
  { symbol: "BNB-USD", name: "BNB USD", exchange: "CRYPTO", country: "GLOBAL", sector: "Exchange Token", aliases: ["BINANCE", "BNB"] },
  { symbol: "XRP-USD", name: "XRP USD", exchange: "CRYPTO", country: "GLOBAL", sector: "Payments & Ripple", aliases: ["RIPPLE", "XRP"] },

  // International ADRs & Mega Caps
  { symbol: "TSM", name: "Taiwan Semiconductor Manufacturing Co.", exchange: "NYSE", country: "GLOBAL", sector: "Semiconductor Foundry", aliases: ["TSMC", "TAIWAN SEMI"] },
  { symbol: "ASML", name: "ASML Holding N.V.", exchange: "NASDAQ", country: "GLOBAL", sector: "Photolithography Machines", aliases: ["ASML", "EUV", "LITHOGRAPHY"] },
  { symbol: "BABA", name: "Alibaba Group Holding Limited", exchange: "NYSE", country: "GLOBAL", sector: "Chinese E-Commerce", aliases: ["ALIBABA", "JACK MA", "TAOBAO"] },
  { symbol: "SONY", name: "Sony Group Corporation", exchange: "NYSE", country: "GLOBAL", sector: "Gaming & Electronics", aliases: ["SONY", "PLAYSTATION"] },
  { symbol: "TM", name: "Toyota Motor Corporation", exchange: "NYSE", country: "GLOBAL", sector: "Automotive", aliases: ["TOYOTA", "HYBRID", "LEXUS"] },
  { symbol: "NVO", name: "Novo Nordisk A/S", exchange: "NYSE", country: "GLOBAL", sector: "Diabetes & GLP-1", aliases: ["NOVO NORDISK", "OZEMPIC", "WEGOVY"] },
  { symbol: "SAP", name: "SAP SE", exchange: "NYSE", country: "GLOBAL", sector: "Enterprise Resource Planning", aliases: ["SAP", "ERP"] },
];

/**
 * Enhanced Search Algorithm:
 * Matches query against symbols, company names, sector, country, and aliases.
 * Handles common typos (e.g. APPL -> AAPL) with alias and edit distance scoring.
 */
export function searchStocks(query: string, limit = 8): StockItem[] {
  const cleanQ = query.trim().toUpperCase();
  if (!cleanQ) {
    return POPULAR_STOCKS.slice(0, limit);
  }

  // Deduplicate helper
  const seen = new Set<string>();
  const scoredItems: { item: StockItem; score: number }[] = [];

  for (const item of STOCK_DATABASE) {
    if (seen.has(item.symbol)) continue;
    seen.add(item.symbol);

    const sym = item.symbol.toUpperCase();
    const name = item.name.toUpperCase();
    const sector = item.sector.toUpperCase();
    const aliases = (item.aliases || []).map((a) => a.toUpperCase());

    let score = 0;

    // 1. Exact symbol match (Highest priority)
    if (sym === cleanQ) {
      score += 1000;
    }
    // 2. Exact alias match (e.g. APPL -> AAPL, GOOGLE -> GOOGL, TATA -> TATAMOTORS)
    else if (aliases.includes(cleanQ)) {
      score += 900;
    }
    // 3. Symbol starts with query (e.g. AAP -> AAPL, NVD -> NVDA)
    else if (sym.startsWith(cleanQ)) {
      score += 800 - (sym.length - cleanQ.length) * 10;
    }
    // 4. Any alias starts with query (e.g. APP -> APPL -> AAPL)
    else if (aliases.some((a) => a.startsWith(cleanQ))) {
      score += 700;
    }
    // 5. Company Name starts with query (e.g. Apple -> Apple Inc.)
    else if (name.startsWith(cleanQ)) {
      score += 650;
    }
    // 6. Company Name word starts with query (e.g. "Bank" -> "HDFC Bank")
    else if (name.split(/\s+/).some((word) => word.startsWith(cleanQ))) {
      score += 600;
    }
    // 7. Symbol contains query
    else if (sym.includes(cleanQ)) {
      score += 400;
    }
    // 8. Company name contains query
    else if (name.includes(cleanQ)) {
      score += 350;
    }
    // 9. Aliases contain query
    else if (aliases.some((a) => a.includes(cleanQ))) {
      score += 300;
    }
    // 10. Sector contains query (e.g. "Tech", "Semi", "Bank")
    else if (sector.includes(cleanQ)) {
      score += 200;
    }
    // 11. Fuzzy / Typo match (for 3+ char queries: e.g., APPL vs AAPL, NVDIA vs NVDA)
    else if (cleanQ.length >= 3) {
      const distance = getLevenshteinDistance(sym, cleanQ);
      if (distance <= 1) {
        score += 500;
      } else if (distance === 2 && cleanQ.length >= 4) {
        score += 250;
      }
    }

    if (score > 0) {
      // Slight boost for popular stocks
      if (item.popular) {
        score += 50;
      }
      scoredItems.push({ item, score });
    }
  }

  scoredItems.sort((a, b) => b.score - a.score);
  return scoredItems.slice(0, limit).map((s) => s.item);
}

/**
 * Fast Levenshtein distance for typo correction
 */
function getLevenshteinDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;

  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = i;
    for (let j = 1; j <= b.length; j++) {
      const val = a[i - 1] === b[j - 1] ? row[j - 1] : Math.min(row[j - 1], prev, row[j]) + 1;
      row[j - 1] = prev;
      prev = val;
    }
    row[b.length] = prev;
  }
  return row[b.length];
}

/**
 * Quick lookup helper
 */
export function getStockBySymbol(symbol: string): StockItem | undefined {
  const s = symbol.trim().toUpperCase();
  return STOCK_DATABASE.find((item) => item.symbol.toUpperCase() === s);
}

