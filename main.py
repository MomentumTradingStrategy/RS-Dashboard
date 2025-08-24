# Relative Strength (RS) Dashboard — Yahoo batch + Nasdaq-100 Top 15
# Columns: RS%, Ticker, Name
# Run: python -m streamlit run main.py --server.port 8080 --server.address 0.0.0.0

# --- Dependencies (installed via requirements.txt; no in-code pip) ---
try:
    import yfinance as yf
except Exception:
    import streamlit as st
    st.error("Missing dependency 'yfinance'. Add it to requirements.txt and restart.")
    st.stop()

import re
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr  # fallback source (Stooq)

# ---------- Page ----------
st.set_page_config(page_title="Relative Strength (RS) Dashboard", layout="wide")
st.title("Relative Strength (RS) Dashboard")
st.caption("See which markets are outperforming or lagging your selected benchmark and lookback period.")

# ---------- Universe (Excel order) ----------
# Note: NASDAQ Composite Index uses Yahoo ticker ^IXIC (was COMP which is a stock).
GROUPS: Dict[str, List[str]] = {
    "US Indices": ["SPY","QQQ","IWM","DIA","^NYA","^IXIC","RSP","QQQE","EDOW","MDY","IWO","IWN","VTI"],
    "US Sectors": ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"],
    "Sub-Sectors / Industry Groups": [
        "GDX","XOP","IYR","XHB","ITB","VNQ","IYE","OIH","XME","XRT","SMH","IBB","KBE","KRE",
        "XTL","XAR","XBI","XHS","KCE","XHE","KIE","XPH","XSD","XSW","XTN","BOTZ","IDNA","IGM","IDRV"
    ],
    "Commodities": ["GLD","SLV","UNG","USO","DBA","CORN","DBB","PALL","URA","UGA","CPER","SOYB","WEAT","DBC","SLX"],
    "Country / Foreign Markets": [
        "IEMG","EEM","EWJ","EWU","EWZ","EWG","EWT","EWH","EWI","EWW",
        "PIN","IDX","EWY","EWA","EWM","EWS","EWC","EWP","EZA","EWL"
    ],
    "Currencies": ["UUP","FXE","FXY","FXB","FXA","FXF","FXC","IBIT","ETHA"],
    "Fixed Income": ["TLT","BND","SHY","IEF","SGOV","IEI","TLH","AGG","MUB","GOVT","IGSB","USHY","IGIB"],
}

# Friendly names (we’ll auto-fill missing via Yahoo)
NAME_MAP: Dict[str, str] = {
    "^IXIC": "NASDAQ Composite Index",
    "XLC":"Comm Svc SPDR","XLY":"Cons Disc SPDR","XLP":"Cons Staples SPDR","XLE":"Energy SPDR",
    "XLF":"Financials SPDR","XLV":"Health Care SPDR","XLI":"Industrials SPDR","XLB":"Materials SPDR",
    "XLRE":"Real Estate SPDR","XLK":"Technology SPDR","XLU":"Utilities SPDR",
    "GDX":"VanEck Gold Miners","XOP":"SPDR Oil & Gas Exp","IYR":"iShares US Real Estate","XHB":"SPDR Homebuilders",
    "ITB":"iShares US Home Const","VNQ":"Vanguard REIT","IYE":"iShares US Energy","OIH":"VanEck Oil Services",
    "XME":"SPDR Metals & Mining","XRT":"SPDR Retail","SMH":"VanEck Semiconductor","IBB":"iShares Biotech",
    "KBE":"SPDR Banks","KRE":"SPDR Regional Banks","XTL":"SPDR Telecom","XAR":"SPDR Aerospace & Def",
    "XBI":"SPDR Biotech","XHS":"SPDR Health Care Svcs","KCE":"SPDR Capital Markets","XHE":"SPDR Health Equip",
    "KIE":"SPDR Insurance","XPH":"SPDR Pharma","XSD":"SPDR Semiconductor","XSW":"SPDR Software & Svcs",
    "XTN":"SPDR Transportation","BOTZ":"Global X Robotics","IDNA":"iShares Genomics","IGM":"iShares Tech & SW",
    "IDRV":"iShares Self-Driving EV",
}

# ---------- Sidebar ----------
with st.sidebar:
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
    lookback = st.selectbox("Lookback", ["1M","3M","6M","YTD","1Y","2Y"], index=0)
    st.markdown("**Sections to show** (Excel order)")
    shown_groups = st.multiselect("Select sections", options=list(GROUPS.keys()), default=list(GROUPS.keys()))
    extra = st.text_input("Extra tickers (comma-separated)", value="TSLA")
    show_ndx_top = st.checkbox("Show Top 15 Growth (Nasdaq-100)", value=True)
    run = st.button("Fetch / Refresh Data")

if not run:
    st.info("Set your options, then click **Fetch / Refresh Data**.")
    st.stop()

# ---------- Helpers ----------
def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _to_y(sym: str) -> str:  # BRK.B -> BRK-B (caret tickers like ^IXIC pass through)
    return sym.replace(".", "-")

def _from_y(sym: str) -> str:
    return sym.replace("-", ".")

def _to_stooq(sym: str) -> str:
    # Stooq format for US stocks/ETFs: TICKER.US ; indices with ^ are spotty -> skip
    if sym.startswith("^"):
        return ""
    base = sym.replace(".", "-")
    if not re.fullmatch(r"[A-Z\-]{1,10}", base):
        return ""
    return f"{base}.US"

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_names_yf(symbols: List[str]) -> Dict[str, str]:
    if not symbols:
        return {}
    names: Dict[str, str] = {}
    tk = yf.Tickers(" ".join(_to_y(s) for s in symbols))
    for ysym, obj in tk.tickers.items():
        nm = ""
        try:
            info = obj.get_info()
            nm = (info.get("shortName") or info.get("longName") or "").strip()
        except Exception:
            try:
                fast = getattr(obj, "fast_info", {}) or {}
                nm = (fast.get("shortName") or fast.get("longName") or "").strip()
            except Exception:
                nm = ""
        if nm:
            names[_from_y(ysym)] = nm
    return names

@st.cache_data(ttl=15*60, show_spinner=True)
def fetch_prices_yf(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Robust price fetch: Yahoo in chunks -> Stooq fallback for stocks/ETFs."""
    if not symbols:
        return pd.DataFrame()

    frames = []
    y_syms = [_to_y(s) for s in symbols]

    # 1) Try Yahoo (chunked to avoid all-or-nothing empties)
    for batch in _chunk(y_syms, 25):
        try:
            df = yf.download(
                tickers=" ".join(batch),
                start=pd.Timestamp(start),
                end=pd.Timestamp(end) + pd.Timedelta(days=1),  # end exclusive
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            if df.empty:
                continue
            close = df["Close"].copy()
            if isinstance(close, pd.Series):
                close = close.to_frame()
                close.columns = [_from_y(batch[0])]
            else:
                close.columns = [_from_y(c) for c in close.columns]
            frames.append(close)
        except Exception:
            continue

    if frames:
        out = pd.concat(frames, axis=1)
        want = [s for s in symbols if s in out.columns]  # preserve original order
        return out[want].sort_index().ffill()

    # 2) Yahoo unavailable: fallback to Stooq for stocks/ETFs (indices with ^ may be skipped)
    cols = []
    for sym in symbols:
        stq = _to_stooq(sym)
        if not stq:
            continue
        try:
            d = pdr.DataReader(
                stq, "stooq",
                start=pd.Timestamp(start),
                end=pd.Timestamp(end) + pd.Timedelta(days=1),
            )
            if not d.empty:
                cols.append(d["Close"].rename(sym).to_frame())
        except Exception:
            continue

    if not cols:
        return pd.DataFrame()

    out = pd.concat(cols, axis=1)
    want = [s for s in symbols if s in out.columns]
    return out[want].sort_index().ffill()

# --- Robust Nasdaq-100 fetch (tries yfinance -> Wikipedia -> static fallback) ---
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fetch_nasdaq100_symbols() -> List[str]:
    # 1) Try yfinance constituents (if available in installed version)
    try:
        ndx = yf.Ticker("^NDX")
        const_obj = getattr(ndx, "constituents", None)
        if const_obj is not None:
            if isinstance(const_obj, pd.DataFrame) and ("Symbol" in const_obj.columns or const_obj.index.name):
                if "Symbol" in const_obj.columns:
                    syms = const_obj["Symbol"].astype(str).str.upper().tolist()
                else:
                    syms = const_obj.index.astype(str).str.upper().tolist()
                return sorted(set(syms))
    except Exception:
        pass

    # 2) Try Wikipedia table
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("ticker" in c or "symbol" in c for c in cols):
                for cand in ["Ticker", "Symbol", "Ticker symbol", "Ticker Symbol"]:
                    if cand in t.columns:
                        syms = t[cand].astype(str).str.replace("\u00a0", " ").str.strip().str.upper()
                        syms = syms.str.replace(r"^NASDAQ:\s*", "", regex=True)
                        syms = syms.str.replace(r"[^A-Z\.\-]", "", regex=True)
                        syms = [s for s in syms if len(s) >= 1]
                        return sorted(set(syms))
    except Exception:
        pass

    # 3) Static fallback list (ok if a few drift)
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
        "ADBE","NFLX","AMD","PEP","CSCO","LIN","QCOM","TMUS","INTC","TXN",
        "AMGN","HON","SBUX","INTU","AMAT","PDD","PYPL","MRVL","ISRG","ADI",
        "MU","BKNG","PANW","GILD","REGN","VRTX","LRCX","KLAC","CSX","MDLZ",
        "ADP","CHTR","ABNB","KDP","SNPS","MAR","MNST","AEP","ORLY","CDNS",
        "CRWD","FTNT","CTAS","ADSK","CCEP","ROP","MELI","NXPI","AZN","PCAR",
        "KHC","WDAY","XEL","ODFL","IDXX","PAYX","MCHP","DXCM","CSGP","MSCI",
        "KDP","TEAM","PDD","EBAY","SIRI","ROST","VRSK","ANSS","VRSN","CTSH",
        "CPRT","FAST","GFS","GEHC","ZS","DDOG","LULU","DG","ALGN","BKR",
        "TTD","FTV","LCID","MRNA","ABNB","FANG","ULTA","BIDU","WBD","EA",
        "NTES","JD","EXC","BKR","ON"
    ]

# ---------- Date window ----------
now = datetime.utcnow().date()
if lookback == "YTD":
    start_date = datetime(now.year, 1, 1).date()
else:
    months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "2Y": 24}[lookback]
    start_date = (now - timedelta(days=int(months * 30.4)))
end_date = now

# ---------- Build symbol list ----------
extra_list = [t.strip().upper() for t in extra.split(",") if t.strip()]
symbols: List[str] = [benchmark]
for g in shown_groups:
    for t in GROUPS[g]:
        if t != benchmark:
            symbols.append(t)
for t in extra_list:
    if t != benchmark:
        symbols.append(t)

# If showing Top 15 Nasdaq-100, make sure those symbols are included in price batch
nasdaq100_symbols: List[str] = []
if show_ndx_top:
    try:
        nasdaq100_symbols = fetch_nasdaq100_symbols()
    except Exception:
        nasdaq100_symbols = []
    for t in nasdaq100_symbols:
        if t and t != benchmark:
            symbols.append(t)

symbols = list(dict.fromkeys(symbols))  # de-dupe, preserve order

# ---------- Fetch prices & compute RS ----------
prices = fetch_prices_yf(symbols, start_date, end_date)

if prices.empty:
    st.warning(
        "No price data returned. Yahoo may be throttled; stocks/ETFs should still load via Stooq fallback. "
        "Tip: start with 'US Indices' only, then add sections."
    )
    st.stop()

if benchmark not in prices.columns:
    st.warning(
        f"Benchmark '{benchmark}' not found in returned data. "
        "Check symbol (indexes often need a caret, e.g., ^NYA, ^IXIC)."
    )
    st.stop()

ff = prices.loc[prices.index >= pd.Timestamp(start_date)]
if ff.shape[0] < 2:
    st.error("Not enough rows in the selected window.")
    st.stop()

# RS vs benchmark (SPY by default): ratio change over lookback
ratio = ff.div(ff[benchmark], axis=0)

alpha_scores = {}
for t in ff.columns:
    if t == benchmark:
        continue
    r = ratio[t].dropna()
    if len(r) >= 2:
        alpha_scores[t] = (r.iloc[-1] / r.iloc[0] - 1.0)

alpha_ser = pd.Series(alpha_scores, name="RS_Score")

# --------- Global percentile rank (1-100) across ALL non-benchmark names ----------
ranks_global = alpha_ser.rank(pct=True).mul(100.0).rename("RS_STS_%")

# Names: NAME_MAP first, then Yahoo for the remainder
need_names = [t for t in alpha_ser.index if not NAME_MAP.get(t)]
auto_names = fetch_names_yf(need_names)

def _name_for(t: str) -> str:
    return NAME_MAP.get(t) or auto_names.get(t, "")

rows = [{"Ticker": t, "RS_STS_%": float(ranks_global.loc[t]), "Name": _name_for(t)} for t in alpha_ser.index]
rs_table = pd.DataFrame(rows)

# ---------- Styling ----------
GREEN = "background-color: rgba(16,185,129,0.85); color: #0b0d12; font-weight:700; text-align:center;"
AMBER = "background-color: rgba(245,158,11,0.85); color: #0b0d12; font-weight:700; text-align:center;"
RED   = "background-color: rgba(244,63,94,0.85); color: #0b0d12; font-weight:700; text-align:center;"

def rs_color(v: float) -> str:
    try:
        vv = float(v)
    except Exception:
        return ""
    if vv >= 75: return GREEN
    if vv >= 50: return AMBER
    return RED

st.markdown(
    """
<style>
  .rs-header{margin:18px 0 6px;font-weight:800;font-size:18px}
</style>
""",
    unsafe_allow_html=True,
)

def fixed_order_df(tickers: List[str], base: pd.DataFrame) -> pd.DataFrame:
    sub = base[base["Ticker"].isin(tickers)].copy()
    order = {t:i for i,t in enumerate(tickers)}   # keep Excel order
    sub["_ord"] = sub["Ticker"].map(order)
    return sub.sort_values("_ord").drop(columns=["_ord"]) if not sub.empty else sub

def render_min_table(df: pd.DataFrame):
    v = df.copy()
    v["RS_STS_%"] = v["RS_STS_%"].round(0).astype(int)
    st.dataframe(
        v[["RS_STS_%","Ticker","Name"]].style.applymap(rs_color, subset=["RS_STS_%"]),
        use_container_width=True,
        hide_index=True,
    )

# ---------- Render sections (DO NOT re-rank; show global RS_STS_% values) ----------
for g in shown_groups:
    tickers = GROUPS[g]
    sub = fixed_order_df(tickers, rs_table)
    if sub.empty:
        continue
    st.markdown(f"<div class='rs-header'>{g}</div>", unsafe_allow_html=True)
    render_min_table(sub)

# ---------- Extra tickers (DO NOT re-rank; keep global RS_STS_%) ----------
if extra_list:
    extra_in_df = [t for t in extra_list if t in rs_table["Ticker"].values]
    if extra_in_df:
        st.markdown("<div class='rs-header'>Extra Tickers</div>", unsafe_allow_html=True)
        sub = fixed_order_df(extra_in_df, rs_table)
        render_min_table(sub)

# ---------- Top 15 Growth (Nasdaq-100) ----------
if show_ndx_top and nasdaq100_symbols:
    ndx_df = rs_table[rs_table["Ticker"].isin(nasdaq100_symbols)].copy()
    if not ndx_df.empty:
        ndx_top15 = ndx_df.sort_values("RS_STS_%", ascending=False).head(15)
        st.markdown("<div class='rs-header'>Top 15 Growth Stocks (Nasdaq-100)</div>", unsafe_allow_html=True)
        render_min_table(ndx_top15)
    else:
        st.info("No RS data for Nasdaq-100 in the selected window.")

# ---------- Footer Explanation ----------
st.caption("RS% = relative strength vs selected Benchmark over the Lookback. RS_STS_% is a global percentile (1–100) across all symbols shown in the dashboard. Colors: Green ≥75, Amber 50–74, Red <50. Names are filled from a local map or Yahoo.")

st.markdown(
    """
**How RS is Calculated:**  
For each symbol, we divide its price by the benchmark’s price for every day in the lookback period.  
We then measure the percentage change in that ratio from the first day to the last day.  
This shows how much the symbol outperformed or underperformed the benchmark.  
Finally, all RS scores are ranked as a percentile (1–100) compared to all symbols shown in the dashboard.
"""
)

