"""Equity research workstation. Every number and label on screen comes from engine.analyze(); pages only render it."""
import html
import numpy as np
import pandas as pd
import streamlit as st

import data as D
import engine as E
import metrics as M
import qual as Q
import rating as RT
import valuation as V

st.set_page_config(page_title="Equity Research", layout="wide", initial_sidebar_state="expanded")
POS, NEG, MUTED = "#12724A", "#B42318", "#5B6676"
RCOL = {"Strong Buy": POS, "Buy": POS, "Hold": MUTED, "Sell": NEG, "Avoid": NEG}
PAGES = ["Overview", "Financials", "Valuation", "Scenarios", "Research", "Data & method"]
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="st-"], .stMarkdown, input, button { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.block-container { padding: 1.1rem 1.6rem 3rem; max-width: 1360px; }
h1, h2, h3, h4 { color: #0F1B2D; font-weight: 600; letter-spacing: -0.01em; }
h3 { font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; color: #5B6676; border-bottom: 1px solid #D9DEE5; padding-bottom: .3rem; margin: 1.3rem 0 .6rem; }
p, li { font-size: .9rem; line-height: 1.5; }
[data-testid="stSidebar"] { background: #0F1B2D; border-right: none; }
[data-testid="stSidebar"] * { color: #D5DBE5; }
[data-testid="stSidebar"] input { color: #0F1B2D !important; background: #fff !important; border-radius: 2px; }
[data-testid="stSidebar"] .brand { font-weight: 600; letter-spacing: .14em; font-size: .78rem; color: #fff; margin: .2rem 0 1rem; }
[data-testid="stSidebar"] label p { font-size: .8rem; }
[data-testid="stFormSubmitButton"] > button, .stButton > button { border-radius: 2px; border: 1px solid #1F3A5F; background: #1F3A5F; color: #fff !important; font-weight: 500; }
.topbar { display: flex; flex-wrap: wrap; gap: 1.6rem; align-items: flex-end; border-bottom: 1px solid #D9DEE5; padding-bottom: .7rem; margin-bottom: .6rem; }
.tb-name { font-size: 1.25rem; font-weight: 600; color: #0F1B2D; line-height: 1.1; }
.tb-name span { font-size: .85rem; color: #5B6676; font-weight: 500; margin-left: .4rem; }
.kv { font-size: .68rem; color: #5B6676; text-transform: uppercase; letter-spacing: .06em; line-height: 1.15; }
.kv b { display: block; font-size: .98rem; color: #1B2430; font-weight: 600; font-variant-numeric: tabular-nums; text-transform: none; letter-spacing: 0; }
.chip { display: inline-block; padding: 1px 8px; border: 1px solid; border-radius: 2px; font-weight: 600; font-size: .85rem; }
table.t { width: 100%; border-collapse: collapse; font-size: .83rem; font-variant-numeric: tabular-nums; margin-bottom: .4rem; }
.t th { text-align: right; font-weight: 600; color: #5B6676; border-bottom: 1px solid #B8C0CC; padding: .3rem .5rem; font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; }
.t td { padding: .3rem .5rem; border-bottom: 1px solid #EDF0F4; text-align: right; vertical-align: top; }
.t th.l, .t td.l { text-align: left; }
.t tr:hover td { background: #F3F5F8; }
.muted { color: #5B6676; } .pos { color: #12724A; } .neg { color: #B42318; }
.na { border: 1px solid #D9DEE5; border-left: 3px solid #8A94A3; background: #fff; padding: .7rem .9rem; font-size: .86rem; color: #2B3544; margin: .4rem 0; }
.note { font-size: .76rem; color: #5B6676; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ formatting and small components
esc = lambda x: html.escape(str(x))
ok = lambda x: x is not None and isinstance(x, (int, float, np.floating)) and np.isfinite(x)


def money(x):
    if not ok(x): return "n/a"
    a, s = abs(x), "-" if x < 0 else ""
    return f"{s}${a / 1e12:,.2f}T" if a >= 1e12 else f"{s}${a / 1e9:,.2f}B" if a >= 1e9 else f"{s}${a / 1e6:,.0f}M" if a >= 1e6 else f"{s}${a:,.0f}"


pct = lambda x, d=1: f"{x * 100:+.{d}f}%" if ok(x) else "n/a"
pct0 = lambda x, d=1: f"{x * 100:.{d}f}%" if ok(x) else "n/a"
mx = lambda x: f"{x:.1f}x" if ok(x) else "n/a"
tone = lambda x, txt: f'<span class="{"pos" if x > 0.0005 else "neg" if x < -0.0005 else "muted"}">{esc(txt)}</span>' if ok(x) else '<span class="muted">n/a</span>'
chip = lambda r: f'<span class="chip" style="color:{RCOL[r]};border-color:{RCOL[r]}">{esc(r)}</span>'


def table(cols, rows, left=(0,)):
    """HTML table (tabular numerals). Cells are pre-formatted HTML or plain text (escaped by the caller)."""
    th = "".join(f'<th class="{"l" if i in left else ""}">{esc(c)}</th>' for i, c in enumerate(cols))
    body = "".join("<tr>" + "".join(f'<td class="{"l" if i in left else ""}">{v}</td>' for i, v in enumerate(r)) + "</tr>" for r in rows)
    st.markdown(f'<table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def na(title, reasons, hint=""):
    """Unavailable-data state: says what is missing and why. Never a blank page."""
    li = "".join(f"<li>{esc(r)}</li>" for r in reasons)
    st.markdown(f'<div class="na"><b>{esc(title)}</b><ul style="margin:.3rem 0 .2rem 1rem;padding:0">{li}</ul><span class="note">{esc(hint)}</span></div>', unsafe_allow_html=True)


def bullets(items):
    for i in items:
        st.markdown(f"- {i}")


def fa(k, v):
    if isinstance(v, str): return esc(v)
    kl = k.lower()
    if any(w in kl for w in ("growth", "roe", "dilution", "change")): return pct(v)
    if "exit" in kl: return mx(v)
    if any(w in kl for w in ("revenue at", "burn", "raise", "dividends")): return money(v)
    if "book value" in kl or "eps" in kl: return f"${v:,.2f}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


# ------------------------------------------------------------------ shell
def sidebar():
    st.sidebar.markdown('<div class="brand">EQUITY RESEARCH</div>', unsafe_allow_html=True)
    with st.sidebar.form("search"):
        t = st.text_input("Ticker", value=st.session_state.get("sym", ""), placeholder="Search ticker").strip().upper()
        go = st.form_submit_button("Load", width="stretch")
    if go and t:
        st.session_state["sym"] = t
    page = st.sidebar.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.sidebar.markdown("---")
    wq = st.sidebar.slider("Qualitative weight (%)", 0, 100, 50, key="wq", help="Share of the rating from the qualitative view, scaled by how much of it has evidence.")
    st.sidebar.markdown('<div class="note">Free data: Yahoo Finance, SEC EDGAR filings. Educational tool, not investment advice.</div>', unsafe_allow_html=True)
    return page, wq / 100


def topbar(c, extra):
    f, rt, i = c["f"], c["rating"], c["info"]
    kv = lambda k, v: f'<div class="kv">{esc(k)}<b>{v}</b></div>'
    st.markdown('<div class="topbar">' + f'<div class="tb-name">{esc(c["name"] or i["symbol"])}<span>{esc(i["symbol"])}</span></div>' + kv("Price", f"${c['price']:,.2f}") +
                kv("Market cap", money(f["mc"])) + kv("Enterprise value", money(c["ev"])) + kv("Sector", esc(i.get("sector") or "n/a")) + kv("Framework", esc(c["fw"]["label"])) +
                kv("Rating", chip(rt["rating"])) + kv("Horizon", esc(rt["horizon"])) + kv("Confidence", f"{esc(c['conf_label'])} {c['conf']:.0%}") + "</div>", unsafe_allow_html=True)


def landing():
    st.markdown("## Equity research")
    na("No company loaded", ["Enter a ticker symbol in the sidebar and choose Load."], "Ratings, valuation and scenarios are computed from free public data (Yahoo Finance and SEC filings). Nothing is estimated where data is missing.")


# ------------------------------------------------------------------ pages
def px_series(c, years=5):
    h = c["hist"]["Close"].dropna()
    h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) is not None else pd.to_datetime(h.index)
    return h.tail(252 * years)


def stmt_frame(c):
    inc, cf, bs = c["inc"], c["cf"], c["bs"]
    if inc is None or inc.empty: return None
    cols = list(inc.columns)
    g = lambda df, names: next((df.loc[n] for n in names if df is not None and not df.empty and n in df.index), None)
    rows = {"Revenue": g(inc, ["Total Revenue", "Operating Revenue"]), "Gross profit": g(inc, ["Gross Profit"]), "Operating income": g(inc, ["Operating Income"]),
            "Net income": g(inc, ["Net Income"]), "Operating cash flow": g(cf, ["Operating Cash Flow"]), "Free cash flow": g(cf, ["Free Cash Flow"]),
            "Stock-based compensation": g(cf, ["Stock Based Compensation"]), "Cash and short-term investments": g(bs, ["Cash Cash Equivalents And Short Term Investments"]),
            "Total debt": g(bs, ["Total Debt"]), "Shareholders' equity": g(bs, ["Stockholders Equity", "Common Stock Equity"]), "Shares outstanding": g(bs, ["Ordinary Shares Number", "Share Issued"])}
    df = pd.DataFrame({k: v.reindex(cols) for k, v in rows.items() if v is not None}).T
    df.columns = [str(pd.to_datetime(x).year) for x in df.columns]
    return df


def page_overview(c, extra):
    rt, s = c["rating"], c["scen"]
    L, R = st.columns([3, 2], gap="large")
    with L:
        st.markdown("### Investment summary")
        st.write(c["thesis"])
        st.markdown('<div class="note">Written from the computed numbers; no AI text unless the review is run on the Research page.</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.markdown("### Reasons to own")
            bullets(c["own"] or ["No dimension scores clearly favorably; see scenarios."])
        with b:
            st.markdown("### Reasons not to own")
            bullets(c["against"] or ["No material negatives identified from the available data."])
        st.markdown("### Price, five years")
        st.line_chart(px_series(c), height=210)
        fr = stmt_frame(c)
        if fr is not None and "Revenue" in fr.index:
            st.markdown("### Revenue and margins")
            x, y = st.columns(2)
            rev = fr.loc["Revenue"].iloc[::-1]
            x.bar_chart(rev / 1e9, height=190)
            m = pd.DataFrame({k: (fr.loc[v] / fr.loc["Revenue"]).iloc[::-1] for k, v in (("Gross margin", "Gross profit"), ("Operating margin", "Operating income")) if v in fr.index})
            if len(m.columns): y.line_chart(m, height=190)
            st.markdown('<div class="note">Revenue in $ billions by fiscal year; margins as a fraction of revenue. Source: reported statements.</div>', unsafe_allow_html=True)
    with R:
        st.markdown("### Rating")
        nt, ntl = c["near_term"]
        table(["", ""], [["Rating", chip(rt["rating"])], ["Score", f"{rt['score']:+.2f}"], ["Expected horizon", esc(rt["horizon"])], ["Confidence", f"{esc(c['conf_label'])} ({c['conf']:.0%})"],
                         ["Framework", esc(c["fw"]["label"])], ["Required return", pct0(c["hurdle"], 0)], ["Near-term setup (not a rating)", esc(ntl)]], left=(0, 1))
        for k in rt["constraints"]:
            st.markdown(f'<div class="na"><b>{esc(k["rule"])}</b>: {esc(k["detail"])}. Rating moved {esc(k["from"])} to {esc(k["to"])}.</div>', unsafe_allow_html=True)
        st.markdown("### Dimensions")
        table(["Dimension", "Weight", "Score"], [[esc(d), f"{w:.0%}", tone(c["dims"][d], f"{c['dims'][d]:+.2f}") if c["dims"][d] is not None else '<span class="muted">not assessed</span>'] for d, w in zip(RT.DIMS, c["fw"]["w"])])
        st.markdown('<div class="note">Weights depend on the framework. Qualitative view blended in at '
                    f'{rt["qual_weight"]:.0%} (scaled by evidence coverage {c["qcov"]:.0%}). Profit measures are skipped, not penalized, for loss-making companies.</div>', unsafe_allow_html=True)
        st.markdown("### Scenarios at a glance")
        if s["ok"]:
            table(["Case", "Total", "Per year", "Probability*"], [[k, tone(v["total"], pct(v["total"], 0)), tone(v["ann"], pct(v["ann"])), pct0(v["prob"], 0)] for k, v in s["rows"].items()])
            st.markdown(f'<div class="note">*Estimates, not statistics. Probability-weighted: {pct(s["ev_ann"])} per year over {V.H} years.</div>', unsafe_allow_html=True)
        else:
            na("Scenarios unavailable", [s["reason"] or "insufficient data"])
        st.markdown("### What could change the rating")
        bullets(c["changes"] or ["Not enough data to identify specific triggers."])


def page_financials(c, extra):
    fr = stmt_frame(c)
    st.markdown("### Annual financial statements")
    if fr is None:
        na("Financial statements unavailable", ["Neither Yahoo Finance nor SEC filings returned annual statements for this ticker.", f"SEC status: {extra['sec_status']}"],
           "SEC covers US 10-K filers. ETFs, foreign issuers and very new listings often have no statements.")
        return
    def fmt(k, v):
        if not ok(v): return '<span class="muted">n/a</span>'
        return f"{v / 1e6:,.0f}" if k == "Shares outstanding" else f"{v / 1e6:,.0f}"
    table(["$ millions (shares in millions)"] + list(fr.columns), [[esc(k)] + [fmt(k, v) for v in fr.loc[k]] for k in fr.index])
    st.markdown("### Margins and growth")
    rev = fr.loc["Revenue"] if "Revenue" in fr.index else None
    rows = []
    for lab, num in (("Gross margin", "Gross profit"), ("Operating margin", "Operating income"), ("Net margin", "Net income"), ("Free-cash-flow margin", "Free cash flow")):
        if rev is not None and num in fr.index: rows.append([lab] + [tone(0, pct0(n / r)) if ok(n) and ok(r) and r else '<span class="muted">n/a</span>' for n, r in zip(fr.loc[num], rev)])
    if rev is not None:
        gr = [(rev.iloc[i] / rev.iloc[i + 1] - 1) if i + 1 < len(rev) and ok(rev.iloc[i]) and ok(rev.iloc[i + 1]) and rev.iloc[i + 1] > 0 else None for i in range(len(rev))]
        rows.append(["Revenue growth"] + [tone(g, pct(g)) if g is not None else '<span class="muted">n/a</span>' for g in gr])
    table([""] + list(fr.columns), rows)
    src = next((d for k, v, d in extra["sources"] if k == "Financial statements"), "")
    st.markdown(f'<div class="note">Source: {esc(next((v for k, v, d in extra["sources"] if k == "Financial statements"), ""))}. {esc(src)}. Annual figures; the latest quarters are not included.</div>', unsafe_allow_html=True)


def page_valuation(c, extra):
    f, info = c["f"], c["info"]
    prof = M.profile_for(info, info.get("sector"))
    st.markdown("### What investors are paying")
    nd = (f["debt0"] or 0) - (f["cash0"] or 0) if f["debt0"] is not None and f["cash0"] is not None else None
    table(["", "Value", "Source"], [["Market capitalization", money(f["mc"]), "price x shares (Yahoo / SEC cover page)"], ["+ Debt (incl. minority interest)", money(f["debt0"]), "latest balance sheet"],
                                   ["- Cash and short-term investments", money(f["cash0"]), "latest balance sheet"], ["= Enterprise value", money(c["ev"]), "computed; Yahoo reports " + money(info.get("enterpriseValue"))]], left=(0, 2))
    st.markdown(f'<div class="note">Framework: {esc(c["fw"]["label"])}. Emphasis: {esc(c["fw"]["emphasis"])}. Multiples are judged against sector rules of thumb, the company\'s own history and peers where available; not every multiple applies to every business.</div>', unsafe_allow_html=True)
    st.markdown("### Multiples and whether they mean anything here")
    ref = {"Forward P/E": "Forward P/E", "Trailing P/E": "Trailing P/E", "EV / EBITDA": "EV / EBITDA", "EV / Sales": "EV / Revenue", "Price / Sales": "Price / Sales", "Price / Book": "Price / Book"}
    rows = []
    for r in c["mult_rows"]:
        band = prof["over"].get(ref.get(r["Multiple"], ""))
        refs = f"{band[0]:g}x to {band[1]:g}x" if band and r["Status"] == "Meaningful" else "n/a"
        val = (pct0(r["Value"], 1) if r["Multiple"] == "FCF yield" else mx(r["Value"])) if r["Status"] == "Meaningful" else '<span class="muted">not meaningful</span>' if r["Status"] == "Not meaningful" else '<span class="muted">unavailable</span>'
        rows.append([esc(r["Multiple"]), val, esc(r["Relevance"]), refs, esc(r["Why"])])
    table(["Multiple", "Value", "Relevance", "Sector range", "Note"], rows, left=(0, 2, 4))
    st.markdown('<div class="note">Negative earnings, EBITDA or cash flow are marked not meaningful instead of being shown as a negative multiple. Ranges are rules of thumb, not live peer data.</div>', unsafe_allow_html=True)
    st.markdown("### Valuation against the company's own history")
    if c["hist_df"] is not None:
        table(["Fiscal year end", "Price", "P/E", "P/S"], [[esc(r["FY end"]), f"${r['Price']:,.2f}", mx(r["P/E"]) if ok(r["P/E"]) else '<span class="muted">n/a</span>', mx(r["P/S"])] for _, r in c["hist_df"].iterrows()])
        for k, v in (c["hist_stats"] or {}).items():
            st.markdown(f'<div class="note">{k}: current {mx(v["current"])} vs fiscal-year-end range {mx(v["low"])} to {mx(v["high"])}, median {mx(v["median"])}; sits at the {v["pct"]:.0%} percentile of {v["n"]} observations.</div>', unsafe_allow_html=True)
    else:
        na("Own-history multiples unavailable", ["Fewer than three fiscal years with price, share count and results."])
    st.markdown("### Peers")
    if c["peer_df"] is not None:
        table(["Metric", "Company", "Peer median", "Difference"], [[esc(r["Metric"]), (pct0(r["Company"], 1) if r["Metric"] in ("Op margin", "Rev growth") else mx(r["Company"])), (pct0(r["Peer median"], 1) if r["Metric"] in ("Op margin", "Rev growth") else mx(r["Peer median"])),
                                                              (pct(r["Premium"], 1) if r["Metric"] not in ("Op margin", "Rev growth") else f"{r['Premium'] * 100:+.1f} pts")] for _, r in c["peer_df"].iterrows()])
        st.markdown('<div class="note">Peers are the top companies Yahoo lists for the same industry; they differ in size and mix, so treat the comparison as context.</div>', unsafe_allow_html=True)
    else:
        na("Peer comparison unavailable", ["Yahoo did not return an industry peer list for this ticker.", "No peer multiples were estimated."])
    st.markdown("### What the current price requires")
    if c["reqg"] is not None:
        st.write(f"To earn the {c['hurdle']:.0%} required return over {V.H} years in the base scenario, year-1 growth of about {c['reqg']:.0%} (fading afterwards) is needed. That looks **{c['plaus'].lower()}**"
                 + (f" against {f['hist_g']:.0%} annual growth over the last three fiscal years." if f["hist_g"] is not None else "."))
        st.markdown('<div class="note">Computed with the same scenario machinery as the Scenarios page: the growth rate that makes the base case earn the required return.</div>', unsafe_allow_html=True)
    else:
        na("Required growth unavailable", [c["scen"]["reason"] or "scenario analysis unavailable"])
    if c["vitems"]:
        st.markdown("### How valuation was scored")
        table(["Component", "Score"], [[esc(n), tone(sc, f"{sc:+.2f}")] for n, sc in c["vitems"]])


def page_scenarios(c, extra):
    s = c["scen"]
    st.markdown("### Bull, base and bear scenarios")
    if not s["ok"]:
        na("Scenarios could not be built", [s["reason"] or "insufficient data"] + [f"Missing: {m}" for m in s.get("missing", [])],
           "Scenarios use explicit assumptions and exit multiples. Nothing is estimated when the inputs are missing.")
        return
    mode = {"earnings": "Forward earnings x exit P/E", "revenue": "Revenue x exit EV/Sales (company is loss-making)", "book": "Book value x exit P/B"}[s["mode"]]
    st.write(f"Method: **{mode}**, {V.H}-year horizon, required return {c['hurdle']:.0%} per year.")
    keys = list(dict.fromkeys(k for v in s["rows"].values() for k in v["assump"]))
    table(["Assumption", "Bear", "Base", "Bull"], [[esc(k)] + [fa(k, s["rows"][case]["assump"].get(k, "n/a")) if k in s["rows"][case]["assump"] else "n/a" for case in ("Bear", "Base", "Bull")] for k in keys])
    st.markdown("### Outcomes")
    table(["", "Bear", "Base", "Bull"], [["Price at horizon"] + [f"${s['rows'][k]['price_h']:,.2f}" for k in ("Bear", "Base", "Bull")],
                                          ["Total return vs today"] + [tone(s["rows"][k]["total"], pct(s["rows"][k]["total"], 0)) for k in ("Bear", "Base", "Bull")],
                                          ["Annualized return"] + [tone(s["rows"][k]["ann"], pct(s["rows"][k]["ann"])) for k in ("Bear", "Base", "Bull")],
                                          ["Probability (estimate)"] + [pct0(s["rows"][k]["prob"], 0) for k in ("Bear", "Base", "Bull")]])
    st.write(f"Probability-weighted outcome: **{pct(s['ev_ann'])} per year** (${s['ev_price']:,.2f} at the horizon), against a {c['hurdle']:.0%} required return.")
    st.markdown(f'<div class="note">{esc(s["probs_note"])} The weighted figure is only as good as these estimates; the base case, not the average, is the central outcome.</div>', unsafe_allow_html=True)
    st.bar_chart(pd.DataFrame({"Price at horizon": [s["rows"][k]["price_h"] for k in ("Bear", "Base", "Bull")]}, index=["Bear", "Base", "Bull"]), height=180)
    st.markdown(f'<div class="note">Current price ${c["price"]:,.2f}.</div>', unsafe_allow_html=True)
    st.markdown("### Limitations")
    bullets(s["limitations"])


def ai_panel(c, sym):
    st.markdown("### AI-written review and independent opinion")
    key, pwd = D._secret("ANTHROPIC_API_KEY"), D._secret("APP_PASSWORD")
    ai = st.session_state.get(f"ai_{sym}")
    if not key:
        na("AI review not enabled", ["Add ANTHROPIC_API_KEY to this app's secrets to have Claude write a review from every metric shown here and give its own opinion."],
           "The API is a separate paid product from a Claude subscription. Everything else in the app is free.")
    else:
        ok_pw = True
        if pwd:
            st.text_input("Access password (protects your API credits)", type="password", key="pw")
            ok_pw = st.session_state.get("pw") == pwd
        else:
            st.markdown('<div class="note">No APP_PASSWORD set: anyone with this app\'s link can trigger paid AI calls. Set APP_PASSWORD in secrets and a spend limit in the Anthropic console.</div>', unsafe_allow_html=True)
        if st.button("Generate AI review", key=f"runai{sym}", disabled=not ok_pw or st.session_state.get("ai_calls", 0) >= 8):
            facts = E.facts_for_ai(c)
            try:
                with st.spinner("Writing review..."):
                    text = ai_call(Q.review_prompt(facts, c["info"].get("longBusinessSummary")), D._secret("ANTHROPIC_MODEL") or "claude-sonnet-5-5", key)
                p = Q.parse_review(text, facts)
                st.session_state[f"ai_{sym}"] = dict(items=p["items"], challenge=p["challenge"], review=p["review"], errors=p["errors"])
                st.session_state["ai_calls"] = st.session_state.get("ai_calls", 0) + 1
                st.rerun()
            except RuntimeError as e:
                st.error(f"AI review unavailable: {e}")
    if not ai or not ai.get("review"):
        if ai:
            for e_ in ai["errors"]: st.caption(e_)
        return
    r, op = ai["review"], ai["review"]["opinion"]
    st.markdown('<div class="note">Written by an AI model from the deterministic outputs on this site plus its general knowledge (labelled where used). It is unverified. Its opinion never changes the calculated rating; the judgment questions it scored feed the qualitative view (shown on this page), and a challenge applies only if you tick the box below.</div>', unsafe_allow_html=True)
    table(["", "Rating", "Notes"], [["Model (calculated)", chip(c["rating"]["rating"]), esc(f"score {c['rating']['score']:+.2f}, confidence {c['conf_label']}")],
                                    ["AI opinion", chip(op["rating"]) if op["rating"] else "n/a", esc(("agrees with the model" if op["agrees"] else "disagrees with the model" if op["agrees"] is False else "") + f"; confidence {op['confidence']:.0%}")]], left=(0, 2))
    st.write(op["reasoning"])
    for title, txt in (("Thesis", r["thesis"]), ("Business", r["business"]), ("Competitive position", r["moat"]["assessment"]), ("Industry and AI", r["industry_and_ai"]),
                       ("Management and capital allocation", r["management"]), ("What the price implies", r["expectations"]), ("What the market may be missing", r["missing"])):
        if txt: st.markdown(f"**{title}.** {esc(txt)}")
    a, b = st.columns(2)
    with a:
        st.markdown("**For**"); bullets([esc(x) for x in r["for_"]])
    with b:
        st.markdown("**Against**"); bullets([esc(x) for x in r["against"]])
    for k in ("bull", "base", "bear"):
        if r["scenarios"].get(k): st.markdown(f"**{k.title()} case.** {esc(r['scenarios'][k])}")
    if r["risks"]: st.markdown("**Thesis-breaking risks**"); bullets([esc(x) for x in r["risks"]])
    if r["monitor"]: st.markdown("**Monitor next quarter**"); bullets([esc(x) for x in r["monitor"]])
    if r["unknowns"]: st.markdown(f'<div class="note">Unknown to the reviewer: {esc("; ".join(r["unknowns"]))}</div>', unsafe_allow_html=True)
    if ai.get("challenge"):
        ch = ai["challenge"]
        st.markdown(f"**Challenge to the numbers ({ch['adjustment']:+.2f}):** {esc(ch['claim'])}  \nEvidence: {esc(ch['evidence'])}")
        st.checkbox("Apply this challenge to the rating (explicit override, at most 0.25)", key=f"ai_apply_{sym}")
    for e_ in ai["errors"]: st.caption(f"Response check: {e_}")


@st.cache_data(ttl=21600, show_spinner=False)
def ai_call(prompt, model, _key):
    text, err = Q.call_llm(prompt, _key, model, timeout=150, max_tokens=4500)
    if err: raise RuntimeError(err)
    return text


def page_research(c, extra):
    sym = c["info"]["symbol"]
    st.markdown("### Ten business questions")
    table(["Question", "Answer", "Source"], [[esc(q["Question"]), esc(q["Answer"]), esc(q["Source"])] for q in c["questions"]], left=(0, 1, 2))
    st.markdown("### Qualitative assessment")
    st.markdown(f'<div class="note">{sum(1 for i in c["qitems"].values() if i["score"] is not None)} of {len(c["qitems"])} questions have evidence (coverage {c["qcov"]:.0%}). Computed items come from reported statements; the rest need business knowledge and stay unassessed unless the AI review or your own judgment supplies them.</div>', unsafe_allow_html=True)
    table(["Question", "Score", "Confidence", "Evidence", "Source"], [[esc(i["label"]), tone(i["score"], f"{i['score']:+.2f}") if i["score"] is not None else '<span class="muted">not assessed</span>', f"{i['conf']:.0%}", esc(i["evidence"]), esc(i["source"])]
                                                                      for i in c["qitems"].values()], left=(0, 3, 4))
    if c["contradictions"] or c["flags"]:
        st.markdown("### Flags")
        bullets([esc(f"{x[1]}: {x[2]}") for x in c["flags"]] + [esc(x) for x in c["contradictions"]])
    ai_panel(c, sym)


def page_data(c, extra):
    st.markdown("### Data sources")
    table(["Item", "Source", "Detail"], [[esc(a), esc(b), esc(d)] for a, b, d in extra["sources"]], left=(0, 1, 2))
    for x in extra["conflicts"]: st.warning(x)
    st.markdown(f'<div class="note">Retrieved {extra["fetched"]:%Y-%m-%d %H:%M} UTC. Yahoo Finance data is unofficial and delayed. SEC data covers US 10-K filers and is annual.</div>', unsafe_allow_html=True)
    cv = c["cov"]
    st.markdown("### Coverage")
    st.write(f"**{len(cv['evaluated'])} of {cv['total']}** quantitative metrics evaluated; {len(cv['not_meaningful'])} not meaningful or not applicable for this company; {len(cv['unavailable'])} unavailable (data not reported or too little history). "
             f"Qualitative rubric: {sum(1 for i in c['qitems'].values() if i['score'] is not None)} of {len(c['qitems'])} assessed.")
    with st.expander("Metric detail"):
        st.write("Not meaningful / not applicable: " + ("; ".join(f"{k} ({v})" for k, v in cv["not_meaningful"].items()) or "none"))
        st.write("Unavailable: " + (", ".join(cv["unavailable"]) or "none"))
    st.markdown("### Rating methodology")
    table(["Rating", "Meaning"], [[chip(k), esc(RT.DEFS[k])] for k in ("Strong Buy", "Buy", "Hold", "Sell", "Avoid")], left=(0, 1))
    st.markdown('<div class="note">Score thresholds: above +0.45 Strong Buy, above +0.18 Buy, above -0.12 Hold, above -0.35 Sell, otherwise Avoid. The same definitions apply on every page. One rating is computed once per analysis; no page produces its own label.</div>', unsafe_allow_html=True)
    table(["Framework"] + list(RT.DIMS), [[esc(v["label"])] + [f"{w:.0%}" for w in v["w"]] for v in RT.FRAMEWORKS.values()], left=(0,))
    st.markdown('<div class="note">Constraints that can cap a rating, each shown with its reason: required-return floor, expected-return ceiling, poor risk/reward, metrics-alone guard, Strong Buy gate, funding constraint, evidence-quality gate. '
                'Losses are handled once (financial health and funding), never again in profitability, valuation and the final verdict.</div>', unsafe_allow_html=True)
    st.markdown("### Not evaluated")
    bullets(["Customer concentration, backlog, contracts and segment detail (need filing text; not in free structured data)", "Addressable-market size", "Management quality and moat type unless the AI review or your own view supplies them",
             "Consensus estimates and revisions beyond what Yahoo returns", "Options-implied volatility, credit ratings, insider-transaction analysis"])


def main():
    page, wq = sidebar()
    sym = st.session_state.get("sym")
    if not sym:
        landing(); return
    with st.spinner(f"Loading {sym}..."):
        info, hist, inc, cf, bs, extra = D.prepare(sym)
    if hist is None or hist.empty:
        st.markdown("## " + esc(sym)); na("No market data", [f"Yahoo Finance returned no price history for '{sym}'.", f"SEC status: {extra['sec_status']}"], "Check the ticker (use BRK-B, not BRK.B) or retry in a minute; Yahoo sometimes throttles."); return
    info["symbol"] = sym
    peers = D.load_peers(sym, info.get("industryKey")) if info.get("industryKey") else None
    try:
        c = E.analyze(info, hist, inc, cf, bs, est=extra["est"], peers=peers, ai=st.session_state.get(f"ai_{sym}"), w_qual=wq, apply_override=bool(st.session_state.get(f"ai_apply_{sym}")))
    except Exception as e:
        st.markdown("## " + esc(sym)); na("The analysis could not be completed", [f"{type(e).__name__}: {e}"], "This is a bug or an unusual data shape; nothing was estimated."); return
    topbar(c, extra)
    {"Overview": page_overview, "Financials": page_financials, "Valuation": page_valuation, "Scenarios": page_scenarios, "Research": page_research, "Data & method": page_data}[page](c, extra)


main()
