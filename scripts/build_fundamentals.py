"""Build data/manifest/fundamentals.csv: YoY growth in net interest income (NII) and standalone PAT.

Levels (Rs crore, standalone) from screener.in quarterly tables (NII = Revenue - Interest), which
match the figures management reads out on the calls. Quarters before Sep 2023 are not on those
tables, so FY24 / FY25Q1 growth comes from the banks' press releases (cited per row).
"""
import csv
from pathlib import Path

Q = ["Sep 2023", "Dec 2023", "Mar 2024", "Jun 2024", "Sep 2024", "Dec 2024", "Mar 2025",
     "Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026", "Sep 2026"]
LEVELS = {  # quarter -> (revenue, interest, net profit)
    "ICICIBANK": [(34920, 16612, 10261), (36695, 18016, 10272), (37948, 18856, 10708), (38996, 19443, 11059),
                  (40537, 20489, 11746), (41300, 20929, 11792), (42431, 21238, 12630), (42947, 21312, 12768),
                  (41758, 20228, 12359), (41966, 20034, 11318), (43275, 20296, 13702), (45671, 21286, 14804),
                  (49110, 23771, 14843)],
    "HDFCBANK": [(67698, 40313, 15976), (70583, 42111, 16373), (71473, 42396, 16512), (73033, 43196, 16175),
                 (74017, 43903, 16821), (76007, 45354, 16736), (77460, 45394, 17616), (77470, 46032, 18155),
                 (76691, 45139, 18641), (76751, 44136, 18654), (76610, 43528, 19221), (79363, 45829, 19060),
                 (83427, 48807, 19510)],
    "AXISBANK": [(26626, 14312, 5864), (27961, 15429, 6071), (29225, 16136, 7130), (30061, 16612, 6035),
                 (30420, 16937, 6918), (30954, 17348, 6304), (31243, 17432, 7118), (31064, 17504, 5806),
                 (30970, 17226, 5090), (32274, 17988, 6490), (32724, 18267, 7071), (33986, 19340, 7114),
                 (36419, 21047, 7003)],
}
FQ_TO_Q = {"FY24Q2": "Sep 2023", "FY24Q3": "Dec 2023", "FY24Q4": "Mar 2024", "FY25Q1": "Jun 2024",
           "FY25Q2": "Sep 2024", "FY25Q3": "Dec 2024", "FY25Q4": "Mar 2025", "FY26Q1": "Jun 2025",
           "FY26Q2": "Sep 2025", "FY26Q3": "Dec 2025", "FY26Q4": "Mar 2026", "FY27Q1": "Jun 2026",
           "FY27Q2": "Sep 2026"}
SCREENER = "screener.in/company/{s}/ (standalone quarterly results)"

# Rows the Screener window can't cover (needs the same quarter a year earlier), from press releases.
MANUAL = {
    "ICICIBANK_FY24Q1": (round(18227 / 13210 * 100 - 100, 1), round(9648 / 6905 * 100 - 100, 1),
                         "ICICI Bank press release Q1 FY2024: NII 18,227 vs 13,210; PAT 9,648 vs 6,905"),
    "ICICIBANK_FY24Q2": (23.8, 35.8, "ICICI Bank Q2 FY2024 results as stated on the call; NII level 18,308 matches Screener"),
    "ICICIBANK_FY24Q3": (13.4, 23.6, "ICICI Bank Q3 FY2024 results as stated on the call; NII level 18,679 matches Screener"),
    "ICICIBANK_FY24Q4": (8.1, 17.4, "ICICI Bank Q4 FY2024 results as stated on the call; NII level 19,092 matches Screener"),
    "ICICIBANK_FY25Q1": (round(19553 / 18227 * 100 - 100, 1), round(11059 / 9648 * 100 - 100, 1),
                         "Screener Jun 2024 vs ICICI press release Q1 FY2024 (NII 18,227, PAT 9,648)"),
    "HDFCBANK_FY24Q1": (round(23599 / 19481 * 100 - 100, 1), 30.0,
                        "HDFC Bank Q1 FY24 results (Business Standard): NII 23,599 vs 19,481; PAT 11,952, +30%"),
    "HDFCBANK_FY25Q1": (None, None, "Not comparable: Jun 2023 base predates the 1 Jul 2023 merger with HDFC Ltd"),
    "AXISBANK_FY24Q1": (27.0, round(5797.10 / 4125.26 * 100 - 100, 1),
                        "Axis Bank press release Q1FY24: NII +27% YoY; PAT 5,797.10 vs 4,125.26"),
    "AXISBANK_FY24Q2": (round(12314.56 / 10360.26 * 100 - 100, 1), 10.0,
                        "Axis Bank press release Q2FY24: NII 12,314.56 vs 10,360.26; PAT 5,864, +10% YoY"),
    "AXISBANK_FY24Q3": (9.0, 3.7, "Axis Bank press release Q3FY24: NII +9% YoY; PAT 6,071, +3.7% YoY"),
    "AXISBANK_FY24Q4": (round(13089 / 11742.2 * 100 - 100, 1), None,
                        "Axis Bank press release Q4FY24: NII 13,089 vs 11,742.2; PAT YoY not meaningful "
                        "(Q4FY23 net loss of 5,728 from the Citibank India acquisition)"),
    "AXISBANK_FY25Q1": (round(13448 / 11959 * 100 - 100, 1), round(6035 / 5797 * 100 - 100, 1),
                        "Screener Jun 2024 vs Axis press release Q1FY24 (NII 11,959, PAT 5,797)"),
}


def yoy(sym, fq):
    q = FQ_TO_Q.get(fq)
    if q is None or Q.index(q) < 4:
        return None
    (r1, i1, p1), (r0, i0, p0) = LEVELS[sym][Q.index(q)], LEVELS[sym][Q.index(q) - 4]
    return round(((r1 - i1) / (r0 - i0) - 1) * 100, 1), round((p1 / p0 - 1) * 100, 1)


root = Path(__file__).resolve().parents[1]  # project root
calls = list(csv.DictReader(open(root / "data/manifest/calls.csv", encoding="utf-8")))
rows = []
for c in calls:
    cid, sym, fq = c["call_id"], c["nse_symbol"], c["fiscal_quarter"]
    if cid in MANUAL:
        nii, pat, src = MANUAL[cid]
    elif (v := yoy(sym, fq)) is not None:
        (nii, pat), src = v, SCREENER.format(s=sym)
    else:
        raise SystemExit(f"no data for {cid}")
    rows.append({"call_id": cid, "yoy_revenue_growth_pct": "" if nii is None else nii,
                 "yoy_profit_growth_pct": "" if pat is None else pat, "source": src})
out = root / "data/manifest/fundamentals.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['call_id']:18} NII {str(r['yoy_revenue_growth_pct']):>6}  PAT {str(r['yoy_profit_growth_pct']):>6}")
