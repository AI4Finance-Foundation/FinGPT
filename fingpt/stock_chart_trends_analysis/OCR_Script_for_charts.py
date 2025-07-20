import easyocr
import cv2
import re
import datetime
import matplotlib.pyplot as plt

def normalize_ocr_texts(ocr_texts):
    cleaned = []
    for t in ocr_texts:
        t = t.replace('|', '/').replace('l', '1')
        substitutions = [
            (r'(^|\s)0\.', r'\1O:'), (r'\bO\.', 'O:'), (r'\bH\.', 'H:'),
            (r'\bL\.', 'L:'), (r'\bC\.', 'C:'), (r'\bV\.', 'V:')
        ]
        for p, r_ in substitutions:
            t = re.sub(p, r_, t)
        t = t.replace('，', ',').replace('：', ':')
        cleaned.append(t.strip())
    return cleaned

def convert_compact_number(val):
    val = val.replace(',', '').strip().lower()
    try:
        if val.endswith('k'):
            return float(val[:-1]) * 1e3
        if val.endswith('m'):
            return float(val[:-1]) * 1e6
        if val.endswith('b'):
            return float(val[:-1]) * 1e9
        return float(val)
    except:
        return None

def extract_dates_with_pos(texts):
    # Handles e.g. "Jul 25", "Jun 2025", "May '25", "12/25", etc.
    patterns = [
        r"\b([A-Za-z]{3,9}[, ]+'\d{2,4})\b",    # May '25
        r"\b([A-Za-z]{3,9} ?\d{1,2}(?:, ?\d{2,4})?)\b", # Jul 25, Jun 2025
        r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)",             # 12/25/2023
        r"(\d{1,2}-\d{1,2}(?:-\d{2,4})?)"
    ]
    date_matches = []
    for idx, t in enumerate(texts):
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                date_matches.append((m.group(1), idx))
                break
    return date_matches

def extract_times_with_pos(texts):
    # Handles 24h, 12h, and ambiguous times
    patterns = [
        (r"(\d{1,2}[:.]\d{2})\s*([APMapm]{2})", "ampm"),   # 2:30 PM
        (r"(\d{1,2}[:.]\d{2})", "24h")                     # 14:00, 23.30
    ]
    results = []
    for idx, t in enumerate(texts):
        found = False
        for pat, mode in patterns:
            for match in re.finditer(pat, t):
                h = match.group(1).replace('.', ':')
                if mode == "ampm":
                    ap = match.group(2)
                    try:
                        tm = datetime.datetime.strptime(f"{h} {ap.upper()}", "%I:%M %p").strftime("%I:%M %p").lstrip("0")
                        results.append((tm, idx))
                        found = True
                    except:
                        continue
                elif mode == "24h":
                    try:
                        tm = datetime.datetime.strptime(h, "%H:%M").strftime("%H:%M")
                        results.append((tm, idx))
                        found = True
                    except:
                        continue
        if found:
            continue
    return results

def infer_previous_trading_day(first_dt):
    # Unchanged from original
    try:
        parts = first_dt.split('/')
        today = datetime.datetime.now()
        if len(parts) >= 2:
            year = today.year if len(parts) == 2 else int(parts[2])
            ref_day = datetime.datetime(year, int(parts[0]), int(parts[1]))
            d = ref_day - datetime.timedelta(days=1)
            while d.weekday() >= 5:
                d -= datetime.timedelta(days=1)
            return f"{d.month}/{d.day}" if len(parts) == 2 else f"{d.month}/{d.day}/{d.year}"
    except:
        return "prev"
    return "prev"

def map_sessions(texts):
    dates_with_idx = extract_dates_with_pos(texts)
    times_with_idx = extract_times_with_pos(texts)
    sessions = []
    all_date_indices = [idx for _, idx in dates_with_idx]
    orphan_times = [(tm, idx) for tm, idx in times_with_idx if idx < (all_date_indices[0] if all_date_indices else len(texts))]
    if orphan_times and all_date_indices:
        prev_date = infer_previous_trading_day(dates_with_idx[0][0])
        orph_group = sorted(set(tm for tm, _ in orphan_times))
        if orph_group:
            sessions.append({'date': prev_date, 'time_range': (orph_group[0], orph_group[-1])})
    for i, (dt, idx) in enumerate(dates_with_idx):
        block_times = []
        start_idx = idx + 1
        end_idx = dates_with_idx[i + 1][1] if i + 1 < len(dates_with_idx) else len(texts)
        for tm, t_idx in times_with_idx:
            if start_idx <= t_idx < end_idx:
                block_times.append(tm)
        block_times = sorted(set(block_times))
        if block_times:
            sessions.append({'date': dt, 'time_range': (block_times[0], block_times[-1])})
        elif block_times == []:
            sessions.append({'date': dt, 'time_range': ("unknown", "unknown")})
    return sessions

def extract_ohlc(texts):
    ohlc = {}
    block = re.search(r'O[: ]?([\d\.]+).*?H[: ]?([\d\.]+).*?L[: ]?([\d\.]+).*?C[: ]?([\d\.]+).*?V[: ]?([\d\.,kKmMbB]+)', " ".join(texts))
    if block:
        ohlc['O'] = float(block.group(1))
        ohlc['H'] = float(block.group(2))
        l_str = block.group(3)
        ohlc['L'] = float(l_str[:-2] + '.' + l_str[-2:]) if l_str.replace('.', '').isdigit() and len(l_str.replace('.', '')) == 4 and '.' not in l_str else float(l_str)
        ohlc['C'] = float(block.group(4))
        ohlc['V'] = convert_compact_number(block.group(5))
    else:
        pats = {'O': r'O[: ]?([\d]{2,4}\.?\d*)', 'H': r'H[: ]?([\d]{2,4}\.?\d*)',
                'L': r'L[: ]?([\d]{2,4}\.?\d*)', 'C': r'C[: ]?([\d]{2,4}\.?\d*)'}
        for label, pat in pats.items():
            for t in texts:
                m = re.search(pat, t)
                if m:
                    num = m.group(1)
                    if num.isdigit() and len(num) == 4:
                        ohlc[label] = float(num[:-2] + '.' + num[-2:])
                    else:
                        try: ohlc[label] = float(num)
                        except: continue
                    break
    return ohlc

def extract_volume(texts):
    for t in texts:
        m = re.search(r'vol undr[\D]*([\d\.,]+[kKmMbB]?)', t, re.IGNORECASE)
        if m:
            v = convert_compact_number(m.group(1))
            if v:
                return v
    candidates = []
    for t in texts:
        found_v = re.findall(r'V[: ]?([\d\.,]+[kKmMbB]?)', t)
        candidates.extend(found_v)
        bars = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?[kKmMbB]\b', t)
        candidates.extend(bars)
    nums = [convert_compact_number(v) for v in set(candidates) if convert_compact_number(v)]
    return max(nums) if nums else None

def extract_price_range(texts, ohlc_vals):
    plausible = []
    if not ohlc_vals:
        return None
    oh_vals = [v for v in ohlc_vals if v is not None and 1.0 < v < 10000]
    if not oh_vals:
        return None
    oh_min, oh_max = min(oh_vals), max(oh_vals)
    for t in texts:
        # Accept prices and also months/dates, skip if letters exist
        if re.fullmatch(r'\d{2,4}(?:\.\d{2})?', t.strip()):
            num = float(t.strip())
            if (oh_min * 0.8) <= num <= (oh_max * 1.2) and 1.0 < num < 1000:
                plausible.append(num)
    plausible = sorted(set(plausible))
    if len(plausible) >= 2:
        return (min(plausible), max(plausible))
    return None

def flexible_ticker_extract(texts):
    # Look for tickers as (TICKER), NASDAQ:TICKER, TICKER only, or company name fallback
    for t in texts:
        if (m := re.search(r'\(([A-Z0-9\-./]{1,10})\)', t)):
            return m.group(1)
        # Exchange-prefixed: NASDAQ:AAPL
        if (m := re.search(r'(NASDAQ|NYSE|BSE|LSE|HKEX|TSX)[:\s\-\.]*([A-Z]{1,8})', t)):
            return m.group(2)
        # All-uppercase tickets
        if (m := re.match(r'^([A-Z]{2,8})$', t.strip())):
            return m.group(1)
    # Company name fallback: first capitalized phrase not all upper
    for t in texts:
        if t.istitle() and not t.isupper() and len(t.strip()) > 2:
            return "unknown", t
    return "unknown"

def parse_chart_metadata(ocr_texts):
    texts = normalize_ocr_texts(ocr_texts)
    metadata = {}
    # Ticker extraction now robust/flexible
    ticker_or_fallback = flexible_ticker_extract(texts)
    if isinstance(ticker_or_fallback, tuple):
        metadata['ticker'] = ticker_or_fallback[0]
        metadata['company_name'] = ticker_or_fallback[1]
    else:
        metadata['ticker'] = ticker_or_fallback
    for line in texts:
        exch = re.search(r'\b(NYSE|BSE|NASDAQ|TSX|LSE|HKEX)\b', line)
        if exch:
            metadata['exchange'] = exch.group(1)
    if any("USD" in t for t in texts):
        metadata["currency"] = "USD"
    elif any("INR" in t for t in texts):
        metadata["currency"] = "INR"
    ohlc = extract_ohlc(texts)
    if ohlc:
        pr = extract_price_range(texts, list(ohlc.values()))
        if pr:
            metadata['price_range'] = pr
        metadata['ohlc'] = ohlc
    price_range = extract_price_range(texts, list(ohlc.values()) if ohlc else [])
    if price_range:
        metadata['price_range'] = price_range
    sessions = map_sessions(texts)
    if sessions:
        metadata['sessions'] = sessions
        metadata['dates'] = [sess['date'] for sess in sessions]
    return metadata

# ---- OCR + Parsing Pipeline ----

reader = easyocr.Reader(['en'])
img_path = "E:\\FinGPT-M\\Datasets\\Stock Charts for OCR\\NLX-2_png.rf.f368e80d3051404e1de82d9022332588.jpg"  # Replace with your image path
results = reader.readtext(img_path)

img = cv2.imread(img_path)
for (bbox, text, confidence) in results:
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    img = cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

for bbox, text, conf in results:
    print(f"{text} ({conf:.2f})")

raw_texts = [text for _, text, _ in results]
metadata = parse_chart_metadata(raw_texts)
print(metadata)
