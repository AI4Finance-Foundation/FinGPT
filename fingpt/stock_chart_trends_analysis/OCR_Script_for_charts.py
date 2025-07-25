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
        for p, r_ in substitutions: t = re.sub(p, r_, t)
        t = t.replace('，', ',').replace('：', ':')
        cleaned.append(t.strip())
    return cleaned

def convert_compact_number(val):
    val = val.replace(',', '').strip().lower()
    try:
        if val.endswith('k'): return float(val[:-1]) * 1e3
        if val.endswith('m'): return float(val[:-1]) * 1e6
        if val.endswith('b'): return float(val[:-1]) * 1e9
        return float(val)
    except: return None

def extract_axis_dates_with_months(texts):
    # E.g. 'May '25', then a line '21' or '28'
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_pat = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.I)
    out = []
    last_month = None
    for idx, t in enumerate(texts):
        tokens = re.split(r"[ ,]", t)
        for tok in tokens:
            if month_pat.fullmatch(tok):
                last_month = tok.title()
            elif last_month and tok.isdigit() and 1 <= int(tok) <= 31:
                out.append((f"{tok} {last_month}", idx))
    return out

def extract_dates_with_pos(texts):
    patterns = [
        r"\b([A-Za-z]{3,9}[, ]+'\d{2,4})\b",
        r"\b([A-Za-z]{3,9}\s*\d{1,2}(?:,?\s*\d{2,4})?)\b",
        r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
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
    pattern = re.compile(r'(\d{1,2}[.:]?\d{2})\s*([APMapm]{2})')
    results = []
    for idx, t in enumerate(texts):
        for h, ap in pattern.findall(t):
            h_fmt = h.replace('.', ':')
            if ':' not in h_fmt and len(h_fmt)==3:
                h_fmt = h_fmt[0]+':' + h_fmt[1:]
            elif ':' not in h_fmt and len(h_fmt)==4:
                h_fmt = h_fmt[:2]+':' + h_fmt[2:]
            try:
                tm = datetime.datetime.strptime(f"{h_fmt} {ap.upper()}", "%I:%M %p").strftime("%I:%M %p").lstrip("0")
                results.append((tm, idx))
            except: continue
    return results

def infer_previous_trading_day(first_dt):
    try:
        # Handles "7/10" and "Jul 10" formats
        if '/' in first_dt:
            parts = first_dt.split('/')
            today = datetime.datetime.now()
            if len(parts) >= 2:
                year = today.year if len(parts) == 2 else int(parts[2])
                ref_day = datetime.datetime(year, int(parts[0]), int(parts[1]))
        else:
            month_lookup = {m: i+1 for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}
            tokens = first_dt.split()
            if len(tokens) == 2 and tokens[0].isdigit() and tokens[1] in month_lookup:
                year = datetime.datetime.now().year
                ref_day = datetime.datetime(year, month_lookup[tokens[1]], int(tokens[0]))
            else:
                return "prev"
        d = ref_day - datetime.timedelta(days=1)
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
        return d.strftime("%m/%d")
    except:
        return "prev"

def map_sessions(texts):
    # Prefer month+day, else fallback to numeric/date
    month_dates = extract_axis_dates_with_months(texts)
    explicit_dates = month_dates if month_dates else extract_dates_with_pos(texts)
    times_with_idx = extract_times_with_pos(texts)
    sessions = []
    all_date_indices = [idx for d, idx in explicit_dates]
    orphan_times = [(tm, idx) for tm, idx in times_with_idx if idx < (all_date_indices[0] if all_date_indices else len(texts))]
    if orphan_times and all_date_indices:
        prev_date = infer_previous_trading_day(explicit_dates[0][0])
        orph_t_sorted = sorted(tm for tm, _ in orphan_times)
        if orph_t_sorted:
            sessions.append({'date': prev_date, 'time_range': (orph_t_sorted[0], orph_t_sorted[-1])})
    # Map between date/month markers
    for i, (dt, idx) in enumerate(explicit_dates):
        block_times = []
        start_idx = idx + 1
        end_idx = explicit_dates[i+1][1] if i+1 < len(explicit_dates) else len(texts)
        for tm, t_idx in times_with_idx:
            if start_idx <= t_idx < end_idx:
                block_times.append(tm)
        block_times = sorted(set(block_times), key=lambda x: datetime.datetime.strptime(x, "%I:%M %p"))
        if block_times:
            sessions.append({'date': dt, 'time_range': (block_times[0], block_times[-1])})
        else:
            sessions.append({'date': dt})
    return sessions

def extract_ohlc(texts):
    block = re.search(r'O[: ]?([\d\.]+).*?H[: ]?([\d\.]+).*?L[: ]?([\d\.]+).*?C[: ]?([\d\.]+)', " ".join(texts))
    if block:
        try:
            o, h, l, c = float(block.group(1)), float(block.group(2)), float(block.group(3)), float(block.group(4))
            return {'O': o, 'H': h, 'L': l, 'C': c}
        except: return {}
    alt = re.search(r'H[ :]?([\d\.]+)\s*L[ :]?([\d\.]+)\s*C[ :]?([\d\.]+)', " ".join(texts))
    if alt:
        try:
            h, l, c = float(alt.group(1)), float(alt.group(2)), float(alt.group(3))
            return {'H': h, 'L': l, 'C': c}
        except: return {}
    return {}

def extract_volume(texts):
    for t in texts:
        m = re.search(r'vol undr[\D]*([\d\.,]+)', t, re.IGNORECASE)
        if m:
            v = convert_compact_number(m.group(1))
            if v: return v
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
    if not ohlc_vals: return None
    oh_vals = [v for v in ohlc_vals if v is not None and 1.0 < abs(v) < 1e7]
    if not oh_vals: return None
    oh_min, oh_max = min(oh_vals), max(oh_vals)
    for t in texts:
        value = t.replace(',', '').replace('-', '').strip()
        if re.fullmatch(r'\d{2,7}(?:\.\d{2})?', value):
            num = float(value)
            if (oh_min * 0.8) <= num <= (oh_max * 1.2):
                plausible.append(num)
    plausible = sorted(set(plausible))
    if len(plausible) >= 2:
        return (min(plausible), max(plausible))
    return None

def flexible_ticker_and_company_extract(texts):
    for t in texts:
        if (m := re.search(r'\(([A-Z0-9\-./]{1,10})\)', t)):
            return {'ticker': m.group(1), 'company_name': None}
        if (m := re.search(r'(NASDAQ|NYSE|BSE|LSE|HKEX|TSX)[:\s\-\.]+([A-Z]{2,8})', t)):
            if m.group(2) not in ['NASDAQ', 'NYSE', 'BSE', 'LSE', 'HKEX', 'TSX']:
                return {'ticker': m.group(2), 'company_name': None}
        if (m := re.match(r'^([A-Z]{2,8})$', t.strip())) and t.strip() not in ['NASDAQ', 'NYSE', 'BSE', 'LSE', 'HKEX', 'TSX']:
            return {'ticker': m.group(1), 'company_name': None}
    return {'ticker': None, 'company_name': None}

def parse_chart_metadata(ocr_texts):
    texts = normalize_ocr_texts(ocr_texts)
    metadata = {}

    id_fields = flexible_ticker_and_company_extract(texts)
    if id_fields['ticker']: metadata['ticker'] = id_fields['ticker']
    if id_fields.get('company_name'): metadata['company_name'] = id_fields['company_name']
    for line in texts:
        exch = re.search(r'\b(NYSE|BSE|NASDAQ|TSX|LSE|HKEX)\b', line)
        if exch: metadata['exchange'] = exch.group(1)
    if any("USD" in t for t in texts): metadata["currency"] = "USD"
    elif any("INR" in t for t in texts): metadata["currency"] = "INR"

    ohlc = extract_ohlc(texts)
    vol = extract_volume(texts)
    if ohlc:
        if vol: ohlc['V'] = vol
        metadata['ohlc'] = ohlc

    price_range = extract_price_range(texts, list(ohlc.values()) if ohlc else [])
    if price_range: metadata['price_range'] = price_range

    sessions = map_sessions(texts)
    if sessions:
        metadata['sessions'] = sessions
        metadata['dates'] = [sess['date'] for sess in sessions]

    return metadata


# General-purpose function to extract metadata from any image source
def extract_image_metadata(image_source):
    """Extract metadata from an image source (path or image object)."""
    # Load image
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise ValueError(f"Unable to load image from path: {image_source}")
        img_path = image_source
    else:
        img = image_source
        img_path = None

    reader = easyocr.Reader(['en'])
    # OCR on full image
    results = reader.readtext(img_path if img_path else img)
    # OCR on cropped and resized axis region
    height = img.shape[0]
    cropped = img[int(height * 0.85):, :]
    resized = cv2.resize(cropped, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    results += reader.readtext(resized)

    # Visualization (optional)
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("OCR Results on Cropped & Resized X-Axis Region")
    plt.show()

    print("\n--- OCR Results ---")
    for bbox, text, conf in results:
        print(f"{text} ({conf:.2f})")

    raw_texts = [text for _, text, _ in results]
    metadata = parse_chart_metadata(raw_texts)
    print(metadata)
    return metadata

# Example usage:
metadata = extract_image_metadata(r"E:\\FinGPT-M\\Datasets\\Stock Charts for OCR\\AIG2_png.rf.1fc60fef6a377e18b489802cb465786b.jpg")