import os
import re
import cv2
import json
import easyocr
import datetime
import matplotlib.pyplot as plt

class StockChartMetadataExtractor:
    def __init__(self, image_source):
        self.image_source = image_source
        self.reader = easyocr.Reader(['en'])
        self.metadata = {}

    def load_image(self):
        img = cv2.imread(self.image_source)
        if img is None:
            raise ValueError(f"Unable to load image from path: {self.image_source}")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray


    def normalize_ocr_texts(self, ocr_texts):
        cleaned = []
        for t in ocr_texts:
            t = t.replace('|', '/')

            substitutions = [
                (r'(^|\s)0\.', r'\1O:'),
                (r'\bO\.', 'O:'),
                (r'\bH\.', 'H:'),
                (r'\bL\.', 'L:'),
                (r'\bC\.', 'C:'),
                (r'\bV\.', 'V:')
            ]
            t = t.replace('0:', 'O:')
            pattern = r'\b0(?!\.\d)([a-zA-Z0-9]*)'
            t = re.sub(pattern, r'O\1', t)
            t = t.replace('.O', '.0')
            for p, r_ in substitutions:
                t = re.sub(p, r_, t)

            t = re.sub(r'(?<=[A-Za-z])1(?=[A-Za-z])', 'l', t)
            t = t.replace(';', ',')
            t = re.sub(r'(?<=[A-Za-z]),?\(', ' (', t)
            t = re.sub(r'\s+', ' ', t)

            final_text = t.strip()
            cleaned.append(final_text)
            print(f"Normalized text: {final_text}")

        return cleaned


    def convert_compact_number(self, val):
        val = val.replace(',', '').strip().lower()
        try:
            if val.endswith('k'): return float(val[:-1]) * 1e3
            if val.endswith('m'): return float(val[:-1]) * 1e6
            if val.endswith('b'): return float(val[:-1]) * 1e9
            return float(val)
        except: return None

    def extract_axis_dates_with_months(self, texts):
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

    def extract_dates_with_pos(self, texts):
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

    def extract_times_with_pos(self, texts):
        pattern = re.compile(r'(\d{1,2}[.:]?\d{2})\s*([APMapm]{2})')
        results = []
        for idx, t in enumerate(texts):
            for h, ap in pattern.findall(t):
                h_fmt = h.replace('.', ':')
                if ':' not in h_fmt and len(h_fmt) == 3:
                    h_fmt = h_fmt[0] + ':' + h_fmt[1:]
                elif ':' not in h_fmt and len(h_fmt) == 4:
                    h_fmt = h_fmt[:2] + ':' + h_fmt[2:]
                try:
                    tm = datetime.datetime.strptime(f"{h_fmt} {ap.upper()}", "%I:%M %p").strftime("%I:%M %p").lstrip("0")
                    results.append((tm, idx))
                except:
                    continue
        return results

    def infer_previous_trading_day(self, first_dt):
        try:
            if '/' in first_dt:
                parts = first_dt.split('/')
                today = datetime.datetime.now()
                if len(parts) >= 2:
                    year = today.year if len(parts) == 2 else int(parts[2])
                    ref_day = datetime.datetime(year, int(parts[0]), int(parts[1]))
            else:
                month_lookup = {m: i + 1 for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
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

    def map_sessions(self, texts):
        month_dates = self.extract_axis_dates_with_months(texts)
        explicit_dates = month_dates if month_dates else self.extract_dates_with_pos(texts)
        times_with_idx = self.extract_times_with_pos(texts)
        sessions = []
        all_date_indices = [idx for d, idx in explicit_dates]
        orphan_times = [(tm, idx) for tm, idx in times_with_idx if idx < (all_date_indices[0] if all_date_indices else len(texts))]
        if orphan_times and all_date_indices:
            prev_date = self.infer_previous_trading_day(explicit_dates[0][0])
            orph_t_sorted = sorted(tm for tm, _ in orphan_times)
            if orph_t_sorted:
                sessions.append({'date': prev_date, 'time_range': (orph_t_sorted[0], orph_t_sorted[-1])})
        for i, (dt, idx) in enumerate(explicit_dates):
            block_times = []
            start_idx = idx + 1
            end_idx = explicit_dates[i + 1][1] if i + 1 < len(explicit_dates) else len(texts)
            for tm, t_idx in times_with_idx:
                if start_idx <= t_idx < end_idx:
                    block_times.append(tm)
            block_times = sorted(set(block_times), key=lambda x: datetime.datetime.strptime(x, "%I:%M %p"))
            if block_times:
                sessions.append({'date': dt, 'time_range': (block_times[0], block_times[-1])})
            else:
                sessions.append({'date': dt})
        return sessions

    # def extract_ohlc(self, texts):
    #     block = re.search(r'O[: ]?([\d\.]+).*?H[: ]?([\d\.]+).*?L[: ]?([\d\.]+).*?C[: ]?([\d\.]+)', " ".join(texts))
    #     if block:
    #         try:
    #             o, h, l, c = float(block.group(1)), float(block.group(2)), float(block.group(3)), float(block.group(4))
    #             return {'O': o, 'H': h, 'L': l, 'C': c}
    #         except:
    #             return {}
    #     return {}

    def extract_ohlc(self, texts):
        joined_text = " ".join(texts)

        block = re.search(
            r'O[: ]?([\d,]+(?:\.\d+)?).*?'
            r'H[: ]?([\d,]+(?:\.\d+)?).*?'
            r'L[: ]?([\d,]+(?:\.\d+)?).*?'
            r'C[: ]?([\d,]+(?:\.\d+)?)',
            joined_text
        )

        if block:
            try:
                o = float(block.group(1).replace(",", ""))
                h = float(block.group(2).replace(",", ""))
                l = float(block.group(3).replace(",", ""))
                c = float(block.group(4).replace(",", ""))

                return {'O': o, 'H': h, 'L': l, 'C': c}
            except ValueError:
                return {}

        return {}


    def extract_volume(self, texts):
        preferred = set()
        fallback = set()

        interval_keywords = {'1d', '5d', '1m', '3m', '6m', 'ytd', '1y', '5y', 'all', 'ty', 'aii'}

        for i, t in enumerate(texts):
            matches = re.findall(r'\b[vV][o0]?[l1]?[^\w\d]{0,2}[:\s]?([\d,.]+[kKmMbB]?)', t)
            preferred.update(matches)

            if re.search(r'\b[vV][o0]?[l1]\b', t) and i + 1 < len(texts):
                next_line = texts[i + 1].strip()
                if re.match(r'^[\d,.]+[kKmMbB]?$', next_line):
                    preferred.add(next_line)

            if not re.search(r'\b[vV][o0]?[l1]\b', t):
                bars = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?[kKmMbB]\b', t)
                fallback.update(bars)

        preferred_vals = [self.convert_compact_number(v) for v in preferred if self.convert_compact_number(v)]
        if preferred_vals:
            return max(preferred_vals)

        fallback_vals = [
            self.convert_compact_number(v)
            for v in fallback
            if v.lower() not in interval_keywords and self.convert_compact_number(v)
        ]
        return max(fallback_vals) if fallback_vals else None


    EXCHANGES   = {"ACRONYM", "ZSE", "ZOBEX", "ZCE", "ZBX", "ZARX", "XTRD", "XTND", "XOFF", "XMIO", "XMFE", "XETRA MIDPOINT", "XETRA", "XBERRY", "XABX", "WSE", "WSAG", "WINS", "WFSE", "WEL SI", "WCHV", "WBC SI", "WBAG", "WABR", "VUB, A.S.", "VNDM", "VMX", "VLEX", "VIRTU SI", "VIRTU OTC", "VFIL", "VFEX", "VFCM", "VERT", "VEQ", "VCMM", "VCM OTF", "VAL", "VAGM MTF", "VAGLOBAL", "UZSE", "USE", "US QUOTE", "UPCOM", "UKRSE", "UICE", "UFEX", "UBS ATS", "UBIS", "UBI", "UBECU", "TWSEF", "TWSE", "TURKDEX", "TTSE", "TSXV DRK", "TSX-V", "TSX DRK", "TSX", "TSE", "TSAF OTC", "TSAF", "TRB", "TRAD-X", "TRADEGATE EXCHANGE", "TPSG", "TPIE", "TPEX", "TOCOM", "TISE", "THEM", "TGE", "TGAG", "TFX", "TFSC", "TFS", "TFEX", "TERM", "TDX", "TD BANK", "TASE", "TAIFEX", "TAH", "T212CY", "T212 IE", "SYFX", "SWAPEX", "SVGEX", "SVEX", "SUPERX EU", "STFL", "STF", "ST", "SSX", "SSWMM", "SSIL", "SSE", "SSBTCO", "SSBI", "SPX", "SPSE", "SPOTEX", "SPIMEX", "SPHR", "SPCEX", "SPBEX", "SPB", "SOHO", "SNUK", "SIX", "SISI", "SIMEX", "SIDAC", "SICOM", "SIBE", "SIB", "SHFE", "SGX-DT", "SGX-BT", "SGX", "SGOE SI", "SGMT", "SGE", "SGAS", "SG SI", "SFOX", "SFMP", "SFE", "SET", "SEND", "SENAF", "SEHK", "SEED CX", "SEDEX", "SEBA", "SEB", "SCIEX", "SB1M", "SASE", "SANT", "SAFEX", "S3FM", "RTSPL", "RTSL", "RSEF (RFS)", "RSEF", "RSE", "ROFEX", "RMS CZ", "RIVERX", "RFQ", "REGS", "REGF", "RCBX", "RCB", "RBS PLC", "RBM", "RBHU-SI", "RBCZ", "RBCCM", "RBC", "RASDAQ", "R4G", "QWIK", "PXE", "PVBL", "PUMA", "PTP", "PTE", "PT LSEG", "PSX", "PSE", "PROS", "PROPEX", "POTC", "POSIT", "PO CAP", "PNGX", "PMX", "PMEX", "PLUS-SX", "PLUS-DX", "PKO BP", "PJSC NDU", "PHLX", "PGT", "PGSL", "PFTS", "PESL", "PEKAO", "PEEL", "PDQX", "PAVE", "OYLD", "OTC-X", "OTCEI", "OTCBB", "OSLSG", "OSLDS", "OSE", "OPTX", "OPRA", "OPEX", "OP", "OMIP", "OMIE", "OMICLEAR", "ODX", "ODE", "NZX", "NYSEDARK", "NYSE", "NYPC", "NYMEX MTF LIMITED", "NYMEX ECM", "NYMEX", "NXVWAP", "NXT", "NXSE", "NXJP", "NXFO", "NWM NV", "NURODARK", "NSXA", "NSX", "NSL", "NSE IFSC", "NSE", "NSDQDARK", "NSC", "NQBXDARK", "NPM", "NP RTS", "NOTC", "NOS", "NOREXECO", "NLX", "NLB", "NIBOR", "NGX", "NGS", "NGM", "NFX", "NEX", "NEWEX", "NEO-N", "NEO-L", "NEO-D", "NEO CONNECT", "NEEQ", "NDXB", "NDX", "NCSE", "NCDEX", "NBF", "NBCT", "NATX", "NASDOTC", "NASDAQ", "NAMEX", "NADEX", "NABE", "NAB", "MUTI", "MUFP", "MUBP SI", "MUBM SI", "MUBL SI", "MUBE SI", "MTS ITALY", "MTA", "MSX", "MSNT", "MSM", "MSIL OTF", "MSEU", "MSEL OTF", "MSE", "MSDM", "MSCX", "MSAX", "MSA", "MPRL", "MOT", "MOSENEX", "MOEX", "MOCX", "MNSE", "MLIX", "MLI", "MLFE", "MLCX", "MIX", "MIV", "MILLENNIUMBCP", "MIHI", "MICEX FAR EAST", "MIBL", "MIBGAS", "MHI", "MHEU", "MGEX", "MFB", "MFAO", "MEXDER", "MESI", "MERVAROS", "MERVAL", "MERJ", "MERF", "MEPD", "MEMXDARK", "MEMX", "MELO", "MEFF", "MEDIP", "ME", "MDX", "MCX", "MBANK", "MATBA", "MARF", "MARCH", "MANUAL OTC", "MAE", "MAC", "MAB", "LXJP", "LTSE", "LTAA", "LSX", "LSE APA", "LSE", "LQNT H20", "LQNT", "LQNF", "LQNA", "LQFI", "LPPM", "LMNX", "LMEC", "LME", "LLBW", "LH", "LFX", "LEVEL", "LDX", "LCX", "LCM", "LBCM", "LATINEX", "LATG", "LAMP", "KSL", "KSE", "KRX SM", "KRX FM", "KOSDAQ", "KONEX", "KISE", "KEX", "KCP", "KCGM", "KCBT", "KAZE", "JX", "JSF", "JSE", "JPX", "JPMX", "JPMI", "JPBX", "J-NET", "JLQD", "JLEQ", "JAX", "JASDAQ", "IXSP", "ITS", "ISX", "ISM", "ISL OTF", "ISE GEM", "ISE ASM", "ISE", "IPSX", "IPEX/GME", "IOM", "INE", "INDX", "INDIA INX", "IMC", "IMAREX", "IFXC MTF", "IFSM", "IFB", "IEX DAX", "IEX", "IEOS", "IDX", "IDEM", "ICEX", "ICE ENDEX SPOT", "ICE ENDEX RM", "ICE ENDEX OTF", "ICE ENDEX OCM", "ICE ENDEX EQUITY", "ICE", "ICBCS", "ICAPS", "IBSC", "IBKR", "IBGH", "IBEX", "IBERCAJA", "IBCO", "IBA", "IATS", "IAB", "HUPX", "HUDEX", "HSBC SI", "HRT", "HPPO", "HPF", "HPC IS", "HPC", "HOSE", "HNX", "HKMEX", "HKFE", "HKEX", "HK GEM", "HENEX S.A.", "HENEX", "HDAT", "GTX", "GTSX", "GSX", "GSPIC", "GSIB", "GSI", "GSET", "GSE", "GSCO", "GRIFFIN MARKETS LIMITED", "GPWB", "GPN", "GOVEX", "GMX", "GMGLDN", "GMGDXB", "GMEX", "GME", "GLPX", "GLPS", "GLOBALCOAL", "GGI", "GFO-X", "GFIA", "GFI SEF", "GFEX", "GET BALTIC", "GEMMA", "GDX", "GCX", "GBS", "GAI", "FXSWAP", "FXM", "FXCM", "FXCLR", "FRTSIL", "FNXB", "FMXX", "FMX NDF", "FLOWDARK", "FIS", "FINN", "FICONEX", "FGM", "FEX", "FBSI", "EXSE", "EXLP", "EXIST", "EXEU", "EXDC", "EXANE BNP PARIBAS", "EXAA", "EWSM", "EVOL", "EUWAX", "EUROFIN", "ETPA", "ETFPLUS", "ESPEED", "ESE", "EPRL", "EPRD", "ENC", "ENAX", "EMTS", "EMLD", "EMBX", "EGM", "EEXCHANGE", "EEX", "EDXM", "EDGXDARK", "EDGX", "EDGO", "EDGADARK", "EDGA", "ECSE", "ECC", "EBS", "EBI", "EBB", "DWSEF", "DVFX", "DSM", "DSE", "DOTS", "DGCX", "DFM", "DDT", "DCX", "DCSX", "DCE", "DCA", "DBX", "DBOT", "DBMID", "DBES", "DARK", "DANSK AMP", "CXE OFF-BOOK", "CXE DARK", "CXA BIDS BLOCK", "CXA BIDS BIDS TWPI", "CXA", "CX LDFX", "CX", "CTSE", "CSZ", "CSX", "CSSEL SI", "CSOB SK", "CSOB", "CSI SI", "CSI", "CSE-PURE", "CSEB", "CSE LISTED", "CSE", "CSD SI", "CSD", "CSAGLB SI", "CSA OHS", "CSA", "CROPEX", "CREDEM SI", "CONTICAP OTF", "COMEX", "CODA", "CNX MTF", "CNODE", "CME (FLOOR)", "CME", "CM SEF", "CM MTF", "CM ETP", "CM", "CLX", "CLST", "CLEARCORP", "CIOI", "CIC", "CIBC WM PLC", "CIBC", "CHI-X", "CGML", "CFIM", "CFI", "CFETSBC", "CFETS", "CFE", "CF", "CENTRAL RISK BOOK", "CDE", "CCX", "C-COM", "CCLUX", "CCGTRIPARTY", "CCGEUROBONDS", "CCGEQUITYDER", "CCGEQUITY", "CCGENERGYDER", "CCGBONDS", "CCGAGRIDER", "CCG", "CBSX", "CBOT (FLOOR)", "CBOT", "CBOE OFF-BOOK", "CBOE OFF EXCHANGE", "CBOE NL CEDX", "CBOE FX", "CBOE EUROPE B.V.", "CBOE EUROPE", "CBOE EU RM", "CBOE EU REGM OFF BOOK", "CBOE EU REGM LIT", "CBOE EU REGM DARK", "CBOE EU OFF EXCHANGE", "CBOE EU LIS", "CBOE EU DXE PERIODIC", "CBOE EU DXE OFF BOOK", "CBOE EU DXE DARK", "CBOE EU DXE", "CBOE EU DARK", "CBOE EU CXE", "CBOE EU BXE OFF BOOK", "CBOE EU BXE", "CBOE DARK", "CBOE  REGM OFF BOOK", "CBOE  REGM LIT", "CBOE  REGM DARK", "CBOE  LIS", "CBOE  EUROPE", "CBOE", "CBLC", "CATS", "CASE", "CAR", "CAPI", "CANTOR", "CANNEX", "CANDEAL", "CACIB UK SI", "CACIB SI", "CABK", "C2OX", "BZXDARK", "BZ WBK", "BYXX", "BYXDARK", "BXE PERIODIC", "BVRD", "BVPASA", "BVMT", "BVM", "BVL", "BVCC", "BVC", "BVB", "BTSL", "BTL", "BTEC CHICAGO", "BSX", "BSSE", "BSE-MTF", "BSEF", "BSE SME", "BSE", "BSAB", "BRVM", "BRUT", "BRM", "BPAG", "BP2S LB SI", "BP2S", "BOX", "BOT", "BOSS", "BOLCHILE", "BOFASE", "BODIVA", "BOATS", "BNYMEL", "BNYM", "BNV", "BNPX", "BNPPF SI", "BNPP SA SI", "BNPP SA LB SI", "BNPP A SI", "BNDS", "BMO", "BMFMS", "BME APA", "BME", "BLSE", "BLPX", "BLOX", "BLKX", "BIVA", "BITGEM", "BISX", "BILV", "BIL", "BIDS", "BHW SI", "BGUK", "BGL", "BGFX", "BGFU", "BGFI", "BGCDML", "BGC SKL", "BETP", "BET", "BESA", "BERN-X", "BEM", "BELEX", "BELEN/MEPX", "BEAM", "BCV", "BCSL SI", "BCSL", "BCSE", "BCR", "BCEE", "BCE", "BCDX", "BCC", "BCBA", "BCAP LX", "BBX", "BBVA", "BBSI", "BBPLC SI", "BBPLC", "BBOK", "BBJ", "BBI SI", "BBI", "BATS", "BATO", "BANA LONDON", "BANA", "BAMLI DAC", "BAMLI", "BAFR", "B3", "AWEX", "AWB", "AUCTIONS", "ATHEXD", "ATHEXC", "ATHEX APA", "ATFUND", "ATDF", "ASX", "ASTROID", "ASE", "ARTEX GM", "ARKX", "AREX", "ARCADARK", "AQX", "AQUIS-EIX", "AQSE", "AQS", "APEX CLEAR", "APEX", "APA CAPI OTF", "ANZBGL", "ANTS", "AMX", "AMPX", "AMEXDARK", "AMEX", "ALTX UGANDA", "ALTX", "ALSI", "ALSE", "ALLT-OTF", "AKIS", "AIX", "AIAF", "AFDLAPA", "ADSM", "ACM", "ACKFI", "ABM", "ABG", "AACB SI", "A2X", "4AX", "42FS", "24EX", "21X"}
    CURRENCIES  = {'INR','USD','EUR','GBP','JPY','CNY','CAD','AUD','SGD','CHF'}

    PAT_BRACKET   = re.compile(r'\(([A-Z0-9\-./]{1,12})\)')
    PAT_SUFFIX    = re.compile(r'([A-Z0-9.\-]{2,12})\.([A-Z]{2,6})$')
    PAT_EX_PREFIX = re.compile(r'(NASDAQ|NYSE|BSE|LSE|HKEX|TSX|NSE)[:\s\-\.]+([A-Z0-9.\-]{2,12})')
    PAT_STANDALONE= re.compile(r'^(?=.*[A-Z])[A-Z0-9.\-]{2,12}$')  # at least one letter

    def looks_numeric(self, tok: str) -> bool:
        return bool(re.fullmatch(r'\d+(?:\.\d+)?', tok))

    def merge_company_name_lines(self, ocr_texts):
        merged = []
        i = 0

        garbage_words = {
            "BUY", "SELL", "VOL", "TRADINGVIEW", "ADJ",
            "OPEN", "HIGH", "LOW", "CLOSE", "MARKET", "USD"
        }

        while i < len(ocr_texts):
            line = ocr_texts[i].strip()

            if i + 1 < len(ocr_texts):
                next_line = ocr_texts[i + 1].strip()

                if not any(ex in line.upper() for ex in self.EXCHANGES):

                    if (
                        next_line
                        and next_line[0].isupper()
                        and not any(word in next_line.upper() for word in garbage_words)
                        and not re.search(r"\d", next_line)
                    ):
                        line = f"{line} {next_line}"
                        i += 1

            merged.append(line)
            i += 1

        print(f"Merged lines for company names: {merged}")
        return merged


    def is_probably_company_name(self, t):
        t_lower = t.lower()
        if any(word in t_lower for word in ["vol", "open", "high", "low", "close", "market", "usd"]):
            return False
        if sum(c.isdigit() for c in t) > len(t) / 2:
            return False
        return True

    def looks_like_garbage(self, text):
        garbage_keywords = {"TRADINGVIEW", "ADJ", "BUY", "SELL", "VOL", "O:", "H:", "L:", "C:"}
        if any(g in text.upper() for g in garbage_keywords):
            return True
        return False

    # def extract_company_name(self, texts):
    #     texts = self.merge_company_name_lines(texts)

    #     if isinstance(self.EXCHANGES, (set, list, tuple)):
    #         exchanges_pattern = r"(" + "|".join(sorted(self.EXCHANGES)) + r")"
    #     else:
    #         exchanges_pattern = self.EXCHANGES

    #     for i, t in enumerate(texts):
    #         t = t.strip()
    #         if not self.is_probably_company_name(t):
    #             continue

    #         t_clean = re.sub(r'\s+', ' ', t)

    #         m = re.match(rf"^(.+?)\s+{exchanges_pattern}\b", t_clean, re.IGNORECASE)
    #         if m:
    #             return m.group(1).strip()

    #         m = re.match(rf"^([A-Za-z0-9 &.,'\-]+)\s+(?:[·\-\.\|]| {2,})\s+{exchanges_pattern}", t_clean, re.IGNORECASE)
    #         if m:
    #             return m.group(1).strip()

    #         m = re.match(r"^(.+?)\s+\(\s*([A-Z]{1,6}(?::[A-Z]{1,6})?)\s*\)$", t_clean)
    #         if m:
    #             return m.group(1).strip()

    #         if re.match(r"^[A-Z]{1,5}\s+[A-Z][a-zA-Z0-9&.,'\-]{2,}", t_clean) and not self.looks_like_garbage(t_clean):
    #             return re.sub(r"^[A-Z]{1,5}\s+", "", t_clean).strip()

    #         if t_clean.isupper() and any(word in t_clean for word in ["LTD", "INC", "CORP", "PLC"]):
    #             return t_clean.strip()

    #         if i + 1 < len(texts):
    #             next_line = texts[i + 1].strip()
    #             if re.search(rf"\b{exchanges_pattern}\b", next_line, re.IGNORECASE):
    #                 return t_clean

    #     return None

    def extract_company_name(self, texts):
        texts = self.merge_company_name_lines(texts)

        if isinstance(self.EXCHANGES, (set, list, tuple)):
            exchanges_pattern = r"(" + "|".join(sorted(self.EXCHANGES)) + r")"
        else:
            exchanges_pattern = self.EXCHANGES

        for i, t in enumerate(texts):
            t = t.strip()
            if not self.is_probably_company_name(t):
                continue

            t_clean = re.sub(r'\s+', ' ', t)
            t_clean = re.sub(r"^\d{1,7}(?:,\d{3})*(?:\.\d+)?\s+", "", t_clean)

            m = re.match(rf"^(.+?)\s+{exchanges_pattern}\b", t_clean, re.IGNORECASE)
            if m:
                return m.group(1).strip()

            m = re.match(rf"^([A-Za-z0-9 &.,'\-]+)\s+(?:[·\-\.\|]| {2,})\s+{exchanges_pattern}", t_clean, re.IGNORECASE)
            if m:
                return m.group(1).strip()

            m = re.match(r"^(.+?)\s+\(\s*([A-Z]{1,6}(?::[A-Z]{1,6})?)\s*\)$", t_clean)
            if m:
                return m.group(1).strip()

            if re.match(r"^[A-Z]{1,5}\s+[A-Z][a-zA-Z0-9&.,'\-]{2,}", t_clean) and not self.looks_like_garbage(t_clean):
                return re.sub(r"^[A-Z]{1,5}\s+", "", t_clean).strip()

            if t_clean.isupper() and any(word in t_clean for word in ["LTD", "INC", "CORP", "PLC"]):
                return t_clean.strip()

            if i + 1 < len(texts):
                next_line = texts[i + 1].strip()
                if re.search(rf"\b{exchanges_pattern}\b", next_line, re.IGNORECASE):
                    return t_clean

        return None

    def extract_ticker(self, texts):
            garbage_keywords = {
                "BUY", "SELL", "VOL", "TRADINGVIEW", "ADJ", "RTH", "UTC",
                "OPEN", "HIGH", "LOW", "CLOSE", "MARKET", "USD",
                "O", "H", "L", "C", "1D", "5D", "1M", "3M", "6M", "YTD", "ALL", "SD"
            }

            def is_garbage(word):
                if sum(c.isdigit() for c in word) > len(word) / 2:
                    return True
                if word.upper() in garbage_keywords:
                    return True
                return False

            possible_tickers = []

            for t in texts:
                tok = t.strip()
                if not tok or is_garbage(tok):
                    continue

                m = re.match(r'^([A-Z0-9]{1,10})\.([A-Z]{2,6})$', tok)
                if m:
                    suffix = m.group(2)
                    if suffix in self.EXCHANGES:
                        possible_tickers.append(m.group(1))
                        continue

                m = re.search(r'\(([A-Z]{1,10})\)', tok)
                if m:
                    ticker = m.group(1)
                    if ticker not in self.EXCHANGES and ticker not in self.CURRENCIES and not is_garbage(ticker):
                        possible_tickers.append(ticker)
                        continue

                m = re.search(r'(NASDAQ|NYSE|BSE|LSE|HKEX|TSX|NSE)[:\s\-]+([A-Z0-9]{1,10})', tok)
                if m:
                    ticker = m.group(2)
                    if ticker not in self.EXCHANGES and ticker not in self.CURRENCIES and not is_garbage(ticker):
                        possible_tickers.append(ticker)
                        continue

                if (
                    re.fullmatch(r'[A-Z]{1,5}\d?', tok)  # 1–5 letters + optional 1 digit
                    and not is_garbage(tok)
                    and tok not in self.EXCHANGES
                    and tok not in self.CURRENCIES
                ):
                    possible_tickers.append(tok)
                    continue

            if possible_tickers:
                return {"ticker": possible_tickers[0], "company_name": None}

            company_name = self.extract_company_name(texts)
            return {"ticker": "N/A", "company_name": company_name}


    # def extract_price_range(self, texts, ohlc_vals):
    #     plausible = []
    #     if not ohlc_vals: return None
    #     oh_vals = [v for v in ohlc_vals if v is not None and 1.0 < abs(v) < 1e7]
    #     if not oh_vals: return None
    #     oh_min, oh_max = min(oh_vals), max(oh_vals)
    #     for t in texts:
    #         value = t.replace(',', '').replace('-', '').strip()
    #         if re.fullmatch(r'\d{2,7}(?:\.\d{1,3})?', value):
    #             num = float(value)
    #             if (oh_min * 0.8) <= num <= (oh_max * 1.2):
    #                 plausible.append(num)
    #     plausible = sorted(set(plausible))
    #     if len(plausible) >= 2:
    #         return (min(plausible), max(plausible))
    #     return None

    def extract_price_range(self, texts, ohlc_vals):
        plausible = []
        if not ohlc_vals:
            return None

        raw_oh_vals = [v for v in ohlc_vals if v is not None and v > 0]

        if not raw_oh_vals:
            return None

        median_price = sorted(raw_oh_vals)[len(raw_oh_vals) // 2]

        oh_vals = [v for v in raw_oh_vals if v <= median_price * 5]

        if not oh_vals:
            return None

        oh_min, oh_max = min(oh_vals), max(oh_vals)

        for t in texts:
            value = t.replace(',', '').replace('-', '').strip()

            if re.fullmatch(r'\d{1,7}(?:\.\d{1,3})?', value):
                num = float(value)

                if (oh_min * 0.8) <= num <= (oh_max * 1.2):
                    plausible.append(num)

        plausible = sorted(set(plausible))
        if len(plausible) >= 2:
            return (min(plausible), max(plausible))

        return None


    def parse_chart_metadata(self, ocr_texts):
        texts = self.normalize_ocr_texts(ocr_texts)
        metadata = {}
        company = self.extract_company_name(texts)
        if company:
            metadata['company_name'] = company

        id_fields = self.extract_ticker(texts)
        if id_fields['ticker']:
            metadata['ticker'] = id_fields['ticker']

        for line in texts:
            pattern = r'\b(' + '|'.join(map(re.escape, self.EXCHANGES)) + r')\b'
            exch = re.search(pattern, line, re.IGNORECASE)
            if exch:
                metadata['exchange'] = exch.group(1)
        if any("USD" in t for t in texts): metadata["currency"] = "USD"
        elif any("INR" in t for t in texts): metadata["currency"] = "INR"
        ohlc = self.extract_ohlc(texts)
        vol = self.extract_volume(texts)
        if ohlc:
            if vol: ohlc['V'] = vol
            metadata['ohlc'] = ohlc
        price_range = self.extract_price_range(texts, list(ohlc.values()) if ohlc else [])
        if price_range: metadata['price_range'] = price_range
        sessions = self.map_sessions(texts)
        if sessions:
            metadata['sessions'] = sessions
            metadata['dates'] = [sess['date'] for sess in sessions]
        return metadata

    # def extract_metadata(self):
    #     img_gray = self.load_image()
    #     results = self.reader.readtext(img_gray)
    #     raw_texts = [text for _, text, _ in results]
    #     metadata = self.parse_chart_metadata(raw_texts)
    #     return metadata


    def extract_metadata(self):
        img_gray = self.load_image()

        results = self.reader.readtext(img_gray)
        raw_texts = [text for _, text, _ in results]

        h, w = img_gray.shape[:2]
        x_axis_crop = img_gray[int(h * 0.9):h, 0:w]  # bottom 10%

        # If image already grayscale, skip conversion
        if len(x_axis_crop.shape) == 3:
            gray = cv2.cvtColor(x_axis_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = x_axis_crop

        _, binarized = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        x_axis_results = self.reader.readtext(binarized)
        x_axis_texts = [text for _, text, _ in x_axis_results]

        raw_texts.extend(x_axis_texts)

        metadata = self.parse_chart_metadata(raw_texts)
        return metadata


    def save_metadata_to_json(self, filename):
        path = "output/"+filename
        if not os.path.exists("output"):
            os.makedirs("output")
        metadata = self.extract_metadata()
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {filename}")
        return metadata