# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import json
from datetime import datetime, date, timedelta
import calendar

# ------------------------------------------------------------
# Küçük yardımcılar
# ------------------------------------------------------------
def _to_int_safe(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, str):
            v = val.strip()
            if v == "":
                return None
            if "." in v:  # "16358.0" gibi
                v = v.split(".")[0]
            return int(v)
        if isinstance(val, float):
            return int(round(val))
        return int(val)
    except Exception:
        return None

def _to_float_safe(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except Exception:
        return None

def normalize_roc(val):
    """
    ROC değerini güvenli biçimde normalize eder.
    Örnekler:
      16358.0 -> "16358"
      "OPET #16358" -> "16358"
      "16358" -> "16358"
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    # sonda .0 varsa at
    if s.endswith(".0"):
        s = s[:-2]
    # sadece ilk rakam blokunu al
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None


# ------------------------------------------------------------
# Supabase entegrasyonu
# ------------------------------------------------------------
try:
    from modules.supabase_client import get_supabase_client  # sadece client
    SUPABASE_ENABLED = True
except ImportError:
    SUPABASE_ENABLED = False
    print("Supabase modülü yüklenemedi. Lokal modda çalışıyor.")


# ------------------------------------------------------------
# ✅ CLOUD DEMO VERİ YÜKLEME (GitHub RAW)
# ------------------------------------------------------------
def load_demo_data_from_cloud():
    """Cloud (GitHub Raw) üzerinden gerçek verileri yükle"""
    try:
        TLAG_DEMO_URL = "https://raw.githubusercontent.com/okilicarslan/tlag-cloud-data/refs/heads/main/tlag_demo.csv"
        COMMENTS_DEMO_URL = "https://raw.githubusercontent.com/okilicarslan/tlag-cloud-data/refs/heads/main/comments_demo.csv"

        import requests
        from io import StringIO

        r1 = requests.get(TLAG_DEMO_URL, timeout=30)
        r1.raise_for_status()
        tlag_df = pd.read_csv(StringIO(r1.text), low_memory=False)

        for col in ["SKOR", "GEÇEN SENE SKOR", "Fark", "TRANSACTION"]:
            if col in tlag_df.columns:
                tlag_df[col] = pd.to_numeric(tlag_df[col], errors="coerce")

        if "ROC_STR" not in tlag_df.columns and "ROC" in tlag_df.columns:
            tlag_df["ROC_STR"] = tlag_df["ROC"].astype(str).str.split(".").str[0]

        r2 = requests.get(COMMENTS_DEMO_URL, timeout=60)
        r2.raise_for_status()
        comments_df = pd.read_csv(
            StringIO(r2.text),
            low_memory=False,
            dtype={"station_code": "string"}  # baştaki sıfırları korur
        )

        if "categories" in comments_df.columns:
            def _to_list(x):
                if isinstance(x, str) and x.strip().startswith("["):
                    try:
                        return json.loads(x)
                    except Exception:
                        return ["GENEL"]
                return [x] if pd.notna(x) else ["GENEL"]
            comments_df["categories"] = comments_df["categories"].apply(_to_list)

        return tlag_df, comments_df, "Cloud verileri başarıyla yüklendi!"
    except Exception as e:
        return None, None, f"Cloud veri yükleme hatası: {str(e)}"


# ------------------------------------------------------------
# Demo verisi oluşturma / export (opsiyonel)
# ------------------------------------------------------------
def create_demo_data_files():
    np.random.seed(42)

    stations_data = [
        # İSTANBUL BÖLGE
        {"ROC": 4001, "İstasyon": "OPET KARTAL", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL ANADOLU", "Site Segment": "My Precious", "SKOR": 0.85, "GEÇEN SENE SKOR": 0.78, "Fark": 7.0, "TRANSACTION": 25000},
        {"ROC": 4002, "İstasyon": "SHELL KADIKÖY", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL ANADOLU", "Site Segment": "My Precious", "SKOR": 0.82, "GEÇEN SENE SKOR": 0.80, "Fark": 2.0, "TRANSACTION": 32000},
        {"ROC": 4003, "İstasyon": "BP ÜSKÜDAR", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL ANADOLU", "Site Segment": "Wasted Talent", "SKOR": 0.68, "GEÇEN SENE SKOR": 0.72, "Fark": -4.0, "TRANSACTION": 18000},
        {"ROC": 4004, "İstasyon": "OPET BEŞİKTAŞ", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL AVRUPA", "Site Segment": "My Precious", "SKOR": 0.88, "GEÇEN SENE SKOR": 0.83, "Fark": 5.0, "TRANSACTION": 28000},
        {"ROC": 4005, "İstasyon": "TOTAL ŞİŞLİ", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL AVRUPA", "Site Segment": "Primitive", "SKOR": 0.65, "GEÇEN SENE SKOR": 0.70, "Fark": -5.0, "TRANSACTION": 15000},
        {"ROC": 4006, "İstasyon": "SHELL BEYOĞLU", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL AVRUPA", "Site Segment": "Wasted Talent", "SKOR": 0.71, "GEÇEN SENE SKOR": 0.68, "Fark": 3.0, "TRANSACTION": 22000},
        {"ROC": 4007, "İstasyon": "OPET ATAŞEHİR", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL ANADOLU", "Site Segment": "My Precious", "SKOR": 0.86, "GEÇEN SENE SKOR": 0.81, "Fark": 5.0, "TRANSACTION": 35000},
        {"ROC": 4008, "İstasyon": "BP MALTEPE", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL ANADOLU", "Site Segment": "Saboteur", "SKOR": 0.52, "GEÇEN SENE SKOR": 0.58, "Fark": -6.0, "TRANSACTION": 12000},
        {"ROC": 4009, "İstasyon": "SHELL PENDİK", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL ANADOLU", "Site Segment": "Primitive", "SKOR": 0.61, "GEÇEN SENE SKOR": 0.64, "Fark": -3.0, "TRANSACTION": 14000},
        {"ROC": 4010, "İstasyon": "OPET BAĞCILAR", "DISTRICT": "İSTANBUL BÖLGE", "NOR": "İSTANBUL AVRUPA", "Site Segment": "Wasted Talent", "SKOR": 0.69, "GEÇEN SENE SKOR": 0.66, "Fark": 3.0, "TRANSACTION": 19000},

        # ANKARA BÖLGE
        {"ROC": 5001, "İstasyon": "OPET YENİMAHALLE", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA KUZEY", "Site Segment": "My Precious", "SKOR": 0.84, "GEÇEN SENE SKOR": 0.79, "Fark": 5.0, "TRANSACTION": 24000},
        {"ROC": 5002, "İstasyon": "SHELL ÇANKAYA", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA GÜNEY", "Site Segment": "My Precious", "SKOR": 0.87, "GEÇEN SENE SKOR": 0.85, "Fark": 2.0, "TRANSACTION": 30000},
        {"ROC": 5003, "İstasyon": "BP KEÇİÖREN", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA KUZEY", "Site Segment": "Primitive", "SKOR": 0.63, "GEÇEN SENE SKOR": 0.67, "Fark": -4.0, "TRANSACTION": 16000},
        {"ROC": 5004, "İstasyon": "TOTAL MAMAK", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA DOĞU", "Site Segment": "Saboteur", "SKOR": 0.48, "GEÇEN SENE SKOR": 0.55, "Fark": -7.0, "TRANSACTION": 11000},
        {"ROC": 5005, "İstasyon": "OPET BATIKÖY", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA BATI", "Site Segment": "Wasted Talent", "SKOR": 0.72, "GEÇEN SENE SKOR": 0.69, "Fark": 3.0, "TRANSACTION": 20000},
        {"ROC": 5006, "İstasyon": "SHELL GÖLBAŞI", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA GÜNEY", "Site Segment": "Primitive", "SKOR": 0.66, "GEÇEN SENE SKOR": 0.71, "Fark": -5.0, "TRANSACTION": 17000},
        {"ROC": 5007, "İstasyon": "BP SİNCAN", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA BATI", "Site Segment": "Saboteur", "SKOR": 0.51, "GEÇEN SENE SKOR": 0.56, "Fark": -5.0, "TRANSACTION": 13000},
        {"ROC": 5008, "İstasyon": "OPET ALTINDAĞ", "DISTRICT": "ANKARA BÖLGE", "NOR": "ANKARA DOĞU", "Site Segment": "Wasted Talent", "SKOR": 0.70, "GEÇEN SENE SKOR": 0.67, "Fark": 3.0, "TRANSACTION": 18000},

        # İZMİR BÖLGE
        {"ROC": 6001, "İstasyon": "OPET BORNOVA", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR KUZEY", "Site Segment": "My Precious", "SKOR": 0.83, "GEÇEN SENE SKOR": 0.80, "Fark": 3.0, "TRANSACTION": 26000},
        {"ROC": 6002, "İstasyon": "SHELL KONAK", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR MERKEZ", "Site Segment": "My Precious", "SKOR": 0.89, "GEÇEN SENE SKOR": 0.86, "Fark": 3.0, "TRANSACTION": 33000},
        {"ROC": 6003, "İstasyon": "BP KARŞIYAKA", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR KUZEY", "Site Segment": "Wasted Talent", "SKOR": 0.74, "GEÇEN SENE SKOR": 0.71, "Fark": 3.0, "TRANSACTION": 21000},
        {"ROC": 6004, "İstasyon": "TOTAL BALÇOVA", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR GÜNEY", "Site Segment": "Primitive", "SKOR": 0.67, "GEÇEN SENE SKOR": 0.72, "Fark": -5.0, "TRANSACTION": 16000},
        {"ROC": 6005, "İstasyon": "OPET BAYRAKLI", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR KUZEY", "Site Segment": "Saboteur", "SKOR": 0.53, "GEÇEN SENE SKOR": 0.59, "Fark": -6.0, "TRANSACTION": 12500},
        {"ROC": 6006, "İstasyon": "SHELL GÜZELBAHÇE", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR GÜNEY", "Site Segment": "Wasted Talent", "SKOR": 0.73, "GEÇEN SENE SKOR": 0.70, "Fark": 3.0, "TRANSACTION": 19500},
        {"ROC": 6007, "İstasyon": "BP ÇIĞLI", "DISTRICT": "İZMİR BÖLGE", "NOR": "İZMİR KUZEY", "Site Segment": "Primitive", "SKOR": 0.64, "GEÇEN SENE SKOR": 0.68, "Fark": -4.0, "TRANSACTION": 15500},

        # BURSA BÖLGE
        {"ROC": 7001, "İstasyon": "OPET NİLÜFER", "DISTRICT": "BURSA BÖLGE", "NOR": "BURSA MERKEZ", "Site Segment": "My Precious", "SKOR": 0.81, "GEÇEN SENE SKOR": 0.78, "Fark": 3.0, "TRANSACTION": 23000},
        {"ROC": 7002, "İstasyon": "SHELL OSMANGAZİ", "DISTRICT": "BURSA BÖLGE", "NOR": "BURSA MERKEZ", "Site Segment": "Primitive", "SKOR": 0.62, "GEÇEN SENE SKOR": 0.66, "Fark": -4.0, "TRANSACTION": 14500},
        {"ROC": 7003, "İstasyon": "BP YILDIRIM", "DISTRICT": "BURSA BÖLGE", "NOR": "BURSA MERKEZ", "Site Segment": "Saboteur", "SKOR": 0.49, "GEÇEN SENE SKOR": 0.54, "Fark": -5.0, "TRANSACTION": 10500},
        {"ROC": 7004, "İstasyon": "TOTAL GEMLİK", "DISTRICT": "BURSA BÖLGE", "NOR": "BURSA GÜNEY", "Site Segment": "Wasted Talent", "SKOR": 0.75, "GEÇEN SENE SKOR": 0.72, "Fark": 3.0, "TRANSACTION": 18500},

        # ANTALYA BÖLGE
        {"ROC": 8001, "İstasyon": "OPET MURATPAŞA", "DISTRICT": "ANTALYA BÖLGE", "NOR": "ANTALYA MERKEZ", "Site Segment": "My Precious", "SKOR": 0.86, "GEÇEN SENE SKOR": 0.82, "Fark": 4.0, "TRANSACTION": 27000},
        {"ROC": 8002, "İstasyon": "SHELL KEPEZ", "DISTRICT": "ANTALYA BÖLGE", "NOR": "ANTALYA MERKEZ", "Site Segment": "Wasted Talent", "SKOR": 0.71, "GEÇEN SENE SKOR": 0.68, "Fark": 3.0, "TRANSACTION": 20500},
        {"ROC": 8003, "İstasyon": "BP ALANYA", "DISTRICT": "ANTALYA BÖLGE", "NOR": "ANTALYA DOĞU", "Site Segment": "Primitive", "SKOR": 0.65, "GEÇEN SENE SKOR": 0.69, "Fark": -4.0, "TRANSACTION": 17000},
        {"ROC": 8004, "İstasyon": "TOTAL KAŞ", "DISTRICT": "ANTALYA BÖLGE", "NOR": "ANTALYA BATI", "Site Segment": "Saboteur", "SKOR": 0.50, "GEÇEN SENE SKOR": 0.57, "Fark": -7.0, "TRANSACTION": 11500},
    ]

    tlag_df = pd.DataFrame(stations_data)
    tlag_df["ROC_STR"] = tlag_df["ROC"].astype(str)

    comments_data = []
    comment_templates = {
        "PERSONEL": [
            ("Personel çok yardımsever ve güleryüzlü", 5),
            ("Çalışanlar ilgisiz davrandı", 2),
            ("Pompacı çok kibar ve hızlıydı", 5),
            ("Kasiyer muameleyi kaba", 1),
            ("Personel profesyonel", 4),
        ],
        "TEMİZLİK": [
            ("İstasyon çok temiz ve bakımlı", 5),
            ("Tuvaletler pis ve bakımsız", 1),
            ("Genel hijyen kötü", 2),
            ("Her yer tertemiz", 5),
            ("Pompalar kirli", 2),
        ],
        "MARKET": [
            ("Market ürün çeşidi bol", 4),
            ("Fiyatlar çok pahalı", 2),
            ("Taze ürünler mevcut", 4),
            ("Market kısmı küçük", 3),
            ("Kaliteli ürünler", 5),
        ],
        "HIZ": [
            ("Çok hızlı servis", 5),
            ("Bekleme süresi uzun", 2),
            ("Kuyruk çok yavaş ilerliyor", 1),
            ("Hızlı ve etkili", 5),
            ("Servis hızı orta", 3),
        ],
        "YAKIT": [
            ("Yakıt kalitesi çok iyi", 5),
            ("Pompa arızalı", 1),
            ("Dolum hızı iyi", 4),
            ("Yakıt problemi yaşadım", 2),
            ("Kaliteli benzin", 5),
        ],
        "GENEL": [
            ("Genel olarak memnunum", 4),
            ("Berbat bir deneyim", 1),
            ("İyi bir istasyon", 4),
            ("Ortalama", 3),
            ("Çok beğendim", 5),
        ],
    }

    for station in stations_data:
        roc = station["ROC"]
        district = station["DISTRICT"]
        nor = station["NOR"]
        num_comments = np.random.randint(3, 9)
        for _ in range(num_comments):
            category = np.random.choice(list(comment_templates.keys()))
            comment_text, score = np.random.choice(comment_templates[category])
            comments_data.append({
                "station_code": str(roc),
                "station_info": f"{station['İstasyon']} #{roc}",
                "comment": comment_text,
                "score": score,
                "categories": json.dumps([category]),
                "dealer": station["İstasyon"].split()[0],
                "territory": nor,
                "district": district,
                "visit_date": pd.date_range("2024-01-01", "2024-12-31", freq="D")[np.random.randint(0, 365)].strftime("%Y-%m-%d"),
            })

    comments_df = pd.DataFrame(comments_data)
    return tlag_df, comments_df

def export_demo_data_files():
    tlag_df, comments_df = create_demo_data_files()
    tlag_df.to_csv("tlag_demo.csv", index=False)
    comments_df.to_csv("comments_demo.csv", index=False)
    st.success("✅ Demo data dosyaları oluşturuldu: tlag_demo.csv, comments_demo.csv")
    st.info("Bu dosyaları GitHub repository'nize yükleyin ve raw URL'lerini kodda güncelleyin.")


# ------------------------------------------------------------
# Sayfa ayarları
# ------------------------------------------------------------
st.set_page_config(
    page_title="TLAG Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header { font-size: clamp(2rem, 5vw, 3rem); font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); cursor: pointer; transition: transform 0.3s; }
    .metric-card:hover { transform: translateY(-5px); }
    .nav-button { background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); color: white; padding: 1rem 2rem; border: none; border-radius: 10px;
        font-size: 1.1rem; font-weight: bold; margin: 0.5rem; cursor: pointer; transition: all 0.3s; }
    .nav-button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .nav-button.active { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .chat-container { max-height: 400px; overflow-y: auto; padding: 1rem; border-radius: 10px; background: #f8f9fa; margin: 1rem 0; }
    .quick-action-btn { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #2c3e50; padding: 0.5rem 1rem; border: none; border-radius: 20px; font-size: 0.9rem; margin: 0.2rem; cursor: pointer; transition: all 0.3s; }
    .quick-action-btn:hover { transform: translateY(-1px); box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .comment-card { background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #4ECDC4; }
    .category-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 15px; margin: 0.25rem; font-size: 0.85rem; font-weight: bold; }
    .category-personel { background: #FF6B6B; color: white; }
    .category-temizlik { background: #4ECDC4; color: white; }
    .category-market { background: #95E1D3; color: dark; }
    .category-hiz { background: #FFA502; color: white; }
    .category-yakit { background: #3742FA; color: white; }
    .category-genel { background: #747D8C; color: white; }
    .performance-excellent { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0; }
    .performance-good { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0; }
    .performance-needs-improvement { background: linear-gradient(135deg, #ff512f 0%, #dd2476 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0; }
    .station-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin: 0.5rem 0; }
    .improvement-card { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 15px; color: #2c3e50; margin: 1rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
    .priority-high { border-left: 5px solid #e74c3c; }
    .priority-medium { border-left: 5px solid #f39c12; }
    .priority-low { border-left: 5px solid #27ae60; }
    @media (max-width: 768px) { .main-header { font-size: 1.5rem; } .metric-card { padding: 1rem; } }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "tlag_data" not in st.session_state: st.session_state.tlag_data = None
if "comments_data" not in st.session_state: st.session_state.comments_data = None
if "analyzed_comments" not in st.session_state: st.session_state.analyzed_comments = None
if "current_view" not in st.session_state: st.session_state.current_view = "main"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "demo_data_loaded" not in st.session_state: st.session_state.demo_data_loaded = False


# ------------------------------------------------------------
# Period yardımcıları
# ------------------------------------------------------------
def _iso_week_bounds(year: int, week: int):
    d = datetime.fromisocalendar(year, week, 1).date()
    return d, d + timedelta(days=6)

def _month_bounds(year: int, month: int):
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    return start, date(year, month, last_day)

def _quarter_of_month(m: int): return ((m - 1) // 3) + 1

def _quarter_bounds(year: int, q: int):
    m1 = (q - 1) * 3 + 1
    start = date(year, m1, 1)
    m3 = m1 + 2
    last_day = calendar.monthrange(year, m3)[1]
    return start, date(year, m3, last_day)

def _half_of_month(m: int): return 1 if m <= 6 else 2

def _half_bounds(year: int, h: int):
    if h == 1: return date(year, 1, 1), date(year, 6, 30)
    else:      return date(year, 7, 1), date(year, 12, 31)

def infer_period_from_filename(filename: str):
    name = filename.lower()
    y = re.search(r"(20\d{2})", name)
    year = int(y.group(1)) if y else date.today().year

    m = re.search(r"w(\d{1,2})", name)
    if m:
        week = int(m.group(1))
        ps, pe = _iso_week_bounds(year, week)
        return dict(period_type="WEEK", year=year, week=week, period_start=ps, period_end=pe)

    m = re.search(r"[-_\.](1[0-2]|0?[1-9])(?=\D|$)", name)
    if m:
        month = int(m.group(1))
        ps, pe = _month_bounds(year, month)
        return dict(period_type="MONTH", year=year, month=month, quarter=_quarter_of_month(month),
                    half=_half_of_month(month), period_start=ps, period_end=pe)

    m = re.search(r"q([1-4])", name)
    if m:
        q = int(m.group(1))
        ps, pe = _quarter_bounds(year, q)
        return dict(period_type="QUARTER", year=year, quarter=q, half=1 if q <= 2 else 2, period_start=ps, period_end=pe)

    m = re.search(r"h([12])", name)
    if m:
        h = int(m.group(1))
        ps, pe = _half_bounds(year, h)
        return dict(period_type="HALF", year=year, half=h, period_start=ps, period_end=pe)

    ps, pe = date(year, 1, 1), date(year, 12, 31)
    return dict(period_type="YEAR", year=year, period_start=ps, period_end=pe)

def attach_period_columns(df: pd.DataFrame, filename: str):
    meta = infer_period_from_filename(filename or "unknown.xlsx")
    for k, v in meta.items():
        df[f"__{k}"] = v
    return df, meta


# ------------------------------------------------------------
# Supabase yardımcıları (period, tlag_data, customer_comments)
# ------------------------------------------------------------
def supabase_upsert_period(meta: dict, source_filename: str):
    if not SUPABASE_ENABLED:
        return None
    client = get_supabase_client()

    sel = (
        client.table("periods")
        .select("id")
        .eq("period_type", meta["period_type"])
        .eq("year", meta["year"])
    )
    if meta.get("week")    is not None: sel = sel.eq("week", meta["week"])
    if meta.get("month")   is not None: sel = sel.eq("month", meta["month"])
    if meta.get("quarter") is not None: sel = sel.eq("quarter", meta["quarter"])
    if meta.get("half")    is not None: sel = sel.eq("half", meta["half"])

    res = sel.limit(1).execute()
    if res.data:
        return res.data[0]["id"]

    payload = {
        "source_filename": source_filename,
        "period_type": meta["period_type"],
        "year": meta["year"],
        "week": meta.get("week"),
        "month": meta.get("month"),
        "quarter": meta.get("quarter"),
        "half": meta.get("half"),
        "period_start": meta["period_start"].isoformat(),
        "period_end": meta["period_end"].isoformat(),
    }
    ins = client.table("periods").insert(payload).select("id").single().execute()
    return ins.data["id"]

def supabase_save_tlag(df: pd.DataFrame, period_id):
    if not SUPABASE_ENABLED or not period_id or df is None or df.empty:
        return
    client = get_supabase_client()
    rows = []
    for _, r in df.iterrows():
        def _num(x): return float(x) if (x is not None and not pd.isna(x)) else None
        roc_val = normalize_roc(r.get("ROC_STR") or r.get("ROC"))
        rows.append(dict(
            roc=roc_val,
            istasyon=r.get("İstasyon"),
            district=r.get("DISTRICT"),
            nor=r.get("NOR"),
            site_segment=r.get("Site Segment"),
            skor=_num(r.get("SKOR")),
            gecen_sene_skor=_num(r.get("GEÇEN SENE SKOR")),
            fark=_num(r.get("Fark")),
            transaction=_num(r.get("TRANSACTION")),
            period_id=period_id
        ))
    if rows:
        client.table("tlag_data").upsert(rows, on_conflict="roc,period_id").execute()

def fetch_tlag_by_period(period_type, year, gran_val=None):
    if not SUPABASE_ENABLED:
        return pd.DataFrame()
    client = get_supabase_client()

    sel = (
        client.table("periods")
        .select("id,period_type,year,week,month,quarter,half,period_start,period_end")
        .eq("period_type", period_type)
        .eq("year", year)
    )
    if period_type == 'WEEK' and gran_val:    sel = sel.eq("week", gran_val)
    if period_type == 'MONTH' and gran_val:   sel = sel.eq("month", gran_val)
    if period_type == 'QUARTER' and gran_val: sel = sel.eq("quarter", gran_val)
    if period_type == 'HALF' and gran_val:    sel = sel.eq("half", gran_val)

    pr = sel.limit(50).execute()
    periods = pr.data or []
    if not periods:
        return pd.DataFrame()

    ids = [p["id"] for p in periods]
    periods_by_id = {p["id"]: p for p in periods}

    res = (
        client.table("tlag_data")
        .select("*")
        .in_("period_id", ids)
        .limit(10000)
        .execute()
    )

    rows = []
    for r in (res.data or []):
        p = periods_by_id.get(r["period_id"], {})
        rows.append({
            'ROC': r.get('roc'),
            'ROC_STR': str(r.get('roc')) if r.get('roc') is not None else None,
            'İstasyon': r.get('istasyon'),
            'DISTRICT': r.get('district'),
            'NOR': r.get('nor'),
            'Site Segment': r.get('site_segment'),
            'SKOR': r.get('skor'),
            'GEÇEN SENE SKOR': r.get('gecen_sene_skor'),
            'Fark': r.get('fark'),
            'TRANSACTION': r.get('transaction'),
            '__period_type': p.get('period_type'),
            '__year': p.get('year'),
            '__week': p.get('week'),
            '__month': p.get('month'),
            '__quarter': p.get('quarter'),
            '__half': p.get('half'),
            '__period_start': p.get('period_start'),
            '__period_end': p.get('period_end'),
        })
    return pd.DataFrame(rows)

def fetch_comments_by_period(period_type, year, gran_val=None):
    if not SUPABASE_ENABLED:
        return pd.DataFrame()
    client = get_supabase_client()

    sel = (
        client.table("periods")
        .select("id")
        .eq("period_type", period_type)
        .eq("year", year)
    )
    if period_type == 'WEEK' and gran_val:    sel = sel.eq("week", gran_val)
    if period_type == 'MONTH' and gran_val:   sel = sel.eq("month", gran_val)
    if period_type == 'QUARTER' and gran_val: sel = sel.eq("quarter", gran_val)
    if period_type == 'HALF' and gran_val:    sel = sel.eq("half", gran_val)

    pr = sel.limit(50).execute()
    ids = [p["id"] for p in (pr.data or [])]
    if not ids:
        return pd.DataFrame()

    res = (
        client.table("customer_comments")
        .select("*")
        .in_("period_id", ids)
        .limit(50000)
        .execute()
    )

    rows = []
    for r in (res.data or []):
        cats = r.get('categories')
        if isinstance(cats, str):
            try:
                cats = json.loads(cats)
            except Exception:
                cats = []
        rows.append({
            'station_code': r.get('roc'),
            'comment': r.get('comment'),
            'score': r.get('score'),
            'categories': cats,
            'dealer': r.get('dealer'),
            'territory': r.get('territory'),
            'district': r.get('district'),
            'visit_date': r.get('visit_date'),
        })
    return pd.DataFrame(rows)

def supabase_save_comments(df: pd.DataFrame, period_id):
    """customer_comments tablosuna yazar (roc=normalize, period_id=uuid)."""
    if not SUPABASE_ENABLED or period_id is None or df is None or df.empty:
        return
    client = get_supabase_client()

    rows = []
    for _, r in df.iterrows():
        cats = r.get("categories")
        if isinstance(cats, list):
            cats_json = json.dumps(cats, ensure_ascii=False)
        else:
            cats_json = json.dumps([cats] if pd.notna(cats) else [], ensure_ascii=False)

        roc_val = normalize_roc(r.get("station_code"))  # ← merkezi ROC normalize
        rows.append(dict(
            roc=roc_val,
            comment=r.get("comment"),
            score=_to_int_safe(r.get("score")),
            categories=cats_json,
            dealer=r.get("dealer"),
            territory=r.get("territory"),
            district=r.get("district"),
            visit_date=str(r.get("visit_date")) if pd.notna(r.get("visit_date")) else None,
            period_id=period_id
        ))

    if rows:
        client.table("customer_comments").insert(rows).execute()


# ------------------------------------------------------------
# Yardımcı fonksiyonlar (temizlik/analiz)
# ------------------------------------------------------------
def clean_data_for_json(df):
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str)
    return df_clean

def format_percentage(value):
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value * 100:.1f}%"

def format_percentage_change(value):
    if pd.isna(value) or value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"

def get_performance_category(score):
    if pd.isna(score): return "Bilinmeyen", "#95a5a6"
    elif score >= 0.80: return "Mükemmel", "#27ae60"
    elif score >= 0.70: return "İyi", "#f39c12"
    elif score >= 0.60: return "Orta", "#e67e22"
    else: return "Gelişim Gerekli", "#e74c3c"

def extract_station_code(station_info):
    if pd.isna(station_info): return None
    match = re.search(r"#(\d+)$", str(station_info))
    return match.group(1) if match else None

def categorize_comment(comment_text):
    if pd.isna(comment_text): return ["GENEL"]
    comment_lower = str(comment_text).lower()
    categories = []
    category_keywords = {
        "PERSONEL": ["personel", "çalışan", "pompacı", "kasiyer", "görevli", "müdür", "yardımsever", "ilgili", "güleryüzlü", "kaba", "ilgisiz", "saygılı"],
        "TEMİZLİK": ["temiz", "kirli", "hijyen", "tuvalet", "pis", "bakım", "tertip", "düzen"],
        "MARKET": ["market", "ürün", "fiyat", "pahalı", "ucuz", "çeşit", "kalite", "taze"],
        "HIZ": ["hızlı", "yavaş", "bekleme", "kuyruk", "süre", "geç", "çabuk", "acele"],
        "YAKIT": ["benzin", "motorin", "lpg", "yakıt", "pompa", "dolum", "depo"],
        "GENEL": ["genel", "güzel", "kötü", "memnun", "beğen", "hoş"]
    }
    for category, keywords in category_keywords.items():
        if any(keyword in comment_lower for keyword in keywords):
            categories.append(category)
    return categories if categories else ["GENEL"]

def enforce_real_districts(df: pd.DataFrame) -> pd.DataFrame:
    if "DISTRICT" not in df.columns:
        return df
    valid = set(df["DISTRICT"].dropna().astype(str).str.strip().unique())
    df["DISTRICT"] = df["DISTRICT"].apply(lambda x: x if (pd.notna(x) and str(x).strip() in valid) else np.nan)
    return df

def load_tlag_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, sheet_name="TLAG DOKUNMA (2)", engine="openpyxl")
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["ROC", "İstasyon"], how="any")
        numeric_columns = ["ROC", "SKOR", "GEÇEN SENE SKOR", "Fark", "TRANSACTION", "NOR HEDEF", "DISTRICT HEDEF", "Geçerli"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        text_columns = ["İstasyon", "NOR", "DISTRICT", "Site Segment"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("nan", np.nan)
        # ROC_STR ve normalize edilmiş ROC_TEXT ekle
        df["ROC_STR"] = df["ROC"].astype(str).str.split(".").str[0]
        df["ROC_TEXT"] = df["ROC_STR"].apply(normalize_roc)
        df = enforce_real_districts(df)
        df_clean = clean_data_for_json(df)
        return df_clean
    except Exception as e:
        st.error(f"Dosya okuma hatası: {str(e)}")
        return None

def load_comments_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=1)
        df = df[df.iloc[:, 0] != "65000 yorum sınırını aştınız."]
        df = df[df.iloc[:, 0] != "birim"]
        column_names = {
            df.columns[0]: "station_info",
            df.columns[1]: "survey_item",
            df.columns[2]: "comment",
            df.columns[3]: "score",
            df.columns[4]: "visit_date",
            df.columns[5]: "hospitality_score",
            df.columns[6]: "dealer",
            df.columns[7]: "territory",
            df.columns[8]: "district",
            df.columns[9]: "country" if len(df.columns) > 9 else "country"
        }
        df = df.rename(columns=column_names)
        # station_code -> normalize
        df["station_code"] = df["station_info"].apply(extract_station_code)
        df["station_code"] = df["station_code"].apply(normalize_roc)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["categories"] = df["comment"].apply(categorize_comment)
        df_clean = clean_data_for_json(df)
        return df_clean
    except Exception as e:
        st.error(f"Yorum dosyası okuma hatası: {str(e)}")
        return None

def merge_comments_with_stations(comments_df, tlag_df):
    try:
        merged = pd.merge(
            comments_df,
            tlag_df[["ROC_STR", "ROC_TEXT", "İstasyon", "NOR", "DISTRICT"]],
            left_on="station_code",
            right_on="ROC_STR",
            how="left"
        )
        merged["NOR_FINAL"] = merged["NOR"].fillna(merged["territory"])
        merged["DISTRICT_FINAL"] = merged["DISTRICT"].fillna(merged["district"])
        return merged
    except Exception as e:
        st.error(f"Veri birleştirme hatası: {str(e)}")
        return comments_df


def analyze_comments_by_category(df, level="district"):
    if df is None or df.empty:
        return {}
    group_col = f"{level.upper()}_FINAL" if f"{level.upper()}_FINAL" in df.columns else level.upper()
    if group_col not in df.columns:
        return {}
    results = {}
    for name, group in df.groupby(group_col):
        if pd.isna(name) or name == "nan" or name == "0":
            continue
        score_dist = group["score"].value_counts().to_dict()
        low_score_comments = group[group["score"] <= 4]
        category_counts = {}
        for cats in low_score_comments["categories"]:
            if isinstance(cats, list):
                for cat in cats:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        category_samples = {}
        for category in category_counts.keys():
            samples = []
            for _, row in low_score_comments.iterrows():
                if isinstance(row["categories"], list) and category in row["categories"]:
                    samples.append({
                        "comment": row["comment"],
                        "score": row["score"],
                        "station": row.get("station_info", "")
                    })
                    if len(samples) >= 3:
                        break
            category_samples[category] = samples
        results[name] = {
            "total_comments": len(group),
            "avg_score": group["score"].mean(),
            "score_distribution": score_dist,
            "low_score_categories": category_counts,
            "category_samples": category_samples,
            "critical_count": len(group[group["score"] <= 2])
        }
    return results

def display_comment_analysis(analysis_data, title="Yorum Analizi"):
    st.markdown(f"### 💬 {title}")
    if not analysis_data:
        st.info("Yorum verisi bulunamadı")
        return
    for idx, (name, data) in enumerate(analysis_data.items()):
        with st.expander(f"📍 {name} - {data['total_comments']} yorum, Ort: {data['avg_score']:.1f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Puan Dağılımı:**")
                score_df = pd.DataFrame(list(data["score_distribution"].items()), columns=["Puan", "Sayı"]).sort_values("Puan")
                fig = px.bar(score_df, x="Puan", y="Sayı", color="Puan", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True, key=f"score_dist_{title}_{idx}_{name}")
            with col2:
                st.markdown("**4 ve Altı Puan Kategorileri:**")
                if data["low_score_categories"]:
                    cat_df = pd.DataFrame(list(data["low_score_categories"].items()), columns=["Kategori", "Sayı"]).sort_values("Sayı", ascending=False)
                    for _, row in cat_df.iterrows():
                        cat_class = f"category-{row['Kategori'].lower()}"
                        st.markdown(f"""
                        <span class="category-badge {cat_class}">
                            {row['Kategori']}: {row['Sayı']}
                        </span>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Düşük puan kategorisi yok")
            if data["category_samples"]:
                st.markdown("**📝 Kategori Detayları (Tıklayın):**")
                for category, samples in data["category_samples"].items():
                    if samples:
                        with st.expander(f"{category} ({len(samples)} örnek)"):
                            for sample in samples:
                                st.markdown(f"""
                                <div class="comment-card">
                                    <strong>Puan: {sample['score']}</strong><br>
                                    <em>{sample['comment']}</em><br>
                                    <small>{sample['station']}</small>
                                </div>
                                """, unsafe_allow_html=True)

def create_clickable_metric(col, title, value, key, df=None):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{title}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Detay", key=f"btn_{key}", use_container_width=True):
            st.session_state[f"show_{key}"] = not st.session_state.get(f"show_{key}", False)
    if st.session_state.get(f"show_{key}", False):
        with st.container():
            st.markdown(f"### 📊 {title} Detayları")
            if key == "total_stations" and df is not None:
                display_df = df[["ROC", "İstasyon", "SKOR", "DISTRICT", "Site Segment"]].copy()
                display_df["SKOR"] = display_df["SKOR"].apply(format_percentage)
                st.dataframe(display_df.sort_values("İstasyon"), use_container_width=True, height=400)
            elif key == "avg_score" and df is not None:
                st.write("**🎯 Ortalamayı En Hızlı Artıracak İstasyonlar:**")
                impact_df = df[df["SKOR"] < 0.7].copy()
                if "TRANSACTION" in impact_df.columns:
                    impact_df["potential_impact"] = (0.7 - impact_df["SKOR"]) * impact_df.get("TRANSACTION", 1)
                    top_impact = impact_df.nlargest(10, "potential_impact")
                    for _, row in top_impact.iterrows():
                        potential_increase = (0.7 - row["SKOR"]) * 100
                        st.write(f"• **{row['İstasyon']}**: {format_percentage(row['SKOR'])} → 70% (+{potential_increase:.1f}%)")
            elif key == "saboteur" and df is not None:
                saboteur_df = df[df["Site Segment"] == "Saboteur"][["İstasyon", "SKOR", "DISTRICT", "NOR"]].copy()
                saboteur_df["SKOR"] = saboteur_df["SKOR"].apply(format_percentage)
                st.dataframe(saboteur_df, use_container_width=True)
            elif key == "precious" and df is not None:
                precious_df = df[df["Site Segment"] == "My Precious"][["İstasyon", "SKOR", "DISTRICT", "NOR"]].copy()
                precious_df["SKOR"] = precious_df["SKOR"].apply(format_percentage)
                st.dataframe(precious_df, use_container_width=True)
            if st.button("❌ Kapat", key=f"close_{key}"):
                st.session_state[f"show_{key}"] = False
                st.rerun()


def calculate_tlag_score(comments_df, group_by="DISTRICT"):
    if comments_df is None or comments_df.empty:
        return pd.DataFrame()
    group_col = f"{group_by}_FINAL" if f"{group_by}_FINAL" in comments_df.columns else group_by
    if group_col not in comments_df.columns:
        return pd.DataFrame()
    results = []
    for name, group in comments_df.groupby(group_col):
        if pd.notna(name) and name != "nan" and name != "0":
            total_responses = len(group)
            five_star_count = len(group[group["score"] == 5])
            tlag_score = (five_star_count / total_responses * 100) if total_responses > 0 else 0
            results.append({
                group_by: name,
                "Toplam_Yanıt": total_responses,
                "5_Puan_Sayısı": five_star_count,
                "TLAG_Skoru_%": round(tlag_score, 1)
            })
    return pd.DataFrame(results)


# ------------------------------------------------------------
# (AI) öneriler & chat
# ------------------------------------------------------------
def ai_recommendations_for_scope(scope_name: str, df_scope: pd.DataFrame, comments_scope: pd.DataFrame):
    try:
        try:
            import openai
        except ImportError:
            return (
                "### 📦 OpenAI Modülü Gerekli\n\n"
                "AI önerileri için `pip install openai` kurun veya secrets’a anahtar ekleyin."
            )
        import os
        api_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("openai", {}).get("api_key") if hasattr(st, "secrets") else None)
        if not api_key:
            return (
                "### 🔑 OpenAI API Anahtarı Gerekli\n\n"
                "ENV: `OPENAI_API_KEY` ya da `.streamlit/secrets.toml` içinde `[openai].api_key`."
            )
        openai.api_key = api_key

        summary = df_scope.assign(SkorY=(df_scope["SKOR"]*100).round(1) if "SKOR" in df_scope.columns else 0).to_dict(orient="records")[:200]
        probs = comments_scope[comments_scope.get("score", 5) <= 4][["comment", "score", "categories"]].head(300).to_dict(orient="records") if not comments_scope.empty else []

        prompt = f"""
        You are an ops analyst. Analyze performance for {scope_name}.
        1) Key drivers dragging score down
        2) Quick wins per category (PERSONEL, TEMİZLİK, MARKET, HIZ, YAKIT, GENEL)
        3) 2-week action plan and expected impact in %.
        Data (stations): {summary}
        Low-score comments: {probs}
        Produce concise, bullet-point Turkish output.
        """
        try:
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"### ❌ AI Çağrısı Başarısız\n\nHata: {str(e)}"


def ai_chat_response(user_message: str, tlag_df=None, comments_df=None):
    try:
        try:
            import openai
        except ImportError:
            return "❌ OpenAI modülü yüklü değil. `pip install openai` komutunu çalıştırın."

        import os
        api_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("openai", {}).get("api_key") if hasattr(st, "secrets") else None)
        if not api_key:
            return "❌ OpenAI API anahtarı bulunamadı. Lütfen API anahtarınızı ayarlayın."

        data_context = ""
        if tlag_df is not None and not tlag_df.empty:
            total_stations = len(tlag_df)
            avg_score = tlag_df["SKOR"].mean() if "SKOR" in tlag_df.columns else 0
            districts = list(tlag_df["DISTRICT"].dropna().unique()) if "DISTRICT" in tlag_df.columns else []
            segments = list(tlag_df["Site Segment"].dropna().unique()) if "Site Segment" in tlag_df.columns else []
            if "SKOR" in tlag_df.columns:
                top5 = tlag_df.nlargest(5, "SKOR")[["İstasyon", "SKOR", "DISTRICT"]].to_dict("records")
                bottom5 = tlag_df.nsmallest(5, "SKOR")[["İstasyon", "SKOR", "DISTRICT"]].to_dict("records")
            else:
                top5, bottom5 = [], []
            data_context += f"""
            TLAG VERİ ÖZETİ:
            - Toplam İstasyon: {total_stations}
            - Ortalama Skor: {avg_score:.1%}
            - District'ler: {', '.join(districts[:10])}
            - Segmentler: {', '.join(segments)}
            - En İyi 5 İstasyon: {top5}
            - En Düşük 5 İstasyon: {bottom5}
            """

        if comments_df is not None and not comments_df.empty:
            total_comments = len(comments_df)
            avg_comment_score = comments_df["score"].mean() if "score" in comments_df.columns else 0
            low_scores = len(comments_df[comments_df["score"] <= 2]) if "score" in comments_df.columns else 0
            all_categories = {}
            if "categories" in comments_df.columns:
                for cats in comments_df["categories"]:
                    if isinstance(cats, list):
                        for cat in cats:
                            all_categories[cat] = all_categories.get(cat, 0) + 1
            data_context += f"""
            YORUM VERİ ÖZETİ:
            - Toplam Yorum: {total_comments}
            - Ortalama Puan: {avg_comment_score:.1f}
            - Düşük Puan (≤2): {low_scores}
            - En Çok Bahsedilen Kategoriler: {dict(sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:5])}
            """

        system_prompt = f"""Sen TLAG Performans Analitik asistanısın. Türk Petrol istasyonları hakkında data analizi yapan, Türkçe konuşan uzman bir analistsin.

Görevlerin:
1. TLAG verileri ve müşteri yorumlarını analiz etmek
2. Performance önerileri vermek
3. Kullanıcının sorularına veri tabanlı cevaplar vermek
4. Grafik ve analiz önerileri sunmak
5. İş kararlarına destek olmak

Mevcut veri durumu:
{data_context if data_context else "Henüz veri yüklenmemiş."}

KURALLAR:
- Her zaman Türkçe yanıt ver
- Somut veriler kullan
- Eğer grafik/analiz önerirsen, hangi verilerin kullanılacağını belirt
- Kısa ve öz cevaplar ver
- Business odaklı düşün"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.chat_history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ AI hatası: {str(e)}"


def generate_ai_chart(user_request: str, tlag_df=None, comments_df=None):
    try:
        if tlag_df is None or tlag_df.empty:
            return None, "Grafik oluşturmak için TLAG verisi gerekli."
        request_lower = user_request.lower()

        if any(word in request_lower for word in ["district", "bölge", "performans dağılım"]):
            if "DISTRICT" in tlag_df.columns and "SKOR" in tlag_df.columns:
                district_avg = tlag_df.groupby("DISTRICT")["SKOR"].mean().reset_index()
                district_avg["Skor_Yüzde"] = district_avg["SKOR"] * 100
                district_avg = district_avg.sort_values("Skor_Yüzde", ascending=False)
                fig = px.bar(
                    district_avg, x="DISTRICT", y="Skor_Yüzde",
                    title="District Bazında Ortalama Performans",
                    labels={"Skor_Yüzde": "Ortalama Skor (%)", "DISTRICT": "District"},
                    color="Skor_Yüzde", color_continuous_scale="RdYlGn"
                )
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Hedef: 70%")
                return fig, "District bazında performans analizi oluşturuldu."

        elif any(word in request_lower for word in ["segment", "saboteur", "precious"]):
            if "Site Segment" in tlag_df.columns and "SKOR" in tlag_df.columns:
                segment_counts = tlag_df["Site Segment"].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index, title="İstasyon Segment Dağılımı")
                return fig, "Segment dağılımı grafiği oluşturuldu."

        elif any(word in request_lower for word in ["skor dağılım", "histogram", "puan dağılım"]):
            if "SKOR" in tlag_df.columns:
                tlag_viz = tlag_df.copy()
                tlag_viz["Skor_Yüzde"] = tlag_viz["SKOR"] * 100
                fig = px.histogram(
                    tlag_viz, x="Skor_Yüzde", nbins=20, title="Performans Skor Dağılımı",
                    labels={"Skor_Yüzde": "Skor (%)", "count": "İstasyon Sayısı"}
                )
                fig.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Hedef: 70%")
                return fig, "Skor dağılımı histogramı oluşturuldu."

        elif any(word in request_lower for word in ["yorum", "comment", "müşteri puan"]):
            if comments_df is not None and not comments_df.empty and "score" in comments_df.columns:
                score_counts = comments_df["score"].value_counts().sort_index()
                fig = px.bar(
                    x=score_counts.index, y=score_counts.values,
                    title="Müşteri Yorumları Puan Dağılımı",
                    labels={"x": "Puan", "y": "Yorum Sayısı"},
                    color=score_counts.index, color_continuous_scale="RdYlGn"
                )
                return fig, "Müşteri yorum puanları grafiği oluşturuldu."

        elif any(word in request_lower for word in ["nor", "bölgesel"]):
            if "NOR" in tlag_df.columns and "SKOR" in tlag_df.columns:
                nor_avg = tlag_df.groupby("NOR")["SKOR"].agg(["mean", "count"]).reset_index()
                nor_avg["Skor_Yüzde"] = nor_avg["mean"] * 100
                nor_avg = nor_avg.sort_values("Skor_Yüzde", ascending=False)
                fig = px.scatter(
                    nor_avg, x="count", y="Skor_Yüzde", size="count", hover_name="NOR",
                    title="NOR Bazında Performans (Büyüklük: İstasyon Sayısı)",
                    labels={"count": "İstasyon Sayısı", "Skor_Yüzde": "Ortalama Skor (%)"}
                )
                return fig, "NOR bazında performans analizi oluşturuldu."

        return None, "Bu tür grafik için uygun veri bulunamadı veya istek anlaşılamadı."
    except Exception as e:
        return None, f"Grafik oluşturma hatası: {str(e)}"


# ------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">📊 TLAG PERFORMANS ANALİTİK</h1>', unsafe_allow_html=True)

    # Sidebar period selector
    st.sidebar.markdown("## 📅 DÖNEM SEÇİMİ")
    period_types = ["WEEK", "MONTH", "QUARTER", "HALF", "YEAR"]
    selected_period_type = st.sidebar.selectbox("Dönem Türü:", period_types, index=1)
    selected_year = st.sidebar.selectbox("Yıl:", [2024, 2025], index=1)

    gran_val = None
    if selected_period_type == "WEEK":
        gran_val = st.sidebar.selectbox("Hafta No:", range(1, 54))
    elif selected_period_type == "MONTH":
        months = {1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"}
        gran_val = st.sidebar.selectbox("Ay:", list(months.keys()), format_func=lambda x: months[x])
    elif selected_period_type == "QUARTER":
        quarters = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
        gran_val = st.sidebar.selectbox("Çeyrek:", list(quarters.keys()), format_func=lambda x: quarters[x])
    elif selected_period_type == "HALF":
        halfs = {1: "İlk Yarı", 2: "İkinci Yarı"}
        gran_val = st.sidebar.selectbox("Yarıyıl:", list(halfs.keys()), format_func=lambda x: halfs[x])

    # Dönemsel veri (Supabase) – seçilen dönemi DB'den oku
    if SUPABASE_ENABLED and st.sidebar.button("🔄 Dönemsel Veri Yükle"):
        try:
            df_from_db = fetch_tlag_by_period(selected_period_type, selected_year, gran_val)
            comments_from_db = fetch_comments_by_period(selected_period_type, selected_year, gran_val)

            if not df_from_db.empty:
                st.session_state.tlag_data = df_from_db
                st.sidebar.success(f"✅ {len(df_from_db)} istasyon verisi yüklendi!")
            else:
                st.sidebar.warning("❌ Seçili dönem için istasyon verisi bulunamadı")

            if not comments_from_db.empty:
                merged_comments = merge_comments_with_stations(
                    comments_from_db, st.session_state.tlag_data or pd.DataFrame()
                )
                st.session_state.comments_data = merged_comments
                district_comments = analyze_comments_by_category(merged_comments, "district")
                nor_comments = analyze_comments_by_category(merged_comments, "nor")
                st.session_state.analyzed_comments = {"district": district_comments, "nor": nor_comments}
                st.sidebar.success(f"✅ {len(comments_from_db)} yorum yüklendi!")
            elif st.session_state.get("comments_data") is None:
                st.sidebar.info("ℹ️ Seçili dönem için yorum bulunamadı")
        except Exception as e:
            st.sidebar.error(f"Veri yükleme hatası: {str(e)}")

    # Dosya yükleme
    st.sidebar.markdown("## 📁 VERİ YÖNETİMİ")
    uploaded_file = st.sidebar.file_uploader("TLAG Excel dosyası:", type=["xlsx", "xls"], help="satis_veri_clean.xlsx dosyanızı yükleyin")

    st.sidebar.markdown("## 💬 MÜŞTERİ YORUMLARI")
    comments_file = st.sidebar.file_uploader("Yorum dosyası (Excel):", type=["xlsx", "xls"], key="comments_uploader", help="Comment YTD All.xlsx dosyanızı yükleyin")

    # Demo veri seçenekleri
    st.sidebar.markdown("## 🚀 DEMO VERİLERİ")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("☁️ Cloud'dan Yükle", key="load_cloud_demo", use_container_width=True):
            with st.spinner("☁️ Cloud'dan demo veriler yükleniyor..."):
                tlag_df, comments_df, message = load_demo_data_from_cloud()
                if tlag_df is not None:
                    st.session_state.tlag_data = tlag_df
                    st.session_state.demo_data_loaded = True
                    if comments_df is not None:
                        merged_comments = merge_comments_with_stations(comments_df, tlag_df)
                        st.session_state.comments_data = merged_comments
                        district_comments = analyze_comments_by_category(merged_comments, "district")
                        nor_comments = analyze_comments_by_category(merged_comments, "nor")
                        st.session_state.analyzed_comments = {"district": district_comments, "nor": nor_comments}
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)

    with col2:
        if st.button("📊 Lokal Demo", key="load_local_demo", use_container_width=True):
            tlag_df, comments_df = create_demo_data_files()
            st.session_state.tlag_data = tlag_df
            st.session_state.demo_data_loaded = True
            merged_comments = merge_comments_with_stations(comments_df, tlag_df)
            st.session_state.comments_data = merged_comments
            district_comments = analyze_comments_by_category(merged_comments, "district")
            nor_comments = analyze_comments_by_category(merged_comments, "nor")
            st.session_state.analyzed_comments = {"district": district_comments, "nor": nor_comments}
            st.sidebar.success("✅ Lokal demo verisi yüklendi!")

    if st.sidebar.button("📥 Demo Dosyalarını Export Et", key="export_demo"):
        export_demo_data_files()

    if st.session_state.demo_data_loaded or st.session_state.tlag_data is not None:
        with st.sidebar.expander("📊 Yüklü Veri Durumu"):
            if st.session_state.tlag_data is not None:
                st.write(f"✅ **TLAG:** {len(st.session_state.tlag_data)} istasyon")
            if st.session_state.comments_data is not None:
                st.write(f"✅ **Yorumlar:** {len(st.session_state.comments_data)} yorum")
            if st.session_state.demo_data_loaded:
                st.info("🚀 Demo verisi aktif")

    with st.sidebar.expander("☁️ Cloud Setup Rehberi"):
        st.markdown("""
        **GitHub Raw URL Yöntemi:**
        1. GitHub'da public repo oluşturun
        2. `tlag_demo.csv` ve `comments_demo.csv` yükleyin
        3. Raw URL'leri alın ve kodda güncelleyin

        **Google Sheets Yöntemi:**
        1. Google Sheets'te veri oluşturun
        2. "Paylaş" → "Bağlantı alan herkes görüntüleyebilir"
        3. URL: `/export?format=csv&gid=SHEET_ID`

        **Örnek Raw URL:**
        `https://raw.githubusercontent.com/username/repo/main/tlag_demo.csv`
        """)

    if st.session_state.demo_data_loaded:
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 0.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <strong>🚀 DEMO MODU AKTİF</strong><br>
            <small>Gerçekçi örnek veriler yüklü</small>
        </div>
        """, unsafe_allow_html=True)

    # TLAG dosyası yüklendiyse işle
    if uploaded_file is not None:
        with st.spinner("📊 Veriler işleniyor..."):
            df = load_tlag_data(uploaded_file)
        if df is not None and not df.empty:
            df, meta = attach_period_columns(df, uploaded_file.name)
            st.session_state.tlag_data = df
            st.sidebar.success(f"✅ {len(df)} istasyon verisi yüklendi!")
            st.sidebar.info(f"📅 Dönem: {meta.get('period_type', 'UNKNOWN')} {meta.get('year', '')}")

            if SUPABASE_ENABLED:
                try:
                    period_id = supabase_upsert_period(meta, uploaded_file.name)
                    supabase_save_tlag(df, period_id)  # ← normalize_roc kullanır
                    st.sidebar.success("✅ Veriler Supabase'e kaydedildi!")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Supabase kayıt başarısız: {str(e)}")

            with st.sidebar.expander("📋 Veri Özeti"):
                st.write(f"**Toplam İstasyon:** {len(df)}")
                if "SKOR" in df.columns:
                    avg_score = df["SKOR"].mean()
                    st.write(f"**Ortalama Skor:** {format_percentage(avg_score)}")
                    st.write(f"**En Yüksek:** {format_percentage(df['SKOR'].max())}")
                    st.write(f"**En Düşük:** {format_percentage(df['SKOR'].min())}")
                if "Site Segment" in df.columns:
                    segments = df["Site Segment"].value_counts()
                    st.write("**Segment Dağılımı:**")
                    for segment, count in segments.items():
                        if pd.notna(segment):
                            st.write(f"- {segment}: {count}")

    # Yorum dosyası yüklendiyse işle
    if comments_file and st.session_state.tlag_data is not None:
        with st.spinner("💬 Yorumlar işleniyor..."):
            comments_df = load_comments_data(comments_file)
            if comments_df is not None:
                comments_df, meta2 = attach_period_columns(comments_df, comments_file.name)
                merged_comments = merge_comments_with_stations(comments_df, st.session_state.tlag_data)
                st.session_state.comments_data = merged_comments
                district_comments = analyze_comments_by_category(merged_comments, "district")
                nor_comments = analyze_comments_by_category(merged_comments, "nor")
                st.session_state.analyzed_comments = {"district": district_comments, "nor": nor_comments}
                st.sidebar.success(f"✅ {len(comments_df)} yorum yüklendi ve analiz edildi!")

                if SUPABASE_ENABLED:
                    try:
                        period_id2 = supabase_upsert_period(meta2, comments_file.name)
                        supabase_save_comments(comments_df, period_id2)  # ← normalize_roc kullanır
                        st.sidebar.success("✅ Yorumlar Supabase'e kaydedildi!")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Yorum Supabase kayıt başarısız: {str(e)}")

                with st.sidebar.expander("💬 Yorum Özeti"):
                    st.write(f"**Toplam Yorum:** {len(comments_df)}")
                    if "score" in comments_df.columns:
                        avg_score = comments_df["score"].mean()
                        st.write(f"**Ortalama Puan:** {avg_score:.1f}")
                        score_dist = comments_df["score"].value_counts().sort_index()
                        st.write("**Puan Dağılımı:**")
                        for score, count in score_dist.items():
                            if pd.notna(score) and score <= 5:
                                st.write(f"- {int(score)} puan: {count}")

    # Navigation
    st.markdown("### 🎯 ANALİZ ALANINA GİT")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏢 District Analiz", key="nav_district", use_container_width=True):
            st.session_state.current_view = "district"
            st.rerun()
    with col2:
        if st.button("📍 NOR Analiz", key="nav_nor", use_container_width=True):
            st.session_state.current_view = "nor"
            st.rerun()
    with col3:
        if st.button("🏪 İstasyon Analiz", key="nav_station", use_container_width=True):
            st.session_state.current_view = "station"
            st.rerun()
    with col4:
        if st.button("🤖 AI Asistan", key="nav_ai", use_container_width=True):
            st.session_state.current_view = "ai_chat"
            st.rerun()

    # Views
    if st.session_state.current_view == "main":
        if st.session_state.tlag_data is not None:
            df = st.session_state.tlag_data
            analyzed_comments = st.session_state.get("analyzed_comments") or {}
            district_comments = analyzed_comments.get("district", {}) if isinstance(analyzed_comments, dict) else {}

            st.markdown("## 📊 ANA METRİKLER")
            col1, col2, col3, col4 = st.columns(4)
            create_clickable_metric(col1, "Toplam İstasyon", len(df), "total_stations", df)
            if "SKOR" in df.columns:
                create_clickable_metric(col2, "Ortalama Skor", format_percentage(df["SKOR"].mean()), "avg_score", df)
            if "Site Segment" in df.columns:
                saboteur_count = len(df[df["Site Segment"] == "Saboteur"])
                precious_count = len(df[df["Site Segment"] == "My Precious"])
                create_clickable_metric(col3, "Saboteur", saboteur_count, "saboteur", df)
                create_clickable_metric(col4, "My Precious", precious_count, "precious", df)

            if "SKOR" in df.columns:
                st.markdown("## 📈 PERFORMANS DAĞILIMI")
                df_viz = df.copy()
                df_viz["Skor_Yüzde"] = df_viz["SKOR"] * 100
                fig_dist = px.histogram(
                    df_viz, x="Skor_Yüzde", nbins=20,
                    title="Performans Skor Dağılımı (%)",
                    labels={"Skor_Yüzde": "Performans Skoru (%)", "count": "İstasyon Sayısı"}
                )
                fig_dist.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Hedef: 70%")
                fig_dist.update_layout(height=400)
                st.plotly_chart(fig_dist, use_container_width=True, key="main_perf_distribution")

            if st.session_state.comments_data is not None:
                st.markdown("## 📊 BÖLGESEL TLAG SKORLARI (Müşteri Yorumlarından)")
                valid_districts = set(df["DISTRICT"].dropna().astype(str).unique())
                district_tlag = calculate_tlag_score(st.session_state.comments_data, "DISTRICT")
                if not district_tlag.empty:
                    district_tlag = district_tlag[district_tlag["DISTRICT"].astype(str).isin(valid_districts)]
                    district_tlag = district_tlag.sort_values("TLAG_Skoru_%", ascending=False)
                    fig_district_tlag = px.bar(
                        district_tlag, x="DISTRICT", y="TLAG_Skoru_%", text="TLAG_Skoru_%",
                        title="District Bazında TLAG Skorları (%)",
                        labels={"TLAG_Skoru_%": "TLAG Skoru (%)", "DISTRICT": "District"},
                        color="TLAG_Skoru_%", color_continuous_scale="RdYlGn"
                    )
                    fig_district_tlag.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_district_tlag.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Hedef: 70%")
                    st.plotly_chart(fig_district_tlag, use_container_width=True, key="main_district_tlag_scores")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### District Detayları")
                        st.dataframe(district_tlag, use_container_width=True)
                    with c2:
                        nor_tlag = calculate_tlag_score(st.session_state.comments_data, "NOR")
                        if not nor_tlag.empty:
                            st.markdown("### NOR Detayları")
                            nor_tlag = nor_tlag.sort_values("TLAG_Skoru_%", ascending=False)
                            st.dataframe(nor_tlag.head(10), use_container_width=True)
        else:
            st.markdown("## 🎯 TLAG PERFORMANS ANALİTİK'E HOŞGELDİNİZ")
            st.info("👈 Sol panelden Excel dosyalarınızı yükleyin veya demo verilerini deneyin")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                ### 📊 ANALİZ ÖZELLİKLERİ
                - ✅ Tıklanabilir metrikler
                - ✅ Müşteri yorumları analizi
                - ✅ Kategori bazlı yorum gruplandırma
                - ✅ District/NOR/İstasyon analizi
                - ✅ AI-powered öneriler
                """)
            with c2:
                st.markdown("""
                ### 💬 YORUM ANALİZİ
                - ✅ Otomatik kategorizasyon
                - ✅ Düşük puan analizi
                - ✅ Fırsat istasyonları tespiti
                - ✅ Gerçek yorum örnekleri
                - ✅ Trend analizi
                """)

    elif st.session_state.current_view == "district":
        if st.session_state.tlag_data is not None:
            df = st.session_state.tlag_data
            analyzed_comments = st.session_state.get("analyzed_comments") or {}
            district_comments = analyzed_comments.get("district", {}) if isinstance(analyzed_comments, dict) else {}
            if st.button("🏠 Ana Sayfaya Dön", key="back_to_main_from_district"):
                st.session_state.current_view = "main"; st.rerun()
            st.markdown("## 🏢 DISTRICT BAZLI ANALİZ")
            if "DISTRICT" in df.columns:
                districts = sorted(df["DISTRICT"].dropna().unique())
                selected_district = st.selectbox("District Seçin:", districts, key="district_selector")
                if selected_district:
                    district_data = df[df["DISTRICT"] == selected_district]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("İstasyon Sayısı", len(district_data))
                    with col2: st.metric("Ortalama Skor", format_percentage(district_data["SKOR"].mean()))
                    with col3:
                        if "Site Segment" in district_data.columns:
                            segments = district_data["Site Segment"].value_counts()
                            st.markdown("**Segment Dağılımı:**")
                            for seg, count in segments.head(3).items():
                                st.write(f"• {seg}: {count}")
                    with col4: st.metric("En Düşük Skor", format_percentage(district_data["SKOR"].min()))
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🏆 En İyi 5 İstasyon")
                        top5 = district_data.nlargest(5, "SKOR")[["İstasyon", "SKOR", "Site Segment"]].copy()
                        top5["SKOR"] = top5["SKOR"].apply(format_percentage)
                        st.dataframe(top5, use_container_width=True)
                    with c2:
                        st.markdown("### ⚠️ En Kötü 5 İstasyon")
                        bottom5 = district_data.nsmallest(5, "SKOR")[["İstasyon", "SKOR", "Site Segment"]].copy()
                        bottom5["SKOR"] = bottom5["SKOR"].apply(format_percentage)
                        st.dataframe(bottom5, use_container_width=True)
                    if st.button(f"🤖 {selected_district} için AI Önerisi Oluştur", key=f"ai_district_{selected_district}"):
                        with st.spinner("AI analizi yapılıyor..."):
                            district_comments_df = st.session_state.comments_data[
                                st.session_state.comments_data["DISTRICT_FINAL"] == selected_district
                            ] if st.session_state.comments_data is not None else pd.DataFrame()
                            ai_result = ai_recommendations_for_scope(selected_district, district_data, district_comments_df)
                            st.markdown("### 🤖 AI Önerileri"); st.markdown(ai_result)
                    if district_comments and selected_district in district_comments:
                        display_comment_analysis({selected_district: district_comments[selected_district]}, title=f"{selected_district} Müşteri Yorumları")

    elif st.session_state.current_view == "nor":
        if st.session_state.tlag_data is not None:
            df = st.session_state.tlag_data
            analyzed_comments = st.session_state.get("analyzed_comments") or {}
            nor_comments = analyzed_comments.get("nor", {}) if isinstance(analyzed_comments, dict) else {}
            if st.button("🏠 Ana Sayfaya Dön", key="back_to_main_from_nor"):
                st.session_state.current_view = "main"; st.rerun()
            st.markdown("## 📍 NOR BAZLI ANALİZ")
            if "NOR" in df.columns:
                nors = sorted(df["NOR"].dropna().unique())
                selected_nor = st.selectbox("NOR Seçin:", nors, key="nor_selector")
                if selected_nor:
                    nor_data = df[df["NOR"] == selected_nor]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("İstasyon Sayısı", len(nor_data))
                    with col2: st.metric("Ortalama Skor", format_percentage(nor_data["SKOR"].mean()))
                    with col3:
                        if "Site Segment" in nor_data.columns:
                            st.metric("En Yaygın Segment", nor_data["Site Segment"].mode().iloc[0] if len(nor_data["Site Segment"].mode()) > 0 else "N/A")
                    with col4: st.metric("Skor Aralığı", f"{format_percentage(nor_data['SKOR'].min())} - {format_percentage(nor_data['SKOR'].max())}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🏆 En İyi 5 İstasyon")
                        top5 = nor_data.nlargest(5, "SKOR")[["İstasyon", "SKOR", "Site Segment", "DISTRICT"]].copy()
                        top5["SKOR"] = top5["SKOR"].apply(format_percentage)
                        st.dataframe(top5, use_container_width=True)
                    with c2:
                        st.markdown("### ⚠️ En Kötü 5 İstasyon")
                        bottom5 = nor_data.nsmallest(5, "SKOR")[["İstasyon", "SKOR", "Site Segment", "DISTRICT"]].copy()
                        bottom5["SKOR"] = bottom5["SKOR"].apply(format_percentage)
                        st.dataframe(bottom5, use_container_width=True)
                    st.markdown("### 📊 NOR Performans Dağılımı")
                    nor_viz = nor_data.copy(); nor_viz["Skor_Yüzde"] = nor_viz["SKOR"] * 100
                    fig_nor = px.box(nor_viz, y="Skor_Yüzde", x="Site Segment", title=f"{selected_nor} - Segment Bazında Performans", labels={"Skor_Yüzde": "Performans Skoru (%)"})
                    st.plotly_chart(fig_nor, use_container_width=True, key=f"nor_perf_{selected_nor}")
                    if st.button(f"🤖 {selected_nor} için AI Önerisi Oluştur", key=f"ai_nor_{selected_nor}"):
                        with st.spinner("AI analizi yapılıyor..."):
                            nor_comments_df = st.session_state.comments_data[
                                st.session_state.comments_data["NOR_FINAL"] == selected_nor
                            ] if st.session_state.comments_data is not None else pd.DataFrame()
                            ai_result = ai_recommendations_for_scope(selected_nor, nor_data, nor_comments_df)
                            st.markdown("### 🤖 AI Önerileri"); st.markdown(ai_result)
                    if nor_comments and selected_nor in nor_comments:
                        display_comment_analysis({selected_nor: nor_comments[selected_nor]}, title=f"{selected_nor} Müşteri Yorumları")

    elif st.session_state.current_view == "station":
        if st.session_state.tlag_data is not None:
            df = st.session_state.tlag_data
            if st.button("🏠 Ana Sayfaya Dön", key="back_to_main_from_station"):
                st.session_state.current_view = "main"; st.rerun()
            st.markdown("## 🏪 İSTASYON DETAY ANALİZİ")
            station_search = st.text_input("🔍 İstasyon ara:", placeholder="İstasyon adı yazın...", key="station_search")
            if station_search:
                filtered_stations = df[df["İstasyon"].str.contains(station_search, case=False, na=False)]["İstasyon"].tolist()
            else:
                filtered_stations = sorted(df["İstasyon"].unique())
            if filtered_stations:
                selected_station = st.selectbox("İstasyon seçin:", filtered_stations, key="station_selector")
                if selected_station:
                    station_row = df[df["İstasyon"] == selected_station].iloc[0]
                    station_data = station_row.to_dict()
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"### 🏢 {selected_station}")
                        st.markdown(f"**ROC:** {station_data.get('ROC_TEXT') or station_data.get('ROC') or 'N/A'}")
                        st.markdown(f"**Bölge:** {station_data.get('DISTRICT', 'N/A')}")
                        st.markdown(f"**NOR:** {station_data.get('NOR', 'N/A')}")
                        st.markdown(f"**Segment:** {station_data.get('Site Segment', 'N/A')}")
                    with col2:
                        current_score = station_data.get("SKOR", 0)
                        previous_score = station_data.get("GEÇEN SENE SKOR", 0)
                        change_percent = (current_score - previous_score) * 100 if previous_score and current_score else 0
                        st.metric("Mevcut Skor", format_percentage(current_score), delta=format_percentage_change(change_percent))
                        st.metric("Geçen Yıl", format_percentage(previous_score))
                    with col3:
                        category, color = get_performance_category(current_score)
                        st.markdown(f"""
                        <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;">
                            {category}
                        </div>
                        """, unsafe_allow_html=True)
                    if st.session_state.comments_data is not None:
                        station_code = normalize_roc(station_data.get("ROC_TEXT") or station_data.get("ROC_STR") or station_data.get("ROC"))
                        station_comments = st.session_state.comments_data[
                            st.session_state.comments_data["station_code"].astype(str) == str(station_code)
                        ]
                        if not station_comments.empty:
                            st.markdown("### 💬 Müşteri Yorumları")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Toplam Yorum", len(station_comments))
                                st.metric("Ortalama Puan", f"{station_comments['score'].mean():.1f}")
                            with c2:
                                score_counts = station_comments["score"].value_counts().sort_index()
                                fig_score = px.bar(x=score_counts.index, y=score_counts.values, title="Puan Dağılımı", labels={"x": "Puan", "y": "Sayı"})
                                st.plotly_chart(fig_score, use_container_width=True, key=f"station_score_{selected_station}")
                            st.markdown("### Son Yorumlar")
                            recent_comments = station_comments.head(5)
                            for _, comment in recent_comments.iterrows():
                                cats = comment["categories"] if isinstance(comment["categories"], list) else ["GENEL"]
                                cat_badges = " ".join([f'<span class="category-badge category-{cat.lower()}">{cat}</span>' for cat in cats])
                                st.markdown(f"""
                                <div class="comment-card">
                                    <strong>Puan: {comment['score']}</strong> {cat_badges}<br>
                                    <em>{comment['comment']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    if st.button(f"🤖 {selected_station} için AI Önerisi Oluştur", key=f"ai_station_{selected_station}"):
                        with st.spinner("AI analizi yapılıyor..."):
                            station_code = normalize_roc(station_data.get("ROC_TEXT") or station_data.get("ROC_STR") or station_data.get("ROC"))
                            station_comments_df = st.session_state.comments_data[
                                st.session_state.comments_data["station_code"].astype(str) == str(station_code)
                            ] if st.session_state.comments_data is not None else pd.DataFrame()
                            station_df = pd.DataFrame([station_data])
                            ai_result = ai_recommendations_for_scope(selected_station, station_df, station_comments_df)
                            st.markdown("### 🤖 AI Önerileri"); st.markdown(ai_result)

    elif st.session_state.current_view == "ai_chat":
        if st.button("🏠 Ana Sayfaya Dön", key="back_to_main_from_ai"):
            st.session_state.current_view = "main"; st.rerun()
        st.markdown("## 🤖 AI ASISTAN")
        st.markdown("*TLAG verilerinizi analiz edebilen akıllı asistanınız*")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📊 Genel Özet", key="quick_summary"):
                st.session_state.chat_history.append({"role": "user", "content": "TLAG verilerimin genel özetini ver. En önemli bulguları listele."})
        with c2:
            if st.button("⚠️ Problem Alanları", key="quick_problems"):
                st.session_state.chat_history.append({"role": "user", "content": "En düşük performanslı district'ler ve istasyonlar hangileri? Sorunları analiz et."})
        with c3:
            if st.button("🎯 İyileştirme Fırsatları", key="quick_opportunities"):
                st.session_state.chat_history.append({"role": "user", "content": "Hangi alanlarda hızlı kazanımlar elde edebilirim? Öncelikli aksiyonları listele."})
        st.markdown("### 💬 Sohbet Geçmişi")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"): st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"): st.markdown(message["content"])
        user_input = st.chat_input("TLAG verileriniz hakkında soru sorun veya analiz isteyin...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤔 Düşünüyor..."):
                    ai_response = ai_chat_response(user_input, st.session_state.tlag_data, st.session_state.comments_data)
                    st.markdown(ai_response)
                    if any(word in user_input.lower() for word in ["grafik", "chart", "görselleştir", "plot", "diagram"]):
                        chart_fig, chart_msg = generate_ai_chart(user_input, st.session_state.tlag_data, st.session_state.comments_data)
                        if chart_fig:
                            st.plotly_chart(chart_fig, use_container_width=True, key=f"ai_chart_{len(st.session_state.chat_history)}")
                            st.info(f"📈 {chart_msg}")
                        else:
                            st.warning(f"⚠️ {chart_msg}")
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            if len(st.session_state.chat_history) > 50:
                st.session_state.chat_history = st.session_state.chat_history[-50:]
            st.rerun()
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🗑️ Sohbeti Temizle", key="clear_chat"):
                st.session_state.chat_history = []; st.rerun()
        with c2:
            st.info(f"💬 {len(st.session_state.chat_history)} mesaj")
        with c3:
            if st.session_state.tlag_data is None:
                st.warning("⚠️ TLAG verisi yok - Sol panelden veri yükleyin")
            else:
                station_count = len(st.session_state.tlag_data)
                comment_count = len(st.session_state.comments_data) if st.session_state.comments_data is not None else 0
                st.success(f"✅ {station_count} istasyon, {comment_count} yorum yüklü")
        with st.expander("💡 Örnek Sorular"):
            st.markdown("""
            **Genel Analiz:**
            - "En kötü performans gösteren 5 district'i listele"
            - "Saboteur segmentindeki istasyonları analiz et"
            - "Müşteri yorumlarındaki en büyük problemler neler?"

            **Grafik İstekleri:**
            - "District bazında performans grafiği oluştur"
            - "Segment dağılımı pasta grafiği göster"
            - "Skor dağılımı histogramı çiz"
            - "Müşteri yorumları puan dağılımını görselleştir"

            **Aksiyonel Öneriler:**
            - "Hangi istasyonlara öncelik vermeliyim?"
            - "PERSONEL kategorisindeki şikayetler nasıl azaltılır?"
            - "Bu ay hangi district'e odaklanmalıyım?"

            **Karşılaştırma:**
            - "Ankara ve İstanbul bölgelerini karşılaştır"
            - "My Precious ile Saboteur arasındaki farklar neler?"
            """)

if __name__ == "__main__":
    main()
