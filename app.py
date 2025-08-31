# app.py - Enhanced TLAG Performance Analytics

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
            if "." in v:
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
    """ROC değerini güvenli biçimde normalize eder."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    # Son 4 haneyi al (müşteri yorumları için)
    m = re.search(r"#(\d{4})$", s)
    if m:
        return m.group(1)
    # Sadece rakam varsa al
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None

def extract_station_code(station_info):
    """Station info'dan son 4 haneli kodu çıkarır"""
    if pd.isna(station_info):
        return None
    s = str(station_info)
    # #5789 formatı
    m = re.search(r"#(\d{4})$", s)
    if m:
        return m.group(1)
    # Son 4 rakamı al
    m = re.search(r"(\d{4})(?=\D*$)", s)
    return m.group(1) if m else None

# ------------------------------------------------------------
# Supabase entegrasyonu
# ------------------------------------------------------------
try:
    from modules.supabase_client import get_supabase_client
    SUPABASE_ENABLED = True
except ImportError:
    SUPABASE_ENABLED = False
    print("Supabase modülü yüklenemedi. Lokal modda çalışıyor.")

# ------------------------------------------------------------
# Demo veri yükleme
# ------------------------------------------------------------
def load_demo_data_from_cloud():
    """Cloud'dan gerçek veriler"""
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
            dtype={"station_code": "string"}
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
# Yorum kategorileri
# ------------------------------------------------------------
def categorize_comment_enhanced(comment_text):
    """Gelişmiş yorum kategorileme"""
    if pd.isna(comment_text):
        return ["GENEL"]
    
    comment_lower = str(comment_text).lower()
    categories = []
    
    category_keywords = {
        "PERSONEL": [
            "personel", "çalışan", "görevli", "müdür", "yardımsever", 
            "güleryüzlü", "kaba", "ilgisiz", "saygılı", "nazik", "kibar"
        ],
        "POMPACI": [
            "pompacı", "yakıt dolu", "benzin dolu", "motorin dolu",
            "cam sil", "kontrol et", "yağ kontrol", "su kontrol"
        ],
        "TEMİZLİK": [
            "temiz", "kirli", "hijyen", "tuvalet", "pis", "bakım", 
            "tertip", "düzen", "leke", "koku"
        ],
        "MARKET": [
            "market", "ürün", "fiyat", "pahalı", "ucuz", "çeşit", 
            "kalite", "taze", "kafe", "restoran"
        ],
        "HIZ": [
            "hızlı", "yavaş", "bekleme", "kuyruk", "süre", "geç", 
            "çabuk", "acele", "bekle", "zaman"
        ],
        "YAKIT": [
            "benzin", "motorin", "lpg", "yakıt", "pompa", "dolum", 
            "depo", "kalite", "verim"
        ],
        "BANKA_KAMPANYA": [
            "kampanya", "indirim", "kart", "bonus", "puan", "fırsat",
            "promosyon", "taksit", "ödeme"
        ],
        "GENEL": [
            "genel", "güzel", "kötü", "memnun", "beğen", "hoş",
            "berbat", "süper", "harika", "mükemmel"
        ]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in comment_lower for keyword in keywords):
            categories.append(category)
    
    return categories if categories else ["GENEL"]

# ------------------------------------------------------------
# Veri yükleme fonksiyonları
# ------------------------------------------------------------
def load_tlag_data(uploaded_file):
    """TLAG Excel dosyasını yükler"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name="TLAG DOKUNMA (2)", engine="openpyxl")
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["ROC", "İstasyon"], how="any")
        
        # Numeric columns
        numeric_columns = ["ROC", "SKOR", "GEÇEN SENE SKOR", "Fark", "TRANSACTION"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Text columns
        text_columns = ["İstasyon", "NOR", "DISTRICT", "Site Segment"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("nan", np.nan)
        
        # ROC normalizasyonu
        df["ROC_STR"] = df["ROC"].astype(str).str.split(".").str[0]
        df["ROC_NORMALIZED"] = df["ROC_STR"].apply(normalize_roc)
        
        return df
    except Exception as e:
        st.error(f"TLAG dosya okuma hatası: {str(e)}")
        return None

def load_comments_data(uploaded_file):
    """Müşteri yorum dosyasını yükler"""
    try:
        df = pd.read_excel(uploaded_file, header=1, engine="openpyxl")
        
        # İlk satırları temizle
        df = df[df.iloc[:, 0] != "65000 yorum sınırını aştınız."]
        df = df[df.iloc[:, 0] != "birim"]
        df = df.dropna(subset=[df.columns[0]], how="all")
        
        # Kolon adlandırması
        column_names = {}
        if len(df.columns) >= 9:
            column_names = {
                df.columns[0]: "station_info",
                df.columns[1]: "survey_item", 
                df.columns[2]: "comment",
                df.columns[3]: "score",
                df.columns[4]: "visit_date",
                df.columns[5]: "hospitality_score",
                df.columns[6]: "dealer",
                df.columns[7]: "territory",
                df.columns[8]: "district"
            }
        
        df = df.rename(columns=column_names)
        
        # Station code çıkarımı
        df["station_code"] = df["station_info"].apply(extract_station_code)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        
        # Gelişmiş kategorizasyon
        df["categories"] = df["comment"].apply(categorize_comment_enhanced)
        
        # 4 puan ama olumlu yorum tespiti
        df["positive_but_4star"] = df.apply(
            lambda row: (
                row["score"] == 4 and 
                isinstance(row["comment"], str) and
                any(word in str(row["comment"]).lower() for word in 
                    ["güzel", "iyi", "memnun", "beğen", "süper", "harika"])
            ), axis=1
        )
        
        return df
    except Exception as e:
        st.error(f"Yorum dosyası okuma hatası: {str(e)}")
        return None

# ------------------------------------------------------------
# Analiz fonksiyonları
# ------------------------------------------------------------
def get_opportunity_stations(df):
    """Fırsat istasyonlarını bulur (My Precious/Primitive <80%)"""
    if df is None or df.empty or "SKOR" not in df.columns:
        return pd.DataFrame()
    
    opportunity_mask = (
        (df["Site Segment"].isin(["My Precious", "Primitive"])) & 
        (df["SKOR"] < 0.80)
    )
    
    return df[opportunity_mask].copy()

def analyze_comments_by_scope(comments_df, scope_col="DISTRICT"):
    """Kapsamlı yorum analizi"""
    if comments_df is None or comments_df.empty:
        return {}
    
    # Scope column belirleme
    if scope_col == "DISTRICT":
        group_col = "DISTRICT_FINAL" if "DISTRICT_FINAL" in comments_df.columns else "district"
    elif scope_col == "NOR":
        group_col = "NOR_FINAL" if "NOR_FINAL" in comments_df.columns else "territory"
    else:
        group_col = scope_col
        
    if group_col not in comments_df.columns:
        return {}
    
    results = {}
    
    for name, group in comments_df.groupby(group_col):
        if pd.isna(name) or name == "nan" or name == "0":
            continue
            
        total_comments = len(group)
        avg_score = group["score"].mean()
        
        # Puan dağılımı
        score_dist = group["score"].value_counts().to_dict()
        
        # Düşük puan yorumları (1-3 puan)
        low_score_comments = group[group["score"] <= 3]
        
        # Kategori analizi
        category_counts = {}
        category_scores = {}
        
        for _, row in group.iterrows():
            if isinstance(row["categories"], list):
                for cat in row["categories"]:
                    if cat not in category_counts:
                        category_counts[cat] = 0
                        category_scores[cat] = []
                    category_counts[cat] += 1
                    category_scores[cat].append(row["score"])
        
        # Kategori ortalama puanları
        category_avg_scores = {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }
        
        # En problemli kategoriler (düşük puan + yüksek frekans)
        problem_categories = {}
        for cat in category_counts:
            if category_avg_scores[cat] < 4.0 and category_counts[cat] >= 3:
                problem_categories[cat] = {
                    "count": category_counts[cat],
                    "avg_score": category_avg_scores[cat],
                    "problem_level": (5 - category_avg_scores[cat]) * category_counts[cat]
                }
        
        # Olumlu ama 4 puan verenler
        positive_4star = group[group.get("positive_but_4star", False) == True]
        
        results[name] = {
            "total_comments": total_comments,
            "avg_score": avg_score,
            "score_distribution": score_dist,
            "category_counts": category_counts,
            "category_avg_scores": category_avg_scores,
            "problem_categories": sorted(
                problem_categories.items(), 
                key=lambda x: x[1]["problem_level"], 
                reverse=True
            )[:3],
            "positive_4star_count": len(positive_4star),
            "critical_issues": len(group[group["score"] <= 2])
        }
    
    return results

def get_top_focus_areas(comments_analysis, top_n=3):
    """En çok odaklanılması gereken alanları döndürür"""
    all_problems = {}
    
    for scope_data in comments_analysis.values():
        for cat, data in scope_data.get("problem_categories", []):
            if cat not in all_problems:
                all_problems[cat] = {"total_impact": 0, "instances": 0}
            all_problems[cat]["total_impact"] += data["problem_level"]
            all_problems[cat]["instances"] += 1
    
    # En yüksek impact'li kategoriler
    sorted_problems = sorted(
        all_problems.items(),
        key=lambda x: x[1]["total_impact"],
        reverse=True
    )
    
    return [item[0] for item in sorted_problems[:top_n]]

# ------------------------------------------------------------
# UI Components
# ------------------------------------------------------------
def create_enhanced_metric_card(col, title, value, key, click_data=None):
    """Gelişmiş metrik kartları"""
    with col:
        st.markdown(f"""
        <div class="metric-card" onclick="toggleDetails('{key}')">
            <h2>{value}</h2>
            <p>{title}</p>
            <small>📊 Detay için tıklayın</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Detayları Göster", key=f"btn_{key}", use_container_width=True):
            st.session_state[f"show_{key}"] = not st.session_state.get(f"show_{key}", False)
    
    # Detay gösterimi
    if st.session_state.get(f"show_{key}", False) and click_data is not None:
        with st.expander(f"📋 {title} Detayları", expanded=True):
            if key == "total_stations":
                display_station_list(click_data)
            elif key == "avg_score": 
                display_score_improvement_analysis(click_data)
            elif key.startswith("segment_"):
                segment_name = key.replace("segment_", "").replace("_", " ")
                display_segment_analysis(click_data, segment_name)
            
            if st.button("❌ Kapat", key=f"close_{key}"):
                st.session_state[f"show_{key}"] = False
                st.rerun()

def display_station_list(df):
    """İstasyon listesi gösterimi"""
    if df is None or df.empty:
        st.info("Veri bulunamadı")
        return
    
    display_cols = ["ROC_STR", "İstasyon", "SKOR", "DISTRICT", "NOR", "Site Segment"]
    available_cols = [col for col in display_cols if col in df.columns]
    
    display_df = df[available_cols].copy()
    
    if "SKOR" in display_df.columns:
        display_df["SKOR_FORMATTED"] = display_df["SKOR"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
        display_df = display_df.drop(columns=["SKOR"])
        display_df = display_df.rename(columns={"SKOR_FORMATTED": "SKOR"})
    
    # İstasyon seçimi için selectbox
    st.markdown("### 🏪 İstasyon Seçin:")
    selected_station = st.selectbox(
        "İstasyon:", 
        df["İstasyon"].tolist(),
        key="station_detail_selector"
    )
    
    if selected_station:
        station_data = df[df["İstasyon"] == selected_station].iloc[0]
        display_station_detail(station_data)
    
    # Tablo gösterimi
    st.markdown("### 📊 Tüm İstasyonlar:")
    st.dataframe(display_df, use_container_width=True, height=400)

def display_station_detail(station_data):
    """Detaylı istasyon bilgileri"""
    st.markdown(f"### 🏪 {station_data['İstasyon']} Detayları")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mevcut Skor", f"{station_data.get('SKOR', 0)*100:.1f}%")
        st.metric("ROC Kodu", station_data.get('ROC_STR', 'N/A'))
    
    with col2:
        st.metric("District", station_data.get('DISTRICT', 'N/A'))
        st.metric("NOR", station_data.get('NOR', 'N/A'))
    
    with col3:
        st.metric("Site Segment", station_data.get('Site Segment', 'N/A'))
        if pd.notna(station_data.get('TRANSACTION')):
            st.metric("Transaction", f"{station_data['TRANSACTION']:,.0f}")
    
    # Yorum analizi (eğer varsa)
    if st.session_state.get("comments_data") is not None:
        station_code = station_data.get("ROC_NORMALIZED") or station_data.get("ROC_STR")
        station_comments = st.session_state.comments_data[
            st.session_state.comments_data["station_code"] == str(station_code)
        ]
        
        if not station_comments.empty:
            display_station_comments(station_comments)

def display_station_comments(comments_df):
    """İstasyon yorumlarını göster"""
    st.markdown("### 💬 Müşteri Yorumları")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Toplam Yorum", len(comments_df))
        st.metric("Ortalama Puan", f"{comments_df['score'].mean():.1f}")
    
    with col2:
        # Puan dağılımı grafiği
        score_counts = comments_df['score'].value_counts().sort_index()
        fig = px.bar(
            x=score_counts.index, 
            y=score_counts.values,
            title="Puan Dağılımı",
            labels={"x": "Puan", "y": "Yorum Sayısı"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Kategori analizi
    st.markdown("#### 📊 Kategori Bazlı Analiz")
    category_scores = {}
    
    for _, row in comments_df.iterrows():
        if isinstance(row["categories"], list):
            for cat in row["categories"]:
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(row["score"])
    
    # Kategori performansı tablosu
    cat_data = []
    for cat, scores in category_scores.items():
        cat_data.append({
            "Kategori": cat,
            "Yorum Sayısı": len(scores),
            "Ortalama Puan": f"{np.mean(scores):.1f}",
            "Problem Seviyesi": "🔴 Yüksek" if np.mean(scores) < 3.5 else "🟡 Orta" if np.mean(scores) < 4.0 else "🟢 Düşük"
        })
    
    cat_df = pd.DataFrame(cat_data).sort_values("Ortalama Puan")
    st.dataframe(cat_df, use_container_width=True)
    
    # Son yorumlar
    st.markdown("#### 💭 Son Yorumlar")
    for _, comment in comments_df.head(5).iterrows():
        with st.container():
            cats = comment["categories"] if isinstance(comment["categories"], list) else ["GENEL"]
            cat_tags = " ".join([f"`{cat}`" for cat in cats])
            
            st.markdown(f"""
            **Puan: {comment['score']}/5** | {cat_tags}
            
            _{comment['comment']}_
            
            ---
            """)

def display_score_improvement_analysis(df):
    """Skor iyileştirme analizi"""
    st.markdown("### 🎯 Skor İyileştirme Stratejisi")
    
    if df is None or df.empty or "SKOR" not in df.columns:
        st.info("Skor verisi bulunamadı")
        return
    
    # Potansiyel impact hesaplama
    improvement_opportunities = df[df["SKOR"] < 0.80].copy()
    
    if "TRANSACTION" in improvement_opportunities.columns:
        improvement_opportunities["potential_impact"] = (
            (0.80 - improvement_opportunities["SKOR"]) * 
            improvement_opportunities["TRANSACTION"] / 1000
        )
        improvement_opportunities = improvement_opportunities.sort_values(
            "potential_impact", ascending=False
        )
    else:
        improvement_opportunities["potential_impact"] = 0.80 - improvement_opportunities["SKOR"]
        improvement_opportunities = improvement_opportunities.sort_values(
            "SKOR", ascending=True
        )
    
    st.markdown("#### 🔥 En Yüksek Impact İstasyonlar (Top 10)")
    
    top_impact = improvement_opportunities.head(10)
    
    for idx, (_, row) in enumerate(top_impact.iterrows(), 1):
        current_score = row["SKOR"] * 100
        target_score = 80
        potential_gain = target_score - current_score
        
        st.markdown(f"""
        **{idx}. {row['İstasyon']}**
        - Mevcut: {current_score:.1f}% → Hedef: {target_score}% (+{potential_gain:.1f} puan)
        - District: {row.get('DISTRICT', 'N/A')} | Segment: {row.get('Site Segment', 'N/A')}
        """)
    
    # Segment bazlı fırsatlar
    st.markdown("#### 🎪 Segment Bazlı Fırsatlar")
    
    segment_opps = improvement_opportunities.groupby("Site Segment").agg({
        "İstasyon": "count",
        "SKOR": "mean",
        "potential_impact": "sum"
    }).reset_index()
    
    segment_opps = segment_opps.sort_values("potential_impact", ascending=False)
    
    for _, row in segment_opps.iterrows():
        avg_score = row["SKOR"] * 100
        st.markdown(f"""
        **{row['Site Segment']}**: {row['İstasyon']} istasyon | Ort: {avg_score:.1f}%
        """)

def display_segment_analysis(df, segment_name):
    """Segment analizi"""
    st.markdown(f"### 🎯 {segment_name} Segment Analizi")
    
    if df is None or df.empty:
        st.info("Veri bulunamadı")
        return
    
    segment_data = df[df["Site Segment"] == segment_name] if "Site Segment" in df.columns else df
    
    if segment_data.empty:
        st.info(f"{segment_name} segmentinde istasyon bulunamadı")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Toplam İstasyon", len(segment_data))
    
    with col2:
        if "SKOR" in segment_data.columns:
            avg_score = segment_data["SKOR"].mean() * 100
            st.metric("Ortalama Skor", f"{avg_score:.1f}%")
    
    with col3:
        if "DISTRICT" in segment_data.columns:
            district_count = segment_data["DISTRICT"].nunique()
            st.metric("District Sayısı", district_count)
    
    # En iyi ve en kötü istasyonlar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 En İyi 5 İstasyon")
        if "SKOR" in segment_data.columns:
            top5 = segment_data.nlargest(5, "SKOR")[["İstasyon", "SKOR", "DISTRICT"]].copy()
            top5["SKOR"] = top5["SKOR"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(top5, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚠️ En Kötü 5 İstasyon")
        if "SKOR" in segment_data.columns:
            bottom5 = segment_data.nsmallest(5, "SKOR")[["İstasyon", "SKOR", "DISTRICT"]].copy()
            bottom5["SKOR"] = bottom5["SKOR"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(bottom5, use_container_width=True)

# ------------------------------------------------------------
# Supabase veri kaydetme
# ------------------------------------------------------------
def save_data_to_supabase(df, comments_df=None, period_meta=None):
    """Veriyi Supabase'e kalıcı olarak kaydeder"""
    if not SUPABASE_ENABLED:
        st.info("Supabase bağlantısı yok - veri geçici olarak hafızada tutulacak")
        return
    
    try:
        client = get_supabase_client()
        
        # Period kaydı
        if period_meta:
            period_id = save_period(client, period_meta)
        else:
            period_id = None
        
        # TLAG verileri kaydet
        if df is not None and not df.empty:
            save_tlag_data(client, df, period_id)
            st.success("✅ TLAG verileri kaydedildi")
        
        # Yorum verileri kaydet
        if comments_df is not None and not comments_df.empty:
            save_comment_data(client, comments_df, period_id)
            st.success("✅ Yorum verileri kaydedildi")
            
    except Exception as e:
        st.error(f"Veri kaydetme hatası: {str(e)}")

# ------------------------------------------------------------
# CSS ve sayfa yapılandırması
# ------------------------------------------------------------
st.set_page_config(
    page_title="TLAG Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { 
        font-size: clamp(2rem, 5vw, 3rem); 
        font-weight: bold; 
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 2rem; 
    }
    
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 1.5rem; 
        border-radius: 15px; 
        color: white; 
        text-align: center; 
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
        cursor: pointer; 
        transition: transform 0.3s; 
    }
    
    .metric-card:hover { 
        transform: translateY(-5px); 
    }
    
    .nav-section {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .nav-button { 
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
        color: white; 
        padding: 1rem 2rem; 
        border: none; 
        border-radius: 10px;
        font-size: 1.1rem; 
        font-weight: bold; 
        margin: 0.5rem; 
        cursor: pointer; 
        transition: all 0.3s; 
        width: 100%;
    }
    
    .nav-button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
    }
    
    .opportunity-card {
        background: linear-gradient(135deg, #FFA502 0%, #FF6B6B 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .category-badge { 
        display: inline-block; 
        padding: 0.25rem 0.75rem; 
        border-radius: 15px; 
        margin: 0.25rem; 
        font-size: 0.85rem; 
        font-weight: bold; 
    }
    
    .category-personel { background: #FF6B6B; color: white; }
    .category-pompaci { background: #3742FA; color: white; }
    .category-temizlik { background: #4ECDC4; color: white; }
    .category-market { background: #95E1D3; color: white; }
    .category-hiz { background: #FFA502; color: white; }
    .category-yakit { background: #2F3542; color: white; }
    .category-banka-kampanya { background: #1e90ff; color: white; }
    .category-genel { background: #747D8C; color: white; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "tlag_data" not in st.session_state:
    st.session_state.tlag_data = None
if "comments_data" not in st.session_state:
    st.session_state.comments_data = None
if "analyzed_comments" not in st.session_state:
    st.session_state.analyzed_comments = None
if "current_view" not in st.session_state:
    st.session_state.current_view = "main"

# ------------------------------------------------------------
# ANA UYGULAMA
# ------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">📊 TLAG PERFORMANS ANALİTİK</h1>', unsafe_allow_html=True)
    
    # SIDEBAR - Veri Yükleme
    st.sidebar.markdown("## 📁 VERİ YÖNETİMİ")
    
    # TLAG dosyası yükleme
    uploaded_tlag = st.sidebar.file_uploader(
        "TLAG Excel dosyası:", 
        type=["xlsx", "xls"], 
        help="İstasyon performans verilerini içeren Excel dosyası"
    )
    
    # Yorum dosyası yükleme
    uploaded_comments = st.sidebar.file_uploader(
        "Müşteri Yorumları Excel dosyası:", 
        type=["xlsx", "xls"], 
        key="comments_uploader",
        help="Müşteri yorum anket sonuçlarını içeren Excel dosyası"
    )
    
    # Demo data yükleme
    st.sidebar.markdown("## 🚀 DEMO VERİLERİ")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("☁️ Cloud Demo", use_container_width=True):
            with st.spinner("Veriler yükleniyor..."):
                tlag_df, comments_df, message = load_demo_data_from_cloud()
                if tlag_df is not None:
                    st.session_state.tlag_data = tlag_df
                    st.sidebar.success("✅ TLAG demo verisi yüklendi")
                    
                    if comments_df is not None:
                        # Yorum verilerini TLAG ile birleştir
                        merged_comments = merge_comments_with_tlag(comments_df, tlag_df)
                        st.session_state.comments_data = merged_comments
                        
                        # Yorum analizini yap
                        district_analysis = analyze_comments_by_scope(merged_comments, "DISTRICT")
                        nor_analysis = analyze_comments_by_scope(merged_comments, "NOR")
                        
                        st.session_state.analyzed_comments = {
                            "district": district_analysis,
                            "nor": nor_analysis
                        }
                        st.sidebar.success("✅ Yorum demo verisi yüklendi")
                else:
                    st.sidebar.error(message)
    
    # Dosya yükleme işlemi
    if uploaded_tlag is not None:
        with st.spinner("TLAG verileri işleniyor..."):
            df = load_tlag_data(uploaded_tlag)
            if df is not None:
                st.session_state.tlag_data = df
                st.sidebar.success(f"✅ {len(df)} istasyon verisi yüklendi")
                
                # Supabase'e kaydet
                save_data_to_supabase(df)
    
    if uploaded_comments is not None and st.session_state.tlag_data is not None:
        with st.spinner("Yorum verileri işleniyor..."):
            comments_df = load_comments_data(uploaded_comments)
            if comments_df is not None:
                # TLAG verisi ile birleştir
                merged_comments = merge_comments_with_tlag(comments_df, st.session_state.tlag_data)
                st.session_state.comments_data = merged_comments
                
                # Analiz yap
                district_analysis = analyze_comments_by_scope(merged_comments, "DISTRICT")
                nor_analysis = analyze_comments_by_scope(merged_comments, "NOR")
                
                st.session_state.analyzed_comments = {
                    "district": district_analysis,
                    "nor": nor_analysis
                }
                
                st.sidebar.success(f"✅ {len(comments_df)} yorum yüklendi ve analiz edildi")
                
                # Supabase'e kaydet
                save_data_to_supabase(None, merged_comments)
    
    # MAIN CONTENT
    if st.session_state.current_view == "main":
        display_main_dashboard()
    elif st.session_state.current_view == "district":
        display_district_analysis()
    elif st.session_state.current_view == "nor":
        display_nor_analysis() 
    elif st.session_state.current_view == "segmentation":
        display_segmentation_analysis()

def merge_comments_with_tlag(comments_df, tlag_df):
    """Yorum verilerini TLAG verisi ile birleştirir"""
    try:
        # ROC kodları üzerinden birleştir
        merged = pd.merge(
            comments_df,
            tlag_df[["ROC_NORMALIZED", "İstasyon", "NOR", "DISTRICT", "Site Segment"]],
            left_on="station_code",
            right_on="ROC_NORMALIZED", 
            how="left"
        )
        
        # Boş alanları doldur
        merged["NOR_FINAL"] = merged["NOR"].fillna(merged["territory"])
        merged["DISTRICT_FINAL"] = merged["DISTRICT"].fillna(merged["district"])
        
        return merged
    except Exception as e:
        st.error(f"Veri birleştirme hatası: {str(e)}")
        return comments_df

def display_main_dashboard():
    """Ana dashboard görünümü"""
    
    if st.session_state.tlag_data is None:
        st.markdown("## 🎯 TLAG PERFORMANS ANALİTİK'E HOŞGELDİNİZ")
        st.info("👈 Sol panelden Excel dosyalarınızı yükleyin veya demo verilerini deneyin")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📊 YENİ ÖZELLİKLER
            - ✅ Kalıcı veri saklama
            - ✅ Tıklanabilir metrikler
            - ✅ Detaylı istasyon analizi
            - ✅ Gelişmiş yorum kategorileme
            - ✅ Fırsat istasyonu tespiti
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 ANALIZ ALANLARI
            - 🏢 District bazlı performans
            - 📍 NOR bazlı analiz
            - 🎪 Site segmentasyon
            - 💬 Müşteri yorum analizi
            - 🤖 AI destekli öneriler
            """)
        return
    
    df = st.session_state.tlag_data
    
    # Ana metrikler
    st.markdown("## 📊 ANA METRİKLER")
    col1, col2, col3, col4 = st.columns(4)
    
    # Toplam istasyon sayısı
    create_enhanced_metric_card(
        col1, "Toplam İstasyon", len(df), "total_stations", df
    )
    
    # Ortalama skor
    if "SKOR" in df.columns:
        avg_score = df["SKOR"].mean() * 100
        create_enhanced_metric_card(
            col2, "Ortalama Skor", f"{avg_score:.1f}%", "avg_score", df
        )
    
    # Segment dağılımı
    if "Site Segment" in df.columns:
        segments = df["Site Segment"].value_counts()
        for idx, (segment, count) in enumerate(segments.head(2).items()):
            if idx == 0:
                create_enhanced_metric_card(
                    col3, segment, count, f"segment_{segment.replace(' ', '_')}", 
                    df[df["Site Segment"] == segment]
                )
            elif idx == 1:
                create_enhanced_metric_card(
                    col4, segment, count, f"segment_{segment.replace(' ', '_')}", 
                    df[df["Site Segment"] == segment]
                )
    
    # Performans dağılımı - District tablosu
    if "DISTRICT" in df.columns and "SKOR" in df.columns:
        st.markdown("## 📈 DISTRICT PERFORMANS TABLOSU")
        
        district_performance = df.groupby("DISTRICT").agg({
            "SKOR": "mean",
            "İstasyon": "count"
        }).reset_index()
        
        district_performance["SKOR_FORMATTED"] = district_performance["SKOR"].apply(
            lambda x: f"{x*100:.1f}%"
        )
        
        district_performance = district_performance.sort_values("SKOR", ascending=False)
        district_performance = district_performance.rename(columns={
            "DISTRICT": "District",
            "İstasyon": "İstasyon Sayısı", 
            "SKOR_FORMATTED": "Ortalama Skor"
        })
        
        # District tablosu ile grafik
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(
                district_performance[["District", "İstasyon Sayısı", "Ortalama Skor"]], 
                use_container_width=True,
                height=400
            )
        
        with col2:
            fig_district = px.bar(
                district_performance,
                x="District", 
                y="SKOR",
                title="District Bazında Ortalama Performans",
                labels={"SKOR": "Ortalama Skor", "District": "District"}
            )
            fig_district.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_district, use_container_width=True)
    
    # Fırsat istasyonları
    opportunity_stations = get_opportunity_stations(df)
    if not opportunity_stations.empty:
        st.markdown("## 🎯 FIRSAT İSTASYONLARI")
        st.markdown("*My Precious veya Primitive segment, 80% altı skor*")
        
        st.markdown(f"**Toplam {len(opportunity_stations)} fırsat istasyonu tespit edildi**")
        
        # En yüksek potansiyelli ilk 10
        if "TRANSACTION" in opportunity_stations.columns:
            opportunity_stations["potential"] = (
                (0.80 - opportunity_stations["SKOR"]) * 
                opportunity_stations["TRANSACTION"] / 1000
            )
            top_opportunities = opportunity_stations.nlargest(10, "potential")
        else:
            top_opportunities = opportunity_stations.nsmallest(10, "SKOR")
        
        for idx, (_, station) in enumerate(top_opportunities.iterrows(), 1):
            current_score = station["SKOR"] * 100
            potential_gain = 80 - current_score
            
            st.markdown(f"""
            <div class="opportunity-card">
                <strong>{idx}. {station['İstasyon']}</strong><br>
                Mevcut: {current_score:.1f}% → Hedef: 80% (+{potential_gain:.1f} puan)<br>
                <small>{station['DISTRICT']} | {station['Site Segment']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Navigasyon
    st.markdown("## 🎯 HANGİ ANALİZİ YAPMAK İSTİYORSUNUZ?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="nav-section">
            <h3>🏢 DISTRICT</h3>
            <p>Bölgesel performans analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("District Analizine Git", key="nav_district", use_container_width=True, type="primary"):
            st.session_state.current_view = "district"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="nav-section">
            <h3>📍 NOR</h3>
            <p>Operasyon bölgesi analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("NOR Analizine Git", key="nav_nor", use_container_width=True, type="primary"):
            st.session_state.current_view = "nor"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="nav-section">
            <h3>🎪 SİTE SEGMENTASYON</h3>
            <p>Segment bazlı performans</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Segmentasyon Analizine Git", key="nav_segmentation", use_container_width=True, type="primary"):
            st.session_state.current_view = "segmentation"
            st.rerun()

def display_district_analysis():
    """District analiz sayfası"""
    if st.button("🏠 Ana Sayfaya Dön", key="back_from_district"):
        st.session_state.current_view = "main"
        st.rerun()
    
    st.markdown("## 🏢 DISTRICT BAZLI ANALİZ")
    
    df = st.session_state.tlag_data
    if df is None or "DISTRICT" not in df.columns:
        st.error("District verisi bulunamadı")
        return
    
    districts = sorted(df["DISTRICT"].dropna().unique())
    selected_district = st.selectbox("District Seçin:", districts, key="district_selector")
    
    if selected_district:
        display_detailed_district_analysis(selected_district, df)

def display_detailed_district_analysis(district_name, df):
    """Detaylı district analizi"""
    district_data = df[df["DISTRICT"] == district_name].copy()
    
    # Temel metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("İstasyon Sayısı", len(district_data))
    
    with col2:
        if "SKOR" in district_data.columns:
            avg_score = district_data["SKOR"].mean() * 100
            st.metric("Ortalama Skor", f"{avg_score:.1f}%")
    
    with col3:
        if "Site Segment" in district_data.columns:
            dominant_segment = district_data["Site Segment"].mode().iloc[0]
            st.metric("Baskın Segment", dominant_segment)
    
    with col4:
        opportunity_count = len(get_opportunity_stations(district_data))
        st.metric("Fırsat İstasyonu", opportunity_count)
    
    # Segment dağılımı
    if "Site Segment" in district_data.columns:
        st.markdown("### 🎪 Segment Dağılımı")
        segment_counts = district_data["Site Segment"].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(segment_counts.reset_index(), use_container_width=True)
        with col2:
            fig_segments = px.pie(
                values=segment_counts.values, 
                names=segment_counts.index,
                title=f"{district_name} Segment Dağılımı"
            )
            st.plotly_chart(fig_segments, use_container_width=True)
    
    # En çok odaklanılması gereken 3 konu
    comments_analysis = st.session_state.get("analyzed_comments", {}).get("district", {})
    if district_name in comments_analysis:
        district_comment_data = comments_analysis[district_name]
        
        st.markdown("### 🎯 En Çok Odaklanılması Gereken 3 Konu")
        
        problem_categories = district_comment_data.get("problem_categories", [])
        if problem_categories:
            for idx, (category, data) in enumerate(problem_categories, 1):
                severity = "🔴 Kritik" if data["avg_score"] < 3.0 else "🟡 Orta" if data["avg_score"] < 4.0 else "🟢 Düşük"
                st.markdown(f"""
                **{idx}. {category}**
                - Ortalama Puan: {data['avg_score']:.1f}
                - Yorum Sayısı: {data['count']}
                - Önem Seviyesi: {severity}
                """)
        else:
            st.info("Yorum verisi bulunamadı")
    
    # En iyi ve en kötü istasyonlar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 En İyi İstasyonlar")
        if "SKOR" in district_data.columns:
            top_stations = district_data.nlargest(5, "SKOR")[["İstasyon", "SKOR", "Site Segment"]].copy()
            top_stations["SKOR"] = top_stations["SKOR"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(top_stations, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ En Kötü İstasyonlar") 
        if "SKOR" in district_data.columns:
            bottom_stations = district_data.nsmallest(5, "SKOR")[["İstasyon", "SKOR", "Site Segment"]].copy()
            bottom_stations["SKOR"] = bottom_stations["SKOR"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(bottom_stations, use_container_width=True)
    
    # Fırsat istasyonları
    opportunity_stations = get_opportunity_stations(district_data)
    if not opportunity_stations.empty:
        st.markdown("### 🎯 Fırsat İstasyonları")
        st.markdown("*My Precious veya Primitive segment, 80% altı skor*")
        
        opp_display = opportunity_stations[["İstasyon", "SKOR", "Site Segment"]].copy()
        opp_display["SKOR"] = opp_display["SKOR"].apply(lambda x: f"{x*100:.1f}%")
        opp_display["Potansiyel Kazanım"] = opportunity_stations["SKOR"].apply(
            lambda x: f"+{(0.80-x)*100:.1f}%" if x < 0.80 else "0%"
        )
        
        st.dataframe(opp_display, use_container_width=True)
    
    # AI Önerileri - Dropdown
    with st.expander("🤖 AI Destekli İyileştirme Önerileri"):
        if st.button(f"🤖 {district_name} için AI Analizi Oluştur", key=f"ai_district_{district_name}"):
            with st.spinner("AI analizi yapılıyor..."):
                # Burada AI analizi yapılabilir
                st.markdown(f"""
                ### 📊 {district_name} AI Analizi
                
                **Temel Bulgular:**
                - Toplam {len(district_data)} istasyon
                - Ortalama skor: {district_data["SKOR"].mean()*100:.1f}%
                - Fırsat istasyonu: {len(opportunity_stations)} adet
                
                **Öncelikli Aksiyonlar:**
                1. En düşük skorlu istasyonlara odaklan
                2. Fırsat istasyonlarını değerlendir
                3. Başarılı istasyonların best practice'lerini kopyala
                
                **Beklenen İyileştirme:** +{(len(opportunity_stations) * 5):.0f} puan
                """)

def display_nor_analysis():
    """NOR analiz sayfası"""
    if st.button("🏠 Ana Sayfaya Dön", key="back_from_nor"):
        st.session_state.current_view = "main"
        st.rerun()
    
    st.markdown("## 📍 NOR BAZLI ANALİZ")
    
    df = st.session_state.tlag_data
    if df is None or "NOR" not in df.columns:
        st.error("NOR verisi bulunamadı")
        return
    
    nors = sorted(df["NOR"].dropna().unique())
    selected_nor = st.selectbox("NOR Seçin:", nors, key="nor_selector")
    
    if selected_nor:
        display_detailed_nor_analysis(selected_nor, df)

def display_detailed_nor_analysis(nor_name, df):
    """Detaylı NOR analizi"""
    nor_data = df[df["NOR"] == nor_name].copy()
    
    # Temel metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("İstasyon Sayısı", len(nor_data))
    
    with col2:
        if "SKOR" in nor_data.columns:
            avg_score = nor_data["SKOR"].mean() * 100
            st.metric("Ortalama Skor", f"{avg_score:.1f}%")
    
    with col3:
        if "DISTRICT" in nor_data.columns:
            district_count = nor_data["DISTRICT"].nunique()
            st.metric("District Sayısı", district_count)
    
    with col4:
        opportunity_count = len(get_opportunity_stations(nor_data))
        st.metric("Fırsat İstasyonu", opportunity_count)
    
    # District dağılımı
    if "DISTRICT" in nor_data.columns:
        st.markdown("### 🏢 District Dağılımı")
        district_performance = nor_data.groupby("DISTRICT").agg({
            "SKOR": "mean",
            "İstasyon": "count"
        }).reset_index()
        
        district_performance["SKOR_FORMATTED"] = district_performance["SKOR"].apply(
            lambda x: f"{x*100:.1f}%"
        )
        
        st.dataframe(district_performance[["DISTRICT", "İstasyon", "SKOR_FORMATTED"]], use_container_width=True)
    
    # Performans dağılım grafiği
    if "SKOR" in nor_data.columns:
        st.markdown("### 📊 Performans Dağılımı")
        
        nor_viz = nor_data.copy()
        nor_viz["Skor_Yüzde"] = nor_viz["SKOR"] * 100
        
        fig_dist = px.histogram(
            nor_viz,
            x="Skor_Yüzde",
            nbins=15,
            title=f"{nor_name} - Performans Dağılımı",
            labels={"Skor_Yüzde": "Skor (%)", "count": "İstasyon Sayısı"}
        )
        fig_dist.add_vline(x=80, line_dash="dash", line_color="orange", annotation_text="Hedef: 80%")
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Fırsat istasyonları detayı
    opportunity_stations = get_opportunity_stations(nor_data)
    if not opportunity_stations.empty:
        st.markdown("### 🎯 Fırsat İstasyonları Detayı")
        
        opp_detail = opportunity_stations[["İstasyon", "SKOR", "DISTRICT", "Site Segment"]].copy()
        opp_detail["SKOR"] = opp_detail["SKOR"].apply(lambda x: f"{x*100:.1f}%")
        opp_detail["Potansiyel"] = opportunity_stations["SKOR"].apply(
            lambda x: f"+{(0.80-x)*100:.1f}%" 
        )
        
        st.dataframe(opp_detail, use_container_width=True)

def display_segmentation_analysis():
    """Segmentasyon analiz sayfası"""
    if st.button("🏠 Ana Sayfaya Dön", key="back_from_segmentation"):
        st.session_state.current_view = "main"
        st.rerun()
    
    st.markdown("## 🎪 SİTE SEGMENTASYON ANALİZİ")
    
    df = st.session_state.tlag_data
    if df is None or "Site Segment" not in df.columns:
        st.error("Site Segment verisi bulunamadı")
        return
    
    # Segment genel bakış
    st.markdown("### 📊 Segment Genel Bakış")
    
    segment_summary = df.groupby("Site Segment").agg({
        "İstasyon": "count",
        "SKOR": ["mean", "min", "max"]
    }).round(3)
    
    segment_summary.columns = ["İstasyon Sayısı", "Ort. Skor", "Min Skor", "Max Skor"]
    segment_summary["Ort. Skor %"] = (segment_summary["Ort. Skor"] * 100).round(1).astype(str) + "%"
    segment_summary["Min Skor %"] = (segment_summary["Min Skor"] * 100).round(1).astype(str) + "%"
    segment_summary["Max Skor %"] = (segment_summary["Max Skor"] * 100).round(1).astype(str) + "%"
    
    st.dataframe(segment_summary, use_container_width=True)
    
    # Segment seçimi
    segments = sorted(df["Site Segment"].dropna().unique())
    selected_segment = st.selectbox("Segment Seçin:", segments, key="segment_selector")
    
    if selected_segment:
        display_detailed_segment_analysis(selected_segment, df)

def display_detailed_segment_analysis(segment_name, df):
    """Detaylı segment analizi"""
    segment_data = df[df["Site Segment"] == segment_name].copy()
    
    st.markdown(f"### 🎯 {segment_name} Detaylı Analizi")
    
    # Segment metrikleri
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam İstasyon", len(segment_data))
    
    with col2:
        if "SKOR" in segment_data.columns:
            avg_score = segment_data["SKOR"].mean() * 100
            st.metric("Ortalama Skor", f"{avg_score:.1f}%")
    
    with col3:
        if "DISTRICT" in segment_data.columns:
            district_count = segment_data["DISTRICT"].nunique()
            st.metric("District Sayısı", district_count)
    
    with col4:
        if "SKOR" in segment_data.columns:
            below_target = len(segment_data[segment_data["SKOR"] < 0.80])
            st.metric("80% Altı İstasyon", below_target)
    
    # District bazlı dağılım
    if "DISTRICT" in segment_data.columns:
        st.markdown("#### 🏢 District Bazlı Dağılım")
        
        district_breakdown = segment_data.groupby("DISTRICT").agg({
            "İstasyon": "count",
            "SKOR": "mean"
        }).reset_index()
        
        district_breakdown["SKOR"] = district_breakdown["SKOR"].apply(lambda x: f"{x*100:.1f}%")
        district_breakdown = district_breakdown.sort_values("İstasyon", ascending=False)
        
        st.dataframe(district_breakdown, use_container_width=True)
    
    # Performans grafiği
    if "SKOR" in segment_data.columns:
        st.markdown("#### 📈 Performans Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            segment_viz = segment_data.copy()
            segment_viz["Skor_Yüzde"] = segment_viz["SKOR"] * 100
            
            fig_box = px.box(
                segment_viz,
                x="DISTRICT",
                y="Skor_Yüzde",
                title=f"{segment_name} - District Bazlı Performans",
                labels={"Skor_Yüzde": "Skor (%)"}
            )
            fig_box.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Histogram
            fig_hist = px.histogram(
                segment_viz,
                x="Skor_Yüzde",
                nbins=15,
                title=f"{segment_name} - Skor Dağılımı",
                labels={"Skor_Yüzde": "Skor (%)", "count": "İstasyon Sayısı"}
            )
            fig_hist.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="Hedef: 80%")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # En iyi ve en kötü performans
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 En İyi Performans")
        if "SKOR" in segment_data.columns:
            top_performers = segment_data.nlargest(10, "SKOR")[["İstasyon", "SKOR", "DISTRICT"]].copy()
            top_performers["SKOR"] = top_performers["SKOR"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(top_performers, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚠️ Gelişim Gereken İstasyonlar")
        if "SKOR" in segment_data.columns:
            low_performers = segment_data.nsmallest(10, "SKOR")[["İstasyon", "SKOR", "DISTRICT"]].copy()
            low_performers["SKOR"] = low_performers["SKOR"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(low_performers, use_container_width=True)

if __name__ == "__main__":
    main()
