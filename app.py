# app.py - Enhanced TLAG Performance Analytics with AI

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
import os

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
# Supabase entegrasyonu - GEÇİCİ OLARAK KAPALI
# ------------------------------------------------------------
SUPABASE_ENABLED = False  # Geçici olarak kapalı

# ------------------------------------------------------------
# OpenAI AI Analiz Fonksiyonları
# ------------------------------------------------------------
def get_openai_api_key():
    """OpenAI API anahtarını al"""
    try:
        # Önce environment variable'dan dene
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        # Sonra Streamlit secrets'tan dene
        if hasattr(st, 'secrets') and 'openai' in st.secrets:
            return st.secrets.openai.api_key
        
        return None
    except:
        return None

def ai_recommendations_for_scope(scope_name, df_scope, comments_scope=None):
    """Belirli bir scope (District/NOR/İstasyon) için AI analizi"""
    try:
        try:
            import openai
        except ImportError:
            return "### 📦 OpenAI Modülü Gerekli\n\nAI önerileri için `pip install openai` kurun."
        
        api_key = get_openai_api_key()
        if not api_key:
            return "### 🔑 OpenAI API Anahtarı Gerekli\n\nSecrets'a OpenAI API anahtarınızı ekleyin."
        
        # Data özeti hazırla
        if df_scope is not None and not df_scope.empty:
            station_count = len(df_scope)
            avg_score = df_scope["SKOR"].mean() * 100 if "SKOR" in df_scope.columns else 0
            
            # En iyi ve en kötü istasyonlar
            if "SKOR" in df_scope.columns:
                best_stations = df_scope.nlargest(3, "SKOR")[["İstasyon", "SKOR"]].to_dict('records')
                worst_stations = df_scope.nsmallest(3, "SKOR")[["İstasyon", "SKOR"]].to_dict('records')
            else:
                best_stations = worst_stations = []
            
            # Segment dağılımı
            if "Site Segment" in df_scope.columns:
                segments = df_scope["Site Segment"].value_counts().to_dict()
            else:
                segments = {}
        else:
            station_count = avg_score = 0
            best_stations = worst_stations = []
            segments = {}
        
        # Yorum analizi özeti
        comment_summary = ""
        if comments_scope is not None and not comments_scope.empty:
            total_comments = len(comments_scope)
            avg_comment_score = comments_scope["score"].mean() if "score" in comments_scope.columns else 0
            
            # Kategori analizi
            category_problems = {}
            if "categories" in comments_scope.columns:
                for _, row in comments_scope.iterrows():
                    if isinstance(row["categories"], list):
                        for cat in row["categories"]:
                            if cat not in category_problems:
                                category_problems[cat] = []
                            category_problems[cat].append(row.get("score", 5))
            
            # Problem kategorileri (düşük puan alan)
            problem_cats = {}
            for cat, scores in category_problems.items():
                avg_cat_score = np.mean(scores)
                if avg_cat_score < 4.0:
                    problem_cats[cat] = {
                        "avg_score": avg_cat_score,
                        "count": len(scores)
                    }
            
            # Düşük puan yorumları örnekleri
            low_score_comments = comments_scope[comments_scope["score"] <= 3]["comment"].head(5).tolist()
            
            comment_summary = f"""
            YORUM ANALİZİ:
            - Toplam yorum: {total_comments}
            - Ortalama puan: {avg_comment_score:.1f}/5
            - Problem kategorileri: {problem_cats}
            - Düşük puan yorum örnekleri: {low_score_comments[:3]}
            """
        
        # AI prompt hazırla
        prompt = f"""Sen bir petrol istasyonu performans analisti olan uzman bir AI'sın. {scope_name} bölgesi için detaylı analiz yapmanı istiyorum.

VERİ ÖZETİ:
- İstasyon sayısı: {station_count}
- Ortalama performans skoru: {avg_score:.1f}%
- Segment dağılımı: {segments}
- En iyi 3 istasyon: {best_stations}
- En kötü 3 istasyon: {worst_stations}

{comment_summary}

GÖREV:
1. Bu verilere dayanarak {scope_name} bölgesi için kapsamlı bir performans analizi yap
2. Ana sorun alanlarını tespit et
3. Her sorun alanı için spesifik, uygulanabilir çözüm önerileri ver
4. Öncelik sıralaması yap (yüksek/orta/düşük)
5. Beklenen iyileştirme yüzdesini tahmin et

ÇIKTI FORMATI:
### 📊 {scope_name} Performans Analizi

**Ana Bulgular:**
- [Temel performans durumu]
- [Kritik sorun alanları]
- [Fırsat alanları]

**Öncelikli Aksiyon Planı:**

🔴 **Yüksek Öncelik:**
1. [Spesifik aksiyon] - Beklenen iyileştirme: X%
2. [Spesifik aksiyon] - Beklenen iyileştirme: X%

🟡 **Orta Öncelik:**
1. [Spesifik aksiyon]
2. [Spesifik aksiyon]

**Detaylı Öneriler:**
- [Her kategori için spesifik öneriler]

**Beklenen Toplam İyileştirme:** +X puan

Türkçe yanıt ver ve pratik, uygulanabilir öneriler sun."""

        # OpenAI API çağrısı
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Eski API syntax'ını dene
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
            
    except Exception as e:
        return f"### ❌ AI Analiz Hatası\n\nHata: {str(e)}\n\nLütfen OpenAI API anahtarınızı kontrol edin."

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
# İyileştirilmiş yorum birleştirme
# ------------------------------------------------------------
def merge_comments_with_tlag(comments_df, tlag_df):
    """Yorum verilerini TLAG verisi ile gelişmiş birleştirme"""
    try:
        st.write("🔍 Yorum verileri birleştiriliyor...")
        st.write(f"📊 TLAG veri: {len(tlag_df)} istasyon")
        st.write(f"💬 Yorum veri: {len(comments_df)} yorum")
        
        # ROC kodları üzerinden birleştir
        merged = pd.merge(
            comments_df,
            tlag_df[["ROC_NORMALIZED", "ROC_STR", "İstasyon", "NOR", "DISTRICT", "Site Segment"]],
            left_on="station_code",
            right_on="ROC_NORMALIZED", 
            how="left"
        )
        
        # Alternatif birleştirme - ROC_STR ile
        not_merged = merged[merged["İstasyon"].isna()]
        if not not_merged.empty:
            st.write(f"⚠️ {len(not_merged)} yorum ROC_NORMALIZED ile eşleşmedi, ROC_STR deneniyor...")
            
            merged2 = pd.merge(
                not_merged[["station_code", "comment", "score", "categories", "dealer", "territory", "district", "positive_but_4star"]],
                tlag_df[["ROC_STR", "İstasyon", "NOR", "DISTRICT", "Site Segment"]],
                left_on="station_code",
                right_on="ROC_STR",
                how="left"
            )
            
            # Eşleşenleri ana veri ile birleştir
            merged_part1 = merged[merged["İstasyon"].notna()]
            merged_part2 = merged2[merged2["İstasyon"].notna()]
            
            if not merged_part2.empty:
                # Kolonları hizala
                merged_part2["ROC_NORMALIZED"] = merged_part2["ROC_STR"]
                merged = pd.concat([merged_part1, merged_part2], ignore_index=True)
        
        # Boş alanları doldur
        merged["NOR_FINAL"] = merged["NOR"].fillna(merged["territory"])
        merged["DISTRICT_FINAL"] = merged["DISTRICT"].fillna(merged["district"])
        
        # Başarı raporu
        successful_matches = len(merged[merged["İstasyon"].notna()])
        st.success(f"✅ {successful_matches}/{len(comments_df)} yorum başarıyla eşleştirildi!")
        
        if successful_matches < len(comments_df):
            st.warning(f"⚠️ {len(comments_df) - successful_matches} yorum eşleştirilemedi")
        
        # District bazlı yorum dağılımı
        district_comment_counts = merged.groupby("DISTRICT_FINAL").size().reset_index(columns=["Yorum_Sayısı"])
        st.write("📊 District bazlı yorum dağılımı:")
        st.dataframe(district_comment_counts, use_container_width=True)
        
        return merged
    except Exception as e:
        st.error(f"Veri birleştirme hatası: {str(e)}")
        return comments_df

# ------------------------------------------------------------
# İyileştirilmiş analiz fonksiyonları
# ------------------------------------------------------------
def analyze_comments_by_scope(comments_df, scope_col="DISTRICT"):
    """Kapsamlı yorum analizi - iyileştirilmiş"""
    if comments_df is None or comments_df.empty:
        return {}
    
    # Scope column belirleme
    if scope_col == "DISTRICT":
        group_col = "DISTRICT_FINAL"
    elif scope_col == "NOR":
        group_col = "NOR_FINAL"
    else:
        group_col = scope_col
        
    if group_col not in comments_df.columns:
        st.warning(f"⚠️ {group_col} kolonu bulunamadı")
        return {}
    
    results = {}
    
    # Null olmayan grupları al
    valid_comments = comments_df[comments_df[group_col].notna() & (comments_df[group_col] != "nan")]
    
    for name, group in valid_comments.groupby(group_col):
        if pd.isna(name) or str(name).strip() == "" or str(name).strip().lower() == "nan":
            continue
            
        total_comments = len(group)
        
        if total_comments == 0:
            continue
            
        avg_score = group["score"].mean()
        
        # Puan dağılımı
        score_dist = group["score"].value_counts().to_dict()
        
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
            cat: np.mean(scores) for cat, scores in category_scores.items() if len(scores) > 0
        }
        
        # En problemli kategoriler (düşük puan + yeterli sayıda yorum)
        problem_categories = {}
        for cat in category_counts:
            if cat in category_avg_scores and category_avg_scores[cat] < 4.0 and category_counts[cat] >= 2:
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

# ------------------------------------------------------------
# Yorum görüntüleme fonksiyonu
# ------------------------------------------------------------
def display_comment_analysis_enhanced(analysis_data, title="Yorum Analizi"):
    """Gelişmiş yorum analizi görüntüleme"""
    if not analysis_data:
        st.info(f"📋 {title} için yorum verisi bulunamadı")
        return
    
    st.markdown(f"### 💬 {title}")
    
    # Özet istatistikler
    total_areas = len(analysis_data)
    total_comments = sum(data["total_comments"] for data in analysis_data.values())
    avg_score_overall = np.mean([data["avg_score"] for data in analysis_data.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam Alan", total_areas)
    with col2:
        st.metric("Toplam Yorum", total_comments)
    with col3:
        st.metric("Genel Ort. Puan", f"{avg_score_overall:.1f}/5")
    
    # Detaylı analiz
    for name, data in analysis_data.items():
        with st.expander(f"📍 {name} - {data['total_comments']} yorum (Ort: {data['avg_score']:.1f}/5)"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Puan Dağılımı:**")
                if data["score_distribution"]:
                    score_df = pd.DataFrame(
                        list(data["score_distribution"].items()), 
                        columns=["Puan", "Sayı"]
                    ).sort_values("Puan")
                    
                    fig = px.bar(
                        score_df, x="Puan", y="Sayı", 
                        color="Puan", color_continuous_scale="RdYlGn",
                        title=f"{name} Puan Dağılımı"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"score_{name}")
            
            with col2:
                st.markdown("**⚠️ Problem Kategorileri:**")
                if data["problem_categories"]:
                    for i, (cat, cat_data) in enumerate(data["problem_categories"], 1):
                        severity = "🔴" if cat_data["avg_score"] < 3.0 else "🟡" if cat_data["avg_score"] < 3.5 else "🟠"
                        st.markdown(f"""
                        **{i}. {cat}** {severity}
                        - Ortalama: {cat_data['avg_score']:.1f}/5
                        - Yorum: {cat_data['count']} adet
                        """)
                else:
                    st.info("Problem kategorisi bulunamadı")
            
            # Kategori bazlı detaylar
            if data["category_counts"]:
                st.markdown("**📋 Tüm Kategoriler:**")
                cat_summary = []
                for cat, count in data["category_counts"].items():
                    avg_score = data["category_avg_scores"].get(cat, 0)
                    status = "🔴 Problem" if avg_score < 3.5 else "🟡 Dikkat" if avg_score < 4.0 else "🟢 İyi"
                    cat_summary.append({
                        "Kategori": cat,
                        "Yorum Sayısı": count,
                        "Ort. Puan": f"{avg_score:.1f}",
                        "Durum": status
                    })
                
                cat_df = pd.DataFrame(cat_summary).sort_values("Ort. Puan")
                st.dataframe(cat_df, use_container_width=True)

# Geri kalan kod (UI components, main dashboard vb.) aynı kalacak - sadece AI entegrasyonunu ve yorum analizini düzelttim.
# Bu noktadan sonra önceki kodun geri kalanını da eklemem gerekiyor...

# ------------------------------------------------------------
# Ana yardımcı fonksiyonlar (devamı)
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

# ------------------------------------------------------------
# UI Components
# ------------------------------------------------------------
def create_enhanced_metric_card(col, title, value, key, click_data=None):
    """Gelişmiş metrik kartları"""
    with col:
        st.markdown(f"""
        <div class="metric-card">
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
    
    # AI Analizi butonu
    if st.button(f"🤖 {station_data['İstasyon']} için AI Analizi", key=f"ai_station_detail_{station_data.get('ROC_STR')}"):
        with st.spinner("AI analizi yapılıyor..."):
            station_code = station_data.get("ROC_NORMALIZED") or station_data.get("ROC_STR")
            station_comments = None
            
            if st.session_state.get("comments_data") is not None:
                station_comments = st.session_state.comments_data[
                    st.session_state.comments_data["station_code"] == str(station_code)
                ]
            
            station_df = pd.DataFrame([station_data])
            ai_result = ai_recommendations_for_scope(
                station_data['İstasyon'], 
                station_df, 
                station_comments
            )
            st.markdown(ai_result)
    
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
# Basitleştirilmiş veri kaydetme - Session Only
# ------------------------------------------------------------
def save_data_to_supabase(df, comments_df=None, period_meta=None):
    """Veri session'da saklanıyor - Supabase devre dışı"""
    if not SUPABASE_ENABLED:
        st.info("ℹ️ Veriler session'da saklanıyor - Supabase şu anda devre dışı")
        return

# CSS ve sayfa yapılandırması - öncekiyle aynı
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
    
    .opportunity-card {
        background: linear-gradient(135deg, #FFA502 0%, #FF6B6B 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "tlag_data" not in st.session_state:
    st.session_state.tlag_data = None
if "comments_data" not in st.session_state:
    st.session_state.comments_data = None
if "analyzed_comments" not in st.session_state:
    st.session_state.analyzed_comments = None
if "current_view" not in st.session_state:
    st.session_state.current_view = "main"

# ANA UYGULAMA
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
    
    if st.sidebar.button("☁️ Cloud Demo", use_container_width=True):
        with st.spinner("Veriler yükleniyor..."):
            tlag_df, comments_df, message = load_demo_data_from_cloud()
            if tlag_df is not None:
                st.session_state.tlag_data = tlag_df
                st.sidebar.success("✅ TLAG demo verisi yüklendi")
                
                if comments_df is not None:
                    # Yorum verilerini TLAG ile birleştir - iyileştirilmiş
                    merged_comments = merge_comments_with_tlag(comments_df, tlag_df)
                    st.session_state.comments_data = merged_comments
                    
                    # Yorum analizini yap - iyileştirilmiş
                    district_analysis = analyze_comments_by_scope(merged_comments, "DISTRICT")
                    nor_analysis = analyze_comments_by_scope(merged_comments, "NOR")
                    
                    st.session_state.analyzed_comments = {
                        "district": district_analysis,
                        "nor": nor_analysis
                    }
                    st.sidebar.success("✅ Yorum demo verisi yüklendi")
                
                # Session'a kaydet
                save_data_to_supabase(tlag_df, merged_comments if comments_df is not None else None)
            else:
                st.sidebar.error(message)
    
    # Dosya yükleme işlemi
    if uploaded_tlag is not None:
        with st.spinner("TLAG verileri işleniyor..."):
            df = load_tlag_data(uploaded_tlag)
            if df is not None:
                st.session_state.tlag_data = df
                st.sidebar.success(f"✅ {len(df)} istasyon verisi yüklendi")
                save_data_to_supabase(df)
    
    if uploaded_comments is not None and st.session_state.tlag_data is not None:
        with st.spinner("Yorum verileri işleniyor..."):
            comments_df = load_comments_data(uploaded_comments)
            if comments_df is not None:
                merged_comments = merge_comments_with_tlag(comments_df, st.session_state.tlag_data)
                st.session_state.comments_data = merged_comments
                
                district_analysis = analyze_comments_by_scope(merged_comments, "DISTRICT")
                nor_analysis = analyze_comments_by_scope(merged_comments, "NOR")
                
                st.session_state.analyzed_comments = {
                    "district": district_analysis,
                    "nor": nor_analysis
                }
                
                st.sidebar.success(f"✅ {len(comments_df)} yorum yüklendi ve analiz edildi")
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

def display_main_dashboard():
    """Ana dashboard görünümü"""
    
    if st.session_state.tlag_data is None:
        st.markdown("## 🎯 TLAG PERFORMANS ANALİTİK'E HOŞGELDİNİZ")
        st.info("👈 Sol panelden Excel dosyalarınızı yükleyin veya demo verilerini deneyin")
        return
    
    df = st.session_state.tlag_data
    
    # Ana metrikler
    st.markdown("## 📊 ANA METRİKLER")
    col1, col2, col3, col4 = st.columns(4)
    
    create_enhanced_metric_card(
        col1, "Toplam İstasyon", len(df), "total_stations", df
    )
    
    if "SKOR" in df.columns:
        avg_score = df["SKOR"].mean() * 100
        create_enhanced_metric_card(
            col2, "Ortalama Skor", f"{avg_score:.1f}%", "avg_score", df
        )
    
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
    
    # District performans tablosu ve yorum analizi
    if "DISTRICT" in df.columns and "SKOR" in df.columns:
        st.markdown("## 📈 DISTRICT PERFORMANS VE YORUM ANALİZİ")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 District Performans Tablosu")
            district_performance = df.groupby("DISTRICT").agg({
                "SKOR": "mean",
                "İstasyon": "count"
            }).reset_index()
            
            district_performance["SKOR_FORMATTED"] = district_performance["SKOR"].apply(
                lambda x: f"{x*100:.1f}%"
            )
            
            district_performance = district_performance.sort_values("SKOR", ascending=False)
            st.dataframe(district_performance[["DISTRICT", "İstasyon", "SKOR_FORMATTED"]], use_container_width=True)
        
        with col2:
            st.markdown("### 💬 District Yorum Analizi")
            if st.session_state.get("analyzed_comments") and st.session_state.analyzed_comments.get("district"):
                display_comment_analysis_enhanced(
                    st.session_state.analyzed_comments["district"], 
                    "District Bazlı Yorumlar"
                )
            else:
                st.info("Yorum verisi yüklenirken analiz edilecek")
    
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
            <p>Bölgesel performans ve AI analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("District Analizine Git", key="nav_district", use_container_width=True, type="primary"):
            st.session_state.current_view = "district"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="nav-section">
            <h3>📍 NOR</h3>
            <p>NOR bazlı performans ve AI analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("NOR Analizine Git", key="nav_nor", use_container_width=True, type="primary"):
            st.session_state.current_view = "nor"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="nav-section">
            <h3>🎪 SİTE SEGMENTASYON</h3>
            <p>Segment bazlı performans analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Segmentasyon Analizine Git", key="nav_segmentation", use_container_width=True, type="primary"):
            st.session_state.current_view = "segmentation"
            st.rerun()

def display_district_analysis():
    """District analiz sayfası - AI entegrasyonu ile"""
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
        district_data = df[df["DISTRICT"] == selected_district].copy()
        
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
                dominant_segment = district_data["Site Segment"].mode().iloc[0] if len(district_data["Site Segment"].mode()) > 0 else "N/A"
                st.metric("Baskın Segment", dominant_segment)
        
        with col4:
            opportunity_count = len(get_opportunity_stations(district_data))
            st.metric("Fırsat İstasyonu", opportunity_count)
        
        # AI Analizi
        st.markdown("### 🤖 AI PERFORMANS ANALİZİ")
        if st.button(f"🤖 {selected_district} için AI Analizi Oluştur", key=f"ai_district_{selected_district}"):
            with st.spinner("AI analizi yapılıyor..."):
                # District yorumlarını al
                district_comments_df = None
                if st.session_state.get("comments_data") is not None:
                    district_comments_df = st.session_state.comments_data[
                        st.session_state.comments_data["DISTRICT_FINAL"] == selected_district
                    ]
                
                ai_result = ai_recommendations_for_scope(
                    selected_district, 
                    district_data, 
                    district_comments_df
                )
                st.markdown(ai_result)
        
        # District yorum analizi
        if st.session_state.get("analyzed_comments") and st.session_state.analyzed_comments.get("district"):
            district_comments = st.session_state.analyzed_comments["district"]
            if selected_district in district_comments:
                display_comment_analysis_enhanced({selected_district: district_comments[selected_district]}, 
                                                f"{selected_district} Müşteri Yorumları")

def display_nor_analysis():
    """NOR analiz sayfası - AI entegrasyonu ile"""
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
        nor_data = df[df["NOR"] == selected_nor].copy()
        
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
        
        # AI Analizi
        st.markdown("### 🤖 AI PERFORMANS ANALİZİ")
        if st.button(f"🤖 {selected_nor} için AI Analizi Oluştur", key=f"ai_nor_{selected_nor}"):
            with st.spinner("AI analizi yapılıyor..."):
                # NOR yorumlarını al
                nor_comments_df = None
                if st.session_state.get("comments_data") is not None:
                    nor_comments_df = st.session_state.comments_data[
                        st.session_state.comments_data["NOR_FINAL"] == selected_nor
                    ]
                
                ai_result = ai_recommendations_for_scope(
                    selected_nor, 
                    nor_data, 
                    nor_comments_df
                )
                st.markdown(ai_result)
        
        # NOR yorum analizi
        if st.session_state.get("analyzed_comments") and st.session_state.analyzed_comments.get("nor"):
            nor_comments = st.session_state.analyzed_comments["nor"]
            if selected_nor in nor_comments:
                display_comment_analysis_enhanced({selected_nor: nor_comments[selected_nor]}, 
                                                f"{selected_nor} Müşteri Yorumları")

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

if __name__ == "__main__":
    main()