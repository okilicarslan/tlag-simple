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
    m = re.search(r"#(\d{4})$", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None

def extract_station_code(station_info):
    """Station info'dan son 4 haneli kodu çıkarır"""
    if pd.isna(station_info):
        return None
    s = str(station_info)
    m = re.search(r"#(\d{4})$", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})(?=\D*$)", s)
    return m.group(1) if m else None

# ------------------------------------------------------------
# Supabase entegrasyonu
# ------------------------------------------------------------
SUPABASE_ENABLED = True

try:
    from supabase import create_client, Client
    
    def init_supabase():
        """Supabase client başlatma"""
        try:
            url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", None)
            key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", None)
            
            if url and key:
                return create_client(url, key)
            return None
        except:
            return None
    
    supabase: Client = init_supabase()
except ImportError:
    supabase = None
    SUPABASE_ENABLED = False

# ------------------------------------------------------------
# OpenAI AI Analiz Fonksiyonları
# ------------------------------------------------------------
def get_openai_api_key():
    """OpenAI API anahtarını al"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        if hasattr(st, 'secrets') and 'openai' in st.secrets:
            return st.secrets.openai.api_key
        return None
    except:
        return None

def ai_chat_interface(df, comments_df=None):
    """AI Chat Interface - Enter ile çalışır"""
    st.markdown("### 🤖 AI TLAG UZMANI")
    
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 15px; color: white;">
            <p style="color: white; margin: 0;">Sorularınızı yazıp Enter'a basın. Örn: "Ankara Güney'in skorunu nasıl artırırım?"</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Form ile Enter desteği
        with st.form(key="ai_chat_form", clear_on_submit=True):
            user_query = st.text_input(
                "Sorunuz:",
                placeholder="İstasyon, NOR veya District analizi için soru yazın...",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns([6, 1])
            with col1:
                submit = st.form_submit_button("🚀 Analiz", use_container_width=True, type="primary")
            with col2:
                if st.form_submit_button("🗑️", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        if submit and user_query:
            with st.spinner("Analiz yapılıyor... (5-10 saniye)"):
                response = get_ai_response(user_query, df, comments_df)
                st.session_state.chat_history.append({
                    "query": user_query,
                    "response": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
        
        # Chat geçmişi
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history[-3:]):  # Son 3 mesaj
                with st.container():
                    st.markdown(f"**🕐 {chat['timestamp']} - Soru:** {chat['query']}")
                    st.info(chat['response'])

def get_ai_response(query, df, comments_df=None):
    """AI analizi - Hızlı ve spesifik"""
    api_key = get_openai_api_key()
    if not api_key:
        return "⚠️ OpenAI API anahtarı bulunamadı."
    
    try:
        from openai import OpenAI
        
        # Sorgu analizi
        query_lower = query.lower()
        target_nor = None
        target_district = None
        
        if "NOR" in df.columns:
            for nor in df["NOR"].dropna().unique():
                if str(nor).lower() in query_lower:
                    target_nor = nor
                    break
        
        if "DISTRICT" in df.columns:
            for district in df["DISTRICT"].dropna().unique():
                if str(district).lower() in query_lower:
                    target_district = district
                    break
        
        # Veri filtreleme
        filtered_df = df.copy()
        filtered_comments = comments_df.copy() if comments_df is not None else pd.DataFrame()
        
        if target_nor:
            filtered_df = df[df["NOR"] == target_nor]
            if not filtered_comments.empty and "NOR_FINAL" in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments["NOR_FINAL"] == target_nor]
        elif target_district:
            filtered_df = df[df["DISTRICT"] == target_district]
            if not filtered_comments.empty and "DISTRICT_FINAL" in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments["DISTRICT_FINAL"] == target_district]
        
        # Kritik veriler
        worst_3 = filtered_df.nsmallest(3, "SKOR")[["İstasyon", "SKOR"]].to_dict('records')
        avg_score = filtered_df['SKOR'].mean() * 100
        
        problem_cats = {}
        if not filtered_comments.empty:
            for _, row in filtered_comments.iterrows():
                if row.get("score", 5) <= 3 and isinstance(row.get("categories"), list):
                    for cat in row["categories"]:
                        if cat not in problem_cats:
                            problem_cats[cat] = 0
                        problem_cats[cat] += 1
        
        top_problems = sorted(problem_cats.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Kısa ve öz prompt
        prompt = f"""TLAG uzmanı olarak analiz et:

VERİ:
- {len(filtered_df)} istasyon, Ort: {avg_score:.1f}%
- En kötü 3: {', '.join([f"{s['İstasyon']} ({s['SKOR']*100:.1f}%)" for s in worst_3])}
- Problem kategoriler: {', '.join([f"{cat}({count})" for cat, count in top_problems])}

Soru: {query}

KISA ve SOMUT yanıt:
1. Hangi istasyona müdahale?
2. Hangi kategoriye odaklan?
3. Beklenen artış?

Gerçek istasyon isimleri kullan. Max 5 cümle."""
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Hata: {str(e)}"
    
    try:
        import openai
        from openai import OpenAI
        
        # Sorgudan NOR/District/İstasyon tespit et
        query_lower = query.lower()
        target_nor = None
        target_district = None
        target_station = None
        
        # NOR kontrolü
        if "NOR" in df.columns:
            for nor in df["NOR"].dropna().unique():
                if str(nor).lower() in query_lower:
                    target_nor = nor
                    break
        
        # District kontrolü
        if "DISTRICT" in df.columns:
            for district in df["DISTRICT"].dropna().unique():
                if str(district).lower() in query_lower:
                    target_district = district
                    break
        
        # İstasyon kontrolü
        if "İstasyon" in df.columns:
            for station in df["İstasyon"].dropna().unique():
                if str(station).lower() in query_lower:
                    target_station = station
                    break
        
        # Hedef bölge verilerini filtrele
        filtered_df = df.copy()
        filtered_comments = comments_df.copy() if comments_df is not None else pd.DataFrame()
        scope_name = "TÜM TÜRKİYE"
        
        if target_station:
            filtered_df = df[df["İstasyon"] == target_station]
            scope_name = f"{target_station} İSTASYONU"
            if not filtered_comments.empty and "İstasyon" in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments["İstasyon"] == target_station]
        elif target_nor:
            filtered_df = df[df["NOR"] == target_nor]
            scope_name = f"{target_nor} NOR BÖLGESİ"
            if not filtered_comments.empty and "NOR_FINAL" in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments["NOR_FINAL"] == target_nor]
        elif target_district:
            filtered_df = df[df["DISTRICT"] == target_district]
            scope_name = f"{target_district} DISTRICT BÖLGESİ"
            if not filtered_comments.empty and "DISTRICT_FINAL" in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments["DISTRICT_FINAL"] == target_district]
        
        # DETAYLI VERİ ANALİZİ
        detailed_analysis = f"""
===== {scope_name} DETAYLI VERİ ANALİZİ =====

📊 İSTASYON VERİLERİ:
- Toplam istasyon sayısı: {len(filtered_df)}
- Ortalama TLAG skoru: {filtered_df['SKOR'].mean()*100:.2f}%
- En yüksek TLAG skoru: {filtered_df['SKOR'].max()*100:.2f}%
- En düşük TLAG skoru: {filtered_df['SKOR'].min()*100:.2f}%
- Standart sapma: {filtered_df['SKOR'].std()*100:.2f}%
"""
        
        # EN DÜŞÜK SKORLU İSTASYONLAR - DETAYLI
        if "SKOR" in filtered_df.columns and len(filtered_df) > 0:
            worst_stations = filtered_df.nsmallest(min(10, len(filtered_df)), "SKOR")
            detailed_analysis += "\n🔴 EN DÜŞÜK SKORLU İSTASYONLAR (Acil müdahale gerekli):\n"
            for idx, (_, row) in enumerate(worst_stations.iterrows(), 1):
                detailed_analysis += f"{idx}. {row['İstasyon']}: {row['SKOR']*100:.1f}% "
                if "Site Segment" in row:
                    detailed_analysis += f"(Segment: {row['Site Segment']})"
                if "TRANSACTION" in row and pd.notna(row['TRANSACTION']):
                    detailed_analysis += f" [Transaction: {row['TRANSACTION']:.0f}]"
                improvement_potential = (0.80 - row['SKOR']) * 100
                if improvement_potential > 0:
                    detailed_analysis += f" → İyileştirme potansiyeli: +{improvement_potential:.1f} puan"
                detailed_analysis += "\n"
        
        # EN YÜKSEK SKORLU İSTASYONLAR
        if "SKOR" in filtered_df.columns and len(filtered_df) > 3:
            best_stations = filtered_df.nlargest(min(5, len(filtered_df)), "SKOR")
            detailed_analysis += "\n✅ EN YÜKSEK SKORLU İSTASYONLAR (Başarı örnekleri):\n"
            for idx, (_, row) in enumerate(best_stations.iterrows(), 1):
                detailed_analysis += f"{idx}. {row['İstasyon']}: {row['SKOR']*100:.1f}%\n"
        
        # SEGMENT ANALİZİ
        if "Site Segment" in filtered_df.columns:
            segment_analysis = filtered_df.groupby("Site Segment").agg({
                "İstasyon": "count",
                "SKOR": ["mean", "min", "max"]
            })
            detailed_analysis += "\n📈 SEGMENT BAZLI ANALİZ:\n"
            for segment in segment_analysis.index:
                count = segment_analysis.loc[segment, ("İstasyon", "count")]
                avg_score = segment_analysis.loc[segment, ("SKOR", "mean")] * 100
                min_score = segment_analysis.loc[segment, ("SKOR", "min")] * 100
                max_score = segment_analysis.loc[segment, ("SKOR", "max")] * 100
                detailed_analysis += f"- {segment}: {count} istasyon, Ortalama: {avg_score:.1f}%, Min: {min_score:.1f}%, Max: {max_score:.1f}%\n"
        
        # FIRSAT İSTASYONLARI
        if "Site Segment" in filtered_df.columns:
            opportunity = filtered_df[
                (filtered_df["Site Segment"].isin(["My Precious", "Primitive"])) & 
                (filtered_df["SKOR"] < 0.80)
            ]
            if not opportunity.empty:
                detailed_analysis += f"\n💎 FIRSAT İSTASYONLARI: {len(opportunity)} adet\n"
                for _, row in opportunity.head(5).iterrows():
                    detailed_analysis += f"- {row['İstasyon']}: {row['SKOR']*100:.1f}% → Hedef: 80% (Potansiyel: +{(0.80-row['SKOR'])*100:.1f} puan)\n"
        
        # MÜŞTERİ YORUMLARI ANALİZİ
        if not filtered_comments.empty and "score" in filtered_comments.columns:
            detailed_analysis += f"""

💬 MÜŞTERİ YORUMLARI ANALİZİ:
- Toplam yorum sayısı: {len(filtered_comments)}
- Ortalama müşteri puanı: {filtered_comments['score'].mean():.2f}/5
- 5 puan veren müşteri sayısı: {len(filtered_comments[filtered_comments['score'] == 5])} ({len(filtered_comments[filtered_comments['score'] == 5])/len(filtered_comments)*100:.1f}%)
- 4 puan veren müşteri sayısı: {len(filtered_comments[filtered_comments['score'] == 4])} ({len(filtered_comments[filtered_comments['score'] == 4])/len(filtered_comments)*100:.1f}%)
- 3 puan veren müşteri sayısı: {len(filtered_comments[filtered_comments['score'] == 3])} ({len(filtered_comments[filtered_comments['score'] == 3])/len(filtered_comments)*100:.1f}%)
- 1-2 puan veren (kritik) müşteri sayısı: {len(filtered_comments[filtered_comments['score'] <= 2])} ({len(filtered_comments[filtered_comments['score'] <= 2])/len(filtered_comments)*100:.1f}%)
"""
            
            # KATEGORİ BAZLI DETAYLI ANALİZ
            category_analysis = {}
            for _, row in filtered_comments.iterrows():
                if isinstance(row.get("categories"), list):
                    for cat in row["categories"]:
                        if cat not in category_analysis:
                            category_analysis[cat] = {
                                "scores": [],
                                "comments": [],
                                "stations": []
                            }
                        category_analysis[cat]["scores"].append(row["score"])
                        if pd.notna(row.get("comment")):
                            category_analysis[cat]["comments"].append(str(row["comment"])[:200])
                        if pd.notna(row.get("İstasyon")):
                            category_analysis[cat]["stations"].append(row["İstasyon"])
            
            # Problem kategorilerini tespit et
            detailed_analysis += "\n🔍 KATEGORİ BAZLI PROBLEM ANALİZİ:\n"
            problem_categories = []
            
            for cat, data in category_analysis.items():
                avg_score = np.mean(data["scores"])
                count = len(data["scores"])
                
                if avg_score < 4.0 and count >= 3:  # Problem kategorisi
                    problem_categories.append((cat, avg_score, count, data))
            
            # Problemleri önem sırasına göre sırala
            problem_categories.sort(key=lambda x: (x[1], -x[2]))  # Önce düşük puan, sonra yüksek sayı
            
            for cat, avg_score, count, data in problem_categories[:5]:  # En problemli 5 kategori
                detailed_analysis += f"\n⚠️ {cat}: {count} yorum, Ortalama: {avg_score:.1f}/5\n"
                
                # Bu kategoride en çok şikayet alan istasyonlar
                if data["stations"]:
                    station_counts = pd.Series(data["stations"]).value_counts().head(3)
                    detailed_analysis += f"   En çok şikayet alan istasyonlar:\n"
                    for station, s_count in station_counts.items():
                        detailed_analysis += f"   - {station}: {s_count} şikayet\n"
                
                # Düşük puanlı yorumlardan örnekler
                low_score_comments = [
                    (score, comment) 
                    for score, comment in zip(data["scores"], data["comments"]) 
                    if score <= 3
                ][:3]
                
                if low_score_comments:
                    detailed_analysis += f"   Örnek şikayetler:\n"
                    for score, comment in low_score_comments:
                        detailed_analysis += f"   • ({score}/5 puan) \"{comment[:100]}...\"\n"
            
            # İyi performans gösteren kategoriler
            good_categories = [(cat, np.mean(data["scores"]), len(data["scores"])) 
                             for cat, data in category_analysis.items() 
                             if np.mean(data["scores"]) >= 4.5 and len(data["scores"]) >= 3]
            
            if good_categories:
                detailed_analysis += "\n✅ İYİ PERFORMANS GÖSTEREN KATEGORİLER:\n"
                for cat, avg_score, count in good_categories[:3]:
                    detailed_analysis += f"- {cat}: {count} yorum, Ortalama: {avg_score:.1f}/5\n"
            
            # İSTASYON BAZLI YORUM ANALİZİ
            if "İstasyon" in filtered_comments.columns:
                station_comments = filtered_comments.groupby("İstasyon").agg({
                    "score": ["mean", "count"],
                    "comment": lambda x: list(x)[:3]  # İlk 3 yorum
                })
                
                detailed_analysis += "\n📍 İSTASYON BAZLI YORUM ANALİZİ:\n"
                
                # En kötü yorum alan istasyonlar
                worst_comment_stations = station_comments.sort_values(("score", "mean")).head(5)
                for station in worst_comment_stations.index:
                    avg = station_comments.loc[station, ("score", "mean")]
                    count = station_comments.loc[station, ("score", "count")]
                    if avg < 4.0:
                        detailed_analysis += f"- {station}: {count} yorum, Ortalama: {avg:.1f}/5 ⚠️\n"
        
        # ÖZET TLAG SKORU HESAPLAMASI
        if not filtered_comments.empty and "score" in filtered_comments.columns:
            score_5_count = len(filtered_comments[filtered_comments["score"] == 5])
            total_surveys = len(filtered_comments)
            calculated_tlag = (score_5_count / total_surveys * 100) if total_surveys > 0 else 0
            detailed_analysis += f"\n📊 HESAPLANAN TLAG SKORU: {calculated_tlag:.2f}% (5 puan veren / toplam anket)\n"
        
        # AI'ya gönderilecek prompt
        prompt = f"""Sen deneyimli bir TLAG (petrol istasyonu) performans uzmanısın. Elindeki GERÇEK VERİLER:

{detailed_analysis}

Kullanıcı sorusu: {query}

GÖREV:
1. Yukarıdaki VERİLERİ DETAYLI İNCELE
2. Sorulan soruya SADECE VERİLERE DAYALI cevap ver
3. GERÇEK İSTASYON İSİMLERİ kullan
4. GERÇEK SKORLAR ve YÜZDELER kullan
5. Somut, ölçülebilir öneriler sun
6. Her öneri için TAHMİNİ SKOR ARTIŞI belirt

ÖNEMLİ:
- Genel tavsiyelerden KAÇIN
- Sadece yukarıdaki verilerde OLAN bilgileri kullan
- İstasyon isimlerini ve skorları doğru kullan
- Problem kategorilerini ve yorumları dikkate al

Türkçe yanıtla ve VERİYE DAYALI öneriler sun."""
        
        # OpenAI API çağrısı
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen bir TLAG performans uzmanısın. SADECE sana verilen gerçek verilere dayanarak analiz yaparsın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Çok düşük temperature = daha tutarlı ve veriye bağlı
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "⚠️ OpenAI modülü kurulu değil. Terminal'de: pip install openai"
    except Exception as e:
        return f"❌ AI analiz hatası: {str(e)}\n\nLütfen OpenAI API anahtarınızı kontrol edin."

def ai_recommendations_for_scope(scope_name, df_scope, comments_scope=None):
    """Belirli bir scope için AI analizi"""
    api_key = get_openai_api_key()
    if not api_key:
        return "### 🔑 OpenAI API Anahtarı Gerekli\n\n.env dosyasına OPENAI_API_KEY ekleyin."
    
    try:
        import openai
        from openai import OpenAI
        
        # Data özeti hazırla
        station_count = len(df_scope) if df_scope is not None else 0
        avg_score = df_scope["SKOR"].mean() * 100 if df_scope is not None and "SKOR" in df_scope.columns else 0
        
        prompt = f"""Sen bir petrol istasyonu performans uzmanısın. {scope_name} için analiz yap.

        VERİ:
        - İstasyon sayısı: {station_count}
        - Ortalama skor: {avg_score:.1f}%
        
        Kısa ve net öneriler ver. Türkçe yanıtla."""
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"### ❌ AI Analiz Hatası\n\n{str(e)}"

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
            tlag_df["ROC_STR"] = tlag_df["ROC"].apply(lambda x: str(int(x)) if pd.notna(x) else str(x))
        
        # Sıralama ekle
        if "SKOR" in tlag_df.columns:
            valid_scores = tlag_df["SKOR"].notna()
            tlag_df.loc[valid_scores, "SIRALAMA"] = tlag_df.loc[valid_scores, "SKOR"].rank(ascending=False, method="min")
            tlag_df["SIRALAMA"] = tlag_df["SIRALAMA"].apply(lambda x: int(x) if pd.notna(x) else None)

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
# TLAG SKOR HESAPLAMA
# ------------------------------------------------------------
def calculate_tlag_score(df):
    """TLAG skorunu hesaplar: 5 puan verilen anket sayısı / toplam anket sayısı"""
    if "score_5_count" in df.columns and "total_surveys" in df.columns:
        df["TLAG_CALCULATED"] = df["score_5_count"] / df["total_surveys"]
        if "SKOR" not in df.columns:
            df["SKOR"] = df["TLAG_CALCULATED"]
    return df

# ------------------------------------------------------------
# Veri yükleme fonksiyonları
# ------------------------------------------------------------
def load_tlag_data(uploaded_file):
    """TLAG Excel dosyasını yükler"""
    try:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        available_sheets = xls.sheet_names
        
        target_sheet = None
        for sheet in available_sheets:
            if "TLAG" in sheet.upper():
                target_sheet = sheet
                break
        
        if target_sheet is None:
            target_sheet = available_sheets[0]
            st.info(f"TLAG sheet bulunamadı, '{target_sheet}' kullanılıyor")
        
        df = pd.read_excel(uploaded_file, sheet_name=target_sheet, engine="openpyxl")
        df.columns = df.columns.str.strip()
        
        # Kolon kontrolü
        required_cols = ["ROC", "İstasyon"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Gerekli kolonlar bulunamadı. Mevcut: {df.columns.tolist()}")
            return None
        
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
        df["ROC_STR"] = df["ROC"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
        df["ROC_NORMALIZED"] = df["ROC_STR"].apply(normalize_roc)
        
        # Sıralama ekle
        if "SKOR" in df.columns:
            valid_scores = df["SKOR"].notna()
            df.loc[valid_scores, "SIRALAMA"] = df.loc[valid_scores, "SKOR"].rank(ascending=False, method="min")
            df["SIRALAMA"] = df["SIRALAMA"].apply(lambda x: int(x) if pd.notna(x) else None)
        
        df = calculate_tlag_score(df)
        
        return df
    except Exception as e:
        st.error(f"TLAG dosya okuma hatası: {str(e)}")
        return None

def load_comments_data(uploaded_file):
    """Müşteri yorum dosyasını yükler"""
    try:
        df = pd.read_excel(uploaded_file, header=1, engine="openpyxl")
        
        df = df[df.iloc[:, 0] != "65000 yorum sınırını aştınız."]
        df = df[df.iloc[:, 0] != "birim"]
        df = df.dropna(subset=[df.columns[0]], how="all")
        
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
        
        df["station_code"] = df["station_info"].apply(extract_station_code)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["categories"] = df["comment"].apply(categorize_comment_enhanced)
        
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
# Yorum birleştirme
# ------------------------------------------------------------
def merge_comments_with_tlag(comments_df, tlag_df):
    """Yorum verilerini TLAG verisi ile birleştir"""
    try:
        st.write("🔍 Yorum verileri birleştiriliyor...")
        
        merged = pd.merge(
            comments_df,
            tlag_df[["ROC_NORMALIZED", "ROC_STR", "İstasyon", "NOR", "DISTRICT", "Site Segment"]],
            left_on="station_code",
            right_on="ROC_NORMALIZED", 
            how="left"
        )
        
        not_merged = merged[merged["İstasyon"].isna()]
        if not not_merged.empty:
            merged2 = pd.merge(
                not_merged[comments_df.columns],
                tlag_df[["ROC_STR", "İstasyon", "NOR", "DISTRICT", "Site Segment"]],
                left_on="station_code",
                right_on="ROC_STR",
                how="left"
            )
            
            merged_part1 = merged[merged["İstasyon"].notna()]
            merged_part2 = merged2[merged2["İstasyon"].notna()]
            
            if not merged_part2.empty:
                merged_part2["ROC_NORMALIZED"] = merged_part2["ROC_STR"]
                merged = pd.concat([merged_part1, merged_part2], ignore_index=True)
        
        merged["NOR_FINAL"] = merged["NOR"].fillna(merged.get("territory", ""))
        merged["DISTRICT_FINAL"] = merged["DISTRICT"].fillna(merged.get("district", ""))
        
        successful_matches = len(merged[merged["İstasyon"].notna()])
        st.success(f"✅ {successful_matches}/{len(comments_df)} yorum eşleştirildi!")
        
        return merged
    except Exception as e:
        st.error(f"Veri birleştirme hatası: {str(e)}")
        return comments_df

# ------------------------------------------------------------
# Analiz fonksiyonları
# ------------------------------------------------------------
def analyze_comments_by_scope(comments_df, scope_col="DISTRICT"):
    """Yorum analizi"""
    if comments_df is None or comments_df.empty:
        return {}
    
    if scope_col == "DISTRICT":
        group_col = "DISTRICT_FINAL"
    elif scope_col == "NOR":
        group_col = "NOR_FINAL"
    else:
        group_col = scope_col
        
    if group_col not in comments_df.columns:
        return {}
    
    results = {}
    valid_comments = comments_df[comments_df[group_col].notna()]
    
    for name, group in valid_comments.groupby(group_col):
        if pd.isna(name) or str(name).strip() == "":
            continue
            
        total_comments = len(group)
        if total_comments == 0:
            continue
            
        avg_score = group["score"].mean()
        score_dist = group["score"].value_counts().to_dict()
        
        category_counts = {}
        category_scores = {}
        
        for _, row in group.iterrows():
            if isinstance(row.get("categories"), list):
                for cat in row["categories"]:
                    if cat not in category_counts:
                        category_counts[cat] = 0
                        category_scores[cat] = []
                    category_counts[cat] += 1
                    category_scores[cat].append(row["score"])
        
        category_avg_scores = {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }
        
        problem_categories = {}
        for cat in category_counts:
            if cat in category_avg_scores and category_avg_scores[cat] < 4.0:
                problem_categories[cat] = {
                    "count": category_counts[cat],
                    "avg_score": category_avg_scores[cat]
                }
        
        results[name] = {
            "total_comments": total_comments,
            "avg_score": avg_score,
            "score_distribution": score_dist,
            "category_counts": category_counts,
            "category_avg_scores": category_avg_scores,
            "problem_categories": problem_categories
        }
    
    return results

# ------------------------------------------------------------
# UI Fonksiyonları
# ------------------------------------------------------------
def display_top_5_lists(df, comments_df=None, scope="Genel"):
    """Top 5 listeler"""
    st.markdown(f"### 🏆 {scope} - TOP 5 LİSTELER")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔴 En Çok Şikayet")
        if comments_df is not None and not comments_df.empty and "İstasyon" in comments_df.columns:
            valid_comments = comments_df[comments_df["İstasyon"].notna()]
            if not valid_comments.empty:
                complaint_counts = valid_comments[valid_comments["score"] <= 3].groupby("İstasyon").size()
                if not complaint_counts.empty:
                    for idx, (station, count) in enumerate(complaint_counts.nlargest(5).items(), 1):
                        st.write(f"{idx}. {station}: **{count}**")
                else:
                    st.info("Şikayet yok")
            else:
                st.info("Veri yok")
        else:
            st.info("Yorum yok")
    
    with col2:
        st.markdown("#### ⚠️ En Kötü Skor")
        if "SKOR" in df.columns:
            worst = df.nsmallest(5, "SKOR")[["İstasyon", "SKOR"]]
            for idx, (_, row) in enumerate(worst.iterrows(), 1):
                st.write(f"{idx}. {row['İstasyon']}: **{row['SKOR']*100:.1f}%**")
    
    with col3:
        st.markdown("#### ✅ En İyi Skor")
        if "SKOR" in df.columns:
            best = df.nlargest(5, "SKOR")[["İstasyon", "SKOR"]]
            for idx, (_, row) in enumerate(best.iterrows(), 1):
                st.write(f"{idx}. {row['İstasyon']}: **{row['SKOR']*100:.1f}%**")

def display_segment_pie_charts(df):
    """Segment dağılımı"""
    st.markdown("### 🎪 SEGMENT DAĞILIMI")
    
    tab1, tab2, tab3 = st.tabs(["📊 Genel", "🏢 District", "📍 NOR"])
    
    with tab1:
        segment_counts = df["Site Segment"].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Genel Segment Dağılımı", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
        selected_segment = st.selectbox("Detay için seçin:", segment_counts.index)
        if selected_segment:
            segment_data = df[df["Site Segment"] == selected_segment]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("İstasyon", len(segment_data))
            with col2:
                st.metric("Ort. Skor", f"{segment_data['SKOR'].mean()*100:.1f}%")
            with col3:
                st.metric("Min-Max", f"{segment_data['SKOR'].min()*100:.0f}%-{segment_data['SKOR'].max()*100:.0f}%")
    
    with tab2:
        if "DISTRICT" in df.columns:
            districts = sorted(df["DISTRICT"].dropna().unique())
            selected_district = st.selectbox("District:", districts)
            if selected_district:
                district_data = df[df["DISTRICT"] == selected_district]
                segment_counts = district_data["Site Segment"].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                            title=f"{selected_district} Segment Dağılımı", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if "NOR" in df.columns:
            nors = sorted(df["NOR"].dropna().unique())
            selected_nor = st.selectbox("NOR:", nors)
            if selected_nor:
                nor_data = df[df["NOR"] == selected_nor]
                segment_counts = nor_data["Site Segment"].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                            title=f"{selected_nor} Segment Dağılımı", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

def display_nor_interactive_chart(nor_data, comments_data=None):
    """NOR puan dağılımı"""
    st.markdown("### 📊 PUAN DAĞILIMI")
    
    if comments_data is not None and not comments_data.empty:
        score_counts = comments_data["score"].value_counts().sort_index()
        
        fig = go.Figure()
        colors = {5: '#28a745', 4: '#90EE90', 3: '#ffc107', 2: '#fd7e14', 1: '#dc3545'}
        
        for score in score_counts.index:
            fig.add_trace(go.Bar(
                x=[str(score)],
                y=[score_counts[score]],
                name=f'{score} Puan',
                marker_color=colors.get(score, '#6c757d'),
                text=[f"{score_counts[score]} anket"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Puan Dağılımı",
            xaxis_title="Puan",
            yaxis_title="Anket Sayısı",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        selected_score = st.selectbox("Yorumları görmek için puan seçin:", [5, 4, 3, 2, 1])
        
        score_comments = comments_data[comments_data["score"] == selected_score]
        if not score_comments.empty:
            st.markdown(f"#### {selected_score} Puan Yorumları ({len(score_comments)} adet)")
            
            for idx, row in score_comments.head(5).iterrows():
                categories_str = ", ".join(row["categories"]) if isinstance(row.get("categories"), list) else "GENEL"
                st.markdown(f"""
                **Kategoriler:** {categories_str}
                _{row['comment']}_
                ---
                """)

def display_comment_analysis_enhanced(analysis_data, title="Yorum Analizi"):
    """Yorum analizi görüntüleme"""
    if not analysis_data:
        st.info(f"📋 {title} için yorum verisi bulunamadı")
        return
    
    st.markdown(f"### 💬 {title}")
    
    total_areas = len(analysis_data)
    total_comments = sum(data["total_comments"] for data in analysis_data.values())
    avg_score_overall = np.mean([data["avg_score"] for data in analysis_data.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam Alan", total_areas)
    with col2:
        st.metric("Toplam Yorum", total_comments)
    with col3:
        st.metric("Ort. Puan", f"{avg_score_overall:.1f}/5")

def get_opportunity_stations(df):
    """Fırsat istasyonları"""
    if df is None or df.empty or "SKOR" not in df.columns or "Site Segment" not in df.columns:
        return pd.DataFrame()
    
    return df[(df["Site Segment"].isin(["My Precious", "Primitive"])) & (df["SKOR"] < 0.80)].copy()

def create_enhanced_metric_card(col, title, value, key, click_data=None):
    """Metrik kartı"""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{title}</p>
            <small>📊 Detay için tıklayın</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Detay", key=f"btn_{key}", use_container_width=True):
            st.session_state[f"show_{key}"] = not st.session_state.get(f"show_{key}", False)

def display_all_stations_detail(df):
    """Tüm istasyonlar detayı"""
    st.markdown("### 📊 TÜM İSTASYONLAR")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        district_filter = st.selectbox("District:", ["Tümü"] + sorted(df["DISTRICT"].dropna().unique().tolist()) if "DISTRICT" in df.columns else ["Tümü"])
    with col2:
        nor_filter = st.selectbox("NOR:", ["Tümü"] + sorted(df["NOR"].dropna().unique().tolist()) if "NOR" in df.columns else ["Tümü"])
    with col3:
        segment_filter = st.selectbox("Segment:", ["Tümü"] + sorted(df["Site Segment"].dropna().unique().tolist()) if "Site Segment" in df.columns else ["Tümü"])
    
    filtered_df = df.copy()
    if district_filter != "Tümü" and "DISTRICT" in df.columns:
        filtered_df = filtered_df[filtered_df["DISTRICT"] == district_filter]
    if nor_filter != "Tümü" and "NOR" in df.columns:
        filtered_df = filtered_df[filtered_df["NOR"] == nor_filter]
    if segment_filter != "Tümü" and "Site Segment" in df.columns:
        filtered_df = filtered_df[filtered_df["Site Segment"] == segment_filter]
    
    st.metric("Filtrelenmiş İstasyon Sayısı", len(filtered_df))
    
    display_cols = ["ROC_STR", "İstasyon", "SKOR", "SIRALAMA", "DISTRICT", "NOR", "Site Segment"]
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    display_df = filtered_df[available_cols].copy()
    if "SKOR" in display_df.columns:
        display_df["SKOR"] = (display_df["SKOR"] * 100).round(1).astype(str) + "%"
    
    st.dataframe(display_df, use_container_width=True, height=400)

def display_score_improvement_detail(df):
    """Skor iyileştirme detayı"""
    st.markdown("### 🎯 SKOR İYİLEŞTİRME")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 En Yüksek Skor")
        if "SKOR" in df.columns:
            top = df.nlargest(5, "SKOR")[["İstasyon", "SKOR"]]
            top["SKOR"] = (top["SKOR"] * 100).round(1).astype(str) + "%"
            st.dataframe(top)
    
    with col2:
        st.markdown("#### ⚠️ En Düşük Skor")
        if "SKOR" in df.columns:
            bottom = df.nsmallest(5, "SKOR")[["İstasyon", "SKOR"]]
            bottom["SKOR"] = (bottom["SKOR"] * 100).round(1).astype(str) + "%"
            st.dataframe(bottom)

def save_data_to_supabase(df, comments_df=None):
    """Supabase'e kaydet"""
    if not SUPABASE_ENABLED or supabase is None:
        return False
    
    try:
        if df is not None:
            df_save = df.copy()
            for col in df_save.columns:
                if df_save[col].dtype == 'object':
                    df_save[col] = df_save[col].astype(str)
            
            data = df_save.to_dict('records')
            response = supabase.table('tlag_data').upsert(data).execute()
            st.success("✅ TLAG verileri kaydedildi")
        
        if comments_df is not None:
            comments_save = comments_df.copy()
            if 'categories' in comments_save.columns:
                comments_save['categories'] = comments_save['categories'].apply(json.dumps)
            
            for col in comments_save.columns:
                if comments_save[col].dtype == 'object':
                    comments_save[col] = comments_save[col].astype(str)
            
            data = comments_save.to_dict('records')
            response = supabase.table('customer_comments').upsert(data).execute()
            st.success("✅ Yorum verileri kaydedildi")
        
        return True
    except Exception as e:
        st.error(f"Kayıt hatası: {str(e)}")
        return False

def load_data_from_supabase():
    """Supabase'den yükle"""
    if not SUPABASE_ENABLED or supabase is None:
        return None, None
    
    try:
        response = supabase.table('tlag_data').select("*").execute()
        tlag_df = pd.DataFrame(response.data) if response.data else None
        
        response = supabase.table('customer_comments').select("*").execute()
        comments_df = pd.DataFrame(response.data) if response.data else None
        
        if comments_df is not None and 'categories' in comments_df.columns:
            comments_df['categories'] = comments_df['categories'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return tlag_df, comments_df
    except Exception as e:
        st.error(f"Yükleme hatası: {str(e)}")
        return None, None

# ------------------------------------------------------------
# Sayfa yapılandırması
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
        font-size: 2.5rem; 
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

# ------------------------------------------------------------
# ANA UYGULAMA
# ------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">📊 TLAG PERFORMANS ANALİTİK SİSTEMİ</h1>', unsafe_allow_html=True)
    
    # SAYFA AÇILDIĞINDA OTOMATİK VERİ YÜKLE
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("Veriler yükleniyor..."):
            # Önce Supabase'den dene
            if SUPABASE_ENABLED and supabase:
                tlag_df, comments_df = load_data_from_supabase()
                if tlag_df is not None:
                    st.session_state.tlag_data = tlag_df
                    if comments_df is not None:
                        st.session_state.comments_data = comments_df
                        # Analiz et
                        district_analysis = analyze_comments_by_scope(comments_df, "DISTRICT")
                        nor_analysis = analyze_comments_by_scope(comments_df, "NOR")
                        st.session_state.analyzed_comments = {
                            "district": district_analysis,
                            "nor": nor_analysis
                        }
                    st.session_state.data_loaded = True
            
            # Supabase'de veri yoksa demo yükle
            if not st.session_state.data_loaded:
                tlag_df, comments_df, _ = load_demo_data_from_cloud()
                if tlag_df is not None:
                    st.session_state.tlag_data = tlag_df
                    if comments_df is not None:
                        merged_comments = merge_comments_with_tlag(comments_df, tlag_df)
                        st.session_state.comments_data = merged_comments
                        district_analysis = analyze_comments_by_scope(merged_comments, "DISTRICT")
                        nor_analysis = analyze_comments_by_scope(merged_comments, "NOR")
                        st.session_state.analyzed_comments = {
                            "district": district_analysis,
                            "nor": nor_analysis
                        }
                    st.session_state.data_loaded = True
                    # Demo veriyi Supabase'e kaydet
                    if SUPABASE_ENABLED and supabase:
                        save_data_to_supabase(tlag_df, st.session_state.comments_data)
    
    # SIDEBAR - VERİ YÖNETİMİ (GİZLENEBİLİR)
    with st.sidebar:
        with st.expander("🔐 Veri Yönetimi (Admin)", expanded=False):
            st.markdown("### 📁 VERİ GÜNCELLEME")
            
            # Manuel veri yükleme
            uploaded_tlag = st.file_uploader("TLAG Excel:", type=["xlsx", "xls"], key="tlag_upload")
            uploaded_comments = st.file_uploader("Yorumlar Excel:", type=["xlsx", "xls"], key="comments_upload")
            
            if uploaded_tlag:
                with st.spinner("TLAG verileri işleniyor..."):
                    df = load_tlag_data(uploaded_tlag)
                    if df is not None:
                        st.session_state.tlag_data = df
                        st.success(f"✅ {len(df)} istasyon")
                        save_data_to_supabase(df, None)
            
            if uploaded_comments and st.session_state.tlag_data is not None:
                with st.spinner("Yorumlar işleniyor..."):
                    comments_df = load_comments_data(uploaded_comments)
                    if comments_df is not None:
                        merged = merge_comments_with_tlag(comments_df, st.session_state.tlag_data)
                        st.session_state.comments_data = merged
                        district_analysis = analyze_comments_by_scope(merged, "DISTRICT")
                        nor_analysis = analyze_comments_by_scope(merged, "NOR")
                        st.session_state.analyzed_comments = {
                            "district": district_analysis,
                            "nor": nor_analysis
                        }
                        st.success(f"✅ {len(comments_df)} yorum")
                        save_data_to_supabase(None, merged)
            
            # Demo veri butonu
            if st.button("🔄 Demo Veri Yükle", use_container_width=True):
                with st.spinner("Demo veriler yükleniyor..."):
                    tlag_df, comments_df, message = load_demo_data_from_cloud()
                    if tlag_df is not None:
                        st.session_state.tlag_data = tlag_df
                        if comments_df is not None:
                            merged = merge_comments_with_tlag(comments_df, tlag_df)
                            st.session_state.comments_data = merged
                            district_analysis = analyze_comments_by_scope(merged, "DISTRICT")
                            nor_analysis = analyze_comments_by_scope(merged, "NOR")
                            st.session_state.analyzed_comments = {
                                "district": district_analysis,
                                "nor": nor_analysis
                            }
                        save_data_to_supabase(tlag_df, st.session_state.comments_data)
                        st.success("✅ Demo veriler yüklendi")
                        st.rerun()
        
        # Kullanıcı bilgisi
        st.markdown("---")
        if st.session_state.get("tlag_data") is not None:
            df = st.session_state.tlag_data
            st.metric("📊 Toplam İstasyon", len(df))
            if "SKOR" in df.columns:
                st.metric("📈 Ort. TLAG", f"{df['SKOR'].mean()*100:.1f}%")
            if st.session_state.get("comments_data") is not None:
                st.metric("💬 Toplam Yorum", len(st.session_state.comments_data))
        
        st.markdown("---")
        st.caption("TLAG Analytics v2.0")
        st.caption("© 2024")
    
    # Ana içerik
    if st.session_state.current_view == "main":
        display_main_dashboard()
    elif st.session_state.current_view == "district":
        display_district_analysis()
    elif st.session_state.current_view == "nor":
        display_nor_analysis()
    elif st.session_state.current_view == "segmentation":
        display_segmentation_analysis()

def display_main_dashboard():
    """Ana dashboard"""
    if st.session_state.tlag_data is None:
        st.markdown("## 🎯 TLAG PERFORMANS ANALİTİK'E HOŞGELDİNİZ")
        st.info("👈 Sol panelden Excel dosyalarınızı yükleyin veya demo verilerini deneyin")
        return
    
    df = st.session_state.tlag_data
    
    # AI Chat Box
    ai_chat_interface(df, st.session_state.get("comments_data"))
    
    # Ana metrikler
    st.markdown("## 📊 ANA METRİKLER")
    col1, col2, col3, col4 = st.columns(4)
    
    create_enhanced_metric_card(col1, "Toplam İstasyon", len(df), "total_stations", df)
    
    if "SKOR" in df.columns:
        avg_score = df["SKOR"].mean() * 100
        create_enhanced_metric_card(col2, "Ortalama Skor", f"{avg_score:.1f}%", "avg_score", df)
    
    if "Site Segment" in df.columns:
        segments = df["Site Segment"].value_counts()
        with col3:
            if len(segments) > 0:
                st.metric(segments.index[0], segments.iloc[0])
        with col4:
            if len(segments) > 1:
                st.metric(segments.index[1], segments.iloc[1])
    
    # Detay gösterimleri
    if st.session_state.get("show_total_stations", False):
        display_all_stations_detail(df)
    
    if st.session_state.get("show_avg_score", False):
        display_score_improvement_detail(df)
    
    # Segment pie chart
    if "Site Segment" in df.columns:
        display_segment_pie_charts(df)
    
    # District performans
    if "DISTRICT" in df.columns and "SKOR" in df.columns:
        st.markdown("## 📈 DISTRICT PERFORMANS")
        
        district_performance = df.groupby("DISTRICT").agg({
            "SKOR": "mean",
            "İstasyon": "count"
        }).reset_index()
        
        district_performance["SKOR_FORMAT"] = (district_performance["SKOR"] * 100).round(1).astype(str) + "%"
        district_performance = district_performance.sort_values("SKOR", ascending=False)
        
        fig = px.bar(district_performance, x="DISTRICT", y="İstasyon",
                    text="SKOR_FORMAT", title="District Performansları")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 listeler
    display_top_5_lists(df, st.session_state.get("comments_data"))
    
    # Fırsat istasyonları
    opportunity_stations = get_opportunity_stations(df)
    if not opportunity_stations.empty:
        st.markdown("## 🎯 FIRSAT İSTASYONLARI")
        st.markdown(f"**{len(opportunity_stations)} fırsat istasyonu** (My Precious/Primitive, <80% skor)")
        
        for idx, (_, station) in enumerate(opportunity_stations.head(5).iterrows(), 1):
            current_score = station["SKOR"] * 100
            st.markdown(f"""
            <div class="opportunity-card">
                <strong>{idx}. {station['İstasyon']}</strong><br>
                Mevcut: {current_score:.1f}% → Hedef: 80%<br>
                <small>{station.get('DISTRICT', '')} | {station['Site Segment']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Navigasyon
    st.markdown("## 🎯 ANALİZ SEÇENEKLERİ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="nav-section"><h3>🏢 DISTRICT</h3></div>', unsafe_allow_html=True)
        if st.button("District Analizi", key="nav_district", use_container_width=True):
            st.session_state.current_view = "district"
            st.rerun()
    
    with col2:
        st.markdown('<div class="nav-section"><h3>📍 NOR</h3></div>', unsafe_allow_html=True)
        if st.button("NOR Analizi", key="nav_nor", use_container_width=True):
            st.session_state.current_view = "nor"
            st.rerun()
    
    with col3:
        st.markdown('<div class="nav-section"><h3>🎪 SEGMENTASYON</h3></div>', unsafe_allow_html=True)
        if st.button("Segmentasyon", key="nav_segmentation", use_container_width=True):
            st.session_state.current_view = "segmentation"
            st.rerun()

def display_district_analysis():
    """District analizi"""
    if st.button("🏠 Ana Sayfa"):
        st.session_state.current_view = "main"
        st.rerun()
    
    st.markdown("## 🏢 DISTRICT ANALİZİ")
    
    df = st.session_state.tlag_data
    if df is None or "DISTRICT" not in df.columns:
        st.error("District verisi yok")
        return
    
    districts = sorted(df["DISTRICT"].dropna().unique())
    selected_district = st.selectbox("District:", districts)
    
    if selected_district:
        district_data = df[df["DISTRICT"] == selected_district].copy()
        
        # District yorumları filtrele
        district_comments = None
        if st.session_state.get("comments_data") is not None:
            district_comments = st.session_state.comments_data[
                st.session_state.comments_data.get("DISTRICT_FINAL", "") == selected_district
            ] if "DISTRICT_FINAL" in st.session_state.comments_data.columns else None
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("İstasyon", len(district_data))
        with col2:
            if "SKOR" in district_data.columns:
                st.metric("Ort. Skor", f"{district_data['SKOR'].mean()*100:.1f}%")
        with col3:
            if "Site Segment" in district_data.columns:
                st.metric("Segment", district_data["Site Segment"].nunique())
        with col4:
            opportunity_count = len(get_opportunity_stations(district_data))
            st.metric("Fırsat", opportunity_count)
        
        display_top_5_lists(district_data, district_comments, selected_district)
        
        # AI Öneriler - Geliştirilmiş
        st.markdown("### 🤖 AI ANALİZİ VE ÖNERİLER")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ai_query = st.text_input(
                "Analiz sorunuz:",
                placeholder=f"{selected_district} için özel soru yazın veya hazır konulardan seçin",
                key="district_ai_query"
            )
        
        with col2:
            ai_topics = [
                "TLAG skorunu hızlı artırma",
                "En düşük skorlu istasyonlar",
                "Müşteri şikayetleri analizi",
                "Personel performansı",
                "Temizlik standartları",
                "Market optimizasyonu"
            ]
            selected_topic = st.selectbox("Hazır konular:", [""] + ai_topics, key="district_topic")
        
        if st.button(f"🤖 Analiz Et", key="district_ai_btn"):
            # Eğer hazır konu seçildiyse onu kullan, yoksa custom query
            final_query = ai_query if ai_query else f"{selected_district} bölgesi için {selected_topic}"
            
            if final_query and final_query != f"{selected_district} bölgesi için ":
                with st.spinner("Veriler analiz ediliyor..."):
                    # Gerçek veri ile AI analizi
                    result = get_ai_response(
                        f"{selected_district} district {final_query}",
                        district_data,
                        district_comments
                    )
                    
                    # Sonucu göster
                    st.markdown("#### 📊 AI Analiz Sonucu:")
                    st.info(result)
                    
                    # Analiz edilen veri özeti
                    with st.expander("📈 Analiz Edilen Veri Detayları"):
                        st.write(f"- İstasyon sayısı: {len(district_data)}")
                        st.write(f"- Ortalama skor: {district_data['SKOR'].mean()*100:.2f}%")
                        if district_comments is not None:
                            st.write(f"- Yorum sayısı: {len(district_comments)}")
                            st.write(f"- Ortalama yorum puanı: {district_comments['score'].mean():.2f}/5")
            else:
                st.warning("Lütfen bir soru yazın veya konu seçin")

def display_nor_analysis():
    """NOR analizi"""
    if st.button("🏠 Ana Sayfa"):
        st.session_state.current_view = "main"
        st.rerun()
    
    st.markdown("## 📍 NOR ANALİZİ")
    
    df = st.session_state.tlag_data
    if df is None or "NOR" not in df.columns:
        st.error("NOR verisi yok")
        return
    
    nors = sorted(df["NOR"].dropna().unique())
    selected_nor = st.selectbox("NOR:", nors)
    
    if selected_nor:
        nor_data = df[df["NOR"] == selected_nor].copy()
        
        # NOR yorumları filtrele
        nor_comments = None
        if st.session_state.get("comments_data") is not None:
            nor_comments = st.session_state.comments_data[
                st.session_state.comments_data.get("NOR_FINAL", "") == selected_nor
            ] if "NOR_FINAL" in st.session_state.comments_data.columns else None
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("İstasyon", len(nor_data))
        with col2:
            if "SKOR" in nor_data.columns:
                st.metric("Ort. Skor", f"{nor_data['SKOR'].mean()*100:.1f}%")
        with col3:
            if nor_comments is not None:
                st.metric("Yorum", len(nor_comments))
            else:
                st.metric("Yorum", 0)
        with col4:
            opportunity_count = len(get_opportunity_stations(nor_data))
            st.metric("Fırsat", opportunity_count)
        
        # İnteraktif chart
        if nor_comments is not None:
            display_nor_interactive_chart(nor_data, nor_comments)
        
        # Top 5 listeler
        display_top_5_lists(nor_data, nor_comments, selected_nor)
        
        # İstasyon detayı
        st.markdown("### 📊 İSTASYON DETAYI")
        selected_station = st.selectbox("İstasyon:", nor_data["İstasyon"].tolist())
        
        if selected_station:
            station_data = nor_data[nor_data["İstasyon"] == selected_station].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skor", f"{station_data['SKOR']*100:.1f}%")
                if "SIRALAMA" in station_data and pd.notna(station_data["SIRALAMA"]):
                    st.markdown(f"### 🏆 Genel Sıralama: **{int(station_data['SIRALAMA'])}**")
            with col2:
                st.metric("Segment", station_data.get("Site Segment", "N/A"))
            with col3:
                if "TRANSACTION" in station_data and pd.notna(station_data["TRANSACTION"]):
                    st.metric("Transaction", f"{station_data['TRANSACTION']:,.0f}")
            
            # İstasyon yorumları
            if nor_comments is not None:
                station_code = station_data.get("ROC_NORMALIZED") or station_data.get("ROC_STR")
                station_comments = nor_comments[nor_comments.get("station_code", "") == str(station_code)]
                
                if not station_comments.empty:
                    st.markdown("#### 💬 İstasyon Yorumları")
                    st.write(f"Toplam {len(station_comments)} yorum, Ortalama: {station_comments['score'].mean():.1f}/5")
                    
                    # Kategori özeti
                    category_summary = {}
                    for _, row in station_comments.iterrows():
                        if isinstance(row.get("categories"), list):
                            for cat in row["categories"]:
                                if cat not in category_summary:
                                    category_summary[cat] = {"count": 0, "total": 0}
                                category_summary[cat]["count"] += 1
                                category_summary[cat]["total"] += row["score"]
                    
                    if category_summary:
                        st.write("**Problem Kategorileri:**")
                        for cat, data in sorted(category_summary.items(), 
                                               key=lambda x: x[1]["total"]/x[1]["count"]):
                            avg = data["total"] / data["count"]
                            if avg < 4.0:
                                st.write(f"- {cat}: {data['count']} yorum, Ort: {avg:.1f}/5 ⚠️")
        
        # AI Öneriler - Geliştirilmiş
        st.markdown("### 🤖 AI ANALİZİ VE ÖNERİLER")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ai_query = st.text_input(
                "Analiz sorunuz:",
                placeholder=f"{selected_nor} için özel soru yazın veya hazır konulardan seçin",
                key="nor_ai_query"
            )
        
        with col2:
            ai_topics = [
                "TLAG skorunu en hızlı artırma yolları",
                "En problemli istasyonlar ve çözümler",
                "Müşteri şikayetleri analizi",
                "Kategori bazlı iyileştirmeler",
                "Fırsat istasyonları stratejisi"
            ]
            selected_topic = st.selectbox("Hazır konular:", [""] + ai_topics, key="nor_topic")
        
        if st.button(f"🤖 Analiz Et", key="nor_ai_btn"):
            # Eğer hazır konu seçildiyse onu kullan, yoksa custom query
            final_query = ai_query if ai_query else f"{selected_nor} için {selected_topic}"
            
            if final_query and final_query != f"{selected_nor} için ":
                with st.spinner("Gerçek veriler analiz ediliyor..."):
                    # Gerçek veri ile AI analizi
                    result = get_ai_response(
                        f"{selected_nor} NOR {final_query}",
                        nor_data,
                        nor_comments
                    )
                    
                    # Sonucu göster
                    st.markdown("#### 📊 AI Analiz Sonucu:")
                    st.success(result)
                    
                    # Analiz edilen veri özeti
                    with st.expander("📈 Analiz Edilen Veri Detayları"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**İstasyon Verileri:**")
                            st.write(f"- Toplam: {len(nor_data)} istasyon")
                            st.write(f"- Ort. skor: {nor_data['SKOR'].mean()*100:.2f}%")
                            st.write(f"- Min skor: {nor_data['SKOR'].min()*100:.2f}%")
                            st.write(f"- Max skor: {nor_data['SKOR'].max()*100:.2f}%")
                        with col2:
                            if nor_comments is not None:
                                st.write(f"**Yorum Verileri:**")
                                st.write(f"- Toplam: {len(nor_comments)} yorum")
                                st.write(f"- Ort. puan: {nor_comments['score'].mean():.2f}/5")
                                st.write(f"- 5 puan: {len(nor_comments[nor_comments['score']==5])}")
                                st.write(f"- 1-2 puan: {len(nor_comments[nor_comments['score']<=2])}")
            else:
                st.warning("Lütfen bir soru yazın veya konu seçin")

def display_segmentation_analysis():
    """Segmentasyon analizi"""
    if st.button("🏠 Ana Sayfa"):
        st.session_state.current_view = "main"
        st.rerun()
    
    st.markdown("## 🎪 SEGMENTASYON ANALİZİ")
    
    df = st.session_state.tlag_data
    if df is None or "Site Segment" not in df.columns:
        st.error("Segment verisi yok")
        return
    
    segment_summary = df.groupby("Site Segment").agg({
        "İstasyon": "count",
        "SKOR": ["mean", "min", "max"] if "SKOR" in df.columns else ["count"]
    }).round(3)
    
    st.dataframe(segment_summary, use_container_width=True)
    
    selected_segment = st.selectbox("Segment:", df["Site Segment"].unique())
    
    if selected_segment:
        segment_data = df[df["Site Segment"] == selected_segment]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 En İyi 5")
            if "SKOR" in segment_data.columns:
                top5 = segment_data.nlargest(5, "SKOR")[["İstasyon", "SKOR"]]
                top5["SKOR"] = (top5["SKOR"] * 100).round(1).astype(str) + "%"
                st.dataframe(top5)
        
        with col2:
            st.markdown("#### ⚠️ En Kötü 5")
            if "SKOR" in segment_data.columns:
                bottom5 = segment_data.nsmallest(5, "SKOR")[["İstasyon", "SKOR"]]
                bottom5["SKOR"] = (bottom5["SKOR"] * 100).round(1).astype(str) + "%"
                st.dataframe(bottom5)

if __name__ == "__main__":
    main()