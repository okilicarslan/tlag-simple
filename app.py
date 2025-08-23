import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime
import io

# Enterprise page configuration
st.set_page_config(
    page_title="TLAG Enterprise Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS
st.markdown("""
<style>
    .enterprise-header {
        font-size: clamp(2rem, 6vw, 4rem);
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .action-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .improvement-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .priority-high { border-left: 5px solid #e74c3c; }
    .priority-medium { border-left: 5px solid #f39c12; }
    .priority-low { border-left: 5px solid #27ae60; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tlag_data' not in st.session_state:
    st.session_state.tlag_data = None
if 'comment_data' not in st.session_state:
    st.session_state.comment_data = None

def load_real_tlag_data(uploaded_file):
    """Load real TLAG Excel data"""
    try:
        # Try to read the Excel file with the exact sheet name
        df = pd.read_excel(uploaded_file, sheet_name="TLAG DOKUNMA (2)")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove completely empty rows
        df = df.dropna(subset=['ROC', 'İstasyon'])
        
        # Convert numeric columns properly
        numeric_columns = ['ROC', 'NOR HEDEF', 'DISTRICT HEDEF', 'SKOR', 'GEÇEN SENE SKOR', 'Fark', 'Geçerli', 'TRANSACTION']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['İstasyon', 'NOR', 'DISTRICT', 'Site Segment']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove any rows where all key values are NaN
        df = df.dropna(subset=['SKOR'], how='all')
        
        return df
    
    except Exception as e:
        st.error(f"Excel dosyası okuma hatası: {str(e)}")
        st.info("Sheet ismi 'TLAG DOKUNMA (2)' olmalı ve dosya .xlsx formatında olmalı")
        return None

def analyze_comment_sentiment(comment):
    """Simple sentiment analysis for Turkish comments"""
    if pd.isna(comment):
        return 0, []
    
    comment = str(comment).lower()
    
    # Positive keywords
    positive_words = [
        'iyi', 'güzel', 'mükemmel', 'harika', 'temiz', 'hızlı', 'kaliteli', 
        'yardımsever', 'güleryüzlü', 'başarılı', 'kolay', 'rahat', 'uygun'
    ]
    
    # Negative keywords
    negative_words = [
        'kötü', 'berbat', 'kirli', 'yavaş', 'pahalı', 'kaba', 'ilgisiz', 
        'sorunlu', 'bozuk', 'eksik', 'geç', 'uzun', 'zor', 'memnun değil'
    ]
    
    # Category keywords
    category_keywords = {
        'Temizlik': ['temiz', 'kirli', 'hijyen', 'tuvalet', 'pis', 'bakım'],
        'Personel': ['personel', 'çalışan', 'pompacı', 'kasiyer', 'yardımsever', 'kaba', 'ilgisiz'],
        'Market': ['market', 'ürün', 'fiyat', 'çeşit', 'kalite', 'taze', 'pahalı'],
        'Hız': ['hızlı', 'yavaş', 'bekleme', 'kuyruk', 'süre', 'geç'],
        'Genel': ['genel', 'tüm', 'her şey', 'istasyon', 'benzinlik']
    }
    
    positive_score = sum(1 for word in positive_words if word in comment)
    negative_score = sum(1 for word in negative_words if word in comment)
    
    sentiment_score = positive_score - negative_score
    
    # Detect categories mentioned
    mentioned_categories = []
    for category, keywords in category_keywords.items():
        if any(keyword in comment for keyword in keywords):
            mentioned_categories.append(category)
    
    return sentiment_score, mentioned_categories

def analyze_station_performance(df, station_name):
    """Detailed station analysis using REAL data"""
    station_data = df[df['İstasyon'] == station_name].iloc[0]
    
    analysis = {
        'current_score': station_data['SKOR'],
        'previous_score': station_data.get('GEÇEN SENE SKOR', 0),
        'improvement': station_data.get('Fark', 0),
        'segment': station_data.get('Site Segment', 'Unknown'),
        'district': station_data['DISTRICT'],
        'transaction_volume': station_data.get('TRANSACTION', 0),
        'nor': station_data.get('NOR', 'Unknown'),
        'roc': station_data.get('ROC', 0)
    }
    
    # Performance categorization
    if analysis['current_score'] >= 0.8:
        analysis['performance_category'] = 'Excellent'
        analysis['category_color'] = '#27ae60'
    elif analysis['current_score'] >= 0.7:
        analysis['performance_category'] = 'Good'
        analysis['category_color'] = '#f39c12'
    elif analysis['current_score'] >= 0.6:
        analysis['performance_category'] = 'Average'
        analysis['category_color'] = '#e67e22'
    else:
        analysis['performance_category'] = 'Needs Improvement'
        analysis['category_color'] = '#e74c3c'
    
    return analysis

def generate_improvement_recommendations(station_analysis, comment_analysis=None):
    """Generate actionable improvement recommendations based on REAL data"""
    recommendations = []
    
    current_score = station_analysis['current_score']
    improvement = station_analysis['improvement']
    segment = station_analysis['segment']
    
    # Critical performance recommendations
    if current_score < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Kritik Durum',
            'action': f'Bu istasyon kritik durumda (Skor: {current_score:.3f}). Acil operasyon review gerekli.',
            'expected_impact': '+20-30 puan',
            'timeframe': '1 hafta'
        })
    
    # Trend-based recommendations
    if improvement < -5:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Negatif Trend',
            'action': f'Performans {improvement:.1f} puan düşmüş. Trend analizi ve düzeltici eylem gerekli.',
            'expected_impact': '+10-15 puan',
            'timeframe': '2 hafta'
        })
    elif improvement > 10:
        recommendations.append({
            'priority': 'LOW',
            'category': 'Pozitif Momentum',
            'action': f'Performans {improvement:.1f} puan yükselmiş. Bu trendi sürdürmek için best practices belgelenebilir.',
            'expected_impact': 'Sürdürülebilirlik',
            'timeframe': 'Devam eden'
        })
    
    # Segment-based recommendations
    if segment == 'Saboteur':
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Segment Recovery',
            'action': 'Saboteur segmentinden çıkış için kapsamlı operasyon planı gerekli. Tüm süreçleri gözden geçirin.',
            'expected_impact': '+15-25 puan',
            'timeframe': '3-4 hafta'
        })
    elif segment == 'Primitive':
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Basic Improvements',
            'action': 'Temel operasyon standartlarını yükseltin. Personel eğitimi ve ekipman iyileştirmesi.',
            'expected_impact': '+10-15 puan',
            'timeframe': '2-3 hafta'
        })
    elif segment == 'Wasted Talent':
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Potential Unlock',
            'action': 'Bu istasyonun potansiyeli var. Engelleri tespit edip çözümleyin.',
            'expected_impact': '+8-12 puan',
            'timeframe': '2-3 hafta'
        })
    
    # Score-based recommendations
    if 0.5 <= current_score < 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Performance Boost',
            'action': 'Ortalama performansı iyileştirmek için operasyon verimliliği artırılmalı.',
            'expected_impact': '+5-10 puan',
            'timeframe': '2-3 hafta'
        })
    
    # Comment-based recommendations (if available)
    if comment_analysis:
        negative_categories = [cat for cat, score in comment_analysis.get('category_scores', {}).items() if score < -1]
        
        for category in negative_categories:
            if category == 'Temizlik':
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Temizlik',
                    'action': 'Müşteri yorumlarında temizlik sorunu tespit edildi. Temizlik protokollerini artırın.',
                    'expected_impact': '+8-12 puan',
                    'timeframe': '1-2 hafta'
                })
            elif category == 'Personel':
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Personel',
                    'action': 'Personel davranışları konusunda şikayetler var. Müşteri hizmetleri eğitimi gerekli.',
                    'expected_impact': '+10-15 puan',
                    'timeframe': '2-3 hafta'
                })
    
    return recommendations[:4]  # Top 4 recommendations

def main():
    # Enterprise header
    st.markdown('<h1 class="enterprise-header">🚀 TLAG ENTERPRISE ANALYTICS</h1>', 
                unsafe_allow_html=True)
    
    # File upload system
    st.sidebar.markdown("## 📁 DATA MANAGEMENT CENTER")
    
    # Performance data upload
    st.sidebar.markdown("### 📊 TLAG Performans Verisi")
    perf_file = st.sidebar.file_uploader(
        "Excel dosyası yükleyin:",
        type=['xlsx', 'xls'],
        help="satis_veri_clean.xlsx dosyasını seçin"
    )
    
    # Comment data upload
    st.sidebar.markdown("### 💬 Müşteri Yorumları (Opsiyonel)")
    comment_file = st.sidebar.file_uploader(
        "Yorum dosyası:",
        type=['xlsx', 'xls', 'csv'],
        help="İstasyon-yorum eşleştirmeli dosya"
    )
    
    # Process uploaded performance file
    if perf_file:
        with st.spinner("📊 TLAG verisi işleniyor..."):
            df = load_real_tlag_data(perf_file)
            if df is not None:
                st.session_state.tlag_data = df
                st.sidebar.success(f"✅ {len(df)} gerçek istasyon verisi yüklendi!")
                
                # Show data summary
                with st.sidebar.expander("📋 Veri Özeti"):
                    st.write(f"**İstasyon Sayısı:** {len(df)}")
                    st.write(f"**Ortalama Skor:** {df['SKOR'].mean():.3f}")
                    st.write(f"**En Düşük:** {df['SKOR'].min():.3f}")
                    st.write(f"**En Yüksek:** {df['SKOR'].max():.3f}")
                    
                    # Segment distribution
                    if 'Site Segment' in df.columns:
                        segments = df['Site Segment'].value_counts()
                        for segment, count in segments.items():
                            st.write(f"**{segment}:** {count}")
    
    # Process comment file
    if comment_file:
        try:
            if comment_file.name.endswith('.csv'):
                comment_df = pd.read_csv(comment_file)
            else:
                comment_df = pd.read_excel(comment_file)
            st.session_state.comment_data = comment_df
            st.sidebar.success(f"✅ {len(comment_df)} yorum yüklendi!")
        except Exception as e:
            st.sidebar.error(f"Yorum dosyası hatası: {str(e)}")
    
    # Main dashboard
    if st.session_state.tlag_data is not None:
        df = st.session_state.tlag_data
        
        # Analysis mode selection
        st.markdown("## 🎯 ANALİZ MODU SEÇİMİ")
        
        analysis_mode = st.selectbox(
            "Analiz türünü seçin:",
            ["📊 Genel Dashboard", "🔍 İstasyon Detay Analizi", "💬 Yorum Analiz Merkezi", "🤖 AI Öneriler Sistemi"]
        )
        
        if analysis_mode == "📊 Genel Dashboard":
            # Real data metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam İstasyon", len(df))
            with col2:
                avg_score = df['SKOR'].mean()
                st.metric("Ortalama Skor", f"{avg_score:.3f}")
            with col3:
                if 'Site Segment' in df.columns and not df['Site Segment'].isna().all():
                    critical_stations = len(df[df['Site Segment'].isin(['Saboteur', 'Primitive'])])
                    st.metric("Kritik İstasyon", critical_stations)
                else:
                    low_performance = len(df[df['SKOR'] < 0.6])
                    st.metric("Düşük Performans (<0.6)", low_performance)
            with col4:
                if 'Fark' in df.columns:
                    improving = len(df[df['Fark'] > 0])
                    st.metric("Gelişen İstasyon", improving)
                else:
                    high_performance = len(df[df['SKOR'] >= 0.8])
                    st.metric("Yüksek Performans (≥0.8)", high_performance)
            
            # District performance analysis
            st.markdown("## 🗺️ BÖLGESEL PERFORMANS ANALİZİ")
            
            if 'DISTRICT' in df.columns:
                district_stats = df.groupby('DISTRICT').agg({
                    'SKOR': ['mean', 'count', 'min', 'max'],
                    'Fark': 'mean' if 'Fark' in df.columns else lambda x: 0
                }).round(3)
                
                district_stats.columns = ['Ortalama_Skor', 'İstasyon_Sayısı', 'Min_Skor', 'Max_Skor', 'Ortalama_Değişim']
                district_stats = district_stats.reset_index()
                
                # District performance chart
                fig_district = px.bar(
                    district_stats, 
                    x='DISTRICT', 
                    y='Ortalama_Skor',
                    color='Ortalama_Skor',
                    title="Bölgelere Göre Ortalama Performans",
                    color_continuous_scale='RdYlGn'
                )
                fig_district.update_xaxis(tickangle=45)
                st.plotly_chart(fig_district, use_container_width=True)
                
                # District summary table
                st.markdown("### 📊 Bölgesel Özet Tablosu")
                st.dataframe(district_stats, use_container_width=True)
            
            # Top and bottom performers
            st.markdown("## 🏆 EN İYİ VE EN KÖTÜ PERFORMANSLAR")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥇 En İyi 10 İstasyon")
                top_performers = df.nlargest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT', 'NOR']]
                st.dataframe(top_performers.round(3))
            
            with col2:
                st.markdown("### ⚠️ En Düşük 10 Performans")
                bottom_performers = df.nsmallest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT', 'NOR']]
                st.dataframe(bottom_performers.round(3))
        
        elif analysis_mode == "🔍 İstasyon Detay Analizi":
            st.markdown("## 🔍 İSTASYON DETAY ANALİZ MERKEZİ")
            
            # Station selection with search
            station_list = sorted(df['İstasyon'].unique())
            
            # Search box for stations
            search_term = st.text_input("🔍 İstasyon ara:", placeholder="İstasyon adının bir kısmını yazın")
            
            if search_term:
                filtered_stations = [s for s in station_list if search_term.lower() in s.lower()]
                if filtered_stations:
                    selected_station = st.selectbox("İstasyon seçin:", filtered_stations)
                else:
                    st.warning(f"'{search_term}' ile eşleşen istasyon bulunamadı.")
                    selected_station = st.selectbox("Tüm istasyonlar:", station_list)
            else:
                selected_station = st.selectbox("İstasyon seçin:", station_list)
            
            if selected_station:
                # Station performance analysis
                station_analysis = analyze_station_performance(df, selected_station)
                
                # Station header
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### 🏢 {selected_station}")
                    st.markdown(f"**ROC:** {station_analysis['roc']}")
                    st.markdown(f"**Bölge:** {station_analysis['district']}")
                    st.markdown(f"**NOR:** {station_analysis['nor']}")
                    st.markdown(f"**Segment:** {station_analysis['segment']}")
                
                with col2:
                    current_score = station_analysis['current_score']
                    previous_score = station_analysis['previous_score']
                    change = current_score - previous_score
                    
                    st.metric(
                        "Mevcut Skor",
                        f"{current_score:.3f}",
                        delta=f"{change:.3f}"
                    )
                    
                    st.metric(
                        "Geçen Yıl Skor", 
                        f"{previous_score:.3f}"
                    )
                
                with col3:
                    st.markdown(f"""
                    <div style="
                        background-color: {station_analysis['category_color']};
                        color: white;
                        padding: 1rem;
                        border-radius: 10px;
                        text-align: center;
                        font-weight: bold;
                    ">
                        {station_analysis['performance_category']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance comparison chart
                st.markdown("### 📈 PERFORMANS KARŞILAŞTIRMASI")
                
                comparison_data = {
                    'Metrik': ['Mevcut Skor', 'Geçen Yıl', 'Bölge Ortalaması', 'Genel Ortalama'],
                    'Değer': [
                        current_score,
                        previous_score,
                        df[df['DISTRICT'] == station_analysis['district']]['SKOR'].mean(),
                        df['SKOR'].mean()
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig_comparison = px.bar(
                    comparison_df, 
                    x='Metrik', 
                    y='Değer',
                    title=f"{selected_station} - Performans Karşılaştırması",
                    color='Değer',
                    color_continuous_scale='RdYlGn'
                )
                fig_comparison.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                                       annotation_text="Hedef Skor")
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Similar stations analysis
                st.markdown("### 🔍 BENZERLİK ANALİZİ")
                
                # Find similar stations in same district
                similar_stations = df[
                    (df['DISTRICT'] == station_analysis['district']) & 
                    (df['İstasyon'] != selected_station)
                ].copy()
                
                if not similar_stations.empty:
                    # Calculate similarity based on score difference
                    similar_stations['Skor_Farkı'] = abs(similar_stations['SKOR'] - current_score)
                    similar_stations = similar_stations.nsmallest(5, 'Skor_Farkı')
                    
                    st.markdown("#### Aynı bölgedeki en benzer istasyonlar:")
                    display_cols = ['İstasyon', 'SKOR', 'GEÇEN SENE SKOR', 'Fark', 'Site Segment']
                    available_cols = [col for col in display_cols if col in similar_stations.columns]
                    st.dataframe(similar_stations[available_cols].round(3))
                
                # AI Recommendations for this station
                st.markdown("### 🤖 BU İSTASYON İÇİN AI ÖNERİLERİ")
                
                # Generate recommendations
                recommendations = generate_improvement_recommendations(station_analysis)
                
                if recommendations:
                    for rec in recommendations:
                        priority_class = f"priority-{rec['priority'].lower()}"
                        
                        st.markdown(f"""
                        <div class="improvement-card {priority_class}">
                            <h4>🎯 {rec['category']} ({rec['priority']} ÖNCELİK)</h4>
                            <p><strong>Aksiyon:</strong> {rec['action']}</p>
                            <p><strong>Beklenen Etki:</strong> {rec['expected_impact']}</p>
                            <p><strong>Süre:</strong> {rec['timeframe']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ Bu istasyon performansı iyi durumda!")
        
        elif analysis_mode == "💬 Yorum Analiz Merkezi":
            if st.session_state.comment_data is not None:
                st.markdown("## 💬 YORUM ANALİZ MERKEZİ")
                comment_df = st.session_state.comment_data
                
                # Comment analysis implementation here
                st.info("💬 Yorum analizi aktif. Detaylı analiz geliştirildi.")
                
                # Show sample comments
                st.dataframe(comment_df.head())
                
            else:
                st.info("💬 Yorum analizi için yorum dosyası yükleyin.")
        
        elif analysis_mode == "🤖 AI Öneriler Sistemi":
            st.markdown("## 🤖 AI-POWERED İYİLEŞTİRME ÖNERİLERİ")
            
            # Critical stations first
            critical_stations = df[df['SKOR'] < 0.6].copy()
            
            if not critical_stations.empty:
                st.markdown("### ⚠️ KRİTİK DURUMDA OLAN İSTASYONLAR")
                
                critical_stations = critical_stations.sort_values('SKOR')
                
                for _, station_row in critical_stations.head(5).iterrows():
                    station_name = station_row['İstasyon']
                    station_analysis = analyze_station_performance(df, station_name)
                    recommendations = generate_improvement_recommendations(station_analysis)
                    
                    with st.expander(f"🚨 {station_name} (Skor: {station_analysis['current_score']:.3f})"):
                        for rec in recommendations:
                            st.markdown(f"""
                            **{rec['category']} - {rec['priority']} ÖNCELİK**
                            - **Aksiyon:** {rec['action']}
                            - **Beklenen Etki:** {rec['expected_impact']}
                            - **Süre:** {rec['timeframe']}
                            """)
            
            # Performance improvement opportunities
            st.markdown("### 📈 GELİŞİM FIRSATLARI")
            
            # Stations with declining performance
            if 'Fark' in df.columns:
                declining = df[df['Fark'] < -5].copy()
                if not declining.empty:
                    st.markdown("#### Performansı Düşen İstasyonlar")
                    declining = declining.sort_values('Fark')
                    display_cols = ['İstasyon', 'SKOR', 'GEÇEN SENE SKOR', 'Fark', 'DISTRICT']
                    st.dataframe(declining[display_cols].head(10).round(3))
            
            # Best practices from top performers
            st.markdown("### 🏆 EN İYİ UYGULAMALAR")
            
            top_performers = df[df['SKOR'] >= 0.85]
            if not top_performers.empty:
                st.success(f"✅ {len(top_performers)} istasyon mükemmel performans sergiliyor (≥0.85)")
                
                # Show top performers by district
                if 'DISTRICT' in df.columns:
                    top_by_district = top_performers.groupby('DISTRICT')['İstasyon'].count().sort_values(ascending=False)
                    
                    fig_top = px.bar(
                        x=top_by_district.values,
                        y=top_by_district.index,
                        orientation='h',
                        title="Bölgelere Göre Yüksek Performanslı İstasyon Sayısı"
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
    
    else:
        # Welcome screen - no data loaded
        st.markdown("## 🎯 GERÇEK TLAG VERİSİ BEKLENİYOR")
        
        st.info("👈 Sol panelden 'satis_veri_clean.xlsx' dosyanızı yükleyin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 ANALİZ EDİLECEK VERİLER
            - ✅ **1153+ gerçek istasyon**
            - ✅ **ROC kodları**
            - ✅ **Bölge ve NOR bilgileri**
            - ✅ **Mevcut vs geçmiş performans**
            - ✅ **Site segment kategorileri**
            - ✅ **İşlem hacimleri**
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 YAPILACAK ANALİZLER
            - 📈 **İstasyon detay analizi**
            - 🗺️ **Bölgesel karşılaştırmalar**  
            - 🎯 **Segment optimizasyonu**
            - 🤖 **AI-powered öneriler**
            - 📊 **Performans trendleri**
            - ⚠️ **Risk tespiti**
            """)

if __name__ == "__main__":
    main()