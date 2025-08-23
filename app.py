import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="TLAG Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    }
    .performance-excellent {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .performance-good {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .performance-needs-improvement {
        background: linear-gradient(135deg, #ff512f 0%, #dd2476 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .station-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .improvement-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .priority-high { border-left: 5px solid #e74c3c; }
    .priority-medium { border-left: 5px solid #f39c12; }
    .priority-low { border-left: 5px solid #27ae60; }
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .metric-card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tlag_data' not in st.session_state:
    st.session_state.tlag_data = None

def format_percentage(value):
    """Convert decimal to percentage format"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value * 100:.1f}%"

def format_percentage_change(value):
    """Format percentage change with + or - sign"""
    if pd.isna(value) or value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"

def get_performance_category(score):
    """Get performance category and color"""
    if pd.isna(score):
        return "Bilinmeyen", "#95a5a6"
    elif score >= 0.80:
        return "Mükemmel", "#27ae60"
    elif score >= 0.70:
        return "İyi", "#f39c12"
    elif score >= 0.60:
        return "Orta", "#e67e22"
    else:
        return "Gelişim Gerekli", "#e74c3c"

def load_tlag_data(uploaded_file):
    """Load TLAG data with proper formatting"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, sheet_name="TLAG DOKUNMA (2)", engine='openpyxl')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove empty rows
        df = df.dropna(subset=['ROC', 'İstasyon'], how='any')
        
        # Convert numeric columns
        numeric_columns = ['ROC', 'SKOR', 'GEÇEN SENE SKOR', 'Fark', 'TRANSACTION', 'NOR HEDEF', 'DISTRICT HEDEF', 'Geçerli']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['İstasyon', 'NOR', 'DISTRICT', 'Site Segment']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', np.nan)
        
        return df
        
    except Exception as e:
        st.error(f"Dosya okuma hatası: {str(e)}")
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

def generate_improvement_recommendations(station_data):
    """Generate AI-powered recommendations"""
    recommendations = []
    
    current_score = station_data.get('SKOR', 0)
    segment = station_data.get('Site Segment', 'Unknown')
    fark = station_data.get('Fark', 0)
    
    # Critical performance
    if current_score < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Kritik Durum',
            'action': f'Bu istasyon kritik durumda ({format_percentage(current_score)}). Acil operasyon review gerekli.',
            'expected_impact': '+15-25%',
            'timeframe': '1-2 hafta'
        })
    
    # Declining performance
    if fark < -5:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Performans Düşüşü',
            'action': f'Performans {fark:.1f}% düşmüş. Trend analizi ve düzeltici eylem gerekli.',
            'expected_impact': '+10-15%',
            'timeframe': '2-3 hafta'
        })
    
    # Segment-based recommendations
    segment_actions = {
        'Saboteur': {
            'priority': 'HIGH',
            'category': 'Segment Recovery',
            'action': 'Saboteur segmentinden çıkış için kapsamlı operasyon planı gerekli.',
            'expected_impact': '+20-30%',
            'timeframe': '3-4 hafta'
        },
        'Primitive': {
            'priority': 'MEDIUM',
            'category': 'Temel İyileştirme',
            'action': 'Temel operasyon standartlarını yükseltin. Personel eğitimi gerekli.',
            'expected_impact': '+10-20%',
            'timeframe': '2-3 hafta'
        },
        'Wasted Talent': {
            'priority': 'MEDIUM',
            'category': 'Potansiyel Açığa Çıkarma',
            'action': 'Bu istasyonun potansiyeli var. Engelleri tespit edip çözümleyin.',
            'expected_impact': '+8-15%',
            'timeframe': '2-4 hafta'
        }
    }
    
    if segment in segment_actions:
        recommendations.append(segment_actions[segment])
    
    # Performance category recommendations
    if 0.5 <= current_score < 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Performans Artışı',
            'action': 'Operasyon verimliliği artırılarak performans yükseltilebilir.',
            'expected_impact': '+5-12%',
            'timeframe': '2-3 hafta'
        })
    elif current_score >= 0.8:
        recommendations.append({
            'priority': 'LOW',
            'category': 'Sürdürülebilirlik',
            'action': 'Mükemmel performans! Best practices diğer istasyonlara yaygınlaştırılabilir.',
            'expected_impact': 'Sürdürülebilirlik',
            'timeframe': 'Devam eden'
        })
    
    return recommendations[:4]  # Top 4 recommendations

def main():
    st.markdown('<h1 class="main-header">📊 TLAG PERFORMANS ANALİTİK</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar file upload
    st.sidebar.markdown("## 📁 VERİ YÖNETİMİ")
    
    uploaded_file = st.sidebar.file_uploader(
        "TLAG Excel dosyası:",
        type=['xlsx', 'xls'],
        help="satis_veri_clean.xlsx dosyanızı yükleyin"
    )
    
    # Demo data option
    if st.sidebar.button("📊 Demo Verisi Yükle"):
        # Generate demo data with percentage formatting
        np.random.seed(42)
        demo_stations = ['KASTAMONU', 'SAMSUN ATAKUM', 'ANKARA YENİMAHALLE', 'İSTANBUL KARTAL', 'İZMİR BORNOVA']
        demo_data = []
        for i, station in enumerate(demo_stations):
            demo_data.append({
                'ROC': 4000 + i,
                'İstasyon': station,
                'DISTRICT': np.random.choice(['ANKARA KUZEY BÖLGE', 'MARMARA BÖLGE', 'İZMİR BÖLGE']),
                'SKOR': np.random.uniform(0.5, 0.9),
                'GEÇEN SENE SKOR': np.random.uniform(0.4, 0.8),
                'Fark': np.random.uniform(-10, 15),
                'Site Segment': np.random.choice(['My Precious', 'Wasted Talent', 'Saboteur']),
                'TRANSACTION': np.random.randint(5000, 50000),
                'NOR': np.random.choice(['KARADENİZ BATI', 'SAMSUN', 'İZMİR MERKEZ'])
            })
        st.session_state.tlag_data = pd.DataFrame(demo_data)
        st.sidebar.success("✅ Demo verisi yüklendi!")
    
    if uploaded_file is not None:
        # Load real data
        with st.spinner("📊 Veriler işleniyor..."):
            df = load_tlag_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.session_state.tlag_data = df
            st.sidebar.success(f"✅ {len(df)} istasyon verisi yüklendi!")
            
            # Sidebar data summary with percentages
            with st.sidebar.expander("📋 Veri Özeti"):
                st.write(f"**Toplam İstasyon:** {len(df)}")
                if 'SKOR' in df.columns:
                    avg_score = df['SKOR'].mean()
                    st.write(f"**Ortalama Skor:** {format_percentage(avg_score)}")
                    st.write(f"**En Yüksek:** {format_percentage(df['SKOR'].max())}")
                    st.write(f"**En Düşük:** {format_percentage(df['SKOR'].min())}")
                
                # Segment distribution
                if 'Site Segment' in df.columns:
                    segments = df['Site Segment'].value_counts()
                    st.write("**Segment Dağılımı:**")
                    for segment, count in segments.items():
                        if pd.notna(segment):
                            st.write(f"- {segment}: {count}")
    
    # Main dashboard
    if st.session_state.tlag_data is not None:
        df = st.session_state.tlag_data
        
        # Analysis mode selection
        st.markdown("## 🎯 ANALİZ TÜRÜ")
        analysis_mode = st.selectbox(
            "Hangi analizi yapmak istiyorsunuz?",
            ["📊 Genel Dashboard", "🔍 İstasyon Detay Analizi", "📈 Performans Karşılaştırması", "🏆 Top/Bottom Performanslar", "🤖 AI Öneriler Sistemi"]
        )
        
        if analysis_mode == "📊 Genel Dashboard":
            # Main metrics with percentage formatting
            st.markdown("## 📊 ANA METRİKLER")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_stations = len(df)
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{total_stations}</h2>
                    <p>Toplam İstasyon</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'SKOR' in df.columns:
                    avg_score = df['SKOR'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>{format_percentage(avg_score)}</h2>
                        <p>Ortalama Skor</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'SKOR' in df.columns:
                    excellent_count = len(df[df['SKOR'] >= 0.80])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>{excellent_count}</h2>
                        <p>Mükemmel (≥80%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'Fark' in df.columns:
                    improving_count = len(df[df['Fark'] > 0])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>{improving_count}</h2>
                        <p>Gelişen İstasyon</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Performance distribution chart
            if 'SKOR' in df.columns:
                st.markdown("## 📈 PERFORMANS DAĞILIMI")
                
                # Create percentage data for visualization
                df_viz = df.copy()
                df_viz['Skor_Yüzde'] = df_viz['SKOR'] * 100
                
                fig_dist = px.histogram(
                    df_viz, 
                    x='Skor_Yüzde', 
                    nbins=20,
                    title="Performans Skor Dağılımı (%)",
                    labels={'Skor_Yüzde': 'Performans Skoru (%)', 'count': 'İstasyon Sayısı'}
                )
                fig_dist.add_vline(x=70, line_dash="dash", line_color="orange", 
                                  annotation_text="Hedef: 70%")
                fig_dist.update_layout(height=400)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # District performance comparison
            if 'DISTRICT' in df.columns and 'SKOR' in df.columns:
                st.markdown("## 🗺️ BÖLGESEL PERFORMANS")
                
                district_stats = df.groupby('DISTRICT').agg({
                    'SKOR': ['mean', 'count', 'min', 'max']
                }).round(4)
                
                district_stats.columns = ['Ortalama', 'İstasyon_Sayısı', 'Min', 'Max']
                district_stats = district_stats.reset_index()
                
                # Create percentage chart
                district_display = district_stats.copy()
                district_display['Ortalama_Yüzde'] = district_display['Ortalama'] * 100
                
                fig_district = px.bar(
                    district_display, 
                    x='DISTRICT', 
                    y='Ortalama_Yüzde',
                    title="Bölgelere Göre Ortalama Performans (%)",
                    labels={'Ortalama_Yüzde': 'Ortalama Skor (%)', 'DISTRICT': 'Bölge'},
                    color='Ortalama_Yüzde',
                    color_continuous_scale='RdYlGn'
                )
                fig_district.update_xaxis(tickangle=45)
                fig_district.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Hedef: 70%")
                fig_district.update_layout(height=500)
                st.plotly_chart(fig_district, use_container_width=True)
                
                # District summary table with percentages
                st.markdown("### 📊 Bölgesel Özet")
                
                summary_data = []
                for _, row in district_stats.iterrows():
                    summary_data.append({
                        'Bölge': row['DISTRICT'],
                        'İstasyon Sayısı': int(row['İstasyon_Sayısı']),
                        'Ortalama Skor': format_percentage(row['Ortalama']),
                        'En Düşük': format_percentage(row['Min']),
                        'En Yüksek': format_percentage(row['Max'])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            # Segment analysis
            if 'Site Segment' in df.columns and not df['Site Segment'].isna().all():
                st.markdown("## 🎯 SEGMENT ANALİZİ")
                
                segment_stats = df.groupby('Site Segment').agg({
                    'SKOR': ['mean', 'count']
                }).round(4)
                
                segment_stats.columns = ['Ortalama_Skor', 'İstasyon_Sayısı']
                segment_stats = segment_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Segment distribution pie chart
                    fig_pie = px.pie(
                        segment_stats,
                        values='İstasyon_Sayısı',
                        names='Site Segment',
                        title="Segment Dağılımı",
                        color_discrete_map={
                            'My Precious': '#27ae60',
                            'Wasted Talent': '#f39c12',
                            'Saboteur': '#e74c3c',
                            'Primitive': '#95a5a6'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Segment performance bar chart
                    segment_display = segment_stats.copy()
                    segment_display['Ortalama_Yüzde'] = segment_display['Ortalama_Skor'] * 100
                    
                    fig_segment = px.bar(
                        segment_display,
                        x='Site Segment',
                        y='Ortalama_Yüzde',
                        title="Segment Bazında Ortalama Performans (%)",
                        color='Site Segment',
                        color_discrete_map={
                            'My Precious': '#27ae60',
                            'Wasted Talent': '#f39c12',
                            'Saboteur': '#e74c3c',
                            'Primitive': '#95a5a6'
                        }
                    )
                    st.plotly_chart(fig_segment, use_container_width=True)
        
        elif analysis_mode == "🔍 İstasyon Detay Analizi":
            st.markdown("## 🔍 İSTASYON DETAY ANALİZİ")
            
            # Station search and selection
            station_search = st.text_input("🔍 İstasyon ara:", placeholder="İstasyon adı yazın...")
            
            if station_search:
                filtered_stations = df[df['İstasyon'].str.contains(station_search, case=False, na=False)]['İstasyon'].tolist()
            else:
                filtered_stations = sorted(df['İstasyon'].unique())
            
            if filtered_stations:
                selected_station = st.selectbox("İstasyon seçin:", filtered_stations)
                
                if selected_station:
                    station_data = df[df['İstasyon'] == selected_station].iloc[0].to_dict()
                    
                    # Station header with percentage formatting
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"### 🏢 {selected_station}")
                        st.markdown(f"**ROC:** {station_data.get('ROC', 'N/A')}")
                        st.markdown(f"**Bölge:** {station_data.get('DISTRICT', 'N/A')}")
                        st.markdown(f"**NOR:** {station_data.get('NOR', 'N/A')}")
                        st.markdown(f"**Segment:** {station_data.get('Site Segment', 'N/A')}")
                    
                    with col2:
                        current_score = station_data.get('SKOR', 0)
                        previous_score = station_data.get('GEÇEN SENE SKOR', 0)
                        change_percent = (current_score - previous_score) * 100
                        
                        st.metric(
                            "Mevcut Skor",
                            format_percentage(current_score),
                            delta=format_percentage_change(change_percent)
                        )
                        
                        st.metric(
                            "Geçen Yıl",
                            format_percentage(previous_score)
                        )
                    
                    with col3:
                        category, color = get_performance_category(current_score)
                        st.markdown(f"""
                        <div style="
                            background-color: {color};
                            color: white;
                            padding: 1rem;
                            border-radius: 10px;
                            text-align: center;
                            font-weight: bold;
                        ">
                            {category}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Performance comparison chart
                    st.markdown("### 📊 PERFORMANS KARŞILAŞTIRMASI")
                    
                    # Calculate comparison metrics
                    district_avg = df[df['DISTRICT'] == station_data.get('DISTRICT', '')]['SKOR'].mean()
                    overall_avg = df['SKOR'].mean()
                    
                    comparison_data = {
                        'Metrik': ['Bu İstasyon', 'Geçen Yıl', 'Bölge Ortalaması', 'Genel Ortalama'],
                        'Skor': [current_score * 100, previous_score * 100, district_avg * 100, overall_avg * 100]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig_comparison = px.bar(
                        comparison_df,
                        x='Metrik',
                        y='Skor',
                        title=f"{selected_station} - Performans Karşılaştırması (%)",
                        labels={'Skor': 'Performans Skoru (%)'},
                        color='Skor',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_comparison.add_hline(y=70, line_dash="dash", line_color="red", 
                                           annotation_text="Hedef: 70%")
                    fig_comparison.update_layout(height=400)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # AI Recommendations for this station
                    st.markdown("### 🤖 AI ÖNERİLERİ")
                    
                    recommendations = generate_improvement_recommendations(station_data)
                    
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
                    
                    # Detailed metrics
                    st.markdown("### 📋 DETAYLI METRİKLER")
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown(f"""
                        **📊 Performans Bilgileri:**
                        - **Mevcut Skor:** {format_percentage(current_score)}
                        - **Geçen Yıl Skor:** {format_percentage(previous_score)}
                        - **Değişim:** {format_percentage_change(change_percent)}
                        - **Kategori:** {category}
                        """)
                    
                    with detail_col2:
                        transaction_volume = station_data.get('TRANSACTION', 0)
                        gecerli = station_data.get('Geçerli', 0)
                        
                        st.markdown(f"""
                        **💰 İşlem Bilgileri:**
                        - **İşlem Hacmi:** {transaction_volume:,} 
                        - **Geçerli İşlem:** {gecerli:,}
                        - **NOR Hedef:** {format_percentage(station_data.get('NOR HEDEF', 0))}
                        - **Bölge Hedef:** {format_percentage(station_data.get('DISTRICT HEDEF', 0))}
                        """)
            
            else:
                st.warning("Arama kriterinize uygun istasyon bulunamadı.")
        
        elif analysis_mode == "🏆 Top/Bottom Performanslar":
            st.markdown("## 🏆 PERFORMANS SIRALAMALARI")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥇 EN İYİ PERFORMANSLAR")
                
                if 'SKOR' in df.columns:
                    top_performers = df.nlargest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT', 'Site Segment']].copy()
                    
                    # Format scores as percentages
                    top_performers['Skor (%)'] = top_performers['SKOR'].apply(format_percentage)
                    top_performers = top_performers.drop('SKOR', axis=1)
                    
                    # Add rank
                    top_performers.insert(0, 'Sıra', range(1, len(top_performers) + 1))
                    
                    st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.markdown("### ⚠️ GELİŞİM GEREKTİREN İSTASYONLAR")
                
                if 'SKOR' in df.columns:
                    bottom_performers = df.nsmallest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT', 'Site Segment']].copy()
                    
                    # Format scores as percentages
                    bottom_performers['Skor (%)'] = bottom_performers['SKOR'].apply(format_percentage)
                    bottom_performers = bottom_performers.drop('SKOR', axis=1)
                    
                    st.dataframe(bottom_performers, use_container_width=True)
            
            # Performance improvement opportunities
            if 'Fark' in df.columns:
                st.markdown("## 📈 GELİŞİM FIRSATLARI")
                
                improvement_col1, improvement_col2 = st.columns(2)
                
                with improvement_col1:
                    st.markdown("### 📈 EN ÇOK GELİŞENLER")
                    
                    top_improvers = df.nlargest(10, 'Fark')[['İstasyon', 'SKOR', 'GEÇEN SENE SKOR', 'Fark', 'DISTRICT']].copy()
                    
                    # Format with percentages
                    display_improvers = top_improvers.copy()
                    display_improvers['Mevcut (%)'] = display_improvers['SKOR'].apply(format_percentage)
                    display_improvers['Geçen Yıl (%)'] = display_improvers['GEÇEN SENE SKOR'].apply(format_percentage)
                    display_improvers['Değişim (%)'] = display_improvers['Fark'].apply(lambda x: f"{x:+.1f}%")
                    
                    final_improvers = display_improvers[['İstasyon', 'Mevcut (%)', 'Geçen Yıl (%)', 'Değişim (%)', 'DISTRICT']]
                    st.dataframe(final_improvers, use_container_width=True)
                
                with improvement_col2:
                    st.markdown("### 📉 DİKKAT GEREKTİRENLER")
                    
                    declining = df.nsmallest(10, 'Fark')[['İstasyon', 'SKOR', 'GEÇEN SENE SKOR', 'Fark', 'DISTRICT']].copy()
                    
                    # Format with percentages
                    display_declining = declining.copy()
                    display_declining['Mevcut (%)'] = display_declining['SKOR'].apply(format_percentage)
                    display_declining['Geçen Yıl (%)'] = display_declining['GEÇEN SENE SKOR'].apply(format_percentage)
                    display_declining['Değişim (%)'] = display_declining['Fark'].apply(lambda x: f"{x:+.1f}%")
                    
                    final_declining = display_declining[['İstasyon', 'Mevcut (%)', 'Geçen Yıl (%)', 'Değişim (%)', 'DISTRICT']]
                    st.dataframe(final_declining, use_container_width=True)
        
        elif analysis_mode == "🤖 AI Öneriler Sistemi":
            st.markdown("## 🤖 AI-POWERED İYİLEŞTİRME ÖNERİLERİ")
            
            # Critical stations analysis
            if 'SKOR' in df.columns:
                critical_stations = df[df['SKOR'] < 0.6].copy()
                
                if not critical_stations.empty:
                    st.markdown(f"### ⚠️ KRİTİK DURUMDA OLAN İSTASYONLAR ({len(critical_stations)} adet)")
                    
                    critical_stations = critical_stations.sort_values('SKOR')
                    
                    for idx, (_, station_row) in enumerate(critical_stations.head(5).iterrows()):
                        station_data = station_row.to_dict()
                        station_name = station_data['İstasyon']
                        current_score = station_data['SKOR']
                        
                        with st.expander(f"🚨 {station_name} (Skor: {format_percentage(current_score)})"):
                            recommendations = generate_improvement_recommendations(station_data)
                            
                            for rec in recommendations:
                                st.markdown(f"""
                                **{rec['category']} - {rec['priority']} ÖNCELİK**
                                - **Aksiyon:** {rec['action']}
                                - **Beklenen Etki:** {rec['expected_impact']}
                                - **Süre:** {rec['timeframe']}
                                """)
                
                # Performance improvement opportunities
                st.markdown("### 📈 GELİŞİM FIRSATLARI")
                
                # Best practices from top performers
                top_performers = df[df['SKOR'] >= 0.85]
                if not top_performers.empty:
                    st.success(f"✅ {len(top_performers)} istasyon mükemmel performans sergiliyor (≥85%)")
                    
                    if 'DISTRICT' in df.columns:
                        top_by_district = top_performers.groupby('DISTRICT')['İstasyon'].count().sort_values(ascending=False)
                        
                        fig_top = px.bar(
                            x=top_by_district.values,
                            y=top_by_district.index,
                            orientation='h',
                            title="Bölgelere Göre Yüksek Performanslı İstasyon Sayısı (≥85%)",
                            labels={'x': 'İstasyon Sayısı', 'y': 'Bölge'}
                        )
                        st.plotly_chart(fig_top, use_container_width=True)
                
                # Segment improvement analysis
                if 'Site Segment' in df.columns:
                    st.markdown("### 🎯 SEGMENT BAZINDA ÖNERİLER")
                    
                    segment_avg = df.groupby('Site Segment')['SKOR'].agg(['mean', 'count']).round(4)
                    segment_avg.columns = ['Ortalama_Skor', 'İstasyon_Sayısı']
                    segment_avg = segment_avg.reset_index()
                    
                    for _, segment_row in segment_avg.iterrows():
                        segment = segment_row['Site Segment']
                        avg_score = segment_row['Ortalama_Skor']
                        count = segment_row['İstasyon_Sayısı']
                        
                        if pd.notna(segment):
                            color = {'My Precious': '#27ae60', 'Wasted Talent': '#f39c12', 
                                   'Saboteur': '#e74c3c', 'Primitive': '#95a5a6'}.get(segment, '#95a5a6')
                            
                            st.markdown(f"""
                            <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                                <h4>{segment} Segmenti</h4>
                                <p><strong>İstasyon Sayısı:</strong> {int(count)}</p>
                                <p><strong>Ortalama Skor:</strong> {format_percentage(avg_score)}</p>
                            </div>
                            """, unsafe_allow_html=True)

    else:
        # Welcome screen - no data loaded
        st.markdown("## 🎯 TLAG PERFORMANS ANALİTİK'E HOŞGELDİNİZ")
        
        st.info("👈 Sol panelden Excel dosyanızı yükleyin veya demo verilerini deneyin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 ANALİZ ÖZELLİKLERİ
            - ✅ **Yüzdelik skorlar** (64.2% formatında)
            - ✅ **İstasyon detay analizi**
            - ✅ **Bölgesel karşılaştırma**
            - ✅ **Performans kategorileri**
            - ✅ **AI-powered öneriler**
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 GELİŞMİŞ ÖZELLİKLER  
            - ✅ **Top/Bottom performanslar**
            - ✅ **Gelişim fırsatları**
            - ✅ **İstasyon arama**
            - ✅ **Yıllık karşılaştırma**
            - ✅ **Mobil uyumlu**
            """)

if __name__ == "__main__":
    main()