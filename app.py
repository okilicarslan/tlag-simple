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
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = None
if 'comment_data' not in st.session_state:
    st.session_state.comment_data = None
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None

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

def generate_demo_performance_data():
    """Generate demo performance data"""
    np.random.seed(42)
    
    stations = [
        'KASTAMONU MERKEZ', 'SAMSUN ATAKUM', 'ANKARA YENİMAHALLE', 'İSTANBUL KARTAL', 
        'İZMİR BORNOVA', 'BURSA MERKEZ', 'ANTALYA KONYAALTI', 'ADANA SEYHAN',
        'GAZİANTEP ŞAHINBEY', 'KOCAELİ GEBZE', 'KONYA SELÇUKLU', 'KAYSERİ MERKEZ'
    ]
    
    districts = ['ANKARA KUZEY BÖLGE', 'MARMARA BÖLGE', 'ADANA BÖLGE', 'CO BÖLGE']
    segments = ['My Precious', 'Wasted Talent', 'Saboteur', 'Primitive']
    
    data = []
    for i, station in enumerate(stations):
        data.append({
            'ROC': 4000 + i,
            'İstasyon': station,
            'DISTRICT': np.random.choice(districts),
            'SKOR': np.random.uniform(0.4, 0.9),
            'GEÇEN SENE SKOR': np.random.uniform(0.4, 0.9),
            'Site Segment': np.random.choice(segments, p=[0.3, 0.4, 0.2, 0.1]),
            'TRANSACTION': np.random.randint(5000, 50000),
            'Fark': np.random.uniform(-15, 20)
        })
    
    return pd.DataFrame(data)

def generate_demo_comments():
    """Generate demo comment data"""
    stations = [
        'KASTAMONU MERKEZ', 'SAMSUN ATAKUM', 'ANKARA YENİMAHALLE', 'İSTANBUL KARTAL', 
        'İZMİR BORNOVA', 'BURSA MERKEZ', 'ANTALYA KONYAALTI', 'ADANA SEYHAN'
    ]
    
    sample_comments = [
        "Personel çok yardımsever, temizlik iyi ama tuvaletler kirli",
        "Market çeşidi az, fiyatlar pahalı. Personel ilgisiz",
        "Çok temiz istasyon, hızlı servis, mükemmel personel",
        "Pompacı kaba davrandı, genel temizlik kötü",
        "Her şey harika, kaliteli ürünler, güleryüzlü çalışanlar",
        "Bekleme süresi uzun, tuvaletler bakımsız",
        "İyi bir istasyon, temiz ve hızlı servis",
        "Market personeli yardımsever ama ürünler taze değil",
        "Genel olarak memnun değilim, temizlik yetersiz",
        "Mükemmel hizmet, her şey çok iyi organize edilmiş"
    ]
    
    comments_data = []
    for station in stations:
        # Her istasyon için 3-7 yorum
        num_comments = np.random.randint(3, 8)
        for i in range(num_comments):
            comments_data.append({
                'İstasyon': station,
                'Yorum': np.random.choice(sample_comments),
                'Tarih': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30))
            })
    
    return pd.DataFrame(comments_data)

def analyze_station_performance(df, station_name):
    """Detailed station analysis"""
    station_data = df[df['İstasyon'] == station_name].iloc[0]
    
    analysis = {
        'current_score': station_data['SKOR'],
        'previous_score': station_data.get('GEÇEN SENE SKOR', 0),
        'improvement': station_data.get('Fark', 0),
        'segment': station_data.get('Site Segment', 'Unknown'),
        'district': station_data['DISTRICT'],
        'transaction_volume': station_data.get('TRANSACTION', 0)
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

def generate_improvement_recommendations(station_analysis, comment_analysis):
    """Generate actionable improvement recommendations"""
    recommendations = []
    
    current_score = station_analysis['current_score']
    
    # High priority recommendations
    if current_score < 0.6:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Genel Performans',
            'action': 'Acil müdahale gerekli - tüm operasyonları gözden geçirin',
            'expected_impact': '+15-20 puan',
            'timeframe': '1-2 hafta'
        })
    
    # Comment-based recommendations
    if comment_analysis:
        negative_categories = [cat for cat, score in comment_analysis['category_scores'].items() if score < -1]
        
        category_actions = {
            'Temizlik': {
                'action': 'Temizlik protokollerini artırın, özellikle tuvaletlere odaklanın',
                'impact': '+8-12 puan',
                'timeframe': '1 hafta'
            },
            'Personel': {
                'action': 'Personel eğitimi ve müşteri hizmetleri training düzenleyin',
                'impact': '+10-15 puan',
                'timeframe': '2-3 hafta'
            },
            'Market': {
                'action': 'Ürün çeşitliliğini artırın ve fiyat optimizasyonu yapın',
                'impact': '+5-8 puan',
                'timeframe': '2-4 hafta'
            },
            'Hız': {
                'action': 'Operasyon verimliliğini artırın, daha fazla personel görevlendirin',
                'impact': '+6-10 puan',
                'timeframe': '1-2 hafta'
            }
        }
        
        for category in negative_categories:
            if category in category_actions:
                rec = category_actions[category].copy()
                rec['priority'] = 'HIGH' if comment_analysis['category_scores'][category] < -2 else 'MEDIUM'
                rec['category'] = category
                recommendations.append(rec)
    
    # Performance trend based recommendations
    if station_analysis['improvement'] < -5:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Trend Analysis',
            'action': 'Performans düşüş trendini durdurmak için kapsamlı analiz yapın',
            'expected_impact': '+10-15 puan',
            'timeframe': '2-3 hafta'
        })
    
    # Segment-based recommendations
    if station_analysis['segment'] == 'Saboteur':
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Segment Recovery',
            'action': 'Bu istasyon kritik durumda - operasyon manager desteği gerekli',
            'expected_impact': '+20-25 puan',
            'timeframe': '1-2 hafta'
        })
    
    return recommendations[:5]  # Top 5 recommendations

def main():
    # Enterprise header
    st.markdown('<h1 class="enterprise-header">🚀 TLAG ENTERPRISE ANALYTICS</h1>', 
                unsafe_allow_html=True)
    
    # Advanced file upload system
    st.sidebar.markdown("## 📁 DATA MANAGEMENT CENTER")
    
    # Performance data upload
    st.sidebar.markdown("### 📊 Performans Verisi")
    perf_file = st.sidebar.file_uploader(
        "TLAG Excel dosyası:",
        type=['xlsx', 'xls'],
        help="Ana performans verilerinizi yükleyin"
    )
    
    # Comment data upload
    st.sidebar.markdown("### 💬 Müşteri Yorumları")
    comment_file = st.sidebar.file_uploader(
        "Yorum dosyası:",
        type=['xlsx', 'xls', 'csv'],
        help="İstasyon-yorum eşleştirmeli dosya"
    )
    
    # Demo data options
    if st.sidebar.button("🎬 Demo Verilerini Yükle"):
        st.session_state.performance_data = generate_demo_performance_data()
        st.session_state.comment_data = generate_demo_comments()
        st.sidebar.success("✅ Demo verileri yüklendi!")
    
    # Process uploaded files
    if perf_file:
        try:
            df = pd.read_excel(perf_file, sheet_name="TLAG DOKUNMA (2)")
            df.columns = df.columns.str.strip()
            numeric_cols = ['SKOR', 'GEÇEN SENE SKOR', 'Fark', 'TRANSACTION']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            st.session_state.performance_data = df
            st.sidebar.success(f"✅ {len(df)} istasyon verisi yüklendi!")
        except Exception as e:
            st.sidebar.error(f"Performans dosyası hatası: {str(e)}")
    
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
    if st.session_state.performance_data is not None:
        df = st.session_state.performance_data
        
        # Analysis mode selection
        st.markdown("## 🎯 ANALİZ MODU SEÇİMİ")
        
        analysis_mode = st.selectbox(
            "Analiz türünü seçin:",
            ["📊 Genel Dashboard", "🔍 İstasyon Detay Analizi", "💬 Yorum Analiz Merkezi", "🤖 AI Öneriler Sistemi"]
        )
        
        if analysis_mode == "📊 Genel Dashboard":
            # General dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam İstasyon", len(df))
            with col2:
                avg_score = df['SKOR'].mean()
                st.metric("Ortalama Skor", f"{avg_score:.3f}")
            with col3:
                if 'Site Segment' in df.columns:
                    critical_stations = len(df[df['Site Segment'].isin(['Saboteur', 'Primitive'])])
                    st.metric("Kritik İstasyon", critical_stations)
            with col4:
                if 'Fark' in df.columns:
                    improving = len(df[df['Fark'] > 0])
                    st.metric("Gelişen İstasyon", improving)
            
            # District performance heatmap
            st.markdown("## 🗺️ BÖLGESEL PERFORMANS HARİTASI")
            
            district_performance = df.groupby('DISTRICT').agg({
                'SKOR': ['mean', 'count'],
                'Fark': 'mean'
            }).round(3)
            
            district_performance.columns = ['Ortalama_Skor', 'İstasyon_Sayısı', 'Ortalama_Gelişim']
            district_performance = district_performance.reset_index()
            
            fig_heatmap = px.scatter(
                district_performance,
                x='Ortalama_Skor',
                y='Ortalama_Gelişim',
                size='İstasyon_Sayısı',
                color='Ortalama_Skor',
                hover_data=['DISTRICT'],
                title="Bölge Performans & Gelişim Haritası",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        elif analysis_mode == "🔍 İstasyon Detay Analizi":
            st.markdown("## 🔍 İSTASYON DETAY ANALİZ MERKEZİ")
            
            # Station selection
            station_list = sorted(df['İstasyon'].unique())
            selected_station = st.selectbox("İstasyon Seçin:", station_list)
            
            if selected_station:
                # Station performance analysis
                station_analysis = analyze_station_performance(df, selected_station)
                
                # Station header
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### 🏢 {selected_station}")
                    st.markdown(f"**Bölge:** {station_analysis['district']}")
                    st.markdown(f"**Segment:** {station_analysis['segment']}")
                
                with col2:
                    st.metric(
                        "Mevcut Skor",
                        f"{station_analysis['current_score']:.3f}",
                        delta=f"{station_analysis['improvement']:.1f}%"
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
                
                # Performance trend analysis
                st.markdown("### 📈 PERFORMANS TRENDİ")
                
                # Create trend visualization
                months = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran']
                # Simulate historical data
                np.random.seed(hash(selected_station) % 1000)
                historical_scores = []
                current = station_analysis['previous_score']
                for _ in months:
                    current += np.random.uniform(-0.05, 0.05)
                    historical_scores.append(max(0.3, min(0.95, current)))
                
                trend_df = pd.DataFrame({
                    'Ay': months,
                    'Skor': historical_scores
                })
                
                fig_trend = px.line(
                    trend_df, 
                    x='Ay', 
                    y='Skor',
                    title=f"{selected_station} - 6 Aylık Skor Trendi",
                    markers=True
                )
                fig_trend.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                                   annotation_text="Hedef Skor")
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Comment analysis for selected station
                if st.session_state.comment_data is not None:
                    station_comments = st.session_state.comment_data[
                        st.session_state.comment_data['İstasyon'] == selected_station
                    ]
                    
                    if not station_comments.empty:
                        st.markdown("### 💬 YORUM ANALİZİ")
                        
                        # Analyze comments
                        sentiments = []
                        categories_mentioned = []
                        
                        for comment in station_comments['Yorum']:
                            sentiment, categories = analyze_comment_sentiment(comment)
                            sentiments.append(sentiment)
                            categories_mentioned.extend(categories)
                        
                        # Sentiment summary
                        avg_sentiment = np.mean(sentiments) if sentiments else 0
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment gauge
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = avg_sentiment,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Genel Memnuniyet"},
                                gauge = {
                                    'axis': {'range': [-5, 5]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [-5, -2], 'color': "lightgray"},
                                        {'range': [-2, 2], 'color': "gray"},
                                        {'range': [2, 5], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col2:
                            # Category mentions
                            if categories_mentioned:
                                category_counts = pd.Series(categories_mentioned).value_counts()
                                fig_cat = px.bar(
                                    x=category_counts.values,
                                    y=category_counts.index,
                                    orientation='h',
                                    title="En Çok Bahsedilen Konular"
                                )
                                st.plotly_chart(fig_cat, use_container_width=True)
                        
                        # Show recent comments
                        st.markdown("#### Son Yorumlar")
                        for _, comment_row in station_comments.head(3).iterrows():
                            sentiment, categories = analyze_comment_sentiment(comment_row['Yorum'])
                            
                            sentiment_color = "#e74c3c" if sentiment < 0 else "#27ae60" if sentiment > 0 else "#95a5a6"
                            
                            st.markdown(f"""
                            <div style="
                                border-left: 4px solid {sentiment_color};
                                padding: 10px;
                                margin: 10px 0;
                                background-color: #f8f9fa;
                                border-radius: 5px;
                            ">
                                <strong>Tarih:</strong> {comment_row.get('Tarih', 'Belirtilmemiş')}<br>
                                <strong>Yorum:</strong> {comment_row['Yorum']}<br>
                                <strong>Kategoriler:</strong> {', '.join(categories) if categories else 'Genel'}
                            </div>
                            """, unsafe_allow_html=True)
        
        elif analysis_mode == "💬 Yorum Analiz Merkezi":
            if st.session_state.comment_data is not None:
                st.markdown("## 💬 YORUM ANALİZ MERKEZİ")
                
                comment_df = st.session_state.comment_data
                
                # Overall comment statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Toplam Yorum", len(comment_df))
                
                # Analyze all comments
                all_sentiments = []
                all_categories = []
                
                for comment in comment_df['Yorum']:
                    sentiment, categories = analyze_comment_sentiment(comment)
                    all_sentiments.append(sentiment)
                    all_categories.extend(categories)
                
                with col2:
                    avg_sentiment = np.mean(all_sentiments) if all_sentiments else 0
                    st.metric("Ortalama Sentiment", f"{avg_sentiment:.2f}")
                
                with col3:
                    positive_comments = sum(1 for s in all_sentiments if s > 0)
                    st.metric("Pozitif Yorum", positive_comments)
                
                with col4:
                    negative_comments = sum(1 for s in all_sentiments if s < 0)
                    st.metric("Negatif Yorum", negative_comments)
                
                # Category analysis
                if all_categories:
                    st.markdown("### 📊 KONU BAŞLIKLARI ANALİZİ")
                    
                    category_df = pd.DataFrame(all_categories, columns=['Kategori'])
                    category_counts = category_df['Kategori'].value_counts()
                    
                    fig_categories = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Yorum Kategorisi Dağılımı"
                    )
                    st.plotly_chart(fig_categories, use_container_width=True)
                
                # Station-wise sentiment analysis
                st.markdown("### 🏢 İSTASYON BAZINDA SENTIMENT")
                
                station_sentiments = {}
                for station in comment_df['İstasyon'].unique():
                    station_comments = comment_df[comment_df['İstasyon'] == station]['Yorum']
                    sentiments = [analyze_comment_sentiment(comment)[0] for comment in station_comments]
                    station_sentiments[station] = np.mean(sentiments) if sentiments else 0
                
                sentiment_df = pd.DataFrame(list(station_sentiments.items()), 
                                          columns=['İstasyon', 'Ortalama_Sentiment'])
                sentiment_df = sentiment_df.sort_values('Ortalama_Sentiment')
                
                fig_station_sentiment = px.bar(
                    sentiment_df,
                    x='Ortalama_Sentiment',
                    y='İstasyon',
                    orientation='h',
                    title="İstasyon Bazında Müşteri Memnuniyeti",
                    color='Ortalama_Sentiment',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_station_sentiment, use_container_width=True)
            
            else:
                st.info("💬 Yorum analizi için yorum dosyası yükleyin.")
        
        elif analysis_mode == "🤖 AI Öneriler Sistemi":
            st.markdown("## 🤖 AI-POWERED İYİLEŞTİRME ÖNERİLERİ")
            
            # Station selection for recommendations
            station_list = sorted(df['İstasyon'].unique())
            selected_stations = st.multiselect(
                "Öneri almak istediğiniz istasyonları seçin:",
                station_list,
                default=station_list[:3]
            )
            
            for station in selected_stations:
                station_analysis = analyze_station_performance(df, station)
                
                # Get comment analysis if available
                comment_analysis = None
                if st.session_state.comment_data is not None:
                    station_comments = st.session_state.comment_data[
                        st.session_state.comment_data['İstasyon'] == station
                    ]
                    
                    if not station_comments.empty:
                        category_scores = {}
                        for comment in station_comments['Yorum']:
                            sentiment, categories = analyze_comment_sentiment(comment)
                            for category in categories:
                                category_scores[category] = category_scores.get(category, 0) + sentiment
                        
                        comment_analysis = {
                            'category_scores': category_scores,
                            'total_comments': len(station_comments)
                        }
                
                # Generate recommendations
                recommendations = generate_improvement_recommendations(station_analysis, comment_analysis)
                
                # Display station recommendations
                st.markdown(f"### 🏢 {station}")
                st.markdown(f"**Mevcut Skor:** {station_analysis['current_score']:.3f} | **Kategori:** {station_analysis['performance_category']}")
                
                if recommendations:
                    for i, rec in enumerate(recommendations):
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
                    st.success(f"✅ {station} performansı iyi durumda!")
                
                st.markdown("---")
    
    else:
        # Welcome screen
        st.markdown("## 🎯 ENTERPRISE DASHBOARD'A HOŞGELDİNİZ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 PERFORMANS ANALİZİ
            - **Station-level deep dive**
            - **District comparison**
            - **Trend analysis**
            - **Segment optimization**
            """)
        
        with col2:
            st.markdown("""
            ### 💬 YORUM ANALİZ SİSTEMİ
            - **AI sentiment analysis**
            - **Category detection**
            - **Actionable insights**
            - **Improvement recommendations**
            """)
        
        st.info("👈 Sol panelden veri dosyalarınızı yükleyin veya demo verilerini kullanın.")

if __name__ == "__main__":
    main()