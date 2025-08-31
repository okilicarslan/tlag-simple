import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import json

# Supabase entegrasyonu
try:
    from modules.supabase_client import (
        get_supabase_client, 
        upload_tlag_data,
        get_historical_data
    )
    SUPABASE_ENABLED = True
except ImportError:
    SUPABASE_ENABLED = False
    print("Supabase modülü yüklenemedi. Lokal modda çalışıyor.")

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
        cursor: pointer;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .comment-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
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
    .category-temizlik { background: #4ECDC4; color: white; }
    .category-market { background: #95E1D3; color: dark; }
    .category-hiz { background: #FFA502; color: white; }
    .category-yakit { background: #3742FA; color: white; }
    .category-genel { background: #747D8C; color: white; }
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
if 'comments_data' not in st.session_state:
    st.session_state.comments_data = None
if 'analyzed_comments' not in st.session_state:
    st.session_state.analyzed_comments = None

def clean_data_for_json(df):
    """Clean DataFrame for JSON serialization"""
    df_clean = df.copy()
    
    # Replace NaN and inf values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    
    # Convert problematic data types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
    
    return df_clean

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

def extract_station_code(station_info):
    """Extract station code from station info string"""
    if pd.isna(station_info):
        return None
    match = re.search(r'#(\d+)$', str(station_info))
    return match.group(1) if match else None

def categorize_comment(comment_text):
    """Categorize comments based on keywords"""
    if pd.isna(comment_text):
        return ['GENEL']
    
    comment_lower = str(comment_text).lower()
    categories = []
    
    # Category mappings
    category_keywords = {
        'PERSONEL': ['personel', 'çalışan', 'pompacı', 'kasiyer', 'görevli', 'müdür', 
                     'yardımsever', 'ilgili', 'güleryüzlü', 'kaba', 'ilgisiz', 'saygılı'],
        'TEMİZLİK': ['temiz', 'kirli', 'hijyen', 'tuvalet', 'pis', 'bakım', 'tertip', 'düzen'],
        'MARKET': ['market', 'ürün', 'fiyat', 'pahalı', 'ucuz', 'çeşit', 'kalite', 'taze'],
        'HIZ': ['hızlı', 'yavaş', 'bekleme', 'kuyruk', 'süre', 'geç', 'çabuk', 'acele'],
        'YAKIT': ['benzin', 'motorin', 'lpg', 'yakıt', 'pompa', 'dolum', 'depo'],
        'GENEL': ['genel', 'güzel', 'kötü', 'memnun', 'beğen', 'hoş']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in comment_lower for keyword in keywords):
            categories.append(category)
    
    return categories if categories else ['GENEL']

def load_tlag_data(uploaded_file):
    """Load TLAG data with proper formatting"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, sheet_name="TLAG DOKUNMA (2)", engine='openpyxl')
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Rename columns for consistency
        column_mapping = {
            'NOR HEDEF': 'NOR HEDEF',
            'DISTRICT HEDEF': 'DISTRICT HEDEF',
            'GEÇEN SENE SKOR': 'GEÇEN SENE SKOR',
            'Fark': 'Fark',
            'Geçerli': 'Geçerli',
            'TRANSACTION': 'TRANSACTION'
        }
        
        # Clean data
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
        
        # Convert ROC to string for matching
        df['ROC_STR'] = df['ROC'].astype(str).str.split('.').str[0]
        
        # Clean data for JSON
        df_clean = clean_data_for_json(df)
        
        # Save to Supabase if enabled
        if SUPABASE_ENABLED:
            try:
                upload_tlag_data(df_clean)
                st.sidebar.success("✅ Veriler Supabase'e kaydedildi!")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Supabase kayıt başarısız: {str(e)}")
        
        return df_clean
        
    except Exception as e:
        st.error(f"Dosya okuma hatası: {str(e)}")
        return None

def load_comments_data(uploaded_file):
    """Load and process comments data"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, header=1)  # Start from second row
        
        # Skip first info row and get real data
        df = df[df.iloc[:, 0] != '65000 yorum sınırını aştınız.']
        df = df[df.iloc[:, 0] != 'birim']
        
        # Rename columns based on position
        column_names = {
            df.columns[0]: 'station_info',
            df.columns[1]: 'survey_item',
            df.columns[2]: 'comment',
            df.columns[3]: 'score',
            df.columns[4]: 'visit_date',
            df.columns[5]: 'hospitality_score',
            df.columns[6]: 'dealer',
            df.columns[7]: 'territory',
            df.columns[8]: 'district',
            df.columns[9]: 'country' if len(df.columns) > 9 else 'country'
        }
        
        df = df.rename(columns=column_names)
        
        # Extract station code
        df['station_code'] = df['station_info'].apply(extract_station_code)
        
        # Convert score to numeric
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        # Categorize comments
        df['categories'] = df['comment'].apply(categorize_comment)
        
        # Clean for JSON
        df_clean = clean_data_for_json(df)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Yorum dosyası okuma hatası: {str(e)}")
        return None

def merge_comments_with_stations(comments_df, tlag_df):
    """Merge comments with station data"""
    try:
        # Match by station code
        merged = pd.merge(
            comments_df,
            tlag_df[['ROC_STR', 'İstasyon', 'NOR', 'DISTRICT']],
            left_on='station_code',
            right_on='ROC_STR',
            how='left'
        )
        
        # Use comment's district/territory if no match
        merged['NOR_FINAL'] = merged['NOR'].fillna(merged['territory'])
        merged['DISTRICT_FINAL'] = merged['DISTRICT'].fillna(merged['district'])
        
        return merged
        
    except Exception as e:
        st.error(f"Veri birleştirme hatası: {str(e)}")
        return comments_df

def analyze_comments_by_category(df, level='district'):
    """Analyze comments by category and score"""
    if df is None or df.empty:
        return {}
    
    # Group by level
    group_col = f'{level.upper()}_FINAL' if f'{level.upper()}_FINAL' in df.columns else level.upper()
    
    if group_col not in df.columns:
        return {}
    
    results = {}
    
    for name, group in df.groupby(group_col):
        if pd.isna(name) or name == 'nan':
            continue
            
        # Score distribution
        score_dist = group['score'].value_counts().to_dict()
        
        # Category analysis for low scores (4 and below)
        low_score_comments = group[group['score'] <= 4]
        
        # Count categories
        category_counts = {}
        for cats in low_score_comments['categories']:
            if isinstance(cats, list):
                for cat in cats:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Get sample comments by category
        category_samples = {}
        for category in category_counts.keys():
            samples = []
            for _, row in low_score_comments.iterrows():
                if isinstance(row['categories'], list) and category in row['categories']:
                    samples.append({
                        'comment': row['comment'],
                        'score': row['score'],
                        'station': row.get('station_info', '')
                    })
                    if len(samples) >= 3:  # Max 3 samples per category
                        break
            category_samples[category] = samples
        
        results[name] = {
            'total_comments': len(group),
            'avg_score': group['score'].mean(),
            'score_distribution': score_dist,
            'low_score_categories': category_counts,
            'category_samples': category_samples,
            'critical_count': len(group[group['score'] <= 2])
        }
    
    return results

def display_comment_analysis(analysis_data, title="Yorum Analizi"):
    """Display comment analysis results"""
    st.markdown(f"### 💬 {title}")
    
    if not analysis_data:
        st.info("Yorum verisi bulunamadı")
        return
    
    for name, data in analysis_data.items():
        with st.expander(f"📍 {name} - {data['total_comments']} yorum, Ort: {data['avg_score']:.1f}"):
            
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Puan Dağılımı:**")
                score_df = pd.DataFrame(
                    list(data['score_distribution'].items()),
                    columns=['Puan', 'Sayı']
                ).sort_values('Puan')
                
                fig = px.bar(score_df, x='Puan', y='Sayı', color='Puan',
                            color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**4 ve Altı Puan Kategorileri:**")
                if data['low_score_categories']:
                    cat_df = pd.DataFrame(
                        list(data['low_score_categories'].items()),
                        columns=['Kategori', 'Sayı']
                    ).sort_values('Sayı', ascending=False)
                    
                    for _, row in cat_df.iterrows():
                        cat_class = f"category-{row['Kategori'].lower()}"
                        st.markdown(f"""
                        <span class="category-badge {cat_class}">
                            {row['Kategori']}: {row['Sayı']}
                        </span>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Düşük puan kategorisi yok")
            
            # Category details with sample comments
            if data['category_samples']:
                st.markdown("**📝 Kategori Detayları (Tıklayın):**")
                
                for category, samples in data['category_samples'].items():
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
    """Create clickable metric card"""
    with col:
        # Metric card
        st.markdown(f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{title}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden button
        if st.button("📊 Detay", key=f"btn_{key}", use_container_width=True):
            st.session_state[f'show_{key}'] = not st.session_state.get(f'show_{key}', False)
    
    # Detail display
    if st.session_state.get(f'show_{key}', False):
        with st.container():
            st.markdown(f"### 📊 {title} Detayları")
            
            if key == "total_stations" and df is not None:
                # Show all stations
                display_df = df[['ROC', 'İstasyon', 'SKOR', 'DISTRICT', 'Site Segment']].copy()
                display_df['SKOR'] = display_df['SKOR'].apply(format_percentage)
                st.dataframe(
                    display_df.sort_values('İstasyon'),
                    use_container_width=True,
                    height=400
                )
                
            elif key == "avg_score" and df is not None:
                # Stations to improve average quickly
                st.write("**🎯 Ortalamayı En Hızlı Artıracak İstasyonlar:**")
                
                impact_df = df[df['SKOR'] < 0.7].copy()
                if 'TRANSACTION' in impact_df.columns:
                    impact_df['potential_impact'] = (0.7 - impact_df['SKOR']) * impact_df.get('TRANSACTION', 1)
                    top_impact = impact_df.nlargest(10, 'potential_impact')
                    
                    for idx, row in top_impact.iterrows():
                        potential_increase = (0.7 - row['SKOR']) * 100
                        st.write(f"• **{row['İstasyon']}**: {format_percentage(row['SKOR'])} → 70% (+{potential_increase:.1f}%)")
            
            elif key == "saboteur" and df is not None:
                # Saboteur segment details
                saboteur_df = df[df['Site Segment'] == 'Saboteur'][['İstasyon', 'SKOR', 'DISTRICT', 'NOR']].copy()
                saboteur_df['SKOR'] = saboteur_df['SKOR'].apply(format_percentage)
                st.dataframe(saboteur_df, use_container_width=True)
            
            elif key == "precious" and df is not None:
                # My Precious segment details
                precious_df = df[df['Site Segment'] == 'My Precious'][['İstasyon', 'SKOR', 'DISTRICT', 'NOR']].copy()
                precious_df['SKOR'] = precious_df['SKOR'].apply(format_percentage)
                st.dataframe(precious_df, use_container_width=True)
            
            if st.button("❌ Kapat", key=f"close_{key}"):
                st.session_state[f'show_{key}'] = False
                st.rerun()

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
    
    return recommendations[:4]

def analyze_district(df, selected_district, comments_analysis=None):
    """District based analysis with comments"""
    district_data = df[df['DISTRICT'] == selected_district]
    
    # District summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("İstasyon Sayısı", len(district_data))
    
    with col2:
        st.metric("Ortalama Skor", format_percentage(district_data['SKOR'].mean()))
    
    with col3:
        if 'Site Segment' in district_data.columns:
            segments = district_data['Site Segment'].value_counts()
            st.markdown("**Segment Dağılımı:**")
            for seg, count in segments.head(3).items():
                st.write(f"• {seg}: {count}")
    
    with col4:
        st.metric("En Düşük Skor", format_percentage(district_data['SKOR'].min()))
    
    # Best and worst stations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 En İyi 5 İstasyon")
        top5 = district_data.nlargest(5, 'SKOR')[['İstasyon', 'SKOR', 'Site Segment']].copy()
        top5['SKOR'] = top5['SKOR'].apply(format_percentage)
        st.dataframe(top5, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ En Kötü 5 İstasyon")
        bottom5 = district_data.nsmallest(5, 'SKOR')[['İstasyon', 'SKOR', 'Site Segment']].copy()
        bottom5['SKOR'] = bottom5['SKOR'].apply(format_percentage)
        st.dataframe(bottom5, use_container_width=True)
    
    # Opportunity stations
    st.markdown("### 💡 Fırsat İstasyonları")
    st.info("My Precious veya Primitive segmentinde olup %80 altında skor alan istasyonlar")
    
    opportunity = district_data[
        ((district_data['Site Segment'] == 'My Precious') | 
         (district_data['Site Segment'] == 'Primitive')) &
        (district_data['SKOR'] < 0.8)
    ]
    
    if not opportunity.empty:
        opp_display = opportunity[['İstasyon', 'SKOR', 'Site Segment', 'TRANSACTION']].copy()
        opp_display['SKOR'] = opp_display['SKOR'].apply(format_percentage)
        st.dataframe(opp_display, use_container_width=True)
    else:
        st.info("Bu district'te fırsat istasyonu bulunmuyor.")
    
    # Comments analysis if available
    if comments_analysis and selected_district in comments_analysis:
        display_comment_analysis(
            {selected_district: comments_analysis[selected_district]},
            title=f"{selected_district} Müşteri Yorumları"
        )

def analyze_nor(df, selected_nor, comments_analysis=None):
    """NOR based analysis with comments"""
    nor_data = df[df['NOR'] == selected_nor]
    
    # NOR summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("İstasyon Sayısı", len(nor_data))
    
    with col2:
        st.metric("Ortalama Skor", format_percentage(nor_data['SKOR'].mean()))
    
    with col3:
        if 'Site Segment' in nor_data.columns:
            st.metric("En Yaygın Segment", 
                     nor_data['Site Segment'].mode().iloc[0] if len(nor_data['Site Segment'].mode()) > 0 else "N/A")
    
    with col4:
        st.metric("Skor Aralığı", 
                 f"{format_percentage(nor_data['SKOR'].min())} - {format_percentage(nor_data['SKOR'].max())}")
    
    # Best and worst stations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 En İyi 5 İstasyon")
        top5 = nor_data.nlargest(5, 'SKOR')[['İstasyon', 'SKOR', 'Site Segment', 'DISTRICT']].copy()
        top5['SKOR'] = top5['SKOR'].apply(format_percentage)
        st.dataframe(top5, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ En Kötü 5 İstasyon")
        bottom5 = nor_data.nsmallest(5, 'SKOR')[['İstasyon', 'SKOR', 'Site Segment', 'DISTRICT']].copy()
        bottom5['SKOR'] = bottom5['SKOR'].apply(format_percentage)
        st.dataframe(bottom5, use_container_width=True)
    
    # Opportunity stations
    st.markdown("### 💡 Fırsat İstasyonları")
    
    opportunity = nor_data[
        ((nor_data['Site Segment'] == 'My Precious') | 
         (nor_data['Site Segment'] == 'Primitive')) &
        (nor_data['SKOR'] < 0.8)
    ]
    
    if not opportunity.empty:
        opp_display = opportunity[['İstasyon', 'SKOR', 'Site Segment', 'DISTRICT']].copy()
        opp_display['SKOR'] = opp_display['SKOR'].apply(format_percentage)
        st.dataframe(opp_display, use_container_width=True)
    else:
        st.info("Bu NOR'da fırsat istasyonu bulunmuyor.")
    
    # NOR performance graph
    st.markdown("### 📊 NOR Performans Dağılımı")
    nor_viz = nor_data.copy()
    nor_viz['Skor_Yüzde'] = nor_viz['SKOR'] * 100
    
    fig_nor = px.box(
        nor_viz,
        y='Skor_Yüzde',
        x='Site Segment',
        title=f"{selected_nor} - Segment Bazında Performans",
        labels={'Skor_Yüzde': 'Performans Skoru (%)'}
    )
    st.plotly_chart(fig_nor, use_container_width=True)
    
    # Comments analysis if available
    if comments_analysis and selected_nor in comments_analysis:
        display_comment_analysis(
            {selected_nor: comments_analysis[selected_nor]},
            title=f"{selected_nor} Müşteri Yorumları"
        )

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
    
    # Comments file upload
    st.sidebar.markdown("## 💬 MÜŞTERİ YORUMLARI")
    
    comments_file = st.sidebar.file_uploader(
        "Yorum dosyası (Excel):",
        type=['xlsx', 'xls'],
        key='comments_uploader',
        help="Comment YTD All.xlsx dosyanızı yükleyin"
    )
    
    # Demo data option
    if st.sidebar.button("📊 Demo Verisi Yükle"):
        # Generate demo data
        np.random.seed(42)
        demo_stations = ['KASTAMONU', 'SAMSUN ATAKUM', 'ANKARA YENİMAHALLE', 'İSTANBUL KARTAL', 'İZMİR BORNOVA']
        demo_data = []
        for i, station in enumerate(demo_stations):
            demo_data.append({
                'ROC': 4000 + i,
                'ROC_STR': str(4000 + i),
                'İstasyon': station,
                'DISTRICT': np.random.choice(['ANKARA KUZEY BÖLGE', 'MARMARA BÖLGE', 'İZMİR BÖLGE']),
                'SKOR': np.random.uniform(0.5, 0.9),
                'GEÇEN SENE SKOR': np.random.uniform(0.4, 0.8),
                'Fark': np.random.uniform(-10, 15),
                'Site Segment': np.random.choice(['My Precious', 'Wasted Talent', 'Saboteur', 'Primitive']),
                'TRANSACTION': np.random.randint(5000, 50000),
                'NOR': np.random.choice(['KARADENİZ BATI', 'SAMSUN', 'İZMİR MERKEZ'])
            })
        st.session_state.tlag_data = pd.DataFrame(demo_data)
        st.sidebar.success("✅ Demo verisi yüklendi!")
    
    # Load TLAG data
    if uploaded_file is not None:
        with st.spinner("📊 Veriler işleniyor..."):
            df = load_tlag_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.session_state.tlag_data = df
            st.sidebar.success(f"✅ {len(df)} istasyon verisi yüklendi!")
            
            # Data summary
            with st.sidebar.expander("📋 Veri Özeti"):
                st.write(f"**Toplam İstasyon:** {len(df)}")
                if 'SKOR' in df.columns:
                    avg_score = df['SKOR'].mean()
                    st.write(f"**Ortalama Skor:** {format_percentage(avg_score)}")
                    st.write(f"**En Yüksek:** {format_percentage(df['SKOR'].max())}")
                    st.write(f"**En Düşük:** {format_percentage(df['SKOR'].min())}")
                
                if 'Site Segment' in df.columns:
                    segments = df['Site Segment'].value_counts()
                    st.write("**Segment Dağılımı:**")
                    for segment, count in segments.items():
                        if pd.notna(segment):
                            st.write(f"- {segment}: {count}")
    
    # Load comments data
    if comments_file and st.session_state.tlag_data is not None:
        with st.spinner("💬 Yorumlar işleniyor..."):
            comments_df = load_comments_data(comments_file)
            
            if comments_df is not None:
                # Merge with station data
                merged_comments = merge_comments_with_stations(comments_df, st.session_state.tlag_data)
                st.session_state.comments_data = merged_comments
                
                # Analyze comments
                district_comments = analyze_comments_by_category(merged_comments, 'district')
                nor_comments = analyze_comments_by_category(merged_comments, 'nor')
                
                st.session_state.analyzed_comments = {
                    'district': district_comments,
                    'nor': nor_comments
                }
                
                st.sidebar.success(f"✅ {len(comments_df)} yorum yüklendi ve analiz edildi!")
                
                # Comments summary
                with st.sidebar.expander("💬 Yorum Özeti"):
                    st.write(f"**Toplam Yorum:** {len(comments_df)}")
                    if 'score' in comments_df.columns:
                        avg_score = comments_df['score'].mean()
                        st.write(f"**Ortalama Puan:** {avg_score:.1f}")
                        
                        score_dist = comments_df['score'].value_counts().sort_index()
                        st.write("**Puan Dağılımı:**")
                        for score, count in score_dist.items():
                            if pd.notna(score) and score <= 5:
                                st.write(f"- {int(score)} puan: {count}")
    
    # Main dashboard
    if st.session_state.tlag_data is not None:
        df = st.session_state.tlag_data
        
        # Get analyzed comments if available
        analyzed_comments = st.session_state.get('analyzed_comments', {})
        district_comments = analyzed_comments.get('district', {})
        nor_comments = analyzed_comments.get('nor', {})
        
        # Analysis tabs
        st.markdown("## 🎯 ANALİZ TÜRÜ")
        
        main_tabs = st.tabs([
            "📊 Genel Dashboard", 
            "🏢 District Analizi", 
            "📍 NOR Analizi", 
            "🎯 Segment Analizi", 
            "🔍 İstasyon Detay", 
            "🏆 Top/Bottom", 
            "💬 Yorum Analizi",
            "🤖 AI Öneriler"
        ])
        
        with main_tabs[0]:
            # GENERAL DASHBOARD
            st.markdown("## 📊 ANA METRİKLER")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Clickable metrics
            create_clickable_metric(col1, "Toplam İstasyon", len(df), "total_stations", df)
            
            if 'SKOR' in df.columns:
                create_clickable_metric(col2, "Ortalama Skor", format_percentage(df['SKOR'].mean()), "avg_score", df)
            
            if 'Site Segment' in df.columns:
                saboteur_count = len(df[df['Site Segment'] == 'Saboteur'])
                precious_count = len(df[df['Site Segment'] == 'My Precious'])
                
                create_clickable_metric(col3, "Saboteur", saboteur_count, "saboteur", df)
                create_clickable_metric(col4, "My Precious", precious_count, "precious", df)
            
            # Performance distribution
            if 'SKOR' in df.columns:
                st.markdown("## 📈 PERFORMANS DAĞILIMI")
                
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
        
        with main_tabs[1]:
            # DISTRICT ANALYSIS
            st.markdown("## 🏢 DISTRICT BAZLI ANALİZ")
            
            if 'DISTRICT' in df.columns:
                districts = sorted(df['DISTRICT'].dropna().unique())
                selected_district = st.selectbox("District Seçin:", districts)
                
                if selected_district:
                    analyze_district(df, selected_district, district_comments)
        
        with main_tabs[2]:
            # NOR ANALYSIS
            st.markdown("## 📍 NOR BAZLI ANALİZ")
            
            if 'NOR' in df.columns:
                nors = sorted(df['NOR'].dropna().unique())
                selected_nor = st.selectbox("NOR Seçin:", nors)
                
                if selected_nor:
                    analyze_nor(df, selected_nor, nor_comments)
        
        with main_tabs[3]:
            # SEGMENT ANALYSIS
            st.markdown("## 🎯 SEGMENT BAZLI ANALİZ")
            
            if 'Site Segment' in df.columns:
                segments = df['Site Segment'].dropna().unique()
                selected_segment = st.selectbox("Segment Seçin:", segments)
                
                if selected_segment:
                    segment_data = df[df['Site Segment'] == selected_segment]
                    
                    st.markdown(f"### {selected_segment} Segmenti Analizi")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("İstasyon Sayısı", len(segment_data))
                    
                    with col2:
                        st.metric("Ortalama Skor", format_percentage(segment_data['SKOR'].mean()))
                    
                    with col3:
                        if 'TRANSACTION' in segment_data.columns:
                            st.metric("Toplam İşlem", f"{segment_data['TRANSACTION'].sum():,}")
                    
                    # District distribution
                    if 'DISTRICT' in segment_data.columns:
                        st.markdown("### District Dağılımı")
                        district_dist = segment_data['DISTRICT'].value_counts()
                        
                        fig_dist = px.bar(
                            x=district_dist.values,
                            y=district_dist.index,
                            orientation='h',
                            title=f"{selected_segment} - District Dağılımı",
                            labels={'x': 'İstasyon Sayısı', 'y': 'District'}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
        
        with main_tabs[4]:
            # STATION DETAIL
            st.markdown("## 🔍 İSTASYON DETAY ANALİZİ")
            
            station_search = st.text_input("🔍 İstasyon ara:", placeholder="İstasyon adı yazın...")
            
            if station_search:
                filtered_stations = df[df['İstasyon'].str.contains(station_search, case=False, na=False)]['İstasyon'].tolist()
            else:
                filtered_stations = sorted(df['İstasyon'].unique())
            
            if filtered_stations:
                selected_station = st.selectbox("İstasyon seçin:", filtered_stations)
                
                if selected_station:
                    station_data = df[df['İstasyon'] == selected_station].iloc[0].to_dict()
                    
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
                    
                    # Station comments if available
                    if st.session_state.comments_data is not None:
                        station_code = str(station_data.get('ROC_STR', ''))
                        station_comments = st.session_state.comments_data[
                            st.session_state.comments_data['station_code'] == station_code
                        ]
                        
                        if not station_comments.empty:
                            st.markdown("### 💬 Müşteri Yorumları")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Toplam Yorum", len(station_comments))
                                st.metric("Ortalama Puan", f"{station_comments['score'].mean():.1f}")
                            
                            with col2:
                                # Score distribution
                                score_counts = station_comments['score'].value_counts().sort_index()
                                fig_score = px.bar(
                                    x=score_counts.index,
                                    y=score_counts.values,
                                    title="Puan Dağılımı",
                                    labels={'x': 'Puan', 'y': 'Sayı'}
                                )
                                st.plotly_chart(fig_score, use_container_width=True)
                            
                            # Recent comments
                            st.markdown("### Son Yorumlar")
                            recent_comments = station_comments.head(5)
                            for _, comment in recent_comments.iterrows():
                                cats = comment['categories'] if isinstance(comment['categories'], list) else ['GENEL']
                                cat_badges = ' '.join([f'<span class="category-badge category-{cat.lower()}">{cat}</span>' for cat in cats])
                                
                                st.markdown(f"""
                                <div class="comment-card">
                                    <strong>Puan: {comment['score']}</strong> {cat_badges}<br>
                                    <em>{comment['comment']}</em>
                                </div>
                                """, unsafe_allow_html=True)
        
        with main_tabs[5]:
            # TOP/BOTTOM PERFORMANCE
            st.markdown("## 🏆 PERFORMANS SIRALAMALARI")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥇 EN İYİ PERFORMANSLAR")
                
                if 'SKOR' in df.columns:
                    top_performers = df.nlargest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT', 'Site Segment']].copy()
                    top_performers['Skor (%)'] = top_performers['SKOR'].apply(format_percentage)
                    top_performers = top_performers.drop('SKOR', axis=1)
                    top_performers.insert(0, 'Sıra', range(1, len(top_performers) + 1))
                    st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.markdown("### ⚠️ GELİŞİM GEREKTİREN İSTASYONLAR")
                
                if 'SKOR' in df.columns:
                    bottom_performers = df.nsmallest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT', 'Site Segment']].copy()
                    bottom_performers['Skor (%)'] = bottom_performers['SKOR'].apply(format_percentage)
                    bottom_performers = bottom_performers.drop('SKOR', axis=1)
                    st.dataframe(bottom_performers, use_container_width=True)
        
        with main_tabs[6]:
            # COMMENTS ANALYSIS
            st.markdown("## 💬 YORUM ANALİZİ")
            
            if st.session_state.comments_data is not None:
                comments_df = st.session_state.comments_data
                
                # Overall statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Toplam Yorum", len(comments_df))
                
                with col2:
                    st.metric("Ortalama Puan", f"{comments_df['score'].mean():.2f}")
                
                with col3:
                    low_scores = len(comments_df[comments_df['score'] <= 2])
                    st.metric("Kritik Yorumlar (≤2)", low_scores)
                
                with col4:
                    high_scores = len(comments_df[comments_df['score'] == 5])
                    st.metric("Mükemmel (5)", high_scores)
                
                # Category analysis
                st.markdown("### 📊 Kategori Analizi (4 ve Altı Puanlar)")
                
                low_score_comments = comments_df[comments_df['score'] <= 4]
                
                # Count all categories
                all_categories = {}
                for cats in low_score_comments['categories']:
                    if isinstance(cats, list):
                        for cat in cats:
                            all_categories[cat] = all_categories.get(cat, 0) + 1
                
                if all_categories:
                    cat_df = pd.DataFrame(
                        list(all_categories.items()),
                        columns=['Kategori', 'Sayı']
                    ).sort_values('Sayı', ascending=False)
                    
                    fig_cat = px.bar(
                        cat_df,
                        x='Sayı',
                        y='Kategori',
                        orientation='h',
                        title="En Çok Bahsedilen Konular (4 ve Altı Puanlar)",
                        color='Sayı',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                    
                    # Sample comments by category
                    st.markdown("### 📝 Kategori Örnekleri")
                    
                    for category in cat_df.head(5)['Kategori']:
                        samples = []
                        for _, row in low_score_comments.iterrows():
                            if isinstance(row['categories'], list) and category in row['categories']:
                                samples.append({
                                    'comment': row['comment'],
                                    'score': row['score'],
                                    'station': row.get('station_info', ''),
                                    'district': row.get('DISTRICT_FINAL', row.get('district', ''))
                                })
                                if len(samples) >= 3:
                                    break
                        
                        if samples:
                            with st.expander(f"{category} - {len(samples)} örnek"):
                                for sample in samples:
                                    st.markdown(f"""
                                    <div class="comment-card">
                                        <strong>Puan: {sample['score']}</strong> | 
                                        <strong>District: {sample['district']}</strong><br>
                                        <em>"{sample['comment']}"</em><br>
                                        <small>{sample['station']}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                # District-based comment analysis
                if district_comments:
                    st.markdown("### 🏢 District Bazlı Yorum Analizi")
                    display_comment_analysis(district_comments, "District Yorumları")
                
                # NOR-based comment analysis
                if nor_comments:
                    st.markdown("### 📍 NOR Bazlı Yorum Analizi")
                    display_comment_analysis(nor_comments, "NOR Yorumları")
            else:
                st.info("💬 Yorum analizi için lütfen yorum dosyasını yükleyin")
        
        with main_tabs[7]:
            # AI RECOMMENDATIONS
            st.markdown("## 🤖 AI-POWERED İYİLEŞTİRME ÖNERİLERİ")
            
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
                            
                            # Show related comments if available
                            if st.session_state.comments_data is not None:
                                station_code = str(station_data.get('ROC_STR', ''))
                                station_comments = st.session_state.comments_data[
                                    (st.session_state.comments_data['station_code'] == station_code) &
                                    (st.session_state.comments_data['score'] <= 3)
                                ]
                                
                                if not station_comments.empty:
                                    st.markdown("**📝 Müşteri Geri Bildirimleri:**")
                                    for _, comment in station_comments.head(3).iterrows():
                                        st.write(f"- Puan {comment['score']}: *{comment['comment']}*")
    
    else:
        # Welcome screen
        st.markdown("## 🎯 TLAG PERFORMANS ANALİTİK'E HOŞGELDİNİZ")
        
        st.info("👈 Sol panelden Excel dosyalarınızı yükleyin veya demo verilerini deneyin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 ANALİZ ÖZELLİKLERİ
            - ✅ **Tıklanabilir metrikler**
            - ✅ **Müşteri yorumları analizi**
            - ✅ **Kategori bazlı yorum gruplandırma**
            - ✅ **District/NOR/İstasyon analizi**
            - ✅ **AI-powered öneriler**
            """)
        
        with col2:
            st.markdown("""
            ### 💬 YORUM ANALİZİ
            - ✅ **Otomatik kategorizasyon**
            - ✅ **Düşük puan analizi**
            - ✅ **Fırsat istasyonları tespiti**
            - ✅ **Gerçek yorum örnekleri**
            - ✅ **Trend analizi**
            """)

if __name__ == "__main__":
    main()