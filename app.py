import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="TLAG Dashboard", layout="wide")

st.title("🚀 TLAG PERFORMANCE DASHBOARD")
st.success("✅ Dashboard başarıyla çalışıyor!")

# Sidebar for file upload
st.sidebar.header("📁 VERİ YÜKLEME")

uploaded_file = st.sidebar.file_uploader(
    "TLAG Excel dosyanızı yükleyin:",
    type=['xlsx', 'xls'],
    help="satis_veri_clean.xlsx dosyanızı seçin"
)

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, sheet_name="TLAG DOKUNMA (2)")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Clean data
        df = df.dropna(subset=['ROC', 'İstasyon'])
        
        # Convert numeric columns
        numeric_cols = ['SKOR', 'GEÇEN SENE SKOR', 'Fark', 'TRANSACTION']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.sidebar.success(f"✅ {len(df)} istasyon yüklendi!")
        
        # Main Dashboard
        st.markdown("## 📊 TLAG PERFORMANS ANALİZİ")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam İstasyon", len(df))
        
        with col2:
            avg_score = df['SKOR'].mean()
            st.metric("Ortalama Skor", f"{avg_score:.3f}")
        
        with col3:
            if 'Site Segment' in df.columns:
                precious = len(df[df['Site Segment'] == 'My Precious'])
                st.metric("My Precious", precious)
            else:
                st.metric("My Precious", "N/A")
        
        with col4:
            improved = len(df[df['Fark'] > 0])
            st.metric("Gelişen İstasyon", improved)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # District performance
            if 'DISTRICT' in df.columns:
                district_avg = df.groupby('DISTRICT')['SKOR'].mean().sort_values(ascending=False)
                fig1 = px.bar(x=district_avg.values, y=district_avg.index, 
                             orientation='h',
                             title="Bölge Bazında Ortalama Skor")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("DISTRICT sütunu bulunamadı")
        
        with col2:
            # Segment performance
            if 'Site Segment' in df.columns:
                segment_counts = df['Site Segment'].value_counts()
                fig2 = px.pie(values=segment_counts.values, names=segment_counts.index,
                             title="Segment Dağılımı")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Score distribution
                fig2 = px.histogram(df, x='SKOR', nbins=20, title="Skor Dağılımı")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Performance vs Transaction Analysis
        if 'TRANSACTION' in df.columns:
            st.markdown("## 💰 PERFORMANS vs İŞLEM HACMİ")
            
            color_column = 'Site Segment' if 'Site Segment' in df.columns else None
            
            fig3 = px.scatter(df, x='TRANSACTION', y='SKOR',
                             color=color_column,
                             hover_data=['İstasyon', 'DISTRICT'] if 'DISTRICT' in df.columns else ['İstasyon'],
                             title="İşlem Hacmi vs Performans")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Top/Bottom Performers
        st.markdown("## 🏆 PERFORMANS ANALİZİ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥇 En İyi Performans")
            top_performers = df.nlargest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT']]
            st.dataframe(top_performers)
        
        with col2:
            if 'Fark' in df.columns:
                st.markdown("### 📈 En Çok Gelişenler")
                top_improvers = df.nlargest(10, 'Fark')[['İstasyon', 'Fark', 'SKOR']]
                st.dataframe(top_improvers)
            else:
                st.markdown("### 📉 En Düşük Performans")
                bottom_performers = df.nsmallest(10, 'SKOR')[['İstasyon', 'SKOR', 'DISTRICT']]
                st.dataframe(bottom_performers)
        
        # Full Data Table
        st.markdown("## 📋 TÜM VERİLER")
        
        # Search functionality
        search_term = st.text_input("İstasyon Ara:", placeholder="İstasyon adı yazın...")
        
        if search_term:
            filtered_df = df[df['İstasyon'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export
        st.markdown("## 💾 VERİ İNDİRME")
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📄 CSV İndir",
            data=csv,
            file_name=f"tlag_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Dosya okuma hatası: {str(e)}")
        st.info("Lütfen doğru Excel formatında dosya yüklediğinizden emin olun.")

else:
    # Demo data when no file uploaded
    st.info("👈 Sol panelden TLAG Excel dosyanızı yükleyin.")
    
    st.markdown("## 🎯 Demo Verileri")
    
    # Demo data
    demo_data = {
        'İstasyon': ['KASTAMONU', 'SAMSUN', 'ANKARA', 'İSTANBUL', 'İZMİR'],
        'SKOR': [0.641, 0.667, 0.810, 0.708, 0.619],
        'DISTRICT': ['ANKARA KUZEY', 'ANKARA KUZEY', 'MARMARA', 'MARMARA', 'İZMİR'],
        'Site Segment': ['My Precious', 'My Precious', 'Wasted Talent', 'My Precious', 'Saboteur'],
        'TRANSACTION': [16358, 15159, 7906, 5957, 4947],
        'Fark': [-1.6, 6.7, -0.4, 14.6, -8.1]
    }
    
    demo_df = pd.DataFrame(demo_data)
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Demo İstasyonları", len(demo_df))
    
    with col2:
        st.metric("Ortalama Skor", f"{demo_df['SKOR'].mean():.3f}")
    
    with col3:
        precious_count = len(demo_df[demo_df['Site Segment'] == 'My Precious'])
        st.metric("My Precious", precious_count)
    
    with col4:
        improved_count = len(demo_df[demo_df['Fark'] > 0])
        st.metric("Gelişen İstasyon", improved_count)
    
    # Demo charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(demo_df, x='İstasyon', y='SKOR', title="Demo İstasyon Performansı")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.pie(demo_df, names='Site Segment', title="Demo Segment Dağılımı")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 📊 Demo Veri Tablosu")
    st.dataframe(demo_df, use_container_width=True)