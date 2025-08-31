import pandas as pd
import re
import streamlit as st

def extract_station_code(station_name):
    """İstasyon adından 4 haneli kodu çıkar"""
    match = re.search(r'#(\d{4})$', str(station_name))
    return match.group(1) if match else None

def categorize_comments(comment_text):
    """Yorumları kategorize et"""
    categories = []
    comment_lower = str(comment_text).lower()
    
    # Kategori keywords
    category_map = {
        'PERSONEL': ['personel', 'çalışan', 'pompacı', 'kasiyer', 'görevli'],
        'TEMİZLİK': ['temiz', 'kirli', 'hijyen', 'tuvalet', 'pis'],
        'MARKET': ['market', 'ürün', 'fiyat', 'pahalı', 'ucuz'],
        'HIZ': ['hızlı', 'yavaş', 'bekleme', 'kuyruk', 'süre'],
        'YAKIT': ['benzin', 'motorin', 'lpg', 'yakıt', 'pompa']
    }
    
    for category, keywords in category_map.items():
        if any(keyword in comment_lower for keyword in keywords):
            categories.append(category)
    
    return categories if categories else ['GENEL']

def analyze_comments(comments_df, stations_df):
    """Yorumları analiz et ve istasyonlarla eşleştir"""
    # Station kodlarını çıkar
    comments_df['station_code'] = comments_df['station_name'].apply(extract_station_code)
    
    # Stations ile eşleştir
    merged = pd.merge(
        comments_df,
        stations_df[['ROC', 'NOR', 'DISTRICT', 'İstasyon']],
        left_on='station_code',
        right_on='ROC',
        how='left'
    )
    
    # Kategorize et
    merged['categories'] = merged['comment'].apply(categorize_comments)
    
    return merged

def get_comment_insights(analyzed_comments, level='district'):
    """Yorum analizinden insights çıkar"""
    insights = {}
    
    if level == 'district':
        grouped = analyzed_comments.groupby('DISTRICT')
    elif level == 'nor':
        grouped = analyzed_comments.groupby('NOR')
    else:
        grouped = analyzed_comments.groupby('İstasyon')
    
    for name, group in grouped:
        # En çok bahsedilen kategoriler
        all_categories = []
        for cats in group['categories']:
            all_categories.extend(cats)
        
        category_counts = pd.Series(all_categories).value_counts()
        
        # Ortalama puan
        avg_rating = group['rating'].mean()
        
        insights[name] = {
            'avg_rating': avg_rating,
            'top_categories': category_counts.head(3).to_dict(),
            'total_comments': len(group),
            'negative_comments': len(group[group['rating'] < 3])
        }
    
    return insights