# modules/supabase_client.py - Basitleştirilmiş versiyon

import streamlit as st
from supabase import create_client, Client

def get_supabase_client():
    """Basit Supabase client"""
    try:
        # Streamlit secrets'tan bilgileri al
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]
        
        # Client oluştur
        supabase: Client = create_client(url, key)
        return supabase
        
    except Exception as e:
        raise Exception(f"Supabase bağlantı hatası: {str(e)}")

def test_connection():
    """Bağlantıyı test et"""
    try:
        client = get_supabase_client()
        # Basit bir SELECT sorgusu
        result = client.table("periods").select("count", count="exact").execute()
        return True, f"Bağlantı başarılı. Periods tablosunda {result.count} kayıt var."
    except Exception as e:
        return False, f"Bağlantı hatası: {str(e)}"
