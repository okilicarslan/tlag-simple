# modules/supabase_client.py
import os
from typing import Optional, List, Dict, Any

try:
    import streamlit as st  # varsa secrets'tan da okuyacağız
except Exception:
    st = None

from functools import lru_cache
from supabase import create_client, Client


def _pick_supabase_key(d: Dict[str, Any]) -> Optional[str]:
    """
    secrets.toml veya ENV içinden uygun anahtarı seç.
    Öncelik: service_role_key > anon_key > key
    """
    return (
        d.get("service_role_key")
        or d.get("anon_key")
        or d.get("key")
        or d.get("SERVICE_ROLE_KEY")
        or d.get("ANON_KEY")
        or d.get("KEY")
    )


def _get_creds() -> (str, str):
    """
    Kimlik bilgilerini ortam değişkenlerinden ya da Streamlit secrets'tan alır.
    - ENV: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY / SUPABASE_KEY
    - secrets.toml: [supabase] url=..., (service_role_key|anon_key|key)=...
    """
    # ENV
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY")
    )

    # Streamlit secrets (varsa)
    if (not url or not key) and (st is not None) and hasattr(st, "secrets"):
        try:
            sb = st.secrets["supabase"]  # KeyError atarsa except'e düşsün
            url = url or sb.get("url") or sb.get("URL")
            key = key or _pick_supabase_key(sb)
        except Exception:
            pass

    if not url or not key:
        raise RuntimeError(
            "Supabase kimlik bilgileri bulunamadı.\n"
            "- ENV değişkenleri: SUPABASE_URL ve (SUPABASE_SERVICE_ROLE_KEY | SUPABASE_ANON_KEY | SUPABASE_KEY)\n"
            "- veya .streamlit/secrets.toml içinde:\n"
            "[supabase]\nurl = \"...\"\nservice_role_key = \"...\"  # (veya anon_key / key)"
        )
    return url, key


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Supabase client (v2) – tek instance (cache).
    """
    url, key = _get_creds()
    return create_client(url, key)


# ----------------- İsteğe bağlı yardımcılar -----------------

def find_period_ids(period_type: str, year: int, gran_val=None) -> List[str]:
    """
    Seçilen döneme uyan period id (uuid) listesini döndürür.
    """
    client = get_supabase_client()
    q = (
        client.table("periods")
        .select("id,period_type,year,week,month,quarter,half")
        .eq("period_type", period_type)
        .eq("year", year)
    )
    if period_type == "WEEK" and gran_val:
        q = q.eq("week", gran_val)
    if period_type == "MONTH" and gran_val:
        q = q.eq("month", gran_val)
    if period_type == "QUARTER" and gran_val:
        q = q.eq("quarter", gran_val)
    if period_type == "HALF" and gran_val:
        q = q.eq("half", gran_val)
    res = q.execute()
    return [r["id"] for r in (res.data or [])]


def upload_tlag_data(df, period_id) -> int:
    """
    TLAG dataframe'ini tlag_data tablosuna yazar.
    Şema: tlag_data(
      roc TEXT, period_id UUID, istasyon TEXT, district TEXT, nor TEXT,
      site_segment TEXT, skor DOUBLE PRECISION, gecen_sene_skor DOUBLE PRECISION,
      fark DOUBLE PRECISION, transaction DOUBLE PRECISION
    )
    NOT: roc DB'de TEXT olmalı. (integer ise: ALTER TABLE ... ALTER COLUMN roc TYPE text USING roc::text;)
    """
    import pandas as pd  # gecikmeli import
    if df is None or df.empty or not period_id:
        return 0

    client = get_supabase_client()
    rows: List[Dict[str, Any]] = []

    def _num(x):
        try:
            return float(x) if (x is not None and not pd.isna(x)) else None
        except Exception:
            return None

    for _, r in df.iterrows():
        # "16358.0" gibi değerler için güvenli string normalizasyonu:
        roc_val = str(r.get("ROC_STR") or r.get("ROC") or "").split(".")[0].strip()

        rows.append(
            dict(
                roc=roc_val,  # TEXT
                istasyon=r.get("İstasyon"),
                district=r.get("DISTRICT"),
                nor=r.get("NOR"),
                site_segment=r.get("Site Segment"),
                skor=_num(r.get("SKOR")),
                gecen_sene_skor=_num(r.get("GEÇEN SENE SKOR")),
                fark=_num(r.get("Fark")),
                transaction=_num(r.get("TRANSACTION")),
                period_id=period_id,
            )
        )

    if rows:
        client.table("tlag_data").upsert(rows, on_conflict="roc,period_id").execute()
    return len(rows)


def get_historical_data(table: str, limit: int = 5000):
    """
    Basit okuma yardımcı fonksiyonu.
    """
    client = get_supabase_client()
    return client.table(table).select("*").limit(limit).execute().data
