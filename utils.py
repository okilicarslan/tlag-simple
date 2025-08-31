import pandas as pd
from io import BytesIO

def add_unique_key(base_key, *args):
    """Her chart için benzersiz key oluştur"""
    return f"{base_key}_{'_'.join(map(str, args))}"

def validate_tlag_data(df):
    """TLAG verisini kontrol et"""
    required_cols = ['ROC', 'İstasyon', 'SKOR']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
    return True

def calculate_tlag_score(comments_df, group_by='DISTRICT'):
    """TLAG skorunu hesapla: (5 puan verenlerin sayısı / toplam yanıt) * 100"""
    if comments_df is None or comments_df.empty:
        return pd.DataFrame()
    
    group_col = f'{group_by}_FINAL' if f'{group_by}_FINAL' in comments_df.columns else group_by
    
    if group_col not in comments_df.columns:
        return pd.DataFrame()
    
    results = []
    for name, group in comments_df.groupby(group_col):
        if pd.notna(name) and name != 'nan' and name != '0':
            total_responses = len(group)
            five_star_count = len(group[group['score'] == 5])
            tlag_score = (five_star_count / total_responses * 100) if total_responses > 0 else 0
            
            results.append({
                group_by: name,
                'Toplam_Yanıt': total_responses,
                '5_Puan_Sayısı': five_star_count,
                'TLAG_Skoru_%': round(tlag_score, 1)
            })
    
    return pd.DataFrame(results)

def export_to_excel(dataframes_dict):
    """DataFrame'leri Excel'e aktar"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()