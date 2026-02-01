import pandas as pd

df = pd.read_csv('data/bilgisayar_statik_veriler.csv')

# DÖNEM 3 satırını bul
for index, row in df.iterrows():
    modul = str(row['MODÜL'])
    konu = str(row['KONU / DOSYA ADI'])
    
    if "BÖLÜM DERSLERİ" in modul and "DÖNEM 3" in konu:
        icerik = str(row['İÇERİK (METİN / TABLO)'])
        print(f"DÖNEM 3 bulundu!")
        print(f"İçerik uzunluğu: {len(icerik)}")
        print(f"İçerik:\n{repr(icerik[:500])}")
        print(f"\n\nSatır sayısı (\\n ile): {icerik.count(chr(10))}")
        print(f"\n\nİlk 5 satır:")
        lines = [l.strip() for l in icerik.split('\n') if l.strip()]
        for i, line in enumerate(lines[:10]):
            print(f"{i+1}: {repr(line)}")
        break
