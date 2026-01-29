import pandas as pd

df = pd.read_csv('data/bilgisayar_statik_veriler.csv')
result = df[df['İÇERİK (METİN / TABLO)'].str.contains('Ayrık Matematik', na=False)]

print(f'Ayrık Matematik içeren satır sayısı: {len(result)}')
print('\n')

for i, row in result.head(3).iterrows():
    print(f'=== Satır {i} ===')
    print(f'MODÜL: {row["MODÜL"]}')
    print(f'KONU: {row["KONU / DOSYA ADI"]}')
    print(f'İÇERİK: {row["İÇERİK (METİN / TABLO)"][:500]}')
    print('---\n')
