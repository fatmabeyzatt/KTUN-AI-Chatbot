# Data Pipeline

Bu klasor veri cekme ve donusturme adimlari icindir.

- `crawler/`: kaynak sayfalari tarama
- `pdf_download/`: PDF indirme
- `pdf_parse/`: PDF metin cikarma
- `normalize/`: alan standardizasyonu
- `config/`: kaynak URL ve ayarlar
- `scripts/`: tek komutlu calistirma scriptleri

Cikti sozlesmesi:
- Yapilandirilmis CSV/JSON dosyalari: `data/processed/*`
- PDF dosyalari: `data/raw/pdf/*`

`ingest.py` hem `data/*.csv|json` (geri uyumluluk) hem de `data/processed/**/*` altini recursive olarak okur.
