# Amazon Product Rating Dashboard

Dashboard pembelajaran dengan alur **Data → Business Intelligence → Machine
Learning → LLM Analyst** menggunakan `amazon.csv`.

## Menjalankan

Di terminal, dari folder `student-doa`:

```bash
pip install -r requirements.txt
python data_prep.py
python ml_model.py
streamlit run app.py
```

`data_prep.py` menyimpan `amazon_clean.csv` dan `data_quality_report.json`.
`ml_model.py` menyimpan pipeline, metrik, feature importance, dan prediksi test
set. Aplikasi juga dapat menyiapkan dataset dan melatih model saat dibuka, jadi
file-file tersebut bukan prasyarat untuk menjalankan Streamlit.

## Pembersihan dan split data

- Kolom harga, diskon, dan jumlah rating diubah ke numerik; rating yang kosong
  atau rusak (contoh `|`) dibuang dan jumlahnya dilaporkan.
- `rating_count` yang kosong diisi median; `main_category` diambil dari segmen
  pertama `category`.
- Label `Rating_Class`: Low untuk rating `< 4.0`, Medium untuk `4.0–4.3`, dan
  High untuk `>= 4.4`.
- `rating` dan `Rating_Class` tidak dimasukkan ke fitur.
- `StratifiedGroupKFold` memakai lima fold (sekitar 80/20), menjaga proporsi
  kelas sejauh mungkin dan memastikan satu `product_id` tidak muncul di train
  dan test sekaligus. Karena pembagian dilakukan per grup, rasio baris aktual
  dapat sedikit berbeda dari 80/20.
- Imputer, one-hot encoder, dan Random Forest berada dalam satu Pipeline;
  preprocessing di-fit hanya menggunakan train set.

## LLM lokal opsional

LLM Analyst mengirim prompt berbasis data produk, probabilitas model, dan
feature importance ke Ollama lokal lewat HTTP dengan modul bawaan `urllib`.
LLM bukan dependensi wajib aplikasi; tab lain tetap dapat digunakan tanpa
Ollama.

```bash
ollama serve
ollama pull llama3.2
```

Atur model melalui environment variable `OLLAMA_MODEL` atau alamat server lewat
`OLLAMA_HOST` bila diperlukan.
