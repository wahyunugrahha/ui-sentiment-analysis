# Sentiment Analysis with IndoBERT

## Requirements

### Folder Structure
- **Folder `Model`:**
  - Harus berisi model hasil pelatihan dengan format nama file sebagai berikut:
    ```
    model_<learning_rate>E<epoch>.pth
    ```
  - Contoh:
    ```
    model_5e-5E4.pth
    ```

### Dependencies
1. **Clone IndoNLU Repository**
   - Anda dapat memperoleh IndoNLU dengan menjalankan perintah berikut:
     ```bash
     [https://github.com/IndoNLP/indonlu]
     ```
2. **Install Requirements**
   - Masuk ke folder `indonlu` dan jalankan:
     ```bash
     cd indonlu
     pip install -r requirements.txt
     ```

### Additional Requirements
- Pastikan Anda memiliki Python 3.8 atau versi yang lebih baru.
- CUDA toolkit (jika menggunakan GPU untuk pelatihan model).
- PyTorch versi terbaru yang kompatibel dengan sistem Anda.

## Usage

1. **Load Model**
   - Simpan file model ke dalam folder `Model`.
   - Gunakan format nama file yang sesuai.

2. **Start Analysis**
   - Jalankan aplikasi analisis sentimen:
     ```bash
     python sentimentAnalysisApp.py
     ```

3. **Testing**
   - Pastikan data testing Anda telah disiapkan dengan format yang sesuai.

## Notes
- Dokumentasi lebih lanjut terkait IndoNLU dapat ditemukan di [repository IndoNLU](https://github.com/IndoNLP/indonlu).
- Pastikan Anda memahami struktur file dan dependensi sebelum memulai proses pelatihan atau pengujian model.
