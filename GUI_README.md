# Medical Waste Feature Extractor GUI

GUI aplikasi untuk ekstraksi fitur sampah medis menggunakan PyQt5. Aplikasi ini mendukung dua mode operasi:

1. **Single Image**: Ekstraksi fitur untuk satu gambar
2. **Batch Processing**: Ekstraksi fitur untuk seluruh dataset

## Instalasi

### Persyaratan Sistem

- Python 3.6 atau lebih baru
- Windows/Linux/MacOS

### Instalasi Dependencies

Jalankan file batch (Windows):

```bash
run_gui.bat
```

Atau install manual:

```bash
pip install PyQt5 opencv-python numpy pandas matplotlib scikit-learn joblib scikit-image
```

## Cara Menggunakan

### Menjalankan GUI

1. **Windows**: Double-click `run_gui.bat`
2. **Manual**: `python medical_waste_gui.py`

### Tab Single Image

1. Klik **"Select Image"** untuk memilih gambar sampah medis
2. (Opsional) Klik **"Select Output Folder"** untuk mengubah folder output
3. Klik **"Extract Features"** untuk memulai ekstraksi
4. Hasil akan ditampilkan di area progress dan disimpan ke folder output

**Format gambar yang didukung**: .jpg, .jpeg, .png, .bmp, .tiff, .tif

### Tab Batch Processing

1. Klik **"Select Dataset Folder"** untuk memilih folder dataset
2. (Opsional) Klik **"Select Output Folder"** untuk mengubah folder output
3. Klik **"Start Batch Processing"** untuk memulai pemrosesan
4. Progress akan ditampilkan secara real-time

**Struktur dataset yang diharapkan**:

```
dataset/
├── ampul/
│   ├── ampul1.jpg
│   ├── ampul2.jpg
│   └── ...
├── gloves/
│   ├── gloves1.jpg
│   └── ...
├── kapas/
│   ├── kapas1.jpg
│   └── ...
└── ...
```

## Output

### Single Image

- `[nama_gambar]_main_output.png`: Hasil Canny edge detection
- `[nama_gambar]_step_by_step.png`: Visualisasi step-by-step processing

### Batch Processing

- Folder per kategori dengan hasil ekstraksi
- `dataset_features_for_ml.csv`: Fitur untuk machine learning
- `main_output_pixels_for_ml.csv`: Nilai pixel main output
- `dataset_metadata.csv`: Metadata dataset

## Fitur Ekstraksi

1. **HSV-based Segmentation**: Menghapus background dan isolasi objek
2. **LBP (Local Binary Pattern)**: Ekstraksi fitur tekstur
3. **Canny Edge Detection**: Ekstraksi fitur bentuk dan kontur
4. **Resize Otomatis**: Semua gambar di-resize ke 256x256 pixel

## Troubleshooting

### Error: "PyQt5 not found"

```bash
pip install PyQt5
```

### Error: "cv2 not found"

```bash
pip install opencv-python
```

### GUI tidak muncul

- Pastikan Python dan PyQt5 terinstall dengan benar
- Coba jalankan dari command line untuk melihat error message

### Batch processing lambat

- Normal untuk dataset besar
- Progress akan ditampilkan secara real-time
- Jangan tutup aplikasi selama processing

## Kontrol Aplikasi

- **ESC**: Keluar dari aplikasi
- **Tab**: Pindah antar tab
- **Enter**: Jalankan fungsi tombol yang sedang fokus

## File Output

### CSV Files

- **dataset_features_for_ml.csv**: Fitur siap untuk machine learning
- **main_output_pixels_for_ml.csv**: Data pixel mentah untuk analisis mendalam
- **dataset_metadata.csv**: Informasi tentang dataset

### Image Files

- **main_output.png**: Hasil utama (Canny edge detection)
- **step_by_step.png**: Visualisasi proses step-by-step

## Tips Penggunaan

1. **Untuk penelitian**: Gunakan batch processing untuk dataset lengkap
2. **Untuk testing**: Gunakan single image untuk test cepat
3. **Performa**: Tutup aplikasi lain saat batch processing dataset besar
4. **Storage**: Pastikan ruang disk cukup untuk output (~2-5MB per gambar)

## Technical Details

### Threading

- Ekstraksi fitur berjalan di background thread
- GUI tetap responsif selama processing
- Progress ditampilkan real-time

### Memory Management

- Gambar diproses satu per satu untuk efisiensi memory
- Automatic garbage collection setelah setiap gambar

### Error Handling

- Validasi input file/folder
- Graceful error handling dengan pesan yang jelas
- Recovery dari error individual dalam batch processing
