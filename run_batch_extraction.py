#!/usr/bin/env python3
"""
Script untuk otomatisasi ekstraksi fitur dari dataset sampah medis dengan struktur folder
dataset -> jenis -> foto-foto

Mengekstrak fitur warna, bentuk, tekstur, dan HOG untuk persiapan machine learning
dan menyimpan hasilnya dalam format CSV.
"""

import os
import sys
import glob
import time
import numpy as np
from pathlib import Path
import pandas as pd
from medical_waste_feature_extractor import MedicalWasteFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DatasetFeatureProcessor:
    def __init__(self, dataset_path, output_path="dataset_features_processed"):
        """
        Inisialisasi processor untuk dataset
        
        Args:
            dataset_path (str): Path ke folder dataset utama
            output_path (str): Path untuk menyimpan hasil ekstraksi fitur
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.extractor = MedicalWasteFeatureExtractor()
        
        # Ekstensi gambar yang didukung
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Statistik pemrosesan
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'categories': {},
            'processing_time': 0
        }
    
    def create_output_structure(self):
        """Buat struktur folder output yang sama dengan input"""
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Folder output dibuat: {self.output_path}")
            return True
        except Exception as e:
            print(f"❌ Error membuat folder output: {e}")
            return False
    
    def get_all_images(self):
        """Dapatkan semua file gambar dari dataset dengan struktur kategori"""
        all_images = []
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset path tidak ditemukan: {self.dataset_path}")
            return all_images
        
        # Iterasi melalui setiap kategori (subfolder)
        for category_path in self.dataset_path.iterdir():
            if category_path.is_dir():
                category_name = category_path.name
                print(f"🔍 Memeriksa kategori: {category_name}")
                
                # Hitung gambar dalam kategori ini
                category_images = []
                for ext in self.image_extensions:
                    pattern = str(category_path / f"*{ext}")
                    category_images.extend(glob.glob(pattern, recursive=False))
                    pattern = str(category_path / f"*{ext.upper()}")
                    category_images.extend(glob.glob(pattern, recursive=False))
                
                self.stats['categories'][category_name] = len(category_images)
                all_images.extend([(img, category_name) for img in category_images])
                
                print(f"   └─ Ditemukan {len(category_images)} gambar")
        
        self.stats['total_images'] = len(all_images)
        return all_images
    
    def process_single_image(self, image_path, category_name):
        """
        Proses satu gambar dan ekstrak semua fitur
        
        Args:
            image_path (str): Path ke gambar
            category_name (str): Nama kategori/jenis
            
        Returns:
            dict: Informasi hasil pemrosesan dan fitur yang diekstrak
        """
        try:
            image_path = Path(image_path)
            
            # Buat folder kategori di output jika belum ada
            category_output_path = self.output_path / category_name
            category_output_path.mkdir(parents=True, exist_ok=True)
            
            # Modifikasi pada fungsi process_single_image (baris 100-120)
              # Proses gambar untuk mendapatkan semua fitur dengan fokus pada main output
            start_time = time.time()
            features = self.extractor.process_image(str(image_path), 
                                                  use_lbp=True, 
                                                  use_hog=False,  # Fokus pada edge detection
                                                  use_edge_detection=True, 
                                                  remove_background=True)
            processing_time = time.time() - start_time
            # Simpan hasil visualisasi main output (Canny edge detection)
            output_filename = f"{image_path.stem}_main_output.png"
            output_path = category_output_path / output_filename
            
            # Gunakan method visualize_and_save dengan fokus pada main output
            main_output_path = self.extractor.visualize_and_save(str(image_path), output_folder=str(category_output_path), create_full_visualization=False)
            
            # Ekstrak fitur untuk machine learning
            extracted_features = self.extract_features_for_ml(features, category_name, str(image_path))
            return {
                'status': 'success',
                'input_path': str(image_path),
                'output_path': main_output_path if main_output_path else str(output_path),
                'category': category_name,
                'features': extracted_features,
                'processing_time': processing_time
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'input_path': str(image_path),
                'category': category_name,
                'error': str(e)
            }
    
    def extract_features_for_ml(self, features, category, image_path):
        """
        Ekstrak dan format fitur untuk machine learning
        
        Args:
            features (dict): Fitur yang diekstrak dari gambar
            category (str): Kategori gambar
            image_path (str): Path gambar
            
        Returns:
            dict: Fitur yang diformat untuk machine learning
        """
        ml_features = {
            'image_path': image_path,
            'category': category
        }
        
        # 1. Ekstrak nilai pixel dari main output (Canny edge detection)
        main_output_pixels = self.extractor.extract_main_output_pixels()
        if main_output_pixels:
            # Tambahkan statistik pixel
            ml_features.update({k: v for k, v in main_output_pixels.items() 
                              if not k.startswith('pixel_') and not k.startswith('norm_pixel_')})
            
            # Simpan semua nilai pixel untuk analisis mendalam (akan disimpan di CSV terpisah)
            ml_features['main_output_pixels'] = main_output_pixels
        
        # 2. Fitur Edge Detection dari Canny
        if 'canny_edge_detection' in features and 'edge_features' in features['canny_edge_detection']:
            edge_features = features['canny_edge_detection']['edge_features']
            for key, value in edge_features.items():
                ml_features[f'edge_{key}'] = value        # 3. Fitur Tekstur LBP
        if 'texture' in features and 'lbp' in features['texture'] and 'histogram' in features['texture']['lbp']:
            lbp_hist = features['texture']['lbp']['histogram']
            # Ambil beberapa bin histogram LBP yang representatif
            num_bins = min(26, len(lbp_hist))  # Batasi jumlah bin
            for i in range(num_bins):
                ml_features[f'lbp_hist_{i+1}'] = lbp_hist[i]
        
        # 4. Fitur HOG (jika ada)
        if 'hog' in features and 'features' in features['hog']:
            hog_features = features['hog']['features']
            
            # Untuk HOG, kita bisa mengambil statistik atau sampel dari vektor fitur
            ml_features['hog_mean'] = np.mean(hog_features)
            ml_features['hog_std'] = np.std(hog_features)
            ml_features['hog_median'] = np.median(hog_features)
            ml_features['hog_min'] = np.min(hog_features)
            ml_features['hog_max'] = np.max(hog_features)
            
            # Ambil beberapa sampel fitur HOG (opsional)
            # Batasi jumlah fitur HOG untuk menghindari dimensi yang terlalu tinggi
            hog_sample_size = min(50, len(hog_features))
            sample_indices = np.linspace(0, len(hog_features)-1, hog_sample_size, dtype=int)
            for i, idx in enumerate(sample_indices):
                ml_features[f'hog_feature_{i+1}'] = hog_features[idx]
            
            # Tambahkan: Simpan nilai per-pixel dari citra HOG
            if 'image' in features['hog']:
                hog_image = features['hog']['image']
                # Flatten HOG image untuk mendapatkan nilai per-pixel
                hog_pixels = hog_image.flatten()
                # Batasi jumlah pixel yang disimpan (ambil sampel)
                pixel_sample_size = min(100, len(hog_pixels))
                pixel_indices = np.linspace(0, len(hog_pixels)-1, pixel_sample_size, dtype=int)
                for i, idx in enumerate(pixel_indices):
                    ml_features[f'hog_pixel_{i+1}'] = hog_pixels[idx]
        
        return ml_features
    
    def save_features_to_csv(self, results):
        """
        Simpan semua fitur ke file CSV untuk analisis dan machine learning
        
        Args:
            results (list): Hasil pemrosesan semua gambar
        """
        try:
            csv_data = []
            
            for result in results:
                if result['status'] == 'success' and 'features' in result:
                    csv_data.append(result['features'])
            
            if csv_data:
                # Buat DataFrame dari data
                df = pd.DataFrame(csv_data)
                
                # Simpan ke CSV
                csv_path = self.output_path / 'dataset_features_for_ml.csv'
                df.to_csv(csv_path, index=False)
                print(f"💾 Fitur disimpan ke: {csv_path}")
                
                # Tambahkan metadata tentang dataset
                class_distribution = df['category'].value_counts()
                print(f"📊 Distribusi kelas:")
                for category, count in class_distribution.items():
                    print(f"   └─ {category}: {count} sampel")
                
                # Simpan metadata dataset
                metadata = {
                    'total_samples': len(df),
                    'total_features': len(df.columns) - 2,
                    'class_distribution': class_distribution.to_dict(),
                    'feature_names': list(df.columns[2:]),  # Semua kolom kecuali image_path dan category
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Simpan metadata ke CSV
                metadata_df = pd.DataFrame([metadata])
                metadata_path = self.output_path / 'dataset_metadata.csv'
                metadata_df.to_csv(metadata_path, index=False)
                print(f"📝 Metadata dataset disimpan ke: {metadata_path}")
                
        except Exception as e:
            print(f"⚠️ Warning: Gagal menyimpan CSV fitur: {e}")
    
    def save_hog_pixels_to_csv(self, results):
        """
        Simpan nilai per-pixel dari citra HOG ke file CSV terpisah
        
        Args:
            results (list): Hasil pemrosesan semua gambar
        """
        try:
            hog_pixel_data = []
            
            for result in results:
                if result['status'] == 'success' and 'features' in result:
                    features = result['features']
                    category = features['category']
                    image_path = features['image_path']
                    
                    # Ekstrak semua kolom yang berhubungan dengan HOG pixel
                    hog_pixel_features = {k: v for k, v in features.items() if k.startswith('hog_pixel_')}
                    
                    if hog_pixel_features:
                        # Tambahkan informasi gambar dan kategori
                        hog_pixel_features['image_path'] = image_path
                        hog_pixel_features['category'] = category
                        hog_pixel_data.append(hog_pixel_features)
            
            if hog_pixel_data:
                # Buat DataFrame dari data
                df = pd.DataFrame(hog_pixel_data)
                
                # Simpan ke CSV
                csv_path = self.output_path / 'hog_pixels_for_ml.csv'
                df.to_csv(csv_path, index=False)
                print(f"💾 Nilai per-pixel HOG disimpan ke: {csv_path}")
                
        except Exception as e:
            print(f"⚠️ Warning: Gagal menyimpan CSV nilai per-pixel HOG: {e}")
    
    def save_main_output_pixels_to_csv(self, results):
        """
        Simpan nilai per-pixel dari main output (Canny edge detection) ke file CSV terpisah
        
        Args:
            results (list): Hasil pemrosesan semua gambar
        """
        try:
            main_output_pixel_data = []
            
            for result in results:
                if result['status'] == 'success' and 'features' in result:
                    features = result['features']
                    category = features['category']
                    image_path = features['image_path']
                    
                    # Ekstrak nilai pixel dari main output
                    if 'main_output_pixels' in features and features['main_output_pixels']:
                        pixel_data = features['main_output_pixels']
                        
                        # Buat record untuk setiap gambar
                        pixel_record = {
                            'image_path': image_path,
                            'category': category
                        }
                        
                        # Tambahkan semua nilai pixel
                        pixel_record.update(pixel_data)
                        main_output_pixel_data.append(pixel_record)
            
            if main_output_pixel_data:
                # Buat DataFrame dari data
                df = pd.DataFrame(main_output_pixel_data)
                
                # Simpan ke CSV
                csv_path = self.output_path / 'main_output_pixels_for_ml.csv'
                df.to_csv(csv_path, index=False)
                print(f"💾 Nilai per-pixel Main Output disimpan ke: {csv_path}")
                print(f"📊 Ukuran data: {len(df)} sampel x {len(df.columns)} fitur pixel")
                
            else:
                print("⚠️ Tidak ada data pixel main output yang dapat disimpan")
                
        except Exception as e:
            print(f"⚠️ Warning: Gagal menyimpan CSV nilai per-pixel Main Output: {e}")
            import traceback
            traceback.print_exc()

    def process_dataset(self):
        """Proses seluruh dataset"""
        print("🚀 Memulai pemrosesan dataset...")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"📁 Output path: {self.output_path}")
        print("=" * 60)
        
        # Buat struktur output
        if not self.create_output_structure():
            return False
        
        # Dapatkan semua gambar
        all_images = self.get_all_images()
        
        if not all_images:
            print("❌ Tidak ada gambar ditemukan dalam dataset!")
            return False
        
        print(f"\n📊 Total gambar ditemukan: {self.stats['total_images']}")
        print("📋 Distribusi per kategori:")
        for category, count in self.stats['categories'].items():
            print(f"   └─ {category}: {count} gambar")
        
        # Proses setiap gambar
        print(f"\n🔄 Memulai ekstraksi fitur...")
        results = []
        start_time = time.time()
        
        for i, (image_path, category_name) in enumerate(all_images, 1):
            print(f"\n[{i}/{len(all_images)}] Memproses: {Path(image_path).name}")
            print(f"   Kategori: {category_name}")
            
            result = self.process_single_image(image_path, category_name)
            results.append(result)
            
            if result['status'] == 'success':
                self.stats['processed'] += 1
                if 'processing_time' in result:
                    print(f"   ⏱️ Waktu pemrosesan: {result['processing_time']:.2f} detik")
                print(f"   ✅ Berhasil")
            else:
                self.stats['failed'] += 1
                print(f"   ❌ Gagal: {result['error']}")
          # Simpan fitur ke CSV
        self.save_features_to_csv(results)
        
        # Simpan nilai per-pixel dari main output ke CSV terpisah
        self.save_main_output_pixels_to_csv(results)
        
        # Simpan nilai per-pixel HOG ke CSV terpisah (jika ada)
        self.save_hog_pixels_to_csv(results)
        
        # Simpan nilai per-pixel main output ke CSV terpisah
        self.save_main_output_pixels_to_csv(results)
        
        # Hitung total waktu pemrosesan
        self.stats['processing_time'] = time.time() - start_time
          # Tampilkan ringkasan
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Tampilkan ringkasan hasil pemrosesan"""
        print("\n" + "=" * 60)
        print("📊 RINGKASAN PEMROSESAN DATASET")
        print("=" * 60)
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"📂 Output: {self.output_path}")
        print(f"🖼️  Total gambar: {self.stats['total_images']}")
        print(f"✅ Berhasil diproses: {self.stats['processed']}")
        print(f"❌ Gagal diproses: {self.stats['failed']}")
        if self.stats['total_images'] > 0:
            success_rate = (self.stats['processed']/self.stats['total_images']*100)
            print(f"📈 Tingkat keberhasilan: {success_rate:.1f}%")
        print(f"⏱️ Total waktu pemrosesan: {self.stats['processing_time']:.2f} detik")
        if self.stats['processed'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['processed']
            print(f"⏱️ Rata-rata waktu per gambar: {avg_time:.2f} detik")
        
        print(f"\n📋 Detail per kategori:")
        for category, total in self.stats['categories'].items():
            print(f"   └─ {category}: {total} gambar")
            
        print(f"\n📊 File CSV untuk machine learning:")
        print(f"   └─ {self.output_path}/dataset_features_for_ml.csv")
        print(f"   └─ {self.output_path}/main_output_pixels_for_ml.csv (NEW!)")
        print(f"   └─ {self.output_path}/hog_pixels_for_ml.csv")
        print(f"   └─ {self.output_path}/dataset_metadata.csv")

def main():
    """Fungsi utama"""
    if len(sys.argv) < 2:
        print("Penggunaan: python run_batch_extraction.py <path_dataset> [output_path]")
        print("Contoh: python run_batch_extraction.py ./dataset ./dataset_output")
        print("\nStruktur dataset yang diharapkan:")
        print("dataset/")
        print("├── kategori1/")
        print("│   ├── gambar1.jpg")
        print("│   ├── gambar2.jpg")
        print("│   └── ...")
        print("├── kategori2/")
        print("│   ├── gambar1.jpg")
        print("│   └── ...")
        print("└── ...")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_output"
    
    # Inisialisasi processor
    processor = DatasetFeatureProcessor(dataset_path, output_path)
    
    # Jalankan pemrosesan
    success = processor.process_dataset()
    
    if success:
        print("\n🎉 Pemrosesan dataset selesai! Data siap untuk machine learning!")
        print(f"📊 CSV untuk machine learning tersedia di: {output_path}")
        sys.exit(0)
    else:
        print("\n💥 Pemrosesan dataset gagal!")
        sys.exit(1)

if __name__ == "__main__":
    main()