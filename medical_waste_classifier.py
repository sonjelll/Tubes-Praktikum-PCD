import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from joblib import load

# Import ekstraksi fitur
from medical_waste_feature_extractor import MedicalWasteFeatureExtractor

class MedicalWasteImageClassifier:
    def __init__(self, model_folder="ml_models_hog_only"):
        """
        Inisialisasi classifier untuk gambar sampah medis
        
        Args:
            model_folder (str): Path ke folder model yang telah dilatih
        """
        self.model_folder = Path(model_folder)
        self.extractor = MedicalWasteFeatureExtractor()
        self.models = {}
        self.scaler = None
        self.classes = None
        
        # Load model dan scaler
        self.load_models()
    
    def load_models(self):
        """
        Muat model KNN dan RandomForest serta scaler dari folder model
        """
        try:
            # Load scaler
            scaler_path = self.model_folder / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = load(scaler_path)
                print(f"✅ Scaler dimuat dari: {scaler_path}")
            else:
                print(f"⚠️ Scaler tidak ditemukan di: {scaler_path}")
                return False
            
            # Load model KNN
            knn_path = self.model_folder / 'knn_model.joblib'
            if knn_path.exists():
                self.models['knn'] = load(knn_path)
                print(f"✅ Model KNN dimuat dari: {knn_path}")
            else:
                print(f"⚠️ Model KNN tidak ditemukan di: {knn_path}")
            
            # Load model RandomForest
            rf_path = self.model_folder / 'randomforest_model.joblib'
            if rf_path.exists():
                self.models['randomforest'] = load(rf_path)
                print(f"✅ Model RandomForest dimuat dari: {rf_path}")
            else:
                print(f"⚠️ Model RandomForest tidak ditemukan di: {rf_path}")
            
            # Pastikan setidaknya satu model berhasil dimuat
            if not self.models:
                print("❌ Tidak ada model yang berhasil dimuat!")
                return False
            
            # Ambil kelas dari salah satu model
            model = next(iter(self.models.values()))
            self.classes = model.classes_
            print(f"📊 Kelas yang dikenali: {self.classes}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saat memuat model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_hog_features(self, image_path):
        """
        Ekstrak fitur HOG dari gambar
        
        Args:
            image_path (str): Path ke gambar yang akan diklasifikasi
            
        Returns:
            dict: Fitur HOG yang diekstrak
        """
        try:
            # Proses gambar untuk mendapatkan fitur HOG
            features = self.extractor.process_image(image_path, use_lbp=False, use_glcm=False)
            
            # Ekstrak nilai per-pixel dari citra HOG
            hog_features = {}
            if 'hog' in features and 'image' in features['hog']:
                hog_image = features['hog']['image']
                # Flatten HOG image untuk mendapatkan nilai per-pixel
                hog_pixels = hog_image.flatten()
                # Batasi jumlah pixel yang disimpan (ambil sampel)
                pixel_sample_size = min(100, len(hog_pixels))
                pixel_indices = np.linspace(0, len(hog_pixels)-1, pixel_sample_size, dtype=int)
                for i, idx in enumerate(pixel_indices):
                    hog_features[f'hog_pixel_{i+1}'] = hog_pixels[idx]
            
            # Visualisasi HOG (opsional)
            if 'hog' in features and 'image' in features['hog']:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
                plt.title("Gambar Asli")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(features['hog']['image'], cmap='gray')
                plt.title("HOG Image")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{Path(image_path).stem}_hog_preview.png")
                plt.close()
            
            return hog_features
            
        except Exception as e:
            print(f"❌ Error saat ekstraksi fitur HOG: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def classify_image(self, image_path, model_type='randomforest'):
        """
        Klasifikasi gambar menggunakan model yang dipilih
        
        Args:
            image_path (str): Path ke gambar yang akan diklasifikasi
            model_type (str): Tipe model ('knn' atau 'randomforest')
            
        Returns:
            dict: Hasil klasifikasi
        """
        try:
            # Validasi model_type
            if model_type not in self.models:
                print(f"❌ Model {model_type} tidak tersedia. Pilihan: {list(self.models.keys())}")
                return None
            
            # Ekstrak fitur HOG
            print(f"🔄 Mengekstrak fitur HOG dari: {image_path}")
            hog_features = self.extract_hog_features(image_path)
            
            if not hog_features:
                print("❌ Gagal mengekstrak fitur HOG!")
                return None
            
            # Konversi fitur ke format yang sesuai untuk model
            features_df = pd.DataFrame([hog_features])
            
            # Normalisasi fitur
            features_scaled = self.scaler.transform(features_df)
            
            # Prediksi menggunakan model
            model = self.models[model_type]
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Buat hasil
            result = {
                'prediction': prediction,
                'probabilities': {cls: prob for cls, prob in zip(self.classes, probabilities)},
                'model_used': model_type,
                'image_path': image_path
            }
            
            # Tampilkan hasil
            print(f"\n✅ Hasil Klasifikasi:")
            print(f"   └─ Gambar: {Path(image_path).name}")
            print(f"   └─ Prediksi: {prediction}")
            print(f"   └─ Model: {model_type}")
            print(f"   └─ Probabilitas:")
            for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"      └─ {cls}: {prob*100:.2f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Error saat klasifikasi: {e}")
            import traceback
            traceback.print_exc()
            return None

def select_image():
    """
    Buka dialog untuk memilih gambar
    
    Returns:
        str: Path ke gambar yang dipilih
    """
    root = Tk()
    root.withdraw()  # Sembunyikan window utama
    
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Sampah Medis",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    
    return file_path if file_path else None

def select_model_folder():
    """
    Buka dialog untuk memilih folder model
    
    Returns:
        str: Path ke folder model yang dipilih
    """
    root = Tk()
    root.withdraw()  # Sembunyikan window utama
    
    folder_path = filedialog.askdirectory(
        title="Pilih Folder Model"
    )
    
    return folder_path if folder_path else "ml_models_hog_only"

def main():
    """
    Fungsi utama program
    """
    print("\n🔍 KLASIFIKASI SAMPAH MEDIS DENGAN HOG, KNN, DAN RANDOMFOREST")
    print("=" * 60)
    
    # Pilih folder model (opsional)
    print("\n📁 Pilih folder model (biarkan kosong untuk default: ml_models_hog_only)")
    model_folder = select_model_folder()
    
    # Inisialisasi classifier
    classifier = MedicalWasteImageClassifier(model_folder)
    
    # Pilih gambar untuk klasifikasi
    print("\n🖼️ Pilih gambar sampah medis untuk diklasifikasi")
    image_path = select_image()
    
    if not image_path:
        print("❌ Tidak ada gambar yang dipilih. Program berhenti.")
        return
    
    # Pilih model untuk klasifikasi
    available_models = list(classifier.models.keys())
    if not available_models:
        print("❌ Tidak ada model yang tersedia. Program berhenti.")
        return
    
    print(f"\n🤖 Pilih model klasifikasi:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")
    
    choice = input("Masukkan nomor pilihan (default: 1): ").strip()
    
    try:
        choice = int(choice) if choice else 1
        if choice < 1 or choice > len(available_models):
            choice = 1
    except ValueError:
        choice = 1
    
    model_type = available_models[choice - 1]
    print(f"🤖 Menggunakan model: {model_type}")
    
    # Klasifikasi gambar
    result = classifier.classify_image(image_path, model_type)
    
    if result:
        print("\n✅ Klasifikasi berhasil!")
        print(f"📊 Hasil prediksi: {result['prediction']}")
    else:
        print("\n❌ Klasifikasi gagal!")

if __name__ == "__main__":
    main()