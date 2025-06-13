import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog, Tk
from joblib import load

# Import ekstraksi fitur
from medical_waste_feature_extractor import MedicalWasteFeatureExtractor

class MedicalWasteImageClassifier:
    def __init__(self, model_path="ml_models"):
        """
        Inisialisasi classifier untuk gambar sampah medis
        
        Args:
            model_path (str): Path ke folder model yang telah dilatih
        """
        self.model_path = Path(model_path)
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
            scaler_path = self.model_path / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = load(scaler_path)
                print(f"✅ Scaler dimuat dari: {scaler_path}")
            else:
                print(f"⚠️ Scaler tidak ditemukan di: {scaler_path}")
                return False
            
            # Load model KNN
            knn_path = self.model_path / 'knn_model.joblib'
            if knn_path.exists():
                self.models['knn'] = load(knn_path)
                print(f"✅ Model KNN dimuat dari: {knn_path}")
            else:
                print(f"⚠️ Model KNN tidak ditemukan di: {knn_path}")
            
            # Load model RandomForest
            rf_path = self.model_path / 'randomforest_model.joblib'
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
        
    def classify_image(self, image_path, model_type='knn', show_hog=True):
        """
        Klasifikasi gambar menggunakan model yang dipilih
        
        Args:
            image_path (str): Path ke gambar yang akan diklasifikasi
            model_type (str): Tipe model ('knn' atau 'randomforest')
            show_hog (bool): Tampilkan visualisasi HOG
            
        Returns:
            dict: Hasil klasifikasi
        """
        try:
            # Validasi model_type
            if model_type not in self.models:
                print(f"❌ Model {model_type} tidak tersedia. Pilihan: {list(self.models.keys())}")
                return None
            
            # Ekstrak fitur dengan alur preprocessing->lbp->hog menggunakan preprocessing manual
            print(f"🔄 Memproses gambar: {image_path}")
            
            # Baca gambar
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Gambar tidak ditemukan: {image_path}")
            
            print("1. Melakukan preprocessing warna manual...")
            # Preprocessing manual seperti di medical_waste_feature_extractor
            processed_image = self.color_preprocessing(image)
            
            print("2. Mengekstrak fitur tekstur LBP...")
            print("3. Mengekstrak fitur HOG...")
            
            # Gunakan ekstraksi fitur dari extractor dengan gambar yang sudah dipreprocess manual
            # Bypass preprocessing di extractor dan langsung ekstrak LBP dan HOG
            lbp_features = self.extractor.texture_features_lbp(processed_image)
            hog_features, hog_image = self.extractor.hog_features(processed_image)
            
            # Gabungkan fitur seperti di extractor
            features = {
                'color_processed': processed_image,
                'texture': {'lbp': lbp_features},
                'hog': {'features': hog_features, 'image': hog_image}
            }
            
            # Gabungkan fitur untuk klasifikasi - HANYA GUNAKAN HOG
            combined_features = {}
            
            if 'hog' in features and 'image' in features['hog']:
                hog_image = features['hog']['image']
                # Flatten HOG image untuk mendapatkan nilai per-pixel
                hog_pixels = hog_image.flatten()
                # Batasi jumlah pixel yang disimpan (ambil sampel)
                pixel_sample_size = min(100, len(hog_pixels))
                pixel_indices = np.linspace(0, len(hog_pixels)-1, pixel_sample_size, dtype=int)
                for i, idx in enumerate(pixel_indices):
                    combined_features[f'hog_pixel_{i+1}'] = hog_pixels[idx]
            
            if not combined_features:
                print("❌ Gagal mengekstrak fitur HOG!")
                return None
            
            # Tampilkan visualisasi jika diminta
            if show_hog:
                # Baca gambar asli untuk ditampilkan
                original_img = cv2.imread(image_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Siapkan visualisasi dengan 3 panel: gambar asli, LBP, dan HOG
                plt.figure(figsize=(18, 6))
                
                # Gambar asli
                plt.subplot(1, 3, 1)
                plt.imshow(original_img)
                plt.title("Gambar Asli")
                plt.axis('off')
                
                # Visualisasi LBP jika tersedia
                if 'texture' in features and 'lbp' in features['texture'] and 'image' in features['texture']['lbp']:
                    plt.subplot(1, 3, 2)
                    plt.imshow(features['texture']['lbp']['image'])
                    plt.title("Fitur LBP")
                    plt.axis('off')
                
                # Visualisasi HOG
                plt.subplot(1, 3, 3)
                plt.imshow(features['hog']['image'], cmap='gray')
                plt.title("Visualisasi HOG")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # Konversi fitur ke format yang sesuai untuk model
            features_df = pd.DataFrame([combined_features])
            
            # Normalisasi fitur
            features_scaled = self.scaler.transform(features_df)
            
            # Prediksi menggunakan model
            model = self.models[model_type]
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Cek apakah probabilitas tertinggi kurang dari 50%
            max_probability = max(probabilities)
            if max_probability < 0.5:
                # Jika semua probabilitas kurang dari 50%, ubah prediksi menjadi bukan sampah medis
                original_prediction = prediction
                prediction = 'bukan sampah medis'
                print(f"ℹ️ Probabilitas tertinggi ({max_probability*100:.2f}%) kurang dari 50%, diklasifikasikan sebagai bukan sampah medis")
            
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
    
    def histogram_equalization(self, image):
        """
        Manual implementation of Histogram Equalization
        
        Args:
            image: Input grayscale image
            
        Returns:
            Histogram equalized image
        """
        # Hitung histogram dari gambar
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # Hitung CDF (Cumulative Distribution Function)
        cdf = np.cumsum(hist)
        
        # Normalisasi CDF ke rentang 0-255
        # Rumus: new_value = (cdf[old_value] - cdf_min) * 255 / (cdf_max - cdf_min)
        cdf_min = cdf[cdf > 0].min()  # Ambil nilai minimum yang tidak nol
        cdf_max = cdf.max()
        
        # Buat lookup table untuk transformasi
        cdf_normalized = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if cdf[i] > 0:
                cdf_normalized[i] = np.round((cdf[i] - cdf_min) * 255.0 / (cdf_max - cdf_min))
            else:
                cdf_normalized[i] = 0        # Terapkan transformasi pada gambar
        equalized_image = cdf_normalized[image]
        
        return equalized_image.astype(np.uint8)
    
    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        """
        Manual implementation of Gaussian Blur
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation for Gaussian kernel
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution
        blurred = self.convolve2d(image, kernel)
        
        # Handle different data types appropriately
        if image.dtype == np.float64 or image.dtype == np.float32:
            return blurred.astype(image.dtype)
        else:
            return blurred.astype(np.uint8)
    
    def create_gaussian_kernel(self, size, sigma):
        """
        Create Gaussian kernel manually
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Calculate Gaussian values
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def convolve2d(self, image, kernel):
        """
        Manual 2D convolution
        """
        # Get dimensions
        image_h, image_w = image.shape
        kernel_h, kernel_w = kernel.shape
        
        # Calculate padding
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2
        
        # Pad image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # Initialize output
        output = np.zeros_like(image, dtype=np.float64)
        
        # Perform convolution
        for i in range(image_h):
            for j in range(image_w):
                # Extract region
                region = padded_image[i:i+kernel_h, j:j+kernel_w]
                  # Apply kernel
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    def color_preprocessing(self, image):
        """
        Manual color preprocessing: grayscale conversion, histogram equalization, dan gaussian blur
        Menggunakan implementasi manual tanpa CLAHE dan bilateral filter
        """
        # Konversi ke grayscale
        if len(image.shape) == 3:
            # Manual RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            gray = np.dot(image[...,:3], [0.114, 0.587, 0.299])  # BGR to grayscale
            gray = gray.astype(np.uint8)
        else:
            gray = image.copy()
            print("   - Mengkonversi ke grayscale secara manual...")
        
        # Peningkatan kontras menggunakan manual histogram equalization
        print("   - Menerapkan histogram equalization manual...")
        contrast_enhanced = self.histogram_equalization(gray)
        
        # Reduksi noise menggunakan manual Gaussian blur
        print("   - Menerapkan Gaussian blur manual...")
        denoised = self.gaussian_blur(contrast_enhanced, kernel_size=5, sigma=1.0)
        
        return denoised
    
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

def main():
    """
    Fungsi utama program
    """
    print("\n🔍 KLASIFIKASI SAMPAH MEDIS DENGAN HOG, KNN, DAN RANDOMFOREST")
    print("=" * 60)
    
    # Gunakan path model default
    model_path = "ml_models"
    
    # Inisialisasi classifier
    classifier = MedicalWasteImageClassifier(model_path)
    
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
    
    # Klasifikasi gambar dengan menampilkan visualisasi HOG
    result = classifier.classify_image(image_path, model_type, show_hog=True)
    
    if result:
        print("\n✅ Klasifikasi berhasil!")
        print(f"📊 Hasil prediksi: {result['prediction']}")
    else:
        print("\n❌ Klasifikasi gagal!")

if __name__ == "__main__":
    main()