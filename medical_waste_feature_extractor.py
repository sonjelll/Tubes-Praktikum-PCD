import cv2
import numpy as np
from skimage import feature, filters, measure
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt
import os

class MedicalWasteFeatureExtractor:
    def __init__(self):
        self.features = {}
        
    def histogram_equalization(self, image):
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        cdf = np.cumsum(hist)
        
        cdf_min = cdf[cdf > 0].min()
        cdf_max = cdf.max()
        
        cdf_normalized = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if cdf[i] > 0:
                cdf_normalized[i] = np.round((cdf[i] - cdf_min) * 255.0 / (cdf_max - cdf_min))
            else:
                cdf_normalized[i] = 0
        equalized_image = cdf_normalized[image]
        
        return equalized_image.astype(np.uint8)
    
    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        blurred = self.convolve2d(image, kernel)
        
        if image.dtype == np.float64 or image.dtype == np.float32:
            return blurred.astype(image.dtype)
        else:
            return blurred.astype(np.uint8)
    
    def create_gaussian_kernel(self, size, sigma):
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def convolve2d(self, image, kernel):
        image_h, image_w = image.shape
        kernel_h, kernel_w = kernel.shape
        
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2
        
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        output = np.zeros_like(image, dtype=np.float64)
        
        for i in range(image_h):
            for j in range(image_w):
                region = padded_image[i:i+kernel_h, j:j+kernel_w]
              
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    def color_preprocessing(self, image):
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
        
        self.features['color_processed'] = denoised
        return denoised
    
    def texture_features_lbp(self, image, radius=3, n_points=24, method='uniform'):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lbp = local_binary_pattern(image, n_points, radius, method)
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Buat visualisasi LBP
        # Normalisasi LBP untuk visualisasi
        lbp_image = (lbp * (255.0 / (n_bins - 1))).astype(np.uint8)
        
        # Buat colormap untuk visualisasi yang lebih baik
        lbp_colored = cv2.applyColorMap(lbp_image, cv2.COLORMAP_JET)
        
        return {
            'lbp': lbp,
            'histogram': hist,
            'image': lbp_colored
        }
    
    def hog_features(self, image, orientations=9, pixels_per_cell=(8, 8), 
                     cells_per_block=(2, 2), visualize=True):
        if visualize:
            hog_features, hog_image = hog(image, orientations=orientations,
                                        pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block,
                                        visualize=True, transform_sqrt=True)
            

            hog_image_normalized = hog_image / np.max(hog_image)
            
            gamma = 0.5  
            hog_image_enhanced = np.power(hog_image_normalized, gamma)
            blurred = self.gaussian_blur(hog_image_enhanced, kernel_size=5, sigma=1.0)
            hog_image_sharpened = hog_image_enhanced + 1.5 * (hog_image_enhanced - blurred)
            
            hog_image_sharpened = np.clip(hog_image_sharpened, 0, 1)
            
            return hog_features, hog_image_sharpened
        else:
            hog_features = hog(image, orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block,
                             visualize=False, transform_sqrt=True)
            return hog_features
    
    def process_image(self, image_path, use_lbp=True, use_hog=True):
        """
        Proses gambar dan ekstrak semua fitur
        
        Args:
            image_path (str): Path ke gambar yang akan diproses
            use_lbp (bool): Jika True, ekstrak fitur tekstur LBP
            use_hog (bool): Jika True, ekstrak fitur HOG
            
        Returns:
            dict: Dictionary berisi semua fitur yang diekstrak
        """
        # Reset fitur
        self.features = {}
        
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak ditemukan: {image_path}")
        
        # 1. Preprocessing warna
        print("1. Melakukan preprocessing warna...")
        processed_image = self.color_preprocessing(image)
        self.features['color_processed'] = processed_image
        
        # Gunakan hasil preprocessing untuk ekstraksi fitur selanjutnya
        processed_image_for_features = processed_image.copy()
        
        # 2. Ekstraksi fitur tekstur LBP
        if use_lbp:
            print("2. Mengekstrak fitur tekstur LBP...")
            texture_features = {}
            lbp_features = self.texture_features_lbp(processed_image_for_features)
            texture_features['lbp'] = lbp_features
            self.features['texture'] = texture_features
        
        # 3. Ekstraksi fitur HOG
        if use_hog:
            print("3. Mengekstrak fitur HOG...")
            hog_features, hog_image = self.hog_features(processed_image_for_features)
            self.features['hog'] = {
                'features': hog_features,
                'image': hog_image
            }
        
        return self.features
    
    def visualize_and_save(self, original_image_path, output_folder=".", create_full_visualization=True):
        # Baca gambar asli
        original = cv2.imread(original_image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Simpan HOG terpisah
        hog_output_path = None
        if 'hog' in self.features:
            # Buat figure baru dengan ukuran yang lebih besar
            plt.figure(figsize=(12, 10))
            
            # Tampilkan HOG dengan colormap yang lebih kontras
            plt.imshow(self.features['hog']['image'], cmap='gray')
            plt.axis('off')
            
            # Simpan hasil visualisasi HOG
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            hog_output_path = os.path.join(output_folder, f"{base_name}_hog.png")
            plt.savefig(hog_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualisasi HOG disimpan di: {hog_output_path}")
        else:
            print("Fitur HOG tidak ditemukan dalam hasil ekstraksi.")
        
        # Hanya buat visualisasi lengkap jika diminta
        if create_full_visualization:
            # Buat subplot dengan ukuran yang sesuai - ubah menjadi 2x2 untuk menampilkan semua fitur
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
            fig.suptitle('Ekstraksi Fitur Sampah Medis', fontsize=16)
            
            # Gambar asli
            axes[0, 0].imshow(original_rgb)
            axes[0, 0].set_title('Gambar Asli')
            axes[0, 0].axis('off')
            
            # Hasil preprocessing warna
            axes[0, 1].imshow(self.features['color_processed'], cmap='gray')
            axes[0, 1].set_title('Preprocessing Warna')
            axes[0, 1].axis('off')
            
            # Fitur tekstur LBP
            if 'texture' in self.features and 'lbp' in self.features['texture'] and 'image' in self.features['texture']['lbp']:
                axes[1, 0].imshow(self.features['texture']['lbp']['image'])
                axes[1, 0].set_title('Fitur LBP')
                axes[1, 0].axis('off')
            
            # HOG
            if 'hog' in self.features:
                axes[1, 1].imshow(self.features['hog']['image'], cmap='gray')
                axes[1, 1].set_title('HOG Features')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Simpan hasil visualisasi
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_features_extracted.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Hasil visualisasi disimpan di: {output_path}")
            
        return hog_output_path if hog_output_path else None



def main():
    """
    Fungsi utama untuk menjalankan ekstraksi fitur
    """
    # Inisialisasi extractor
    extractor = MedicalWasteFeatureExtractor()
    
    # Input path gambar
    image_path = input("Masukkan path gambar sampah medis: ").strip()
    
    # Cek apakah file ada
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} tidak ditemukan!")
        return
    
    try:
        # Proses gambar
        features = extractor.process_image(image_path)
        
        # Visualisasi dan simpan hasil
        output_path = extractor.visualize_and_save(image_path)
        
        
        print(f"\nProses selesai! Hasil disimpan di folder saat ini.")
        
    except Exception as e:
        print(f"Error saat memproses gambar: {str(e)}")

if __name__ == "__main__":
    main()