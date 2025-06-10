import cv2
import numpy as np
from skimage import feature, filters, measure
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import matplotlib.pyplot as plt
from scipy import ndimage
import os

class MedicalWasteFeatureExtractor:
    def __init__(self):
        self.features = {}
        
    def color_preprocessing(self, image):
        """
        Ekstraksi fitur warna: konversi ke grayscale, peningkatan kontras, dan reduksi noise
        """
        # Konversi ke grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Peningkatan kontras menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Reduksi noise menggunakan Gaussian blur
        denoised = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)
        
        # Reduksi noise tambahan menggunakan bilateral filter
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        self.features['color_processed'] = denoised
        return denoised
    
    def shape_features(self, image):
        """
        Ekstraksi fitur bentuk: Hu moments, aspect ratio, dan analisis kontur
        """
        # Threshold untuk mendapatkan binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfologi untuk membersihkan noise
        kernel = np.ones((3,3), np.uint8)  # Kembali ke kernel 3x3 seperti semula
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Cari kontur
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Kembali ke CHAIN_APPROX_SIMPLE
        
        shape_features = []
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if contours:
            # Ambil kontur terbesar
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Buat mask dari kontur terbesar
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Hu Moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Aspect Ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h
            
            # Area dan Perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
                
            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
            
            shape_features = {
                'hu_moments': hu_moments,
                'aspect_ratio': aspect_ratio,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'solidity': solidity,
                'contour': largest_contour,
                'mask': mask  # Tambahkan mask untuk background removal
            }
        
        self.features['shape'] = shape_features
        return shape_features
    
    def texture_features_lbp(self, image, radius=3, n_points=24):
        """
        Ekstraksi fitur tekstur menggunakan Local Binary Pattern (LBP)
        """
        # Hitung LBP
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Hitung histogram LBP
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalisasi histogram
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return lbp, hist
    
    def texture_features_glcm(self, image, distances=[1], angles=[0, 45, 90, 135]):
        """
        Ekstraksi fitur tekstur menggunakan Gray Level Co-occurrence Matrix (GLCM)
        """
        # Konversi sudut ke radian
        angles_rad = [np.radians(angle) for angle in angles]
        
        # Hitung GLCM
        glcm = graycomatrix(image, distances=distances, angles=angles_rad, 
                           levels=256, symmetric=True, normed=True)
        
        # Ekstrak properti GLCM
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        glcm_features = {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation
        }
        
        return glcm_features
    
    def hog_features(self, image, orientations=9, pixels_per_cell=(8, 8), 
                     cells_per_block=(2, 2), visualize=True):
        """
        Ekstraksi fitur HOG (Histogram of Oriented Gradients)
        """
        if visualize:
            hog_features, hog_image = hog(image, orientations=orientations,
                                        pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block,
                                        visualize=True, transform_sqrt=True)
            
            # Meningkatkan visibilitas garis HOG
            # Normalisasi gambar HOG ke rentang 0-1
            hog_image_normalized = hog_image / np.max(hog_image)
            
            # Meningkatkan kontras dengan power-law transformation (gamma correction)
            gamma = 0.5  # Nilai gamma < 1 akan meningkatkan kontras area gelap
            hog_image_enhanced = np.power(hog_image_normalized, gamma)
            
            # Meningkatkan ketajaman garis dengan unsharp masking
            blurred = ndimage.gaussian_filter(hog_image_enhanced, sigma=1.0)
            hog_image_sharpened = hog_image_enhanced + 1.5 * (hog_image_enhanced - blurred)
            
            # Clip nilai untuk memastikan tetap dalam rentang 0-1
            hog_image_sharpened = np.clip(hog_image_sharpened, 0, 1)
            
            return hog_features, hog_image_sharpened
        else:
            hog_features = hog(image, orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block,
                             visualize=False, transform_sqrt=True)
            return hog_features
    
    def process_image(self, image_path, use_lbp=True, use_glcm=False):  # Ubah default use_glcm menjadi False
        """
        Proses lengkap ekstraksi fitur dari gambar sampah medis
        """
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        print(f"Memproses gambar: {image_path}")
        
        # 1. Preprocessing warna
        print("1. Melakukan preprocessing warna...")
        processed_image = self.color_preprocessing(image)
        
        # 2. Ekstraksi fitur bentuk
        print("2. Mengekstrak fitur bentuk...")
        shape_features = self.shape_features(processed_image)
        
        # Gunakan gambar yang sudah diproses untuk ekstraksi fitur selanjutnya
        processed_image_for_features = processed_image
        
        # 3. Ekstraksi fitur tekstur
        print("3. Mengekstrak fitur tekstur...")
        texture_features = {}
        
        if use_lbp:
            lbp_image, lbp_hist = self.texture_features_lbp(processed_image_for_features)
            texture_features['lbp'] = {
                'image': lbp_image,
                'histogram': lbp_hist
            }
        
        # Hapus bagian GLCM
        # if use_glcm:
        #     glcm_features = self.texture_features_glcm(processed_image_for_features)
        #     texture_features['glcm'] = glcm_features
        
        self.features['texture'] = texture_features
        
        # 4. Ekstraksi fitur HOG
        print("4. Mengekstrak fitur HOG...")
        hog_features, hog_image = self.hog_features(processed_image_for_features)
        self.features['hog'] = {
            'features': hog_features,
            'image': hog_image
        }
        
        return self.features
    
    def visualize_and_save(self, original_image_path, output_folder=".", create_full_visualization=True):
        """
        Visualisasi dan simpan hasil ekstraksi fitur
        
        Args:
            original_image_path (str): Path ke gambar asli
            output_folder (str): Folder untuk menyimpan hasil visualisasi
            create_full_visualization (bool): Jika True, buat file features_extracted.png
        """
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
            # Buat subplot dengan ukuran yang sesuai - ubah menjadi 2x3 untuk menampilkan semua fitur
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                
            fig.suptitle('Ekstraksi Fitur Sampah Medis', fontsize=16)
            
            # Gambar asli
            axes[0, 0].imshow(original_rgb)
            axes[0, 0].set_title('Gambar Asli')
            axes[0, 0].axis('off')
            
            # Hasil preprocessing warna
            axes[0, 1].imshow(self.features['color_processed'], cmap='gray')
            axes[0, 1].set_title('Preprocessing Warna')
            axes[0, 1].axis('off')
            
            # Fitur bentuk (mask)
            if 'shape' in self.features and 'mask' in self.features['shape']:
                axes[0, 2].imshow(self.features['shape']['mask'], cmap='gray')
                axes[0, 2].set_title('Fitur Bentuk (Mask)')
                axes[0, 2].axis('off')
            
            # Fitur tekstur LBP
            if 'texture' in self.features and 'lbp' in self.features['texture']:
                axes[1, 0].imshow(self.features['texture']['lbp']['image'], cmap='gray')
                axes[1, 0].set_title('Fitur LBP')
                axes[1, 0].axis('off')
            
            # HOG
            if 'hog' in self.features:
                axes[1, 1].imshow(self.features['hog']['image'], cmap='gray')
                axes[1, 1].set_title('HOG Features')
                axes[1, 1].axis('off')
            
            # Kosongkan subplot yang tidak digunakan
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Simpan hasil visualisasi
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_features_extracted.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Hasil visualisasi disimpan di: {output_path}")
            
        return hog_output_path if hog_output_path else None
    
    def print_feature_summary(self):
        """
        Cetak ringkasan fitur yang diekstrak
        """
        print("\n=== RINGKASAN FITUR YANG DIEKSTRAK ===")
        
        # Fitur bentuk
        if 'shape' in self.features and self.features['shape']:
            print("\nFitur Bentuk:")
            shape = self.features['shape']
            print(f"  - Aspect Ratio: {shape['aspect_ratio']:.4f}")
            print(f"  - Area: {shape['area']:.2f}")
            print(f"  - Perimeter: {shape['perimeter']:.2f}")
            print(f"  - Circularity: {shape['circularity']:.4f}")
            print(f"  - Solidity: {shape['solidity']:.4f}")
            print(f"  - Hu Moments: {shape['hu_moments'][:3]}...")  # Tampilkan 3 pertama
        
        # Hapus bagian GLCM
        # if 'texture' in self.features and 'glcm' in self.features['texture']:
        #     print("\nFitur Tekstur (GLCM):")
        #     glcm = self.features['texture']['glcm']
        #     for key, value in glcm.items():
        #         print(f"  - {key.capitalize()}: {value:.4f}")
        
        # Fitur HOG
        if 'hog' in self.features:
            print(f"\nFitur HOG:")
            print(f"  - Jumlah fitur: {len(self.features['hog']['features'])}")
            print(f"  - Rata-rata nilai: {np.mean(self.features['hog']['features']):.4f}")
            print(f"  - Standar deviasi: {np.std(self.features['hog']['features']):.4f}")


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
        features = extractor.process_image(image_path, use_lbp=True, use_glcm=True)
        
        # Visualisasi dan simpan hasil
        output_path = extractor.visualize_and_save(image_path)
        
        # Cetak ringkasan fitur
        extractor.print_feature_summary()
        
        print(f"\nProses selesai! Hasil disimpan di folder saat ini.")
        
    except Exception as e:
        print(f"Error saat memproses gambar: {str(e)}")

if __name__ == "__main__":
    main()