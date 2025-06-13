import cv2
import numpy as np
from skimage.feature import local_binary_pattern
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
    
    def hsv_based_segmentation(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definisikan range HSV untuk objek putih/terang (seperti kapas, perban)
        # Range untuk putih: H=any, S=low, V=high
        lower_white = np.array([0, 0, 180])    # Saturation rendah, Value tinggi
        upper_white = np.array([180, 60, 255])  # Toleransi untuk berbagai kondisi pencahayaan
        
        # Buat mask untuk objek putih
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Tambahan mask untuk objek terang lainnya
        lower_light = np.array([0, 0, 160])
        upper_light = np.array([180, 80, 255])
        light_mask = cv2.inRange(hsv, lower_light, upper_light)
        
        # Gabungkan mask
        combined_mask = cv2.bitwise_or(white_mask, light_mask)
        
        # Morphological operations untuk membersihkan mask
        kernel = np.ones((5,5), np.uint8)
        
        # Opening untuk menghapus noise kecil
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Closing untuk mengisi gap dalam objek
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Dilasi untuk memperbesar area objek
        cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)
        
        # Aplikasikan mask ke gambar asli
        cleaned_image = cv2.bitwise_and(image, image, mask=cleaned_mask)        # Konversi ke grayscale untuk processing selanjutnya
        gray_cleaned = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
        
        # Hanya hapus border tepi yang sangat tipis (1-2 pixel) 
        # untuk menghindari menghapus objek kapas
        gray_cleaned[0:2, :] = 0      # Top
        gray_cleaned[-2:, :] = 0     # Bottom
        gray_cleaned[:, 0:2] = 0     # Left
        gray_cleaned[:, -2:] = 0     # Right
        
        # Hitung area objek dan background
        object_area = np.sum(cleaned_mask > 0)
        total_area = cleaned_mask.shape[0] * cleaned_mask.shape[1]
        background_area = total_area - object_area
        
        # Ekstrak kontur objek
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Hitung properties objek
        object_properties = {}
        if contours:
            # Ambil kontur terbesar (objek utama)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Hitung properties
            object_properties = {
                'area': cv2.contourArea(largest_contour),
                'perimeter': cv2.arcLength(largest_contour, True),
                'contour_count': len(contours),
                'bounding_rect': cv2.boundingRect(largest_contour)
            }
              # Hitung compactness (keliling^2 / area)
            if object_properties['area'] > 0:
                object_properties['compactness'] = (object_properties['perimeter'] ** 2) / object_properties['area']
            else:
                object_properties['compactness'] = 0
        
        return {
            'mask': cleaned_mask,
            'cleaned_image': gray_cleaned,
            'original_with_mask': cleaned_image,
            'object_area': object_area,
            'background_area': background_area,
            'area_ratio': object_area / total_area if total_area > 0 else 0,
            'object_properties': object_properties,
            'contours': contours
        }
    
    def color_preprocessing(self, image, remove_background=True):
        """
        Preprocessing menggunakan HSV color space untuk segmentasi objek yang lebih baik
        """
        if len(image.shape) == 3:
            # Konversi ke HSV untuk segmentasi yang lebih baik
            print("   - Mengkonversi ke HSV color space...")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Ekstrak channel Value (brightness) untuk analisis intensitas
            value_channel = hsv[:, :, 2]
            
            # Ekstrak channel Saturation untuk deteksi objek
            saturation_channel = hsv[:, :, 1]
            
            # Simpan informasi HSV
            self.features['hsv_channels'] = {
                'hue': hsv[:, :, 0],
                'saturation': saturation_channel,
                'value': value_channel,
                'hsv_full': hsv
            }
            
            # Gunakan Value channel sebagai basis untuk segmentasi objek putih
            gray = value_channel.copy()
        else:
            gray = image.copy()
            
        # Background removal menggunakan segmentasi HSV yang lebih canggih
        if remove_background:
            print("   - Menghapus background menggunakan HSV-based segmentation...")
            bg_removal_result = self.hsv_based_segmentation(image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            self.features['background_removal'] = bg_removal_result
            processed_image = bg_removal_result['cleaned_image']
        else:
            processed_image = gray
        
        # Peningkatan kontras menggunakan CLAHE untuk hasil yang lebih baik
        print("   - Menerapkan CLAHE untuk peningkatan kontras...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(processed_image)
        
        # Reduksi noise menggunakan bilateral filter untuk mempertahankan edges
        print("   - Menerapkan bilateral filter untuk noise reduction...")
        denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)        
        self.features['color_processed'] = denoised
        return denoised
    
    def texture_features_lbp(self, image, radius=3, n_points=24, method='uniform'):
        # Pastikan input adalah grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pastikan input image berukuran 256x256 tanpa mengubah aspek ratio
        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Lakukan LBP pada gambar yang sudah dibersihkan
        lbp = local_binary_pattern(image, n_points, radius, method)
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)        # Normalisasi LBP untuk visualisasi (0-255)
        lbp_normalized = (lbp * (255.0 / (n_bins - 1))).astype(np.uint8)
        
        # Buat colormap untuk visualisasi yang lebih baik
        lbp_colored = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_JET)
        
        # Pastikan output tetap 256x256 tanpa border
        if lbp_colored.shape[:2] != (256, 256):
            lbp_colored = cv2.resize(lbp_colored, (256, 256), interpolation=cv2.INTER_AREA)
        
        return {
            'lbp': lbp,
            'histogram': hist,
            'image': lbp_colored  # Sudah 256x256 tanpa border
        }
    
    def process_image(self, image_path, use_lbp=True, use_edge_detection=True, remove_background=True):
        """
        Proses gambar dengan urutan: Preprocessed → LBP → Canny Edge Detection
        Semua output berukuran 256x256
        
        Args:
            image_path (str): Path ke gambar yang akan diproses
            use_lbp (bool): Jika True, ekstrak fitur tekstur LBP
            use_edge_detection (bool): Jika True, ekstrak fitur edge detection (output utama)
            remove_background (bool): Jika True, hapus background sebelum ekstraksi fitur
            
        Returns:
            dict: Dictionary berisi semua fitur yang diekstrak dengan fokus pada edge detection
        """
        # Reset fitur
        self.features = {}
        
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak ditemukan: {image_path}")
        
        # Resize gambar ke ukuran standar 256x256
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        
        print(f"📸 Memproses gambar: {image_path} (resized to 256x256)")
        
        # 1. Preprocessing warna dengan HSV-based segmentation
        print("1. Melakukan HSV-based preprocessing dan segmentasi objek...")
        processed_image = self.color_preprocessing(image, remove_background=remove_background)
        self.features['color_processed'] = processed_image        # 2. Ekstraksi fitur tekstur LBP terlebih dahulu
        if use_lbp:
            print("2. Mengekstrak fitur tekstur LBP...")
            texture_features = {}
            
            # Gunakan gambar yang sudah dibersihkan dari background removal
            if 'background_removal' in self.features and self.features['background_removal']:
                # Gunakan cleaned_image dari HSV segmentation yang sudah menghilangkan background
                cleaned_for_lbp = self.features['background_removal']['cleaned_image']
            else:
                cleaned_for_lbp = processed_image
            
            lbp_features = self.texture_features_lbp(cleaned_for_lbp)
            texture_features['lbp'] = lbp_features
            self.features['texture'] = texture_features    
        if use_edge_detection:
            print("3. Mengekstrak fitur Canny edge detection (OUTPUT UTAMA)...")
            
            # Gunakan optimal Canny edge detection
            canny_results = self.optimal_canny_edge_detection(processed_image)
            
            self.features['canny_edge_detection'] = canny_results
            # Set sebagai output utama
            self.features['main_output'] = canny_results['edges']
        
        print("✅ Ekstraksi fitur selesai!")
        return self.features

    def visualize_and_save(self, original_image_path, output_folder=".", create_full_visualization=True):
        """
        Visualisasi dan simpan hasil ekstraksi fitur dengan fokus pada Canny edge detection
        """
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
          # Buat visualisasi step-by-step
        step_by_step_path = self.visualize_step_by_step(original_image_path, output_folder, base_name)
        
        # Simpan Main Output sebagai file terpisah
        main_output_path = None
        if 'main_output' in self.features:
            main_output_path = os.path.join(output_folder, f"{base_name}_main_output.png")
            cv2.imwrite(main_output_path, self.features['main_output'])
            
            print(f"✅ Main output disimpan di: {main_output_path}")
        else:
            print("⚠️ Main output tidak ditemukan dalam hasil ekstraksi.")
        
        return main_output_path

    def optimal_canny_edge_detection(self, image):
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Hitung threshold otomatis berdasarkan histogram
        mean_intensity = np.mean(blurred[blurred > 0])
        
        # Threshold adaptif berdasarkan intensitas rata-rata objek
        if mean_intensity > 150:  # Objek sangat terang (seperti kapas putih)
            low_threshold = int(mean_intensity * 0.3)
            high_threshold = int(mean_intensity * 0.6)
        elif mean_intensity > 100:  # Objek sedang
            low_threshold = int(mean_intensity * 0.4)
            high_threshold = int(mean_intensity * 0.8)
        else:  # Objek gelap atau kontras rendah
            low_threshold = 30
            high_threshold = 80
        
        # Batasi threshold dalam range yang masuk akal
        low_threshold = max(20, min(low_threshold, 100))
        high_threshold = max(50, min(high_threshold, 200))     
        
        # 3. Canny edge detection dengan threshold adaptif
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
          
        
        # 4. Ekstrak kontur untuk analisis bentuk dari raw edges
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter kontur berdasarkan area minimum (menghapus noise)
        min_area = 100 
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # 6. Buat visualisasi dengan kontur menggunakan raw edges
        edge_visualization = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if filtered_contours:
            cv2.drawContours(edge_visualization, filtered_contours, -1, (0, 255, 0), 2)
        
        # 8. Ekstrak features untuk machine learning menggunakan raw edges
        edge_features = self.extract_canny_features(edges, filtered_contours)
        
        # 9. Hitung statistik edge detection menggunakan raw edges
        edge_stats = {
            'total_edge_pixels': np.sum(edges > 0),
            'edge_density': np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]),
            'mean_edge_intensity': np.mean(edges[edges > 0]) if np.any(edges > 0) else 0,
            'contour_count': len(filtered_contours),
            'threshold_used': {'low': low_threshold, 'high': high_threshold}
        }
        
        return {
            'edges': edges,                          # Output utama: raw canny edges
            'raw_edges': edges,                      # Raw Canny output
            'edge_visualization': edge_visualization,  # Visualisasi dengan kontur
            'contours': filtered_contours,           # Kontur yang sudah difilter
            'edge_features': edge_features,          # Features untuk ML
            'edge_stats': edge_stats,               # Statistik edge detection
            'preprocessing_info': {
                'mean_intensity': mean_intensity,
                'blur_applied': True,
                'morphological_ops': False           # Tidak menggunakan morphological ops
            }
        }
    def extract_canny_features(self, edges, contours):
        """
        Ekstrak nilai pixel dari citra biner Canny edge detection untuk machine learning
        
        Args:
            edges: Binary edge image dari Canny (256x256)
            contours: List of filtered contours (tidak digunakan)
            
        Returns:
            dict: Nilai pixel dalam format yang siap untuk machine learning
        """
        # Pastikan ukuran gambar adalah 256x256
        if edges.shape != (256, 256):
            edges = cv2.resize(edges, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Flatten citra biner menjadi array 1D (65536 pixels)
        pixels_flat = edges.flatten()
        
        # Normalisasi ke nilai 0 dan 1 (biner)
        pixels_binary = (pixels_flat > 0).astype(np.uint8)
        
        # Buat dictionary dengan nilai pixel sebagai features
        pixel_features = {}
        
        # Setiap pixel menjadi satu feature untuk machine learning
        for i, pixel_val in enumerate(pixels_binary):
            pixel_features[f'pixel_{i+1}'] = int(pixel_val)
        
        return pixel_features
    
    def extract_main_output_pixels(self):
        """
        Ekstrak nilai pixel dari main output (Canny edge detection) - sama seperti extract_canny_features
        
        Returns:
            dict: Berisi nilai pixel biner untuk machine learning
        """
        if 'main_output' not in self.features:
            return None
        
        main_output = self.features['main_output']
        
        # Pastikan ukuran gambar adalah 256x256
        if main_output.shape != (256, 256):
            main_output = cv2.resize(main_output, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Flatten citra biner menjadi array 1D (65536 pixels)
        pixels_flat = main_output.flatten()
        
        # Normalisasi ke nilai 0 dan 1 (biner)
        pixels_binary = (pixels_flat > 0).astype(np.uint8)
        
        # Buat dictionary dengan nilai pixel sebagai features
        pixel_features = {}
        
        # Setiap pixel menjadi satu feature untuk machine learning
        for i, pixel_val in enumerate(pixels_binary):
            pixel_features[f'pixel_{i+1}'] = int(pixel_val)
        
        return pixel_features

    def visualize_step_by_step(self, original_image_path, output_folder=".", base_name=None):
        """
        Membuat visualisasi step-by-step dari proses ekstraksi fitur
        
        Args:
            original_image_path (str): Path ke gambar asli
            output_folder (str): Folder untuk menyimpan hasil
            base_name (str): Base name untuk file output
            
        Returns:
            str: Path ke file step-by-step yang disimpan
        """
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
          # Baca gambar asli dan resize ke 256x256
        original = cv2.imread(original_image_path)
        original = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)        # Buat figure dengan 4 subplot horizontal untuk urutan: Original → Preprocessed → LBP → Enhanced Canny
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Step-by-Step Medical Waste Feature Extraction: {base_name}', fontsize=14, fontweight='bold')
        
        # 1. Original Image (256x256)
        axes[0].imshow(original_rgb)
        axes[0].set_title('1. Original Image\n(256x256)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Preprocessed Image (HSV + Segmentation, 256x256)
        if 'background_removal' in self.features and self.features['background_removal']:
            # Gunakan cleaned_image dari HSV segmentation yang sudah menghilangkan background
            processed = self.features['background_removal']['cleaned_image']
            processed = cv2.resize(processed, (256, 256), interpolation=cv2.INTER_AREA)
            axes[1].imshow(processed, cmap='gray')
            axes[1].set_title('2. Preprocessed\n(HSV + Segmentation)', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        elif 'color_processed' in self.features:
            processed = cv2.resize(self.features['color_processed'], (256, 256), interpolation=cv2.INTER_AREA)
            axes[1].imshow(processed, cmap='gray')
            axes[1].set_title('2. Preprocessed\n(HSV + Segmentation)', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No Processed\nImage Data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('2. Preprocessed', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        
        # 3. LBP Texture Features (sudah 256x256 dari fungsi texture_features_lbp)
        if 'texture' in self.features and 'lbp' in self.features['texture'] and 'image' in self.features['texture']['lbp']:
            lbp_image = self.features['texture']['lbp']['image']
            # LBP image sudah dipastikan 256x256 dari fungsi texture_features_lbp
            axes[2].imshow(lbp_image)
            axes[2].set_title('3. LBP Texture\n(Local Binary Pattern)', fontsize=12, fontweight='bold')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'No LBP\nTexture Data', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('3. LBP Texture', fontsize=12, fontweight='bold')
            axes[2].axis('off')
          # 4. Main Output - Canny Edge Detection (256x256)
        if 'main_output' in self.features:
            main_output = cv2.resize(self.features['main_output'], (256, 256), interpolation=cv2.INTER_NEAREST)
            axes[3].imshow(main_output, cmap='gray')
            axes[3].set_title('4. Canny Edge Detection\n', fontsize=12, fontweight='bold', color='red')
            axes[3].axis('off')
        else:
            axes[3].text(0.5, 0.5, 'No Main Output\nData', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('4. MAIN OUTPUT', fontsize=12, fontweight='bold', color='red')
            axes[3].axis('off')
        
        # Simpan figure
        plt.tight_layout()
        step_by_step_path = os.path.join(output_folder, f"{base_name}_step_by_step.png")
        plt.savefig(step_by_step_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualisasi step-by-step disimpan di: {step_by_step_path}")
        return step_by_step_path

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