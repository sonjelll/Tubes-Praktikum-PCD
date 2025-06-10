#!/usr/bin/env python3
"""
Script untuk menjalankan ekstraksi fitur sampah medis dengan mudah
"""

import os
import sys
from medical_waste_feature_extractor import MedicalWasteFeatureExtractor

def run_extraction(image_path):
    """
    Jalankan ekstraksi fitur untuk gambar yang diberikan
    """
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} tidak ditemukan!")
        return False
    
    try:
        print(f"Memulai ekstraksi fitur untuk: {image_path}")
        print("=" * 50)
        
        # Inisialisasi extractor
        extractor = MedicalWasteFeatureExtractor()
        
        # Proses gambar
        features = extractor.process_image(image_path)
        
        # Visualisasi dan simpan hasil
        output_path = extractor.visualize_and_save(image_path)
        
        # Cetak ringkasan fitur
        extractor.print_feature_summary()
        
        print("\n" + "=" * 50)
        print("✅ Ekstraksi fitur berhasil!")
        print(f"📁 Hasil disimpan di: {os.path.dirname(output_path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saat memproses gambar: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Penggunaan: python run_feature_extraction.py <path_gambar>")
        print("Contoh: python run_feature_extraction.py sample_medical_waste.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = run_extraction(image_path)
    
    if not success:
        sys.exit(1)