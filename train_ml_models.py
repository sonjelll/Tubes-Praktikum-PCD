#!/usr/bin/env python3
"""
Script untuk melatih model machine learning (KNN dan RandomForest) menggunakan
nilai per-pixel HOG yang telah diekstrak dari dataset sampah medis.

Menggunakan file CSV hasil ekstraksi nilai per-pixel HOG untuk melatih model klasifikasi.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import untuk machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

class MedicalWasteClassifier:
    def __init__(self, hog_pixels_path, output_path=None):
        """
        Inisialisasi classifier untuk dataset sampah medis menggunakan hanya HOG pixels
        
        Args:
            hog_pixels_path (str): Path ke file CSV nilai per-pixel HOG
            output_path (str, optional): Path untuk menyimpan hasil model
        """
        self.hog_pixels_path = Path(hog_pixels_path)
        
        if output_path:
            self.output_path = Path(output_path)
        else:
            # Default output path adalah folder yang sama dengan file HOG pixels
            self.output_path = self.hog_pixels_path.parent / 'ml_models_hog_only'
        
        # Buat folder output jika belum ada
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi model
        self.knn = None
        self.rf = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classes = None
    
    def load_data(self):
        """
        Muat data dari file CSV HOG pixels
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            print(f"📊 Membaca data HOG pixels dari: {self.hog_pixels_path}")
            if not self.hog_pixels_path.exists():
                print(f"❌ File CSV HOG pixels tidak ditemukan: {self.hog_pixels_path}")
                return False
            
            # Baca data dari CSV HOG pixels
            df = pd.read_csv(self.hog_pixels_path)
            
            # Pisahkan fitur dan target
            X = df.drop(['image_path', 'category'], axis=1)
            y = df['category']
            
            # Cek missing values dan handle jika ada
            if X.isnull().values.any():
                print("⚠️ Terdapat missing values, melakukan imputasi...")
                X = X.fillna(0)  # Ganti dengan 0 atau strategi lain
            
            # Simpan nama kolom untuk feature importance nanti
            self.feature_names = X.columns
            
            # Simpan kelas unik
            self.classes = y.unique()
            
            # Normalisasi fitur
            print("🔄 Melakukan normalisasi fitur HOG pixels...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data menjadi training dan testing
            print("🔄 Membagi data menjadi training dan testing (80:20)...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"✅ Data HOG pixels berhasil dimuat: {len(X)} sampel, {len(self.classes)} kelas")
            print(f"   └─ Training: {len(self.X_train)} sampel")
            print(f"   └─ Testing: {len(self.X_test)} sampel")
            print(f"   └─ Jumlah fitur HOG pixels: {X.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saat memuat data: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Metode lainnya tetap sama
    def train_knn(self, n_neighbors=5, use_grid_search=True):
        """
        Latih model KNN dengan GridSearch opsional
        
        Args:
            n_neighbors (int): Jumlah tetangga untuk KNN jika tidak menggunakan GridSearch
            use_grid_search (bool): Apakah menggunakan GridSearch untuk tuning parameter
            
        Returns:
            float: Akurasi model pada data testing
        """
        print("\n🔄 Melatih model KNN...")
        
        if use_grid_search:
            print("   └─ Menggunakan GridSearchCV untuk tuning parameter...")
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            
            grid_search = GridSearchCV(
                KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"   └─ Parameter terbaik: {grid_search.best_params_}")
            self.knn = grid_search.best_estimator_
        else:
            print(f"   └─ Menggunakan n_neighbors={n_neighbors}")
            self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.knn.fit(self.X_train, self.y_train)
        
        # Evaluasi model
        y_pred = self.knn.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"✅ Akurasi KNN: {accuracy*100:.2f}%")
        
        return accuracy
    
    def train_random_forest(self, n_estimators=100, use_grid_search=True):
        """
        Latih model RandomForest dengan GridSearch opsional
        
        Args:
            n_estimators (int): Jumlah estimator untuk RandomForest jika tidak menggunakan GridSearch
            use_grid_search (bool): Apakah menggunakan GridSearch untuk tuning parameter
            
        Returns:
            float: Akurasi model pada data testing
        """
        print("\n🔄 Melatih model RandomForest...")
        
        if use_grid_search:
            print("   └─ Menggunakan GridSearchCV untuk tuning parameter...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42), param_grid, cv=5, 
                scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"   └─ Parameter terbaik: {grid_search.best_params_}")
            self.rf = grid_search.best_estimator_
        else:
            print(f"   └─ Menggunakan n_estimators={n_estimators}")
            self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            self.rf.fit(self.X_train, self.y_train)
        
        # Evaluasi model
        y_pred = self.rf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"✅ Akurasi RandomForest: {accuracy*100:.2f}%")
        
        return accuracy
    
    def save_confusion_matrix(self, model, model_name):
        """
        Simpan confusion matrix untuk model
        
        Args:
            model: Model yang akan dievaluasi
            model_name (str): Nama model untuk penamaan file
        """
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'Confusion Matrix - {model_name} (Accuracy: {accuracy_score(self.y_test, y_pred)*100:.2f}%)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        cm_path = self.output_path / f'{model_name.lower()}_confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        
        print(f"📊 Confusion Matrix {model_name} disimpan ke: {cm_path}")
    
    def save_classification_report(self):
        """
        Simpan classification report untuk kedua model
        """
        report_path = self.output_path / 'classification_report.txt'
        
        with open(report_path, 'w') as f:
            # KNN Report
            if self.knn:
                y_pred_knn = self.knn.predict(self.X_test)
                f.write("=== CLASSIFICATION REPORT - KNN ===\n")
                f.write(classification_report(self.y_test, y_pred_knn, target_names=self.classes))
                f.write("\n\n")
            
            # RandomForest Report
            if self.rf:
                y_pred_rf = self.rf.predict(self.X_test)
                f.write("=== CLASSIFICATION REPORT - RANDOM FOREST ===\n")
                f.write(classification_report(self.y_test, y_pred_rf, target_names=self.classes))
        
        print(f"📝 Classification Report disimpan ke: {report_path}")
    
    def save_feature_importance(self):
        """
        Simpan dan visualisasikan feature importance dari RandomForest
        """
        if not self.rf or not hasattr(self.rf, 'feature_importances_'):
            print("⚠️ Model RandomForest tidak tersedia atau tidak memiliki feature_importances_")
            return
        
        # Buat DataFrame feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Simpan ke CSV
        csv_path = self.output_path / 'feature_importance.csv'
        feature_importance.to_csv(csv_path, index=False)
        print(f"📊 Feature Importance disimpan ke: {csv_path}")
        
        # Visualisasi feature importance (top 20)
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top 20 Feature Importance - RandomForest')
        plt.tight_layout()
        
        img_path = self.output_path / 'feature_importance.png'
        plt.savefig(img_path)
        plt.close()
        
        print(f"📊 Visualisasi Feature Importance disimpan ke: {img_path}")
    
    def save_models(self):
        """
        Simpan model yang telah dilatih
        """
        # Simpan KNN model
        if self.knn:
            knn_path = self.output_path / 'knn_model.joblib'
            dump(self.knn, knn_path)
            print(f"✅ Model KNN disimpan ke: {knn_path}")
        
        # Simpan RandomForest model
        if self.rf:
            rf_path = self.output_path / 'randomforest_model.joblib'
            dump(self.rf, rf_path)
            print(f"✅ Model RandomForest disimpan ke: {rf_path}")
        
        # Simpan Scaler
        if self.scaler:
            scaler_path = self.output_path / 'scaler.joblib'
            dump(self.scaler, scaler_path)
            print(f"✅ Scaler disimpan ke: {scaler_path}")
    
    def run_training(self, use_grid_search=False):
        """
        Jalankan seluruh proses pelatihan dan evaluasi
        
        Args:
            use_grid_search (bool): Apakah menggunakan GridSearch untuk tuning parameter
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        print("\n🤖 Memulai proses machine learning...")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Train KNN
        self.train_knn(use_grid_search=use_grid_search)
        
        # Train RandomForest
        self.train_random_forest(use_grid_search=use_grid_search)
        
        # Save confusion matrices
        if self.knn:
            self.save_confusion_matrix(self.knn, "KNN")
        
        if self.rf:
            self.save_confusion_matrix(self.rf, "RandomForest")
        
        # Save classification report
        self.save_classification_report()
        
        # Save feature importance for RandomForest
        self.save_feature_importance()
        
        # Save models
        self.save_models()
        
        print("\n✅ Pelatihan model machine learning selesai!")
        print(f"📊 Hasil disimpan di: {self.output_path}")
        
        return True

def main():
    """Fungsi utama"""
    if len(sys.argv) < 2:
        print("Penggunaan: python train_ml_models.py <path_hog_pixels_csv> [output_path]")
        print("Contoh: python train_ml_models.py ./dataset_output/hog_pixels_for_ml.csv ./ml_models_hog_only")
        sys.exit(1)
    
    hog_pixels_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Inisialisasi classifier hanya dengan HOG pixels
    classifier = MedicalWasteClassifier(hog_pixels_path, output_path)
    
    # Tanya apakah ingin menggunakan GridSearch
    use_grid_search = input("Gunakan GridSearch untuk tuning parameter? (y/n, default: n): ").strip().lower() == 'y'
    
    # Jalankan pelatihan
    success = classifier.run_training(use_grid_search=use_grid_search)
    
    if success:
        print("\n🎉 Pelatihan model machine learning dengan HOG pixels berhasil!")
        sys.exit(0)
    else:
        print("\n💥 Pelatihan model machine learning gagal!")
        sys.exit(1)

if __name__ == "__main__":
    main()