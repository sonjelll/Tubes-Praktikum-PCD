#!/usr/bin/env python3
"""
GUI untuk ekstraksi fitur sampah medis menggunakan PyQt5
Mendukung ekstraksi fitur tunggal dan batch processing
"""

import sys
import os
import threading
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QTextEdit, QProgressBar, QGroupBox,
                            QGridLayout, QMessageBox, QSplitter)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QFont, QIcon

from medical_waste_feature_extractor import MedicalWasteFeatureExtractor
from run_batch_extraction import DatasetFeatureProcessor

class SingleImageExtractorThread(QThread):
    """Thread untuk ekstraksi fitur gambar tunggal"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, image_path, output_folder):
        super().__init__()
        self.image_path = image_path
        self.output_folder = output_folder
        
    def run(self):
        try:
            self.progress_signal.emit("🚀 Memulai ekstraksi fitur...")
            
            # Inisialisasi extractor
            extractor = MedicalWasteFeatureExtractor()
            
            self.progress_signal.emit("📸 Memproses gambar...")
            
            # Proses gambar
            features = extractor.process_image(self.image_path)
            
            self.progress_signal.emit("🎨 Membuat visualisasi...")
            
            # Visualisasi dan simpan hasil
            output_path = extractor.visualize_and_save(
                self.image_path, 
                output_folder=self.output_folder,
                create_full_visualization=False
            )
            
            self.progress_signal.emit("✅ Ekstraksi fitur selesai!")
            self.finished_signal.emit(True, output_path)
            
        except Exception as e:
            self.progress_signal.emit(f"❌ Error: {str(e)}")
            self.finished_signal.emit(False, str(e))

class BatchProcessorThread(QThread):
    """Thread untuk batch processing dataset"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, dataset_path, output_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.output_path = output_path
        
    def run(self):
        try:
            self.progress_signal.emit("🚀 Memulai batch processing...")
            
            # Inisialisasi processor
            processor = DatasetFeatureProcessor(self.dataset_path, self.output_path)
            
            # Custom progress callback
            original_print = print
            def custom_print(*args, **kwargs):
                message = " ".join(str(arg) for arg in args)
                self.progress_signal.emit(message)
                original_print(*args, **kwargs)
            
            # Redirect print untuk progress
            import builtins
            builtins.print = custom_print
            
            # Jalankan pemrosesan
            success = processor.process_dataset()
            
            # Restore print
            builtins.print = original_print
            
            if success:
                self.progress_signal.emit("✅ Batch processing selesai!")
                self.finished_signal.emit(True, self.output_path)
            else:
                self.progress_signal.emit("❌ Batch processing gagal!")
                self.finished_signal.emit(False, "Processing failed")
                
        except Exception as e:
            self.progress_signal.emit(f"❌ Error: {str(e)}")
            self.finished_signal.emit(False, str(e))

class MedicalWasteGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Inisialisasi UI"""
        self.setWindowTitle("Medical Waste Feature Extractor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget dengan tab
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout utama
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("Medical Waste Feature Extractor")
        header_label.setAlignment(Qt.AlignCenter)
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        main_layout.addWidget(header_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: Single Image
        self.single_tab = self.create_single_image_tab()
        self.tab_widget.addTab(self.single_tab, "Single Image")
        
        # Tab 2: Batch Processing
        self.batch_tab = self.create_batch_processing_tab()
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_single_image_tab(self):
        """Buat tab untuk ekstraksi fitur gambar tunggal"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        
        # Image selection
        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        input_layout.addWidget(QLabel("Image:"), 0, 0)
        input_layout.addWidget(self.image_path_label, 0, 1)
        
        self.select_image_btn = QPushButton("Select Image")
        self.select_image_btn.clicked.connect(self.select_image)
        input_layout.addWidget(self.select_image_btn, 0, 2)
        
        # Output folder selection
        self.output_folder_label = QLabel("Current directory")
        self.output_folder_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        input_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        input_layout.addWidget(self.output_folder_label, 1, 1)
        
        self.select_output_btn = QPushButton("Select Output Folder")
        self.select_output_btn.clicked.connect(self.select_output_folder)
        input_layout.addWidget(self.select_output_btn, 1, 2)
        
        layout.addWidget(input_group)
        
        # Image preview section
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(300)
        self.image_preview.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        preview_layout.addWidget(self.image_preview)
        
        layout.addWidget(preview_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.extract_btn = QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.extract_single_image)
        self.extract_btn.setEnabled(False)
        control_layout.addWidget(self.extract_btn)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.single_progress_text = QTextEdit()
        self.single_progress_text.setMaximumHeight(150)
        self.single_progress_text.setReadOnly(True)
        progress_layout.addWidget(self.single_progress_text)
        
        layout.addWidget(progress_group)
        
        # Variables
        self.selected_image_path = None
        self.selected_output_folder = "."
        
        return tab
    
    def create_batch_processing_tab(self):
        """Buat tab untuk batch processing"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input section
        input_group = QGroupBox("Dataset Configuration")
        input_layout = QGridLayout(input_group)
        
        # Dataset folder selection
        self.dataset_path_label = QLabel("No dataset folder selected")
        self.dataset_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        input_layout.addWidget(QLabel("Dataset Folder:"), 0, 0)
        input_layout.addWidget(self.dataset_path_label, 0, 1)
        
        self.select_dataset_btn = QPushButton("Select Dataset Folder")
        self.select_dataset_btn.clicked.connect(self.select_dataset_folder)
        input_layout.addWidget(self.select_dataset_btn, 0, 2)
        
        # Output folder selection
        self.batch_output_label = QLabel("dataset_output")
        self.batch_output_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        input_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        input_layout.addWidget(self.batch_output_label, 1, 1)
        
        self.select_batch_output_btn = QPushButton("Select Output Folder")
        self.select_batch_output_btn.clicked.connect(self.select_batch_output_folder)
        input_layout.addWidget(self.select_batch_output_btn, 1, 2)
        
        layout.addWidget(input_group)
        
        # Info section
        info_group = QGroupBox("Dataset Structure Expected")
        info_layout = QVBoxLayout(info_group)
        
        info_text = """
Dataset structure should be:
dataset/
├── category1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── category2/
│   ├── image1.jpg
│   └── ...
└── ...

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
        """
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-family: monospace; background-color: #f8f8f8; padding: 10px;")
        info_layout.addWidget(info_label)
        
        layout.addWidget(info_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.batch_process_btn = QPushButton("Start Batch Processing")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        self.batch_process_btn.setEnabled(False)
        control_layout.addWidget(self.batch_process_btn)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        progress_layout.addWidget(self.batch_progress_bar)
        
        self.batch_progress_text = QTextEdit()
        self.batch_progress_text.setReadOnly(True)
        progress_layout.addWidget(self.batch_progress_text)
        
        layout.addWidget(progress_group)
        
        # Variables
        self.selected_dataset_path = None
        self.selected_batch_output = "dataset_output"
        
        return tab
    
    def select_image(self):
        """Pilih gambar untuk ekstraksi fitur tunggal"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Medical Waste Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.image_path_label.setText(os.path.basename(file_path))
            self.extract_btn.setEnabled(True)
            
            # Show image preview
            self.show_image_preview(file_path)
            
    def show_image_preview(self, image_path):
        """Tampilkan preview gambar"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale image to fit preview area while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)
        else:
            self.image_preview.setText("Unable to load image preview")
    
    def select_output_folder(self):
        """Pilih folder output untuk gambar tunggal"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            "."
        )
        
        if folder_path:
            self.selected_output_folder = folder_path
            self.output_folder_label.setText(folder_path)
    
    def select_dataset_folder(self):
        """Pilih folder dataset untuk batch processing"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder",
            "."
        )
        
        if folder_path:
            self.selected_dataset_path = folder_path
            self.dataset_path_label.setText(folder_path)
            self.batch_process_btn.setEnabled(True)
    
    def select_batch_output_folder(self):
        """Pilih folder output untuk batch processing"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            "."
        )
        
        if folder_path:
            self.selected_batch_output = folder_path
            self.batch_output_label.setText(folder_path)
    
    def extract_single_image(self):
        """Mulai ekstraksi fitur gambar tunggal"""
        if not self.selected_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return
        
        # Disable button during processing
        self.extract_btn.setEnabled(False)
        self.single_progress_text.clear()
        
        # Start extraction thread
        self.extraction_thread = SingleImageExtractorThread(
            self.selected_image_path,
            self.selected_output_folder
        )
        self.extraction_thread.progress_signal.connect(self.update_single_progress)
        self.extraction_thread.finished_signal.connect(self.single_extraction_finished)
        self.extraction_thread.start()
    
    def start_batch_processing(self):
        """Mulai batch processing"""
        if not self.selected_dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
        
        # Confirm batch processing
        reply = QMessageBox.question(
            self,
            "Confirm Batch Processing",
            f"Start batch processing for dataset:\n{self.selected_dataset_path}\n\nThis may take a while. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Disable button during processing
        self.batch_process_btn.setEnabled(False)
        self.batch_progress_text.clear()
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start batch processing thread
        self.batch_thread = BatchProcessorThread(
            self.selected_dataset_path,
            self.selected_batch_output
        )
        self.batch_thread.progress_signal.connect(self.update_batch_progress)
        self.batch_thread.finished_signal.connect(self.batch_processing_finished)
        self.batch_thread.start()
    
    def update_single_progress(self, message):
        """Update progress untuk ekstraksi tunggal"""
        self.single_progress_text.append(message)
        self.single_progress_text.ensureCursorVisible()
        self.statusBar().showMessage(message)
    
    def update_batch_progress(self, message):
        """Update progress untuk batch processing"""
        self.batch_progress_text.append(message)
        self.batch_progress_text.ensureCursorVisible()
        self.statusBar().showMessage(message)
    
    def single_extraction_finished(self, success, result):
        """Callback ketika ekstraksi tunggal selesai"""
        self.extract_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(
                self,
                "Success",
                f"Feature extraction completed successfully!\n\nOutput saved to:\n{result}"
            )
            self.statusBar().showMessage("Feature extraction completed successfully!")
        else:
            QMessageBox.critical(
                self,
                "Error",
                f"Feature extraction failed:\n{result}"
            )
            self.statusBar().showMessage("Feature extraction failed!")
    
    def batch_processing_finished(self, success, result):
        """Callback ketika batch processing selesai"""
        self.batch_process_btn.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(
                self,
                "Success",
                f"Batch processing completed successfully!\n\nResults saved to:\n{result}"
            )
            self.statusBar().showMessage("Batch processing completed successfully!")
        else:
            QMessageBox.critical(
                self,
                "Error",
                f"Batch processing failed:\n{result}"
            )
            self.statusBar().showMessage("Batch processing failed!")

def main():
    """Fungsi utama"""
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Medical Waste Feature Extractor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Medical Waste Analysis")
    
    # Create and show main window
    window = MedicalWasteGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
