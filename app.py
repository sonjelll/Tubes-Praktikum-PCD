from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class PETBottleDetector:
    def __init__(self):
        # Basic parameters
        self.min_area = 500
        self.max_area = 50000
        self.min_aspect_ratio = 1.0
        self.max_aspect_ratio = 6.0
        self.min_solidity = 0.3
        self.min_extent = 0.2
        self.morph_kernel_size = 5
        
        # Template matching parameters
        self.template_scales = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
        self.template_threshold = 0.6
        self.nms_threshold = 0.3
        
        # Create bottle templates programmatically
        self.bottle_templates = self.create_bottle_templates()
        
        # Feature detection parameters
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def create_bottle_templates(self):
        """Create synthetic bottle templates for matching"""
        templates = []
        
        # Template 1: Standard bottle shape
        template1 = np.zeros((120, 40), dtype=np.uint8)
        # Neck (top)
        cv2.rectangle(template1, (15, 0), (25, 20), 255, -1)
        # Body (middle-bottom)
        cv2.rectangle(template1, (5, 20), (35, 100), 255, -1)
        # Bottom curve
        cv2.ellipse(template1, (20, 110), (15, 10), 0, 0, 180, 255, -1)
        templates.append(template1)
        
        # Template 2: Wide bottle
        template2 = np.zeros((100, 50), dtype=np.uint8)
        cv2.rectangle(template2, (20, 0), (30, 15), 255, -1)  # Neck
        cv2.rectangle(template2, (5, 15), (45, 85), 255, -1)   # Body
        cv2.ellipse(template2, (25, 90), (20, 10), 0, 0, 180, 255, -1)
        templates.append(template2)
        
        # Template 3: Tall bottle
        template3 = np.zeros((150, 35), dtype=np.uint8)
        cv2.rectangle(template3, (12, 0), (23, 25), 255, -1)   # Neck
        cv2.rectangle(template3, (3, 25), (32, 130), 255, -1)  # Body
        cv2.ellipse(template3, (17, 140), (14, 10), 0, 0, 180, 255, -1)
        templates.append(template3)
        
        return templates
    
    def extract_candidates_from_mask(self, mask):
        """Extract bottle candidates from binary mask"""
        candidates = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Basic geometric properties
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Calculate additional properties
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            extent = area / (w * h) if (w * h) > 0 else 0
            
            if solidity >= self.min_solidity and extent >= self.min_extent:
                candidates.append({
                    'contour': contour,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'bounding_box': (x, y, w, h),
                    'solidity': solidity,
                    'extent': extent,
                    'method': 'contour_based'
                })
        
        return candidates
    
    def advanced_template_matching(self, image):
        """Advanced multi-scale template matching"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = []
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        for template_idx, template in enumerate(self.bottle_templates):
            for scale in self.template_scales:
                # Resize template
                template_h, template_w = template.shape
                new_h, new_w = int(template_h * scale), int(template_w * scale)
                
                if new_h > gray.shape[0] or new_w > gray.shape[1]:
                    continue
                    
                scaled_template = cv2.resize(template, (new_w, new_h))
                
                # Multiple matching methods
                methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
                
                for method in methods:
                    result = cv2.matchTemplate(enhanced_gray, scaled_template, method)
                    
                    if method == cv2.TM_SQDIFF_NORMED:
                        locations = np.where(result <= (1 - self.template_threshold))
                        scores = 1 - result[locations]
                    else:
                        locations = np.where(result >= self.template_threshold)
                        scores = result[locations]
                    
                    # Process detected locations
                    for i, (y, x) in enumerate(zip(locations[0], locations[1])):
                        score = scores[i] if len(scores) > i else 0
                        
                        # Calculate bounding box
                        x1, y1 = x, y
                        x2, y2 = x + new_w, y + new_h
                        
                        # Basic validation
                        w, h = new_w, new_h
                        aspect_ratio = h / w if w > 0 else 0
                        area = w * h
                        
                        if (self.min_area <= area <= self.max_area and
                            self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                            
                            candidates.append({
                                'bounding_box': (x1, y1, w, h),
                                'score': float(score),
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'solidity': 0.8,  # Default value for template matches
                                'extent': 0.7,    # Default value for template matches
                                'method': f'template_{template_idx}_{method}',
                                'scale': scale
                            })
        
        return candidates
    
    def apply_non_max_suppression(self, candidates):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not candidates:
            return []
        
        # Convert to format needed for NMS
        boxes = []
        scores = []
        
        for candidate in candidates:
            x, y, w, h = candidate['bounding_box']
            boxes.append([x, y, x + w, y + h])
            scores.append(candidate.get('score', 0.5))
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                   score_threshold=0.3, nms_threshold=self.nms_threshold)
        
        # Return filtered candidates
        filtered_candidates = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                filtered_candidates.append(candidates[i])
        
        return filtered_candidates
    
    def color_segmentation(self, image):
        """Segment image based on typical bottle colors"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common bottle colors
        color_ranges = [
            # Clear/transparent (white-ish)
            ([0, 0, 200], [180, 30, 255]),
            # Blue bottles
            ([100, 50, 50], [130, 255, 255]),
            # Green bottles
            ([40, 50, 50], [80, 255, 255]),
            # Brown bottles
            ([10, 50, 20], [20, 255, 200]),
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morph_kernel_size, self.morph_kernel_size))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def detect_bottles(self, image):
        """Main detection method combining multiple approaches"""
        all_candidates = []
        processed_images = {}
        
        # 1. Color segmentation approach
        color_mask = self.color_segmentation(image)
        processed_images['combined_mask'] = color_mask
        color_candidates = self.extract_candidates_from_mask(color_mask)
        all_candidates.extend(color_candidates)
        
        # 2. Grayscale processing approach
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images['grayscale'] = gray
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images['enhanced'] = enhanced
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Morphological operations on edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        edge_candidates = self.extract_candidates_from_mask(edges_closed)
        all_candidates.extend(edge_candidates)
        
        # 3. Template matching approach
        template_candidates = self.advanced_template_matching(image)
        all_candidates.extend(template_candidates)
        
        # 4. Apply Non-Maximum Suppression
        final_candidates = self.apply_non_max_suppression(all_candidates)
        
        return final_candidates, processed_images
    
    def update_parameters(self, params):
        """Update detection parameters"""
        if 'min_area' in params:
            self.min_area = params['min_area']
        if 'max_area' in params:
            self.max_area = params['max_area']
        if 'min_aspect_ratio' in params:
            self.min_aspect_ratio = params['min_aspect_ratio']
        if 'max_aspect_ratio' in params:
            self.max_aspect_ratio = params['max_aspect_ratio']
        if 'min_solidity' in params:
            self.min_solidity = params['min_solidity']
        if 'min_extent' in params:
            self.min_extent = params['min_extent']

# Initialize detector
detector = PETBottleDetector()

def save_image_base64(image, folder, filename):
    """Save image and return base64 encoded string"""
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    
    # Convert to base64 for web display
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'success': False, 'error': 'Invalid image file'})
            
            # Detect bottles
            detections, processed_images = detector.detect_bottles(image)
            
            # Create result image with bounding boxes
            result_image = image.copy()
            detection_data = []
            
            for i, detection in enumerate(detections):
                x, y, w, h = detection['bounding_box']
                
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"Bottle {i+1}"
                cv2.putText(result_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Collect detection data
                detection_data.append({
                    'id': i + 1,
                    'area': detection['area'],
                    'aspect_ratio': detection['aspect_ratio'],
                    'solidity': detection['solidity'],
                    'extent': detection['extent'],
                    'position': {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    },
                    'method': detection.get('method', 'unknown')
                })
            
            # Save images and convert to base64
            base_filename = os.path.splitext(filename)[0]
            
            original_b64 = save_image_base64(image, app.config['PROCESSED_FOLDER'], 
                                           f"{base_filename}_original.jpg")
            result_b64 = save_image_base64(result_image, app.config['PROCESSED_FOLDER'], 
                                         f"{base_filename}_result.jpg")
            
            processed_b64 = {}
            for key, img in processed_images.items():
                processed_b64[key] = save_image_base64(img, app.config['PROCESSED_FOLDER'], 
                                                     f"{base_filename}_{key}.jpg")
            
            return jsonify({
                'success': True,
                'detection_count': len(detections),
                'detections': detection_data,
                'original_image': original_b64,
                'result_image': result_b64,
                'processed_images': processed_b64
            })
        
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/adjust_parameters', methods=['POST'])
def adjust_parameters():
    try:
        params = request.get_json()
        detector.update_parameters(params)
        return jsonify({'success': True, 'message': 'Parameters updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)