"""
PATH - Comprehensive Radiological Image Diagnostic Tool
Author: Umair Masood Awan
Contributor: Tooba Noor ul Ieman
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import cv2
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Mathematical/Image Processing
from scipy import ndimage, signal, fft
from skimage import filters, feature, morphology, segmentation, exposure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import radon, hough_line, hough_circle
from scipy.stats import entropy, kurtosis, skew
from scipy.ndimage import gaussian_filter, laplace, convolve

# Machine Learning
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Advanced
import pywt
from scipy.spatial.distance import directed_hausdorff
import monai
from monai.networks.nets import UNet, AttentionUnet
import antropy

# Configuration
class Config:
    """Configuration parameters for PATH system"""
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 16
    NUM_CLASSES = 6  # cyst, tumor, cancer, abscess, hematoma, lipoma
    CLASS_NAMES = ['cyst', 'tumor', 'cancer', 'abscess', 'hematoma', 'lipoma']
    
    # Model parameters
    CNN_FILTERS = [32, 64, 128, 256]
    TRANSFORMER_DIM = 768
    TRANSFORMER_HEADS = 12
    DROPOUT_RATE = 0.3
    
    # Radiomics features
    RADIOMICS_FEATURES = {
        'first_order': ['mean', 'variance', 'skewness', 'kurtosis', 'energy', 'entropy'],
        'texture': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm'],
        'shape': ['area', 'perimeter', 'compactness', 'eccentricity', 'solidity']
    }

# =============================================
# 1. PREPROCESSING MODULE
# =============================================

class Preprocessor:
    """Comprehensive medical image preprocessing"""
    
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load various medical image formats"""
        filepath = Path(filepath)
        
        if filepath.suffix in ['.dcm', '.DCM']:
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
        elif filepath.suffix in ['.nii', '.nii.gz']:
            img = nib.load(filepath).get_fdata()
        else:
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        
        return img.astype(np.float32)
    
    @staticmethod
    def normalize_ct(img: np.ndarray) -> np.ndarray:
        """Normalize CT images using Hounsfield units"""
        # Standard CT normalization ranges
        img = np.clip(img, -1000, 1000)
        img = (img + 1000) / 2000  # Normalize to [0, 1]
        return img
    
    @staticmethod
    def normalize_mri(img: np.ndarray) -> np.ndarray:
        """Normalize MRI images using Z-score"""
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-8)
    
    @staticmethod
    def window_level_adjustment(img: np.ndarray, window: float = 400, level: float = 50) -> np.ndarray:
        """Apply window/level optimization for medical images"""
        img_min = level - window / 2
        img_max = level + window / 2
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min + 1e-8)
        return img
    
    @staticmethod
    def histogram_equalization(img: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization"""
        img_normalized = exposure.equalize_adapthist(img, clip_limit=0.03)
        return img_normalized
    
    @staticmethod
    def gabor_filter_bank(img: np.ndarray, 
                         frequencies: List[float] = [0.1, 0.25, 0.4],
                         thetas: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
        """Apply Gabor filter bank for texture analysis"""
        gabor_responses = []
        for freq in frequencies:
            for theta in thetas:
                real, imag = filters.gabor(img, frequency=freq, theta=theta)
                gabor_responses.append(np.sqrt(real**2 + imag**2))
        return np.stack(gabor_responses, axis=-1)

# =============================================
# 2. FEATURE EXTRACTION MODULE
# =============================================

class FeatureExtractor:
    """Extract comprehensive radiomic features"""
    
    @staticmethod
    def first_order_statistics(img: np.ndarray) -> Dict:
        """First-order statistical features"""
        features = {
            'mean': np.mean(img),
            'variance': np.var(img),
            'skewness': skew(img.flatten()),
            'kurtosis': kurtosis(img.flatten()),
            'energy': np.sum(img**2),
            'entropy': entropy(img.flatten())
        }
        return features
    
    @staticmethod
    def texture_features(img: np.ndarray) -> Dict:
        """Haralick and GLCM texture features"""
        img_int = (img * 255).astype(np.uint8)
        glcm = graycomatrix(img_int, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'asm': graycoprops(glcm, 'ASM').mean()
        }
        return features
    
    @staticmethod
    def morphological_features(mask: np.ndarray) -> Dict:
        """Morphological shape features"""
        from skimage.measure import regionprops
        
        props = regionprops(mask.astype(int))[0]
        
        features = {
            'area': props.area,
            'perimeter': props.perimeter,
            'compactness': (props.perimeter**2) / (4 * np.pi * props.area),
            'eccentricity': props.eccentricity,
            'solidity': props.solidity
        }
        return features
    
    @staticmethod
    def fractal_analysis(img: np.ndarray) -> Dict:
        """Fractal dimension and complexity analysis"""
        # Box-counting method for fractal dimension
        def boxcount(img, k):
            s = 2**k
            return np.add.reduceat(
                np.add.reduceat(img, np.arange(0, img.shape[0], s), axis=0),
                np.arange(0, img.shape[1], s), axis=1).any(axis=(1, 2)).sum()
        
        sizes = []
        counts = []
        
        for k in range(1, 8):
            sizes.append(2**k)
            counts.append(boxcount(img > np.mean(img), k))
        
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        fractal_dim = -coeffs[0]
        
        # Higuchi fractal dimension
        def higuchi_fd(x, kmax=10):
            n = len(x)
            lk = []
            for k in range(1, kmax+1):
                lm = []
                for m in range(k):
                    idx = np.arange(m, n, k)
                    lm.append(np.sum(np.abs(np.diff(x[idx]))) * (n - 1) / (len(idx) * k))
                lk.append(np.log(np.mean(lm)))
            coeffs = np.polyfit(np.log(range(1, kmax+1)), lk, 1)
            return coeffs[0]
        
        higuchi_dim = higuchi_fd(img.flatten())
        
        return {
            'fractal_dimension_box': fractal_dim,
            'fractal_dimension_higuchi': higuchi_dim,
            'lacunarity': np.var(img) / (np.mean(img)**2 + 1e-8)
        }
    
    @staticmethod
    def fourier_features(img: np.ndarray) -> Dict:
        """Fourier transform-based features"""
        fft_img = fft.fft2(img)
        fft_shift = fft.fftshift(fft_img)
        magnitude = np.abs(fft_shift)
        
        # Power spectral density
        psd = np.abs(fft_img)**2
        
        features = {
            'fourier_energy': np.sum(magnitude**2),
            'fourier_entropy': entropy(magnitude.flatten()),
            'psd_mean': np.mean(psd),
            'psd_std': np.std(psd)
        }
        return features
    
    @staticmethod
    def wavelet_features(img: np.ndarray, wavelet='db4', level=3) -> Dict:
        """Wavelet transform features"""
        coeffs = pywt.wavedec2(img, wavelet, level=level)
        
        features = {}
        for i, coeff in enumerate(coeffs):
            if i == 0:
                features[f'wavelet_approx_mean'] = np.mean(coeff)
                features[f'wavelet_approx_std'] = np.std(coeff)
            else:
                features[f'wavelet_detail_{i}_mean'] = np.mean(coeff[0])
                features[f'wavelet_detail_{i}_std'] = np.std(coeff[0])
        
        return features

# =============================================
# 3. DEEP LEARNING MODELS
# =============================================

class HybridCNNViT(nn.Module):
    """Hybrid CNN-Vision Transformer architecture"""
    
    def __init__(self, config: Config):
        super().__init__()
        
        # CNN Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Vision Transformer component
        self.vit_encoder = VisionTransformer(
            image_size=128,
            patch_size=16,
            num_classes=config.NUM_CLASSES,
            dim=config.TRANSFORMER_DIM,
            depth=6,
            heads=config.TRANSFORMER_HEADS,
            mlp_dim=2048,
            dropout=config.DROPOUT_RATE,
            emb_dropout=config.DROPOUT_RATE
        )
        
        # Attention mechanism
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(256)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 + config.TRANSFORMER_DIM, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, config.NUM_CLASSES),
            nn.Softmax(dim=1)
        )
        
        # Monte Carlo Dropout for uncertainty
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
    
    def forward(self, x, mc_dropout: bool = False):
        # CNN features
        cnn_features = self.cnn_encoder(x)
        
        # Apply attention
        cnn_features = self.channel_attention(cnn_features)
        cnn_features = self.spatial_attention(cnn_features)
        
        # Flatten CNN features
        cnn_flat = F.adaptive_avg_pool2d(cnn_features, (1, 1)).view(x.size(0), -1)
        
        # ViT features
        vit_features = self.vit_encoder(x)
        
        # Concatenate features
        combined = torch.cat([cnn_flat, vit_features], dim=1)
        
        # Apply Monte Carlo dropout if requested
        if mc_dropout:
            combined = self.dropout(combined)
        
        # Classification
        output = self.classifier(combined)
        
        return output

class VisionTransformer(nn.Module):
    """Simplified Vision Transformer"""
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout),
            depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(img.shape[0], -1, p*p)
        
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        
        return x

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

# =============================================
# 4. LOSS FUNCTIONS
# =============================================

class CombinedLoss(nn.Module):
    """Combined Dice, Focal, and Cross-Entropy loss"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def dice_loss(self, pred, target, smooth=1e-8):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(pred * target_one_hot)
        union = torch.sum(pred) + torch.sum(target_one_hot)
        
        return 1 - (2. * intersection + smooth) / (union + smooth)
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        ce = F.cross_entropy(pred, target)
        
        return self.alpha * dice + self.beta * focal + self.gamma * ce

# =============================================
# 5. UNCERTAINTY QUANTIFICATION
# =============================================

class UncertaintyQuantifier:
    """Monte Carlo Dropout and Ensemble uncertainty measures"""
    
    @staticmethod
    def monte_carlo_dropout(model, x, n_samples=50):
        """Monte Carlo dropout sampling"""
        model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(x, mc_dropout=True)
                predictions.append(pred.cpu().numpy())
        
        model.eval()
        predictions = np.array(predictions)
        
        # Calculate uncertainty metrics
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    @staticmethod
    def ensemble_uncertainty(models, x, n_samples=10):
        """Ensemble uncertainty from multiple models"""
        all_predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                preds = []
                for _ in range(n_samples):
                    pred = model(x, mc_dropout=True)
                    preds.append(pred.cpu().numpy())
                all_predictions.append(np.mean(preds, axis=0))
        
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0)
        
        return mean_pred, uncertainty

# =============================================
# 6. MAIN PATH CLASSIFIER
# =============================================

class PATHClassifier:
    """Main PATH diagnostic system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.deep_model = HybridCNNViT(config).to(self.device)
        self.ensemble_models = []
        
        # Initialize traditional ML models
        self.ml_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Calibrated classifiers
        self.calibrated_models = {}
        
    def extract_radiomics_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """Extract comprehensive radiomics features"""
        features = {}
        
        # First-order statistics
        features.update(self.feature_extractor.first_order_statistics(image))
        
        # Texture features
        features.update(self.feature_extractor.texture_features(image))
        
        # Fourier features
        features.update(self.feature_extractor.fourier_features(image))
        
        # Wavelet features
        features.update(self.feature_extractor.wavelet_features(image))
        
        # Fractal analysis
        features.update(self.feature_extractor.fractal_analysis(image))
        
        # If mask provided, add morphological features
        if mask is not None:
            features.update(self.feature_extractor.morphological_features(mask))
        
        return features
    
    def train_deep_model(self, train_loader, val_loader, epochs=50):
        """Train deep learning model"""
        optimizer = torch.optim.AdamW(self.deep_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        criterion = CombinedLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.deep_model.train()
            train_loss = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.deep_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.deep_model.eval()
            val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.deep_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / len(val_loader.dataset)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.deep_model.state_dict(), 'best_model.pth')
            
            scheduler.step(avg_val_loss)
    
    def train_ml_models(self, X_train, y_train, X_val, y_val):
        """Train traditional ML models with calibration"""
        for name, model in self.ml_models.items():
            # Cross-validation training
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f'{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Calibrate classifier
            calibrated = CalibratedClassifierCV(model, cv='prefit')
            calibrated.fit(X_val, y_val)
            self.calibrated_models[name] = calibrated
    
    def predict_with_uncertainty(self, image: np.ndarray, n_samples=50):
        """Make prediction with uncertainty quantification"""
        # Preprocess image
        processed = self.preprocessor.normalize_mri(image)
        processed = self.preprocessor.histogram_equalization(processed)
        processed_tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Deep learning prediction with uncertainty
        deep_pred, deep_uncertainty = UncertaintyQuantifier.monte_carlo_dropout(
            self.deep_model, processed_tensor, n_samples
        )
        
        # Extract radiomics features
        features = self.extract_radiomics_features(image)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # ML model predictions
        ml_predictions = []
        ml_confidences = []
        
        for name, model in self.calibrated_models.items():
            pred = model.predict_proba(feature_vector)
            ml_predictions.append(pred)
            ml_confidences.append(np.max(pred, axis=1))
        
        # Ensemble predictions
        ensemble_pred = np.mean(ml_predictions + [deep_pred], axis=0)
        ensemble_confidence = 1 - np.mean(ml_confidences + [deep_uncertainty], axis=0)
        
        # Final prediction
        final_prediction = np.argmax(ensemble_pred, axis=1)
        confidence_score = ensemble_confidence[0][final_prediction[0]]
        
        return {
            'prediction': final_prediction[0],
            'class_name': self.config.CLASS_NAMES[final_prediction[0]],
            'probabilities': ensemble_pred[0],
            'confidence': confidence_score,
            'uncertainty': deep_uncertainty[0],
            'features': features
        }
    
    def generate_diagnostic_report(self, image_path: str):
        """Generate comprehensive diagnostic report"""
        # Load image
        image = self.preprocessor.load_image(image_path)
        
        # Get prediction with uncertainty
        result = self.predict_with_uncertainty(image)
        
        # Generate report
        report = {
            'patient_id': Path(image_path).stem,
            'image_dimensions': image.shape,
            'diagnosis': result['class_name'],
            'confidence_score': float(result['confidence']),
            'probability_distribution': {
                cls: float(prob) for cls, prob in zip(self.config.CLASS_NAMES, result['probabilities'])
            },
            'uncertainty_metrics': {
                'aleatoric_uncertainty': float(np.mean(result['uncertainty'])),
                'confidence_interval': float(1.96 * np.std(result['uncertainty']))
            },
            'key_features': {
                'texture_contrast': float(result['features'].get('contrast', 0)),
                'fractal_dimension': float(result['features'].get('fractal_dimension_box', 0)),
                'morphological_eccentricity': float(result['features'].get('eccentricity', 0))
            },
            'recommendations': self._generate_recommendations(result)
        }
        
        return report
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate clinical recommendations based on diagnosis"""
        diagnosis = result['class_name']
        confidence = result['confidence']
        
        recommendations = []
        
        if confidence < 0.7:
            recommendations.append("Consider additional imaging for confirmation")
            recommendations.append("Consult with senior radiologist")
        
        if diagnosis == 'cancer':
            recommendations.append("Urgent biopsy recommended")
            recommendations.append("Oncology consultation required")
        elif diagnosis == 'tumor':
            recommendations.append("Follow-up MRI in 3 months")
            recommendations.append("Consider contrast-enhanced imaging")
        elif diagnosis == 'cyst':
            recommendations.append("Routine follow-up in 6 months")
            recommendations.append("No immediate intervention needed")
        
        return recommendations

# =============================================
# 7. EVALUATION METRICS
# =============================================

class EvaluationMetrics:
    """Comprehensive evaluation metrics for medical imaging"""
    
    @staticmethod
    def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        intersection = np.logical_and(mask1, mask2).sum()
        return (2. * intersection) / (mask1.sum() + mask2.sum() + 1e-8)
    
    @staticmethod
    def hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Hausdorff distance"""
        coords1 = np.argwhere(mask1)
        coords2 = np.argwhere(mask2)
        
        if len(coords1) == 0 or len(coords2) == 0:
            return 0
        
        dist1 = directed_hausdorff(coords1, coords2)[0]
        dist2 = directed_hausdorff(coords2, coords1)[0]
        
        return max(dist1, dist2)
    
    @staticmethod
    def jaccard_index(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Jaccard index"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-8)
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_prob=None):
        """Calculate all classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics.update({'precision': precision, 'recall': recall, 'f1_score': f1})
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Calibration metrics if probabilities provided
        if y_prob is not None:
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=10)
            metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        
        return metrics

# =============================================
# 8. UTILITY FUNCTIONS
# =============================================

def visualize_results(image: np.ndarray, mask: np.ndarray, prediction: Dict):
    """Visualize image with segmentation and prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation overlay
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, alpha=0.3, cmap='Reds')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Prediction probabilities
    classes = list(prediction['probability_distribution'].keys())
    probabilities = list(prediction['probability_distribution'].values())
    
    axes[2].barh(classes, probabilities)
    axes[2].set_xlabel('Probability')
    axes[2].set_title(f'Diagnosis: {prediction["diagnosis"]}\nConfidence: {prediction["confidence_score"]:.2%}')
    axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================
# 9. MAIN EXECUTION
# =============================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("PATH - Radiological Image Diagnostic Tool")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Initialize classifier
    classifier = PATHClassifier(config)
    
    # Example usage
    image_path = "sample_mri.dcm"  # Replace with actual path
    
    try:
        # Generate diagnostic report
        report = classifier.generate_diagnostic_report(image_path)
        
        # Print report
        print("\n" + "=" * 60)
        print("DIAGNOSTIC REPORT")
        print("=" * 60)
        
        print(f"\nPatient ID: {report['patient_id']}")
        print(f"Diagnosis: {report['diagnosis']}")
        print(f"Confidence: {report['confidence_score']:.2%}")
        
        print("\nProbability Distribution:")
        for cls, prob in report['probability_distribution'].items():
            print(f"  {cls}: {prob:.2%}")
        
        print("\nKey Features:")
        for feature, value in report['key_features'].items():
            print(f"  {feature}: {value:.4f}")
        
        print("\nClinical Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
        
        # For visualization (if image data available)
        # image = classifier.preprocessor.load_image(image_path)
        # visualize_results(image, segmentation_mask, report)
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
