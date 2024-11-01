import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pytesseract
import cv2
import numpy as np
import os
from datetime import datetime
import json
import requests
from PIL import Image
import logging
import io
from urllib.parse import urlparse

class TextExtractorSaver:
    def __init__(self, output_dir="extracted_texts"):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"extraction_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

    def download_image(self, url):
        """Download image from URL"""
        try:
            # Send request with headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Convert to opencv format
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from URL")
                
            return image
            
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {str(e)}")
            return None

    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Apply various preprocessing techniques
            # 1. Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 2. Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            # 3. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(denoised)
            
            return contrast
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return None

    def extract_and_save(self, source, lang='eng', is_url=False):
        """Extract text from image source (URL or local path) and save results"""
        try:
            # Get image based on source type
            if is_url:
                image = self.download_image(source)
                filename = os.path.basename(urlparse(source).path) or 'url_image'
            else:
                image = cv2.imread(source)
                filename = os.path.basename(source)
                
            if image is None:
                raise ValueError(f"Could not load image from: {source}")
            
            # Preprocess image
            processed = self.preprocess_image(image)
            if processed is None:
                raise ValueError("Image preprocessing failed")
            
            # Extract text using Tesseract with optimized configuration
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?()-_\'\" "' 
            text = pytesseract.image_to_string(processed, config=config)
            
            # Clean extracted text
            cleaned_text = self.clean_text(text)
            
            # Generate filenames
            base_name = os.path.splitext(filename)[0]
            text_file = os.path.join(self.results_dir, f"{base_name}_text.txt")
            info_file = os.path.join(self.results_dir, f"{base_name}_info.json")
            processed_image_file = os.path.join(self.results_dir, f"{base_name}_processed.png")
            
            # Save results
            # 1. Save extracted text
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # 2. Save processing info
            info = {
                'source': source,
                'is_url': is_url,
                'timestamp': datetime.now().isoformat(),
                'language': lang,
                'characters_extracted': len(cleaned_text),
                'words_extracted': len(cleaned_text.split()),
                'success': True
            }
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            # 3. Save processed image
            cv2.imwrite(processed_image_file, processed)
            
            self.logger.info(f"Successfully processed {source}")
            return cleaned_text, info
            
        except Exception as e:
            self.logger.error(f"Error processing {source}: {str(e)}")
            return None, {'success': False, 'error': str(e)}

    def clean_text(self, text):
        """Clean extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text

    def process_mixed_sources(self, sources):
        """Process multiple sources (URLs and local files)"""
        results = {
            'total_sources': len(sources),
            'successful': 0,
            'failed': 0,
            'sources_processed': []
        }
        
        for source in sources:
            # Check if source is URL
            is_url = source.lower().startswith(('http://', 'https://'))
            
            # Process source
            text, info = self.extract_and_save(source, is_url=is_url)
            
            if info['success']:
                results['successful'] += 1
                results['sources_processed'].append({
                    'source': source,
                    'type': 'url' if is_url else 'local',
                    'status': 'success',
                    'words_extracted': info['words_extracted']
                })
            else:
                results['failed'] += 1
                results['sources_processed'].append({
                    'source': source,
                    'type': 'url' if is_url else 'local',
                    'status': 'failed',
                    'error': info.get('error', 'Unknown error')
                })
        
        # Save summary report
        summary_file = os.path.join(self.results_dir, 'extraction_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        return results

def main():
    # Initialize extractor
    extractor = TextExtractorSaver()
    
    # Example sources (mix of URLs and local files)
    sources = [
        "https://www.jantra.org.np/wp-content/uploads/2023/04/img2.jpg",
        # "path/to/local/image.png",
        # "https://example.com/image2.jpg",
        # "path/to/another/local/image.jpg"
    ]
    
    # Process sources
    results = extractor.process_mixed_sources(sources)
    
    # Print summary
    print("\nExtraction Summary:")
    print(f"Total sources processed: {results['total_sources']}")
    print(f"Successful extractions: {results['successful']}")
    print(f"Failed extractions: {results['failed']}")
    print(f"\nResults saved in: {extractor.results_dir}")

if __name__ == "__main__":
    main()