import torch
import torchvision.models as models
import pytesseract
import cv2
import numpy as np
import os
from datetime import datetime
import json
import requests
from PIL import Image
import logging
from urllib.parse import urlparse
from pytesseract import Output
import re
from collections import defaultdict

class FormattedTextExtractor:
    def __init__(self, output_dir="extracted_texts"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"extraction_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

    def download_image(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from URL")
            return image
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {str(e)}")
            return None

    def preprocess_image(self, image):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(denoised)
            
            return contrast
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return None

    def analyze_layout(self, image):
        """Analyze document layout using Tesseract"""
        try:
            # Get detailed OCR data including bounding boxes and confidence
            ocr_data = pytesseract.image_to_data(
                image, 
                output_type=Output.DICT,
                config='--psm 6'
            )
            
            # Organize text elements by their position
            layout_data = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                # Skip empty text
                if int(ocr_data['conf'][i]) > 0:
                    layout_data.append({
                        'text': ocr_data['text'][i],
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'conf': ocr_data['conf'][i],
                        'block_num': ocr_data['block_num'][i],
                        'par_num': ocr_data['par_num'][i],
                        'line_num': ocr_data['line_num'][i],
                        'word_num': ocr_data['word_num'][i]
                    })
            
            return layout_data
        except Exception as e:
            self.logger.error(f"Error in layout analysis: {str(e)}")
            return None

    def format_text_by_layout(self, layout_data):
        """Format text based on layout analysis"""
        try:
            # Group text elements by blocks, paragraphs, and lines
            formatted_text = []
            current_block = defaultdict(lambda: defaultdict(list))
            
            # Sort by vertical position first, then horizontal
            layout_data.sort(key=lambda x: (x['block_num'], x['par_num'], x['line_num'], x['x']))
            
            # Group text elements
            for element in layout_data:
                block = element['block_num']
                par = element['par_num']
                line = element['line_num']
                current_block[block][par].append(element)
            
            # Format text with proper spacing and layout
            for block in sorted(current_block.keys()):
                for par in sorted(current_block[block].keys()):
                    paragraph_text = []
                    current_line = -1
                    line_text = []
                    
                    for element in current_block[block][par]:
                        if element['line_num'] != current_line:
                            if line_text:
                                paragraph_text.append(' '.join(line_text))
                                line_text = []
                            current_line = element['line_num']
                        
                        # Add appropriate spacing based on x-coordinates
                        if line_text:
                            prev_element = current_block[block][par][len(line_text)-1]
                            space_width = element['x'] - (prev_element['x'] + prev_element['width'])
                            
                            # Add extra spaces for larger gaps
                            if space_width > element['width'] * 0.5:
                                num_spaces = max(1, int(space_width / (element['width'] * 0.3)))
                                line_text.append(' ' * num_spaces)
                        
                        line_text.append(element['text'])
                    
                    if line_text:
                        paragraph_text.append(' '.join(line_text))
                    
                    # Add paragraph to formatted text
                    formatted_text.append('\n'.join(paragraph_text))
                
                # Add block separator
                formatted_text.append('\n\n')
            
            return '\n'.join(formatted_text)
            
        except Exception as e:
            self.logger.error(f"Error in text formatting: {str(e)}")
            return None

    def extract_and_save(self, source, is_url=False):
        """Extract formatted text from image source and save results"""
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
            
            # Analyze layout
            layout_data = self.analyze_layout(processed)
            if layout_data is None:
                raise ValueError("Layout analysis failed")
            
            # Format text based on layout
            formatted_text = self.format_text_by_layout(layout_data)
            if formatted_text is None:
                raise ValueError("Text formatting failed")
            
            # Generate filenames
            base_name = os.path.splitext(filename)[0]
            text_file = os.path.join(self.results_dir, f"{base_name}_formatted.txt")
            html_file = os.path.join(self.results_dir, f"{base_name}_formatted.html")
            info_file = os.path.join(self.results_dir, f"{base_name}_info.json")
            
            # Save formatted text
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            # Save HTML version with styling
            html_content = self.create_html_output(layout_data, image.shape)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save processing info
            info = {
                'source': source,
                'is_url': is_url,
                'timestamp': datetime.now().isoformat(),
                'text_file': text_file,
                'html_file': html_file,
                'image_size': image.shape,
                'blocks_detected': len(set(x['block_num'] for x in layout_data)),
                'success': True
            }
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            return formatted_text, info
            
        except Exception as e:
            self.logger.error(f"Error processing {source}: {str(e)}")
            return None, {'success': False, 'error': str(e)}

    def create_html_output(self, layout_data, image_shape):
        """Create HTML representation of the formatted text"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .text-block {
                    position: absolute;
                    font-family: Arial, sans-serif;
                }
                .container {
                    position: relative;
                    margin: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        # Calculate scaling factor for positioning
        scale_factor = 1.0
        
        # Add text blocks with positioning
        for element in layout_data:
            x = element['x'] * scale_factor
            y = element['y'] * scale_factor
            
            html += f"""
                <div class="text-block" style="
                    left: {x}px;
                    top: {y}px;
                    width: {element['width']}px;
                    font-size: {element['height']}px;
                ">{element['text']}</div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

def main():
    # Initialize extractor
    extractor = FormattedTextExtractor()
    
    # Example usage
    source = "img2_processed.png"  # or URL
    is_url = False
    
    # Extract and save formatted text
    formatted_text, info = extractor.extract_and_save(source, is_url)
    
    if info['success']:
        print(f"Extraction successful!")
        print(f"Results saved in: {extractor.results_dir}")
        print("\nFormatted Text Preview:")
        print("-" * 50)
        print(formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text)
    else:
        print(f"Extraction failed: {info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()