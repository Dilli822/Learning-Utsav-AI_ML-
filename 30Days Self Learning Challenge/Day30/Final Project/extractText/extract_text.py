import cv2
import pytesseract
import os
from datetime import datetime
import json
import logging
import pyttsx3

class TextExtractor:
    def __init__(self, output_dir="extracted_texts"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"extraction_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

    def capture_image_from_camera(self):
        cap = cv2.VideoCapture(0)  # Open the default camera
        if not cap.isOpened():
            self.logger.error("Could not open camera. Please check the camera connection.")
            return None
        
        self.logger.info("Press 'c' to capture an image, or 'q' to quit.")
        image_filename = None
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to capture image from camera.")
                break
            
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                image_filename = os.path.join(self.results_dir, 'captured_image.png')
                cv2.imwrite(image_filename, frame)
                self.logger.info(f"Image saved as {image_filename}")
                break
            
            if key & 0xFF == ord('q'):
                self.logger.info("Camera closed without capturing.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return image_filename

    def extract_text_from_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from: {image_path}")

            # Perform OCR on the image
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            self.logger.error(f"Error in text extraction: {str(e)}")
            return None

    def extract_and_save(self, use_camera=False):
        try:
            image_path = None

            if use_camera:
                image_path = self.capture_image_from_camera()
                if image_path is None:
                    raise ValueError("No image captured from camera.")

            extracted_text = self.extract_text_from_image(image_path)
            if extracted_text is None:
                raise ValueError("Text extraction failed")

            # Save the extracted text
            text_file = os.path.join(self.results_dir, 'extracted_text.txt')
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            info = {
                'timestamp': datetime.now().isoformat(),
                'text_file': text_file,
                'success': True
            }
            info_file = os.path.join(self.results_dir, 'info.json')
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            return extracted_text, info
            
        except Exception as e:
            self.logger.error(f"Error processing the image: {str(e)}")
            return None, {'success': False, 'error': str(e)}

def text_to_speech(text):
    """Convert the provided text to speech."""
    engine = pyttsx3.init()  # Initialize the text-to-speech engine
    engine.setProperty('rate', 150)  # Set the speech rate to 150 words per minute
    engine.say(text)         # Pass the text to the engine
    engine.runAndWait()      # Wait for the speech to finish

# Example usage
if __name__ == "__main__":
    extractor = TextExtractor()
    extracted_text, info = extractor.extract_and_save(use_camera=True)
    
    if info['success']:
        print("Extracted Text:")
        print(extracted_text)
        print("Metadata saved in:", info['text_file'])    

        # Convert the extracted text to speech
        if extracted_text:  # Check if text extraction was successful
            text_to_speech(extracted_text)
        else:
            print("No text was extracted from the captured image.")
    else:
        print("Error:", info['error'])
