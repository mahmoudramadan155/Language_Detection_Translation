from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from string import ascii_letters, digits, punctuation
from utils import *

# Note that I only used the ready model from hugging face in the flask_app for quick response, and I do not have a GPU.
# But I can replace it with what we trained previously.

# Load environment variables from .env file
_ = load_dotenv(override=True)
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Set Hugging Face API token in the environment
os.environ['HF_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# Define repository IDs for translation and language detection
repo_id_trans_en_ar = "Helsinki-NLP/opus-mt-en-ar"
repo_id_trans_ar_en = "Helsinki-NLP/opus-mt-ar-en"
repo_id_lang_det = 'papluca/xlm-roberta-base-language-detection'

# Initialize Flask app
app = Flask(__name__)

# Define API-1, Endpoint for language detection
@app.route('/detect_language', methods=['POST'])
def detect_language():
    # Mapping of language codes to language names
    map_values = {'en': 'English', 'ar': 'Arabic'}
    
    # Extract input sentence from request
    data = request.get_json()
    input_sentence = data.get('sentence')
    
    # If input sentence is a string, convert it to a list
    if isinstance(input_sentence, str):
        input_sentence = [input_sentence]
    
    # Check if input sentence is not empty or contains only special characters
    if not input_sentence or all(char in digits+punctuation for txt in input_sentence  for char in txt):
        return {"error": "Invalid input sentence. Please provide a valid sentence."}

    # Perform language detection
    lang_det_output = detection(input_sentence, repo_id_lang_det)
    
    # Extract detected languages and map them to language names
    det = [map_values[i[0]['label']] for i in lang_det_output]
    
    # Return detected language(s)
    return {"detected_language": det}

# Define API-2, Endpoint for translation
@app.route('/translate', methods=['POST'])
def translate():
    # Extract input sentence from request
    data = request.get_json()
    input_sentence = data.get('sentence')
    
    # If input sentence is a string, convert it to a list
    if isinstance(input_sentence, str):
        input_sentence = [input_sentence]
    
    # Check if input sentence is not empty or contains only special characters
    if not input_sentence or all(char in digits+punctuation for txt in input_sentence  for char in txt):
        return {"error": "Invalid input sentence. Please provide a valid sentence."}

    # Perform language detection and translation
    output = detection_and_translation(input_sentence, repo_id_lang_det, repo_id_trans_en_ar, repo_id_trans_ar_en)
    
    # Return translated sentence(s)
    return output

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

# python app.py
# http://localhost:5000/detect_language
# http://localhost:5000/translate
# 
# Example usage:
# 1.
# http://127.0.0.1:5000/translate
# {"sentence": ["This castle is amazing", "انا لا اشعر بالعطش"]}
# http://127.0.0.1:5000/detect_language
# {"sentence": ["This castle is amazing", "انا لا اشعر بالعطش"]}

# 2.
# http://127.0.0.1:5000/translate
# {"sentence": ["This castle is amazing", "انا لا اشعر بالعطش"]}
# http://127.0.0.1:5000/detect_language
# {"sentence": ["This castle is amazing", "انا لا اشعر بالعطش"]}