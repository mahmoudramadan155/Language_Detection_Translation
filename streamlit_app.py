import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
import os
from string import ascii_letters, digits, punctuation

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

## Load environment variables from .env file
_ = load_dotenv(override=True)
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# Set Hugging Face API token in the environment
os.environ['HF_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# Define repository IDs for translation and language detection
repo_id_trans_en_ar = "Helsinki-NLP/opus-mt-en-ar"
repo_id_trans_ar_en = "Helsinki-NLP/opus-mt-ar-en"
repo_id_lang_det = 'papluca/xlm-roberta-base-language-detection'

def detect_language(input_sentence):
    # Mapping of language codes to language names
    map_values = {'en': 'English', 'ar': 'Arabic'}
    
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

def translate(input_sentence):
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

def main():
        
    st.set_page_config(page_title="Language Detection and Translation")
    st.title("Detection----Translation...💁 ")
    st.subheader("I can help you in language detection and translation")

    input_sentence = st.text_area("Please paste the 'sentence' here...",key="1")
    
    options  = ('Det','Trans')
    choice   = st.selectbox('Select whether you need to do choice Det or Trans: ',options)
    st.write('You selected:', choice )

    submit=st.button("Help me with the analysis",key="2")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            if choice =='Det':
                Det = detect_language(input_sentence)
                st.write(Det)

            if choice =='Trans':
                Trans = translate(input_sentence)
                st.write(Trans)
                
        st.success("Hope I was able to save your time❤️")

#Invoking main function
if __name__ == '__main__':
    main()
