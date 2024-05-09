from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def translate(text,model_ckpt):
    translator = pipeline(
        "translation",
        model=model_ckpt
    )
    translated_text = translator(text)
    return translated_text

def detection(text, model_ckpt):
    lang_det = pipeline(
        "text-classification",
        model=model_ckpt
    )
    lang_det_text = lang_det(text, top_k=1, truncation=True)
    return lang_det_text

def detection_and_translation(texts, repo_id_lang_det, repo_id_trans_en_ar, repo_id_trans_ar_en):
    out = []
    for input_text in texts:
        lang_det_output = detection(input_text, repo_id_lang_det)
        det = lang_det_output[0]['label']
        if det == 'en':
            trans = translate(input_text, repo_id_trans_en_ar)
            out.append({'En': input_text, 'Ar': trans[0]['translation_text']})
        if det == 'ar':
            trans = translate(input_text, repo_id_trans_ar_en)
            out.append({'Ar': input_text, 'En': trans[0]['translation_text']})
    return out

