import requests
import json
import pandas as pd
from json_repair import repair_json
import pyarabic.araby as araby

def clean_arabic_text(text):
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    return text

def load_data(url="https://raw.githubusercontent.com/NoorBayan/Burhan/main/corpus/metaphors_data.json"):
    response = requests.get(url)
    fixed_json_string = repair_json(response.text)
    data = json.loads(fixed_json_string)

    records = []
    effort_map = {"Low": 0, "Medium": 1, "High": 2}

    for item in data:
        ayah = item.get('metadata', {}).get('ayah_text_uthmani', '')
        similes = item.get('rhetorical_analysis', {}).get('similes', [])
        
        if not similes: continue
        
        for metaphor in similes:
            classification = metaphor.get('classification', {})
            effort = classification.get('processing_effort')
            segment = metaphor.get('simile_identity', {}).get('segment_text', '')
            
            if effort in effort_map and ayah:
                combined_text = f"{ayah} [SEP] {segment}" if segment else ayah
                records.append({
                    'text': combined_text, 
                    'label_text': effort,
                    'label': effort_map[effort]
                })
                break 

    df = pd.DataFrame(records)
    df['clean_text'] = df['text'].apply(clean_arabic_text)
    
    label_encoder = {0: "Low", 1: "Medium", 2: "High"}
    
    return df, label_encoder
