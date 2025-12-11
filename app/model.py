import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
import random
import sys
import csv


# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    sys.exit(1)

# Load Sentence Transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded successfully!")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    sys.exit(1)

# Check for CUDA
if torch.cuda.is_available():
    print("CUDA available, using GPU.")
    model = model.to('cuda')
else:
    print("CUDA not available, using CPU.")

# Load datasets
try:
    df = pd.read_csv(r'C:\Arman\Study\Projects\medicine_recommendation_system\data\dataset.csv', quoting=csv.QUOTE_NONNUMERIC)
    if df.empty:
        raise ValueError("Dataset is empty.")
    required_columns = ['Disease', 'Symptoms', 'Medications', 'Precautions', 'Description']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("dataset.csv missing required columns.")
except FileNotFoundError:
    print("Error: dataset.csv not found in project directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

try:
    med_df = pd.read_csv(r'C:\Arman\Study\Projects\medicine_recommendation_system\data\medications_data.csv', quoting=csv.QUOTE_NONNUMERIC)
    if med_df.empty:
        raise ValueError("Medications dataset is empty.")
    if not all(col in med_df.columns for col in ['Medication', 'Purpose']):
        raise ValueError("medications.csv missing required columns.")
    medication_purposes = dict(zip(med_df['Medication'], med_df['Purpose']))
except FileNotFoundError:
    print("Error: medications.csv not found in project directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading medications: {e}")
    sys.exit(1)

try:
    prec_df = pd.read_csv(r'C:\Arman\Study\Projects\medicine_recommendation_system\data\precautions.csv', quoting=csv.QUOTE_NONNUMERIC)
    if prec_df.empty:
        raise ValueError("Precautions dataset is empty.")
    if not all(col in prec_df.columns for col in ['Precaution', 'Template']):
        raise ValueError("precautions.csv missing required columns.")
    precaution_templates = prec_df.groupby('Precaution')['Template'].apply(list).to_dict()
except FileNotFoundError:
    print("Error: precautions.csv not found in project directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading precautions: {e}")
    sys.exit(1)

# Encode symptoms
try:
    df['Symptom_Text'] = df['Symptoms'].apply(lambda x: ' '.join(x.split(',')))
    symptom_embeddings = model.encode(df['Symptom_Text'].tolist(), convert_to_tensor=True)
except Exception as e:
    print(f"Error encoding dataset symptoms: {e}")
    sys.exit(1)

# Helper functions
def generate_medication_response(medications):
    meds = [m.strip() for m in medications.split(',')]
    responses = []
    for med in meds:
        purpose = medication_purposes.get(med, 'help with your symptoms')
        responses.append(f"You can take {med} to {purpose}.")
    return " ".join(responses)

def generate_precaution_response(precautions):
    precs = [p.strip() for p in precautions.split(',')]
    responses = []
    for prec in precs:
        template = random.choice(precaution_templates.get(prec, [f"{prec.replace(',', ' and ')}."]))
        responses.append(template)
    return " ".join(responses)

def is_affirmative(response):
    response = response.lower().strip()
    affirmative_keywords = ['yes', 'yeah', 'sure', 'ok', 'okay', 'please', 'yep']
    doc = nlp(response)
    for token in doc:
        if token.text in affirmative_keywords:
            return True
    return False

def diagnose_disease(user_input):
    if not user_input.strip():
        return None, "Input is empty. Please provide symptoms."
    try:
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, symptom_embeddings)[0]
        best_idx = cos_scores.argmax().item()
        if cos_scores[best_idx] < 0.3:
            return None, "Symptoms do not match any known disease closely enough."
        row = df.iloc[best_idx]
        return {
            'Disease': row['Disease'],
            'Medications': row['Medications'],
            'Precautions': row['Precautions'],
            'Description': row['Description']
        }, None
    except Exception as e:
        return None, f"Error processing input: {e}"