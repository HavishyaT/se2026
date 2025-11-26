import os
os.environ["USE_SCIPY"] = "0"

import re
import string
import sys
import warnings
from urllib.parse import urlparse
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np

# Hugging Face / PyTorch
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# Optional GUI file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except Exception:
    TKINTER_AVAILABLE = False

warnings.filterwarnings("ignore")


def simple_train_test_split(df, test_size=0.15, random_state=42):
    """Simple numpy-based train/test split (no sklearn)."""
    np.random.seed(random_state)
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def compute_f1_binary(labels, preds):
    """Compute F1 for binary labels (0/1) without sklearn."""
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(float(f1), 4)


class Complete5StagePipeline:
    """5-stage pipeline with DistilBERT fine-tuning integration (no sklearn/scipy)."""

    def __init__(
        self,
        transformer_model_name="distilbert-base-uncased",
        transformer_output_dir="distilbert_finetuned",
        use_saved_transformer=False,
        freeze_transformer_layers=False,   
        transformer_train_args=None
    ):
      
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }

        self.sensational_words = {
            'shocking': 4, 'unbelievable': 4, 'incredible': 3, 'outrageous': 4,
            'devastating': 4, 'horrifying': 5, 'terrifying': 5, 'explosive': 4,
            'bombshell': 4, 'angry': 2, 'furious': 3, 'heartbreaking': 3,
            'tragic': 3, 'disaster': 3, 'catastrophe': 4, 'nightmare': 3,
            'crisis': 3, 'panic': 3, 'chaos': 3
        }

        self.fake_indicators = {
            'breaking': 2, 'exclusive': 2, 'leaked': 3, 'coverup': 5,
            'conspiracy': 5, 'wake up': 4, 'hoax': 5, 'scam': 4,
            'fraud': 4, 'lie': 3, 'fake': 3, 'exposed': 2
        }

        self.credibility_indicators = {
            'according to': 3, 'reported': 2, 'stated': 2, 'confirmed': 3,
            'research': 4, 'study': 4, 'university': 4, 'journal': 4,
            'published': 3, 'professor': 3, 'expert': 3, 'official': 3
        }

        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|percent|%))?',
            'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        }

        self.claim_indicators = [
            'claim', 'alleged', 'reportedly', 'according to', 'stated that',
            'says', 'said', 'announced', 'research shows', 'study finds'
        ]

        self.credible_domains = {
            'reuters.com': 95, 'apnews.com': 95, 'bbc.com': 90, 'npr.org': 88,
            'nature.com': 98, 'science.org': 98, 'nih.gov': 98, 'cdc.gov': 97,
            'gov': 90, 'edu': 85
        }

        self.knowledge_base = {
            'earth is flat': {'verified': False, 'confidence': 100},
            'vaccines cause autism': {'verified': False, 'confidence': 99},
            'climate change is real': {'verified': True, 'confidence': 97},
            'moon landing was faked': {'verified': False, 'confidence': 99},
            'covid-19 is real': {'verified': True, 'confidence': 100},
        }

        self.headline_templates = {
            'neutral': [
                'Reports indicate: {entity} {action}',
                'Study shows: {entity} {action}',
                'Officials announce: {entity} {action}',
                '{entity} {action}: What experts say',
                'News: {entity} {action}',
            ],
            'credible': [
                'Verified: {entity} {action}',
                'Confirmed by experts: {entity} {action}',
                'Research confirms: {entity} {action}',
                '{entity} {action}: Official statement',
                'Fact-checked: {entity} {action}',
            ],
            'cautious': [
                'Alleged: {entity} {action}',
                'Claims emerge: {entity} {action}',
                'Unverified reports: {entity} {action}',
                '{entity} {action}: Verification pending',
                'Questionable: {entity} {action}',
            ]
        }

        self.action_words = [
            'announces breakthrough', 'faces challenge', 'achieves milestone',
            'reveals findings', 'addresses concerns', 'launches initiative',
            'confirms status', 'updates policy', 'demonstrates impact',
            'establishes record', 'implements changes', 'reports progress'
        ]

        self.annotation_labels = {
            'credibility': {
                (0, 30): ' LOW CREDIBILITY',
                (30, 50): ' QUESTIONABLE',
                (50, 70): ' MODERATE',
                (70, 85): ' HIGH CREDIBILITY',
                (85, 101): ' VERIFIED'
            },
            'bias': {
                (0, 20): 'Minimal bias',
                (20, 40): 'Some bias present',
                (40, 60): 'Notable bias',
                (60, 80): 'Strong bias',
                (80, 101): 'Extreme bias'
            },
            'fake_risk': {
                (0, 20): 'Very Low Risk',
                (20, 40): 'Low Risk',
                (40, 60): 'Moderate Risk',
                (60, 80): 'High Risk',
                (80, 101): 'Very High Risk'
            }
        }

        self.text_column = None
        self.df = None

        self.transformer_model_name = transformer_model_name
        self.transformer_output_dir = transformer_output_dir
        self.use_saved_transformer = use_saved_transformer
        self.freeze_transformer_layers = freeze_transformer_layers

        default_train_args = {
            'output_dir': self.transformer_output_dir,
            'num_train_epochs': 1,                
            'per_device_train_batch_size': 8,    
            'per_device_eval_batch_size': 32,
            'gradient_accumulation_steps': 4,     
            'weight_decay': 0.01,
            'evaluation_strategy': 'epoch',
            'save_strategy': 'epoch',
            'logging_strategy': 'epoch',
            'learning_rate': 2e-5,
            'fp16': False,                        
            'load_best_model_at_end': True,
            'metric_for_best_model': 'f1',
            'greater_is_better': True,
            'seed': 42
        }
        self.transformer_train_args = transformer_train_args or default_train_args

       
        self.tokenizer = None
        self.transformer_model = None
        self.label_column = None

    def select_file_from_args(self):
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            if os.path.exists(file_path):
                return file_path
            else:
                print(f" File not found: {file_path}")
                return None
        if TKINTER_AVAILABLE:
            try:
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select CSV File",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                root.destroy()
                return file_path if file_path else None
            except Exception as e:
                print(f" File dialog error: {e}")
        print("\n Please enter the CSV file path:")
        file_path = input("File path: ").strip().strip('"').strip("'")
        if os.path.exists(file_path):
            return file_path
        print(f" File not found: {file_path}")
        return None

    def load_data(self, file_path):
        print(f"\n Loading: {file_path}")
        try:
            self.df = pd.read_csv(file_path)
            print(f" Loaded {len(self.df):,} records")
            print(f" Columns: {list(self.df.columns)}\n")
            return True
        except Exception as e:
            print(f" ERROR loading CSV: {e}")
            return False

    def detect_label_column(self):
        """Detect a likely label column and return its name or None."""
        if self.df is None:
            return None
        common_labels = ['label', 'labels', 'target', 'truth', 'is_fake', 'fake', 'class', 'y']
        cols = list(self.df.columns)
        for col in cols:
            if col.lower() in common_labels:
                self.label_column = col
                return col

        for col in cols:
            try:
                nunique = self.df[col].nunique(dropna=True)
                if nunique == 2:
                    self.label_column = col
                    return col
            except Exception:
                continue
        return None

    def map_label(self, v):
        if pd.isna(v):
            return None
        s = str(v).strip().lower()
        if s in {'1', 'true', 'fake', 'yes', 'y', 't', 'fake_news'}:
            return 1
        if s in {'0', 'false', 'real', 'no', 'n', 'f'}:
            return 0
        try:
            val = float(s)
            return int(round(val))
        except Exception:
            return None

    def preprocess_for_transformer(self, texts):
        return self.tokenizer(list(texts), truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        """Trainer metric function: logits, labels -> accuracy & f1"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        f1 = compute_f1_binary(labels, preds)
        return {'accuracy': round(acc, 4), 'f1': f1}

    def train_transformer(self):
        """Train/fine-tune DistilBERT on dataset (no sklearn)."""
        print("\nStarting transformer training/fine-tuning...")

        label_col = self.detect_label_column()
        if not label_col:
            raise ValueError(
                "No label column found. Add a binary label column (e.g. 'label' with 0/1 or 'fake'/'real')."
            )

        df = self.df[[self.text_column, label_col]].dropna().copy()
        df['__label__'] = df[label_col].apply(self.map_label)
        df = df.dropna(subset=['__label__']).reset_index(drop=True)
        df['__label__'] = df['__label__'].astype(int)

        if len(df) < 20:
            raise ValueError("Insufficient labeled rows for fine-tuning (need >= 20).")

        train_df, eval_df = simple_train_test_split(df, test_size=0.10, random_state=42)
        print(f"Transformer training: train={len(train_df)}, eval={len(eval_df)}")

        print("Loading tokenizer and model:", self.transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.transformer_model_name, num_labels=2)

        if self.freeze_transformer_layers:
            for name, param in model.named_parameters():

                if not name.startswith("classifier") and "distilbert" in name:
                    param.requires_grad = False
            print("Frozen transformer base layers; only classifier head will be trained.")

   
        train_ds = Dataset.from_pandas(train_df.rename(columns={self.text_column: 'text'}))
        eval_ds = Dataset.from_pandas(eval_df.rename(columns={self.text_column: 'text'}))

        def tokenize_batch(batch):
            tokens = self.tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
            tokens['labels'] = batch['__label__']
            return tokens

        train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(tokenize_batch, batched=True, remove_columns=eval_ds.column_names)

        training_args = TrainingArguments(**self.transformer_train_args)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train
        trainer.train()
        print("Training complete. Evaluating...")
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

        trainer.save_model(self.transformer_output_dir)
        self.transformer_model = model
        return trainer, metrics

    def load_transformer(self, model_dir=None):
        model_dir = model_dir or self.transformer_output_dir
        if not os.path.exists(model_dir):
            raise ValueError(f"Model dir not found: {model_dir}. Train first or provide correct dir.")
        print("Loading model from", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return True

    def predict_transformer_on_series(self, texts, batch_size=32):
        if self.transformer_model is None or self.tokenizer is None:
            raise ValueError("Transformer not loaded/trained.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_model.to(device)
        self.transformer_model.eval()
        all_probs = []
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i+batch_size])
            enc = self.tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt', max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = self.transformer_model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                all_probs.extend(probs[:, 1].tolist())
                all_preds.extend(preds.tolist())
        return all_preds, all_probs

    def auto_detect_text_column(self):
        print(" Auto-detecting text column...")
        for col in self.df.columns:
            if any(kw in col.lower() for kw in ['text', 'content', 'message', 'article', 'news', 'headline', 'title']):
                self.text_column = col
                print(f" Detected: '{self.text_column}'\n")
                return True
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    avg_len = self.df[col].astype(str).str.len().mean()
                    if avg_len > 50:
                        self.text_column = col
                        print(f" Detected: '{self.text_column}'\n")
                        return True
                except Exception:
                    pass
        print(" No text column found!")
        return False

    def preprocess_text(self, text):
        if not text or not isinstance(text, str):
            return ''
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        words = [w for w in text.split() if w not in self.stopwords]
        return ' '.join(words)

    def stage1_preprocessing(self):
        print("="*70)
        print("STAGE 1: TEXT PREPROCESSING")
        print("="*70)
        print(f"\n Processing {len(self.df):,} records...")
        self.df['original_text'] = self.df[self.text_column].astype(str)
        self.df['processed_text'] = self.df['original_text'].apply(self.preprocess_text)
        self.df['original_words'] = self.df['original_text'].str.split().str.len()
        self.df['processed_words'] = self.df['processed_text'].str.split().str.len()
        self.df['reduction_percent'] = (
            (self.df['original_words'] - self.df['processed_words']) / self.df['original_words'] * 100
        ).fillna(0).round(2)
        print(" Preprocessing complete!")
        print(f"\n Avg word reduction: {self.df['reduction_percent'].mean():.1f}%")
        print(f"Total words: {self.df['original_words'].sum():,} → {self.df['processed_words'].sum():,}")
        return True

    def calculate_sensationalism(self, text):
        if not text:
            return 0
        score = 0
        text_lower = text.lower()
        for word, weight in self.sensational_words.items():
            if word in text_lower:
                score += weight * 3
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            score += 25
        score += min(text.count('!') * 5, 20)
        return min(score, 100)

    def calculate_fake_probability(self, text):
        if not text:
            return 50
        text_lower = text.lower()
        fake_score = sum(weight for word, weight in self.fake_indicators.items() if word in text_lower)
        credibility_score = sum(weight for word, weight in self.credibility_indicators.items() if word in text_lower)
        if text.count('!') > 3:
            fake_score += 10
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            fake_score += 15
        total = fake_score + credibility_score
        if total == 0:
            return 50
        return min((fake_score / total) * 100, 100)

    def calculate_bias(self, text):
        if not text:
            return 0, 'neutral'
        text_lower = text.lower()
        left_keywords = ['liberal', 'progressive', 'democrat', 'left-wing']
        right_keywords = ['conservative', 'republican', 'right-wing', 'traditional']
        left_score = sum(1 for w in left_keywords if w in text_lower)
        right_score = sum(1 for w in right_keywords if w in text_lower)
        bias_type = 'neutral'
        if left_score > right_score + 1:
            bias_type = 'left-leaning'
        elif right_score > left_score + 1:
            bias_type = 'right-leaning'
        return min((left_score + right_score) * 8, 100), bias_type

    def stage2_fake_bias_detection(self):
        print("\n" + "="*70)
        print("STAGE 2: FAKE/BIAS DETECTION (with DistilBERT if available)")
        print("="*70)
        print(f"\n Analyzing {len(self.df):,} records...")
        self.df['sensationalism_score'] = self.df['original_text'].apply(self.calculate_sensationalism)
        self.df['fake_probability'] = self.df['original_text'].apply(self.calculate_fake_probability)
        bias_results = self.df['original_text'].apply(self.calculate_bias)
        self.df['bias_score'] = bias_results.apply(lambda x: x[0])
        self.df['bias_type'] = bias_results.apply(lambda x: x[1])

        label_col = self.detect_label_column()
        if label_col and not self.use_saved_transformer:
            try:
                trainer, metrics = self.train_transformer()
                print("Transformer trained. Metrics:", metrics)

                self.load_transformer(self.transformer_output_dir)
            except Exception as e:
                print("Transformer training failed or skipped:", e)
                self.transformer_model = None
                self.tokenizer = None
        else:
            if self.use_saved_transformer:
                try:
                    self.load_transformer(self.transformer_output_dir)
                except Exception as e:
                    print("Failed to load saved transformer:", e)
                    self.transformer_model = None
                    self.tokenizer = None
            else:
                print("No label column detected and not using saved model. Transformer not available.")
                self.transformer_model = None
                self.tokenizer = None

        if self.transformer_model is not None and self.tokenizer is not None:
            print("Running transformer predictions for dataset (may take time)...")
            preds, probs = self.predict_transformer_on_series(self.df['original_text'].astype(str).tolist(), batch_size=32)
            self.df['transformer_pred'] = preds
            self.df['transformer_prob'] = probs
            self.df['combined_fake_probability'] = (
                0.65 * (self.df['transformer_prob'] * 100) + 0.35 * self.df['fake_probability']
            ).clip(0, 100)
        else:
            self.df['transformer_pred'] = np.nan
            self.df['transformer_prob'] = np.nan
            self.df['combined_fake_probability'] = self.df['fake_probability']

        print(" Analysis complete!")
        high_fake = (self.df['combined_fake_probability'] > 70).sum()
        print(f"\n High fake probability (>70%): {high_fake:,} ({high_fake/len(self.df)*100:.1f}%)")
        return True

    def extract_entities(self, text):
        if not text:
            return {}
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))[:5]
        return entities

    def extract_claims(self, text):
        if not text:
            return []
        claims = []
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            if any(indicator in sentence.lower() for indicator in self.claim_indicators):
                claims.append({
                    'claim': sentence[:150],
                    'has_numbers': bool(re.search(r'\d+', sentence))
                })
        return claims

    def check_knowledge_base(self, text):
        if not text:
            return []
        text_lower = text.lower()
        matches = []
        for fact, info in self.knowledge_base.items():
            if fact in text_lower:
                matches.append({
                    'fact': fact,
                    'verified': info['verified'],
                    'confidence': info['confidence']
                })
        return matches

    def extract_sources(self, text):
        if not text:
            return []
        urls = re.findall(self.entity_patterns['url'], text)
        sources = []
        for url in urls[:3]:
            try:
                domain = urlparse(url).netloc.lower().replace('www.', '')
                credibility = 50
                if domain in self.credible_domains:
                    credibility = self.credible_domains[domain]
                elif domain.endswith('.gov'):
                    credibility = 90
                elif domain.endswith('.edu'):
                    credibility = 85
                sources.append({
                    'domain': domain,
                    'url': url,
                    'credibility': credibility
                })
            except Exception:
                continue
        return sources

    def calculate_verifiability(self, entities, claims, kb_matches, sources):
        score = 50
        entity_count = sum(len(v) for v in entities.values())
        score += min(entity_count * 3, 20)
        score += min(len(claims) * 5, 15)
        if kb_matches:
            verified = [m for m in kb_matches if m['verified']]
            false_claims = [m for m in kb_matches if not m['verified']]
            score += len(verified) * 10
            score -= len(false_claims) * 20
        if sources:
            avg_cred = np.mean([s['credibility'] for s in sources])
            score += (avg_cred - 50) * 0.3
        else:
            score -= 10
        return max(0, min(100, round(score, 2)))

    def stage3_fact_verification(self):
        print("\n" + "="*70)
        print("STAGE 3: FACT VERIFICATION")
        print("="*70)
        print(f"\n Analyzing {len(self.df):,} records...")
        entities_list, claims_list, kb_matches_list, sources_list, verifiability_list = [], [], [], [], []
        for idx, text in enumerate(self.df['original_text']):
            if idx % 500 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(self.df)} ({idx/len(self.df)*100:.0f}%)")
            entities = self.extract_entities(str(text))
            claims = self.extract_claims(str(text))
            kb_matches = self.check_knowledge_base(str(text))
            sources = self.extract_sources(str(text))
            verifiability = self.calculate_verifiability(entities, claims, kb_matches, sources)
            entities_list.append(entities)
            claims_list.append(claims)
            kb_matches_list.append(kb_matches)
            sources_list.append(sources)
            verifiability_list.append(verifiability)
        self.df['entities'] = entities_list
        self.df['entity_count'] = [sum(len(v) for v in e.values()) for e in entities_list]
        self.df['claims'] = claims_list
        self.df['claim_count'] = [len(c) for c in claims_list]
        self.df['kb_matches'] = kb_matches_list
        self.df['sources'] = sources_list
        self.df['source_count'] = [len(s) for s in sources_list]
        self.df['verifiability_score'] = verifiability_list
        self.df['verification_status'] = pd.cut(
            self.df['verifiability_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Unverifiable', 'Low', 'Moderate', 'High']
        )
        print("✓ Fact verification complete!")
        high_verify = (self.df['verifiability_score'] >= 70).sum()
        print(f"\n Highly verifiable: {high_verify:,} ({high_verify/len(self.df)*100:.1f}%)")
        return True

    def get_main_entity(self, entities):
        if not entities or not isinstance(entities, dict):
            return 'The subject'
        for entity_type in ['person', 'organization', 'location']:
            if entity_type in entities and entities[entity_type]:
                return entities[entity_type][0]
        if 'number' in entities and entities['number']:
            return entities['number'][0]
        return 'The subject'

    def select_headline_template(self, verifiability_score, fake_probability):
        if verifiability_score >= 70 and fake_probability < 30:
            return 'credible'
        elif verifiability_score < 50 or fake_probability > 70:
            return 'cautious'
        else:
            return 'neutral'

    def generate_headline_candidate(self, text, entities, verifiability_score, fake_probability):
        template_type = self.select_headline_template(verifiability_score, fake_probability)
        templates = self.headline_templates[template_type]
        template = templates[hash(text) % len(templates)]
        entity = self.get_main_entity(entities)
        action = self.action_words[hash(text + str(verifiability_score)) % len(self.action_words)]
        return template.format(entity=entity, action=action)

    def rank_candidates(self, candidates, verifiability_score, fake_probability, sensationalism_score):
        ranked = []
        for i, candidate in enumerate(candidates):
            score = 50
            length = len(candidate)
            if 50 <= length <= 80:
                score += 20
            else:
                score -= abs(length - 65) / 5
            if verifiability_score >= 70:
                if any(word in candidate.lower() for word in ['verified', 'confirmed', 'research']):
                    score += 15
            else:
                if any(word in candidate.lower() for word in ['alleged', 'claims', 'reports']):
                    score += 15
            sensational_count = sum(1 for word in self.sensational_words.keys() if word in candidate.lower())
            score -= sensational_count * 10
            fake_indicator_count = sum(1 for word in self.fake_indicators.keys() if word in candidate.lower())
            score -= fake_indicator_count * 15
            if i > 0:
                similarity = sum(1 for word in candidates[0].split() if word in candidate.split()) / max(len(candidate.split()), 1)
                if similarity < 0.7:
                    score += 10
            ranked.append({
                'headline': candidate,
                'rank_score': max(0, min(100, score)),
                'template_type': self.select_headline_template(verifiability_score, fake_probability)
            })
        return sorted(ranked, key=lambda x: x['rank_score'], reverse=True)

    def stage4_headline_rewriting(self):
        print("\n" + "="*70)
        print("STAGE 4: HEADLINE REWRITING & RANKING")
        print("="*70)
        print(f"\n Generating headlines for {len(self.df):,} records...")
        candidates_list = []
        top_headlines_list = []
        for idx, row in self.df.iterrows():
            if idx % 500 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(self.df)} ({idx/len(self.df)*100:.0f}%)")
            text = row['original_text']
            entities = row.get('entities', {})
            verifiability = row.get('verifiability_score', 50)
            fake_prob = row.get('combined_fake_probability', row.get('fake_probability', 50))
            sensationalism = row.get('sensationalism_score', 0)
            candidates = []
            for _ in range(5):
                candidate = self.generate_headline_candidate(text + str(_), entities, verifiability, fake_prob)
                candidates.append(candidate)
            ranked = self.rank_candidates(candidates, verifiability, fake_prob, sensationalism)
            candidates_list.append(ranked)
            top_headlines_list.append(ranked[0]['headline'] if ranked else 'No headline generated')
        self.df['headline_candidates'] = candidates_list
        self.df['rewritten_headline'] = top_headlines_list
        self.df['headline_rank_score'] = [c[0]['rank_score'] if c else 0 for c in candidates_list]
        print("✓ Headline rewriting complete!")
        return True

    def get_annotation_label(self, category, score):
        if category not in self.annotation_labels:
            return 'Unknown'
        for score_range, label in self.annotation_labels[category].items():
            if score_range[0] <= score <= score_range[1]:
                return label
        return list(self.annotation_labels[category].values())[0]

    def calculate_overall_confidence(self, verifiability, fake_prob, bias_score):
        confidence = verifiability * 0.5 + (100 - fake_prob) * 0.3 + (100 - bias_score) * 0.2
        return round(confidence, 2)

    def generate_source_references(self, sources):
        if not sources or not isinstance(sources, list):
            return 'No sources found'
        references = []
        for i, source in enumerate(sources[:3], 1):
            if isinstance(source, dict):
                domain = source.get('domain', 'Unknown')
                credibility = source.get('credibility', 0)
                cred_label = '✓✓✓' if credibility >= 85 else '✓✓' if credibility >= 70 else '✓' if credibility >= 50 else '⚠️'
                references.append(f"[{i}] {domain} ({cred_label} {credibility}/100)")
        return ' | '.join(references) if references else 'No sources found'

    def generate_warning_flags(self, fake_prob, verifiability, bias_score):
        flags = []
        if fake_prob > 70:
            flags.append(' HIGH FAKE RISK')
        elif fake_prob > 50:
            flags.append(' MODERATE FAKE RISK')
        if verifiability < 30:
            flags.append(' UNVERIFIABLE')
        if bias_score > 70:
            flags.append(' STRONG BIAS DETECTED')
        return ' | '.join(flags) if flags else 'No warnings'

    def stage5_output_generation(self):
        print("\n" + "="*70)
        print("STAGE 5: OUTPUT GENERATION")
        print("="*70)
        print(f"\n Generating final outputs for {len(self.df):,} records...")
        self.df['credibility_annotation'] = self.df['verifiability_score'].apply(lambda x: self.get_annotation_label('credibility', x))
        self.df['bias_annotation'] = self.df['bias_score'].apply(lambda x: self.get_annotation_label('bias', x))
        self.df['fake_risk_annotation'] = self.df['combined_fake_probability'].apply(lambda x: self.get_annotation_label('fake_risk', x))
        self.df['overall_confidence'] = self.df.apply(lambda row: self.calculate_overall_confidence(row['verifiability_score'], row['combined_fake_probability'], row['bias_score']), axis=1)
        self.df['source_references'] = self.df['sources'].apply(self.generate_source_references)
        self.df['warning_flags'] = self.df.apply(lambda row: self.generate_warning_flags(row['combined_fake_probability'], row['verifiability_score'], row['bias_score']), axis=1)
        self.df['final_headline_with_annotations'] = self.df.apply(lambda row: (
            f"{row['rewritten_headline']}\n"
            f"├─ Credibility: {row['credibility_annotation']}\n"
            f"├─ Fake Risk: {row['fake_risk_annotation']}\n"
            f"├─ Bias: {row['bias_annotation']} ({row['bias_type']})\n"
            f"├─ Overall Confidence: {row['overall_confidence']:.1f}%\n"
            f"├─ Sources: {row['source_references']}\n"
            f"└─ Warnings: {row['warning_flags']}"
        ), axis=1)
        self.df['quality_tier'] = pd.cut(self.df['overall_confidence'], bins=[0, 40, 60, 80, 100], labels=['Poor', 'Fair', 'Good', 'Excellent'])
        print(" Output generation complete!")
        excellent_count = (self.df['quality_tier'] == 'Excellent').sum()
        print(f"\n Excellent quality outputs: {excellent_count:,} ({excellent_count/len(self.df)*100:.1f}%)")
        return True

    def print_summary(self):
        print("\n" + "="*70)
        print("COMPLETE 5-STAGE ANALYSIS SUMMARY")
        print("="*70)
        print(f"\n Dataset: {len(self.df):,} records")
        print(f"\n STAGE 1 - Preprocessing:")
        print(f"  Avg word reduction: {self.df['reduction_percent'].mean():.1f}%")
        print(f"\n STAGE 2 - Fake/Bias Detection:")
        print(f"  High sensationalism (>60): {(self.df['sensationalism_score'] > 60).sum():,}")
        print(f"  High fake probability (>70): {(self.df['combined_fake_probability'] > 70).sum():,}")
        print(f"  Bias distribution:")
        for bias, count in self.df['bias_type'].value_counts().items():
            print(f"    {bias}: {count:,} ({count/len(self.df)*100:.1f}%)")
        print(f"\n STAGE 3 - Fact Verification:")
        print(f"  Records with entities: {(self.df['entity_count'] > 0).sum():,}")
        print(f"  Records with claims: {(self.df['claim_count'] > 0).sum():,}")
        print(f"  Records with sources: {(self.df['source_count'] > 0).sum():,}")
        print(f"  Verification status:")
        for status, count in self.df['verification_status'].value_counts().items():
            print(f"    {status}: {count:,} ({count/len(self.df)*100:.1f}%)")
        print(f"\n  STAGE 4 - Headline Rewriting:")
        print(f"  Avg headline rank score: {self.df['headline_rank_score'].mean():.1f}/100")
        print(f"  Headlines generated: {(self.df['rewritten_headline'].notna()).sum():,}")
        print(f"\n STAGE 5 - Output Generation:")
        print(f"  Avg overall confidence: {self.df['overall_confidence'].mean():.1f}%")
        print(f"  Quality tier distribution:")
        for tier, count in self.df['quality_tier'].value_counts().items():
            print(f"    {tier}: {count:,} ({count/len(self.df)*100:.1f}%)")
        print("\n" + "="*70)
        print("SAMPLE RESULT (First Record)")
        print("="*70)
        row = self.df.iloc[0]
        print(f"\n ORIGINAL TEXT:")
        print(f"{row['original_text'][:150]}...")
        print(f"\n ANALYSIS SCORES:")
        print(f"  Sensationalism: {row['sensationalism_score']:.1f}/100")
        print(f"  Fake Probability (heuristic): {row['fake_probability']:.1f}%")
        print(f"  Transformer Prob (if available): {row.get('transformer_prob', 'N/A')}")
        print(f"  Combined Fake Probability: {row['combined_fake_probability']:.1f}%")
        print(f"  Bias: {row['bias_type']} ({row['bias_score']:.1f}/100)")
        print(f"  Verifiability: {row['verifiability_score']:.1f}/100")
        print(f"  Overall Confidence: {row['overall_confidence']:.1f}%")
        print(f"\n FINAL OUTPUT:")
        print(row['final_headline_with_annotations'])
        if isinstance(row['headline_candidates'], list) and row['headline_candidates']:
            print(f"\n ALTERNATIVE HEADLINES:")
            for i, cand in enumerate(row['headline_candidates'][:3], 1):
                print(f"  {i}. [{cand['rank_score']:.1f}] {cand['headline']}")

    def save_results(self, filename='complete_5stage_analysis.csv'):
        print(f"\n Saving to: {filename}")
        save_df = self.df.copy()
        for col in ['entities', 'claims', 'kb_matches', 'sources', 'headline_candidates']:
            if col in save_df.columns:
                save_df[col] = save_df[col].astype(str)
        save_df.to_csv(filename, index=False)
        print(f" Saved successfully! Location: {os.path.abspath(filename)}")
        hr_filename = filename.replace('.csv', '_readable.csv')
        readable_cols = [
            'original_text', 'rewritten_headline', 'credibility_annotation',
            'fake_risk_annotation', 'bias_annotation', 'bias_type',
            'overall_confidence', 'source_references', 'warning_flags',
            'quality_tier', 'final_headline_with_annotations'
        ]
        readable_df = save_df[[col for col in readable_cols if col in save_df.columns]]
        readable_df.to_csv(hr_filename, index=False)
        print(f" Human-readable version saved: {hr_filename}")

    def run_complete_pipeline(self):
        print("\n" + "="*70)
        print(" COMPLETE 5-STAGE TEXT ANALYSIS & OUTPUT GENERATION PIPELINE (DistilBERT)")
        print("="*70)
        input_file = self.select_file_from_args()
        if not input_file:
            print(" No file selected")
            return False
        if not self.load_data(input_file):
            return False
        if not self.auto_detect_text_column():
            return False
        # Stage 1
        self.stage1_preprocessing()
        # Stage 2: may train/load transformer
        self.stage2_fake_bias_detection()
        # Stage 3-5
        self.stage3_fact_verification()
        self.stage4_headline_rewriting()
        self.stage5_output_generation()
        self.print_summary()
        output = input("\n Enter output filename (default: complete_5stage_analysis.csv): ").strip()
        self.save_results(output if output else 'complete_5stage_analysis.csv')
        print("\n" + "="*70)
        print(" ALL 5 STAGES COMPLETE!")
        print("="*70)
        return True


def main():

    pipeline = Complete5StagePipeline(freeze_transformer_layers=False, use_saved_transformer=False)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
