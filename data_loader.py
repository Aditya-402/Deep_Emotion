import spacy
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
from typing import List, Tuple
from config import Config

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def preprocess(self, text: str) -> List[str]:
        doc = self.nlp(text.lower())
        
        tokens = []
        for token in doc:
            if token.is_alpha or token.text in ["!", "?", "..."]:
                tokens.append(token.lemma_)
        
        return tokens

class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2vec_model: Word2Vec, config: Config):
        self.texts = texts
        self.labels = labels
        self.word2vec_model = word2vec_model
        self.config = config
        self.preprocessor = TextPreprocessor()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.preprocessor.preprocess(text)
        
        embeddings = []
        for token in tokens[:self.config.MAX_SEQUENCE_LENGTH]:
            try:
                embedding = self.word2vec_model.wv[token]
            except KeyError:
                embedding = self.word2vec_model.wv.vectors.mean(axis=0)
            embeddings.append(embedding)
            
        if len(embeddings) < self.config.MAX_SEQUENCE_LENGTH:
            padding = [np.zeros(self.config.EMBEDDING_DIM) for _ in range(
                self.config.MAX_SEQUENCE_LENGTH - len(embeddings))]
            embeddings.extend(padding)
        
        embeddings_tensor = torch.FloatTensor(np.array(embeddings))
        
        return {
            'embeddings': embeddings_tensor,
            'label': torch.LongTensor([label])[0]
        }

def load_and_split_data(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = load_dataset("dair-ai/emotion")
    
    texts = dataset['train']['text']
    labels = dataset['train']['label']
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, 
        train_size=config.TRAIN_RATIO, 
        stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        train_size=0.5,
        stratify=temp_labels
    )
    
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for text in train_texts]
    word2vec_model = Word2Vec(
        sentences=processed_texts,
        vector_size=config.EMBEDDING_DIM,
        window=5,
        min_count=1,
        workers=4
    )
    word2vec_model.save(config.WORD2VEC_PATH)
    
    train_dataset = EmotionDataset(train_texts, train_labels, word2vec_model, config)
    val_dataset = EmotionDataset(val_texts, val_labels, word2vec_model, config)
    test_dataset = EmotionDataset(test_texts, test_labels, word2vec_model, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    return train_loader, val_loader, test_loader