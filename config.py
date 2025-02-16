from dataclasses import dataclass

@dataclass
class Config:
    MAX_SEQUENCE_LENGTH = 64
    BATCH_SIZE = 64
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    NUM_CLASSES = 6
    NUM_LAYERS = 3
    DROPOUT = 0.5
    
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 5
    GRADIENT_CLIP = 1.0
    
    WORD2VEC_PATH = "word2vec_model.bin"
    MODEL_SAVE_PATH = "emotion_model.pt"
    
    DEVICE = "cuda"
    
    EMOTIONS = {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "love",
        4: "sadness",
        5: "surprise"
    }