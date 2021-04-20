import transformers
import tokenizers
import os

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 50
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = '/home/levi/Desktop/KSE/input/train.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    'vocab.txt',
    lowercase = True
)
