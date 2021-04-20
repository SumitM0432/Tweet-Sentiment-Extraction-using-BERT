import torch
import config
import engine
import preproc
import dataset
from model import TweetModel
import pandas as pd
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).dropna()
    
    dfx = preproc.preprocessing(dfx)
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size = 0.1,
        random_state = 42,
        stratify = dfx.sentiment.values
        )
    
    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)
    
    train_dataset = dataset.TweetDataset(
        tweet = df_train.text.values,
        sentiment = df_train.sentiment.values,
        selected_text = df_train.selected_text.values
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = 4
    )
    
    valid_dataset = dataset.TweetDataset(
        tweet = df_valid.text.values,
        sentiment = df_valid.sentiment.values,
        selected_text = df_valid.selected_text.values
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = 1
    )
    
    device = torch.device('cuda')
    model = TweetModel()
    model.to(device)
    
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    
    optimizer = AdamW(model.parameters(), lr = 2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
        )
    
    best_jaccard = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard, _ = engine.eval_fn(valid_data_loader, model, device)
        print ("Epoch -- ", epoch)
        print (f"Jaccard Score = {jaccard}")
        if jaccard > best_jaccard:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_jaccard = jaccard