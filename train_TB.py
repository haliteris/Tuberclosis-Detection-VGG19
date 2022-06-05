import argparse
import json
import numpy as np
import os
import torch
from pathlib import Path
from evaluate_TB import run_model
from loader_TB import load_data
from model_TB import TBNet

def train(rundir, epochs, learning_rate, use_gpu):
    
    train_loader, valid_loader = load_data(use_gpu)
    
    model = TBNet()
    
    if use_gpu:
        print("using GPU")
        model = model.cuda()
    else:
        print("Something is wrong, I am not using the GPU")

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)
    
    best_val_loss = float('inf')
    print("best_val_loss",best_val_loss)

    for epoch in range(epochs):
        print('starting epoch {}.'.format(epoch+1))
        
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.3f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = f'val{val_loss:0.3f}_train{train_loss:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            torch.save(model.state_dict(), save_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', default="LL/vol01_sagitleft_line3/vgg19E-e06", type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default='store_true')
    parser.add_argument('--learning_rate', default=1e-06, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(args.rundir, args.epochs, args.learning_rate, args.gpu)