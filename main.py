import glob
from tqdm import tqdm
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary
from torch import nn
import pandas as pd

import utils
import training 
import validation
# from model import R2Plus1D
import r2p1d_5
import r3d_5
import cnn3d

if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1) 
    
    num_epoch = 100
    batch_size = 6
    person_path = glob.glob('/data/spet5167/biovid/cropped_tensor_2/*')

    train_loader, val_loader = utils.load_data(person_path, batch_size, method='Predefined', n_classes = 5)

    model_r3d_10 = r3d_5.generate_model(model_depth = 10,
							  n_classes = 5,
							  n_input_channels = 1,
							  conv1_t_size = 7,
							  conv1_t_stride = 1, 
							  no_max_pool = False)
    
    model_r3d_18 = r3d_5.generate_model(model_depth = 18,
							  n_classes = 5,
							  n_input_channels = 1,
							  conv1_t_size = 7,
							  conv1_t_stride = 1, 
							  no_max_pool = False)
    
    model_r3d_34 = r3d_5.generate_model(model_depth = 34,
							  n_classes = 5,
							  n_input_channels = 1,
							  conv1_t_size = 7,
							  conv1_t_stride = 1, 
							  no_max_pool = False)

    model_r2p1d_10 = r2p1d_5.generate_model(model_depth = 10,
							  n_classes = 5,
							  n_input_channels = 1,
							  conv1_t_size = 7,
							  conv1_t_stride = 1, 
							  no_max_pool = False)
    
    model_r2p1d_18 = r2p1d_5.generate_model(model_depth = 18,
							  n_classes = 5,
							  n_input_channels = 1,
							  conv1_t_size = 7,
							  conv1_t_stride = 1, 
							  no_max_pool = False)
    
    model_r2p1d_34 = r2p1d_5.generate_model(model_depth = 34,
							  n_classes = 5,
							  n_input_channels = 1,
							  conv1_t_size = 7,
							  conv1_t_stride = 1, 
							  no_max_pool = False)
    
    # model_cnn3d = cnn3d.generate_model()
    
    models = {'r2p1d_10':model_r2p1d_10, 'r2p1d_18':model_r2p1d_18, 'r2p1d_34':model_r2p1d_34, 
              'r3d_10':model_r3d_10, 'r3d_18':model_r3d_18, 'r3d_34':model_r3d_34}

    for key, model in models.items(): 
        os.makedirs(f'outputs_7/output_{key}', exist_ok=True)
        model = model.cuda()

        print(f"Start Training Model: {key}")
        summary(model, (1, 25, 256, 256))

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=5e-3, weight_decay=1e-5)
        print("PyTorch Version:", torch.__version__)
        print("CUDA Available:", torch.cuda.is_available())
        print('Number of parameters:', sum(param.numel() for param in model.parameters()))

        batch_logger = []
        epoch_logger = []
        val_logger = []

        training_stats = []

        best_val_loss = float('inf')

        for epoch in range(1, num_epoch+1):
            train_loss, train_acc = training.train_epoch(epoch, train_loader, model, criterion, optimizer, batch_logger, epoch_logger, n_classes = 5)
            val_loss, val_acc, conf_matrix = validation.validation_epoch(epoch, val_loader, model, criterion, val_logger, n_classes = 5)

            training_stats.append({
                'epoch': epoch, 
                'train_loss': train_loss,
                'train_acc': train_acc, 
                'val_loss': val_loss, 
                'val_acc': val_acc, 
                'conf_matrix': conf_matrix
            })

            if epoch % 5 == 0: 
                torch.save(model.state_dict(), f'outputs_8/output_{key}/checkpoint_epoch_{epoch}.pth')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'outputs_8/output_{key}/best_model.pth')

        batch_df = pd.DataFrame(batch_logger)
        batch_df.to_csv(f'outputs_8/output_{key}/batch_logger.csv', index=False)
        epoch_df = pd.DataFrame(epoch_logger)
        epoch_df.to_csv(f'outputs_8/output_{key}/epoch_logger.csv', index=False)
        val_df = pd.DataFrame(val_logger)
        val_df.to_csv(f'outputs_8/output_{key}/val_logger.csv', index=False)
        training_df = pd.DataFrame(training_stats)
        training_df.to_csv(f'outputs_8/output_{key}/training_stats.csv', index=False)

    

