import os
import sys
sys.path.append('./datasets')
sys.path.append('./models')
from TSforecastingDatasets import PatchTSDataset
from TSforecastingModels import PatchTransformer

from datetime import datetime, timedelta
import numpy as np
import random
import time
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary
from tqdm import tqdm

def generate_train_val_dates(global_start_date, global_end_date, validation_split_fraction):
    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(global_start_date, '%Y%m%d')
    end_date = datetime.strptime(global_end_date, '%Y%m%d')
    
    # Calculate the total number of days in the date range
    total_days = (end_date - start_date).days
    
    # Calculate the number of days for the validation set
    validation_days = int(total_days * validation_split_fraction)
    
    # Calculate the start date and end date for the training set
    train_start_date = start_date
    train_end_date = start_date + timedelta(days=total_days - validation_days)
    
    # Calculate the start date and end date for the validation set
    val_start_date = train_end_date + timedelta(days=1)
    val_end_date = end_date
    
    # Format the dates as strings in 'YYYYMMDD' format
    train_start_date_str = train_start_date.strftime('%Y%m%d')
    train_end_date_str = train_end_date.strftime('%Y%m%d')
    val_start_date_str = val_start_date.strftime('%Y%m%d')
    val_end_date_str = val_end_date.strftime('%Y%m%d')
    
    return train_start_date_str, train_end_date_str, val_start_date_str, val_end_date_str

'''##########################################################################|
|############################################################################|
|////////////////////////////////////////////////////////////////////////////|
|////////////////////////////////////////////////////////////////////////////|
|////////////////////////////////////////////////////////////////////////////|
|############################################################################|
|##########################################################################'''

def trainSingleGPU(args):
    # Set RandomSeed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    useValSet = False
    trainStartDate = args.trainStartDate
    trainEndDate = args.trainEndDate
    if args.validationSplit > 0 and args.validationSplit < 1:
        useValSet = True
        trainStartDate, trainEndDate, valStartDate, valEndDate = generate_train_val_dates(
            args.trainStartDate, args.trainEndDate, args.validationSplit)
        print(trainStartDate, trainEndDate, valStartDate, valEndDate)

    # Make Dataset
    trainDataset = PatchTSDataset(args.dataDir, trainStartDate, trainEndDate,
                                  contWindowsFilePath=os.path.join(os.getcwd(), 'datasets', 'data', 'contWindows.pkl'))
    
    # Create DataLoader
    trainDataLoader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True)

    if useValSet:
        # Make Dataset
        normalizationParams = trainDataset.getNormalizationParams()
        valDataset = PatchTSDataset(args.dataDir, valStartDate, valEndDate,
                                    normalization_params=normalizationParams,
                                    contWindowsFilePath=os.path.join(os.getcwd(), 'datasets', 'data', 'contWindowsVal.pkl'))
        # Create DataLoader
        valDataLoader = DataLoader(valDataset, batch_size=args.batchSize, shuffle=True)
    
    ### Setup gpu device
    device = torch.device('cuda', 0)
    ### Create model
    history_shape = trainDataset.__getitem__(0)[0].shape
    target_shape = trainDataset.__getitem__(0)[1].shape
    print(history_shape, target_shape)
    input_features = history_shape[1]
    output_steps = target_shape[0]
    model = PatchTransformer(in_features=input_features, context_window=history_shape[0],
                             d_model=128, nhead=16, num_layers=3, target_window=output_steps)
    ### Put the model to the device
    model.to(device)
    ### Print summary of the model
    summary(model, [(args.batchSize, history_shape[0], history_shape[1])])
    ### Define loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ### Check if training is to be resumed
    if args.resumeTraining and os.path.isfile(args.ckptSavePath):
        checkpoint = torch.load(args.ckptSavePath)
        startEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        running_loss = checkpoint['loss']
        loss_y = checkpoint['lossHistory']
        val_loss_y = checkpoint['valLossHistory']
        times = checkpoint['timeHistory']
        print(f'Resuming training from epoch {startEpoch} with loss {running_loss}')
    else:
        startEpoch = 0
        # Training steps and losses tracking
        loss_y = []
        # Validation steps and losses tracking
        val_loss_y = []
        # Measure time
        times = []
    
    # Define your patience and threshold as input parameters
    patience = args.esPatience  # Number of epochs to wait for improvement
    threshold = args.esThreshold  # Minimum improvement required in validation loss
    # Initialize variables for early stopping
    best_val_loss = float('inf')  # Set to positive infinity initially
    no_improvement_count = 0  # Counter for consecutive epochs with no improvement

    ### Start training loop
    for epoch in range(startEpoch, args.numEpochs):
        start_epoch = time.time()
        running_loss = 0.0
        model.train()  # Set the model to training mode
        progress_bar = tqdm(enumerate(trainDataLoader, 0), desc=f'Epoch [{epoch + 1}/{args.numEpochs}]', ncols=100, unit="batch", total=len(trainDataLoader))
        for i,batch in progress_bar:
            # Transfer batch data to device
            history = batch[0].float().to(device)
            target = batch[1].float().to(device)
            # Forward pass
            outputs = model(history)
            loss = criterion(outputs, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute loss and set progress bar
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1), refresh=False)  # Update the progress bar with the current loss
        progress_bar.close()
        avg_validation_loss = None
        if useValSet:
            # Validation loop
            model.eval()  # Set the model to evaluation mode
            validation_loss = 0.0
            with torch.no_grad():
                for i, batch in enumerate(valDataLoader, 0):
                    # Transfer batch data to device
                    history = batch[0].float().to(device)
                    target = batch[1].float().to(device)
                    # Forward pass
                    outputs = model(history)
                    loss = criterion(outputs, target)
                    # Compute validation loss
                    validation_loss += loss.item()
            # Calculate average validation loss
            avg_validation_loss = validation_loss / len(valDataLoader)
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)
        print(f'Epoch [{epoch + 1}/{args.numEpochs}] | Training Loss: {running_loss / len(trainDataLoader):.4f} | Val Loss Improvement: {best_val_loss - avg_validation_loss:.6f}')
        ### Append losses of each step into the lists
        loss_y.append(running_loss / len(trainDataLoader))
        val_loss_y.append(avg_validation_loss)
        # Save checkpoint every ckptInterval epochs
        if args.ckptInterval is not None and args.ckptSavePath is not None:
            if (epoch + 1) % args.ckptInterval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(trainDataLoader),
                    'lossHistory': loss_y,
                    'valLossHistory': val_loss_y,
                    'timeHistory': times
                }, args.ckptSavePath)
                print(f'Saved checkpoint at epoch {epoch + 1}')
        if useValSet:
            # Check if the validation loss has improved
            val_loss_improvement = best_val_loss - avg_validation_loss
            if val_loss_improvement >= threshold:
                best_val_loss = avg_validation_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            # If there's no improvement for 'patience' epochs, stop training
            if no_improvement_count >= patience:
                print(f'Early stopping! No improvement for {patience} epochs.')
                break
    ### Save the trained model
    torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(trainDataLoader),
                    'lossHistory': loss_y,
                    'valLossHistory': val_loss_y,
                    'timeHistory': times
                }, args.modelSavePath)
    # Measure time
    avg_time = sum(times)/(args.numEpochs)
    print(f'\n------- Avg_time: {avg_time} sec -------\n')

    ### Plot the losses
    epoch_x = np.linspace(1, epoch+1, epoch+1).astype(int)
    plt.plot(epoch_x, loss_y, label='Training Loss')
    if useValSet:
        plt.plot(epoch_x, val_loss_y, label='Validation Loss')
    plt.legend()
    plt.savefig(f'loss.pdf')

'''##########################################################################|
|############################################################################|
|////////////////////////////////////////////////////////////////////////////|
|////////////////////////////////////////////////////////////////////////////|
|////////////////////////////////////////////////////////////////////////////|
|############################################################################|
|##########################################################################'''

def trainMultiGPU(gpu, args):
    rank = gpu
    print(rank)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '7788'
    # os.environ['WORLD_SIZE'] = str(args.nGPUs)
    os.environ['RANK'] = str(rank)
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.nGPUs,
        rank=rank,
    )
    dist.barrier()
    # Measure time
    start_init = time.time()

    # Set RandomSeed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    useValSet = False
    trainStartDate = args.trainStartDate
    trainEndDate = args.trainEndDate
    if args.validationSplit > 0 and args.validationSplit < 1:
        useValSet = True
        trainStartDate, trainEndDate, valStartDate, valEndDate = generate_train_val_dates(
            args.trainStartDate, args.trainEndDate, args.validationSplit)
        print(trainStartDate, trainEndDate, valStartDate, valEndDate)

    # Make Dataset
    trainDataset = PatchTSDataset(args.dataDir, trainStartDate, trainEndDate,
                                  contWindowsFilePath=os.path.join(os.getcwd(), 'datasets', 'data', 'contWindows.pkl'))
    sampler = DistributedSampler(trainDataset, num_replicas=args.nGPUs, rank=rank)
    # Create DataLoader
    trainDataLoader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=False,
                                 num_workers=0, drop_last=True, pin_memory=True, sampler=sampler)

    if useValSet:
        # Make Dataset
        normalizationParams = trainDataset.getNormalizationParams()
        valDataset = PatchTSDataset(args.dataDir, valStartDate, valEndDate,
                                    normalization_params=normalizationParams,
                                    contWindowsFilePath=os.path.join(os.getcwd(), 'datasets', 'data', 'contWindowsVal.pkl'))
        valSampler = DistributedSampler(valDataset, num_replicas=args.nGPUs, rank=rank)
        # Create DataLoader
        valDataLoader = DataLoader(valDataset, batch_size=args.batchSize, shuffle=False,
                                   num_workers=0, drop_last=True, pin_memory=True, sampler=valSampler)
    
    # Initialization
    ### Setup gpu device
    device = torch.device('cuda', rank)
    ### Create model
    history_shape = trainDataset.__getitem__(0)[0].shape
    target_shape = trainDataset.__getitem__(0)[1].shape
    if rank==0:
        print(history_shape, target_shape)
    input_features = history_shape[1]
    output_steps = target_shape[0]
    model = PatchTransformer(in_features=input_features, context_window=history_shape[0],
                             d_model=128, nhead=16, num_layers=3, target_window=output_steps)
    ### Put the model to the device
    model.to(device)
    ### Apply DDP
    modelDDP = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model = modelDDP.module
    ### Define loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ### Check if training is to be resumed
    if args.resumeTraining and os.path.isfile(args.ckptSavePath):
        checkpoint = torch.load(args.ckptSavePath)
        startEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        running_loss = checkpoint['loss']
        loss_y = checkpoint['lossHistory']
        val_loss_y = checkpoint['valLossHistory']
        times = checkpoint['timeHistory']
        if rank==0:
            print(f'Resuming training from epoch {startEpoch} with loss {running_loss}')
    else:
        startEpoch = 0
        # Training steps and losses tracking
        loss_y = []
        # Validation steps and losses tracking
        val_loss_y = []
        # Measure time
        times = []
    
    # Define your patience and threshold as input parameters
    patience = args.esPatience  # Number of epochs to wait for improvement
    threshold = args.esThreshold  # Minimum improvement required in validation loss
    # Initialize variables for early stopping
    best_val_loss = float('inf')  # Set to positive infinity initially
    no_improvement_count = 0  # Counter for consecutive epochs with no improvement

    # Measure time
    torch.cuda.synchronize()
    end_init = time.time()
    init_time = end_init - start_init
    if rank==0:
        print(f'\n------- Init_time: {init_time} sec -------\n')
    ### Start training loop
    for epoch in range(startEpoch, args.numEpochs):
        start_epoch = time.time()
        running_loss = 0.0
        model.train()  # Set the model to training mode
        progress_bar = tqdm(enumerate(trainDataLoader, 0), desc=f'Epoch [{epoch + 1}/{args.numEpochs}]', ncols=100, unit="batch", total=len(trainDataLoader))
        for i, batch in progress_bar:
            # Transfer batch data to device
            history = batch[0].float().to(device)
            target = batch[1].float().to(device)
            # Forward pass
            outputs = model(history)
            loss = criterion(outputs, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute loss and set progress bar
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1), refresh=False)  # Update the progress bar with the current loss
        progress_bar.close()
        avg_validation_loss = None
        if useValSet:
            # Validation loop
            model.eval()  # Set the model to evaluation mode
            validation_loss = 0.0
            with torch.no_grad():
                for i, batch in enumerate(valDataLoader, 0):
                    # Transfer batch data to device
                    history = batch[0].float().to(device)
                    target = batch[1].float().to(device)
                    # Forward pass
                    outputs = model(history)
                    loss = criterion(outputs, target)
                    # Compute validation loss
                    validation_loss += loss.item()
            # Calculate average validation loss
            avg_validation_loss = validation_loss / len(valDataLoader)
            # Synchronize the loss across all GPUs
            dist.barrier()
            avg_validation_loss_tensor = torch.tensor(avg_validation_loss, device=device)
            dist.broadcast(avg_validation_loss_tensor, 0)
            avg_validation_loss = avg_validation_loss_tensor.item()
            dist.barrier()
            # Check if the validation loss has improved
            val_loss_improvement = best_val_loss - avg_validation_loss
            if val_loss_improvement >= threshold:
                best_val_loss = avg_validation_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if rank == 0:
                    print(f'No significant validation set improvement seen! Improvement = {val_loss_improvement}')
            # If there's no improvement for 'patience' epochs, stop training
            if no_improvement_count >= patience and rank==0:
                early_stopping_decision = 1
                print(f'Early stopping condition encountered! No improvement for {patience} epochs on rank {rank}.')
            else:
                early_stopping_decision = 0
            # Broadcast Early Stopping ACROSS ALL GPUs
            dist.barrier()
            early_stopping_decision_tensor = torch.tensor(early_stopping_decision, device=device)
            dist.broadcast(early_stopping_decision_tensor, 0)
            early_stopping_decision = early_stopping_decision_tensor.item()
            dist.barrier()
        # Measure time
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)
        print(f'Rank: {rank} | Epoch [{epoch + 1}/{args.numEpochs}] | Training Loss: {running_loss / len(trainDataLoader):.4f} | Val Loss Improvement: {val_loss_improvement:.6f}')
        ### Append losses of each step into the lists
        loss_y.append(running_loss / len(trainDataLoader))
        val_loss_y.append(avg_validation_loss)
        dist.barrier()
        # Reduce losses and times across all GPUs
        loss_y_tensor = torch.tensor(loss_y, device=device)
        val_loss_y_tensor = torch.tensor(val_loss_y, device=device)
        times_tensor = torch.tensor(times, device=device)
        dist.all_reduce(loss_y_tensor, op=dist.ReduceOp.SUM)
        if not None in val_loss_y:
            dist.all_reduce(val_loss_y_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(times_tensor, op=dist.ReduceOp.SUM)
        loss_y_tensor = torch.div(loss_y_tensor, args.nGPUs)
        if not None in val_loss_y:
            val_loss_y_tensor = torch.div(val_loss_y_tensor, args.nGPUs)
        times_tensor = torch.div(times_tensor, args.nGPUs)
        # Save checkpoint every ckptInterval epochs - only from GPU: 0
        if rank==0 and args.ckptInterval is not None and args.ckptSavePath is not None:
            if (epoch + 1) % args.ckptInterval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(trainDataLoader),
                    'lossHistory': loss_y_tensor.tolist(),
                    'valLossHistory': val_loss_y_tensor.tolist(),
                    'timeHistory': times_tensor.tolist()
                }, args.ckptSavePath)
                print(f'Saved checkpoint at epoch {epoch + 1}')
        # If GPU 0 has decided to stop early, break the training loop
        if early_stopping_decision > 0:
            print(f'Early stopping decision taken on rank {rank}!')
            break
    ### Save the trained model
    if rank==0:
        torch.save(model.state_dict(), args.modelSavePath)
        # Measure time
        avg_time = sum(times_tensor.tolist())/(args.numEpochs)
        print(f'\n------- Avg_time: {avg_time} sec -------\n')

        ### Plot the losses
        epoch_x = np.linspace(1, epoch+1, epoch+1).astype(int)
        plt.plot(epoch_x, loss_y_tensor.tolist(), label='Train Loss')
        if useValSet:
            plt.plot(epoch_x, val_loss_y_tensor.tolist(), label='Val Loss')
        plt.legend()
        plt.savefig(f'lossPatchTS.pdf')