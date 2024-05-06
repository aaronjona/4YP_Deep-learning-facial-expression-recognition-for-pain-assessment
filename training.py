import torch
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, 
                train_loader, 
                model, 
                criterion, 
                optimizer, 
                batch_logger, 
                epoch_logger, 
                n_classes = 5):
    print('train at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (data, labels) in enumerate(train_loader):
        data_time.update(time.time() - end_time)

        data, labels = data.cuda(), labels.cuda()

        outputs = model(data)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels, n_classes)

        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
        #                                                  i + 1,
        #                                                  len(train_loader),
        #                                                  batch_time=batch_time,
        #                                                  data_time=data_time,
        #                                                  loss=losses,
        #                                                  acc=accuracies))
        
        batch_logger.append({
            'epoch': epoch,
            'batch': i+1,
            'iter': (epoch - 1) * len(train_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val
        })

    # Done training the epoch    
            
    print(f'Epoch: {epoch}; Time: {batch_time.sum}; Total Loss: {losses.sum}; Average Loss: {losses.avg}; Accuracy: {accuracies.avg}')

    epoch_logger.append({
        'epoch': epoch, 
        'total_loss': losses.sum,
        'average_loss': losses.avg,
        'average_accuracy': accuracies.avg
    })


    return losses.avg, accuracies.avg


