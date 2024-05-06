import torch
from utils import AverageMeter, calculate_accuracy
from sklearn.metrics import confusion_matrix

def validation_epoch(epoch, val_loader, model, criterion, val_logger, n_classes = 5):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    all_labels = []
    all_pred = []

    with torch.no_grad():
        for data, labels in val_loader: 
            data, labels = data.cuda(), labels.cuda()

            outputs = model(data)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels, n_classes)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

            if n_classes == 5:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (torch.sigmoid(outputs) >= 0.5).long()

            all_pred.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        conf_matrix = confusion_matrix(all_labels, all_pred)

        print(f'Epoch: {epoch}; Total Loss: {losses.sum}; Average Loss: {losses.avg}; Accuracy: {accuracies.avg}')

        val_logger.append({
            'epoch': epoch, 
            'total_loss': losses.sum,
            'average_loss': losses.avg,
            'average_accuracy': accuracies.avg,
            'conf_matrix': conf_matrix
        })

    return losses.avg, accuracies.avg, conf_matrix

