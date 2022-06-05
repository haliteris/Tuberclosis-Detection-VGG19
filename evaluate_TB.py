import torch
from sklearn import metrics
from torch.autograd import Variable

def run_model(model, loader, train=False, optimizer=None):
    preds     = []
    labels    = []
    preds_acc = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()

        vol, label = batch
        
        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)
        
        if(pred_npy>0.5):
            preds_acc.append(1.0)
        else:
            preds_acc.append(0.0)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)
    
    acc = (metrics.accuracy_score(labels,preds_acc))*100
    print("Accuracy=",round(acc,1))
    
    return avg_loss, auc, preds, labels