import torch
import torch.autograd as autograd
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


def train_model(train_data, dev_data, model, args):

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)

    model.train()

    for epoch in range(1, args.epochs+1):

        print "-------------\nEpoch {}:\n".format(epoch)

        loss = run_epoch(train_data, True, model, optimizer, args)

        print 'Train MSE loss: {:.6f}\n'.format(loss)

        # val_loss = run_epoch(dev_data, False, model, optimizer, args)
        # print('Val MSE loss: {:.6f}'.format( val_loss))

        # Save model
        torch.save(model, args.save_path)


def run_epoch(data, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    losses = []

    if is_training:
        model.train()
    else:
        model.eval()

    loss_function = torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True)

    for batch in tqdm(data_loader):

        q_tensors, candidate_tensors = autograd.Variable(batch['qid_tensors']), autograd.Variable(batch['candidate_tensors'])
        print "Len candidate tensors"
        print len(candidate_tensors)
        targets = autograd.Variable(torch.from_numpy(np.zeros(len(candidate_tensors))))

        if is_training:
            optimizer.zero_grad()

        cosine_similarities = model(q_tensors, candidate_tensors)

        loss = loss_function(cosine_similarities, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss