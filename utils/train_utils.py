import torch
import torch.autograd as autograd
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


def train_model(train_data, dev_data, model, args):

    if args.cuda:
        model = model.cuda()

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

    print "GOT HERE"
    for batch in data_loader:

        q_title_tensors = autograd.Variable(batch['qid_title_tensor'])
        q_body_tensors = autograd.Variable(batch['qid_body_tensor'])
        candidate_title_tensors = autograd.Variable(torch.stack(batch['candidate_title_tensors']))
        candidate_body_tensors = autograd.Variable(torch.stack(batch['candidate_body_tensors']))

        targets = autograd.Variable(torch.LongTensor([0]*args.batch_size))

        if args.cuda:
            q_title_tensors = q_title_tensors.cuda()
            q_title_tensors = q_title_tensors.cuda()
            q_body_tensors = q_body_tensors.cuda()
            candidate_title_tensors = candidate_title_tensors.cuda()
            candidate_body_tensors = candidate_body_tensors.cuda()
            targets = targets.cuda()

        if is_training:
            optimizer.zero_grad()

        cosine_similarities = model(q_title_tensors, q_body_tensors, candidate_title_tensors, candidate_body_tensors)

        loss = loss_function(cosine_similarities, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss