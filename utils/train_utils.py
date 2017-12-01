import torch
import torch.autograd as autograd
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.evaluation_utils as eval_utils
import utils.misc_utils as misc_utils

NUM_NEGATIVE_EXCEPTION_MESSAGE = "The number of negative examples desired ({}) is larger than that available ({})."

# TODO: add dropout
def train_model(train_data, dev_data, model, args):

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)

    model.train()

    for epoch in range(1, args.epochs+1):

        print "-------------\nEpoch {}:\n".format(epoch)

        loss = run_epoch(train_data, True, model, optimizer, args)

        print 'Train MSE loss: {:.6f}\n'.format(loss)

        eval_utils.evaluate_model(dev_data, model, args)

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

        q_title_tensors = autograd.Variable(batch['qid_title_tensor'])
        q_body_tensors = autograd.Variable(batch['qid_body_tensor'])

        candidate_title_tensors = torch.stack(batch['candidate_title_tensors'])
        candidate_body_tensors = torch.stack(batch['candidate_body_tensors'])
        if args.debug: misc_utils.print_shape_tensor('candidate_body_tensors', candidate_body_tensors)

        # Generate random sampling of negative examples
        # We do - 1 because candidate_title_tensors includes the title tensor for the query itself (at position 0)
        num_available_candidates = candidate_title_tensors.shape[0] - 1
        num_negative = min(num_available_candidates, args.num_negative)
        inds = np.zeros((num_negative + 1, args.batch_size, args.len_query),dtype=np.long)
        for i in xrange(args.batch_size):
            # we do + 1 so we are only choosing among the negative examples (0 is reserved for the query itself)
            random_sample = np.expand_dims(np.random.choice(num_available_candidates, num_negative) + 1,1)
            inds[1:,i,:] = random_sample.repeat(args.len_query, 1)
        inds = torch.LongTensor(inds)


        selected_candidate_title_tensors = autograd.Variable(candidate_title_tensors.gather(0,inds))
        selected_candidate_body_tensors = autograd.Variable(candidate_body_tensors.gather(0,inds))
        if args.debug: misc_utils.print_shape_variable('selected_candidate_body_tensors', selected_candidate_body_tensors)


        targets = autograd.Variable(torch.LongTensor([0]*args.batch_size))
        if args.debug: misc_utils.print_shape_variable('targets', targets)

        if args.cuda:
            q_title_tensors = q_title_tensors.cuda()
            q_body_tensors = q_body_tensors.cuda()
            selected_candidate_title_tensors = selected_candidate_title_tensors.cuda()
            selected_candidate_body_tensors = selected_candidate_body_tensors.cuda()
            targets = targets.cuda()

        if is_training:
            optimizer.zero_grad()

        cosine_similarities = model(q_title_tensors, q_body_tensors, selected_candidate_title_tensors, selected_candidate_body_tensors)
        if args.debug: misc_utils.print_shape_variable('cosine_similarities', cosine_similarities)


        loss = loss_function(cosine_similarities, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss