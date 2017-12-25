import torch
import torch.autograd as autograd
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.evaluation_utils as eval_utils
from models.Adversary import Adversary
from models.LabelPredictor import LabelPredictor
from itertools import izip

def train_model(label_predictor_train_data, adversary_train_data_generator, dev_data, encoder_model, domain_classifier_model, args):

    if args.cuda:
        encoder_model = encoder_model.cuda()
        domain_classifier_model = domain_classifier_model.cuda()

    encoder_optimizer = torch.optim.Adam([
        {'params': encoder_model.encoder.parameters()},
        {'params': encoder_model.reconstructor.parameters(), 'lr': args.reconstructor_lr}] , lr=args.encoder_lr)
    domain_classifier_lr = -args.domain_classifier_lr # we set the learning rate to negative to train the adversary
    domain_classifier_optimizer = torch.optim.Adam(domain_classifier_model.parameters(), lr=domain_classifier_lr)

    label_predictor = LabelPredictor(encoder_model)
    adversary = Adversary(encoder_model, domain_classifier_model)

    label_predictor.train()
    adversary.train()

    for epoch in range(1, args.epochs+1):

        print "-------------\nEpoch {}:\n".format(epoch)
        
        loss = run_epoch(label_predictor_train_data, adversary_train_data_generator, True, label_predictor, adversary, encoder_optimizer, domain_classifier_optimizer, args)
        print 'Train loss: {:.6f}\n'.format(loss)

        eval_utils.evaluate_model(dev_data, label_predictor, args)

        # Save model
        torch.save(encoder_model, join(args.save_path,'encoder_epoch_{}.pt'.format(epoch)))
        torch.save(domain_classifier_model, join(args.save_path,'domain_classifier_epoch_{}.pt'.format(epoch)))

def mse_loss(input, target):
    return torch.sum((input - target)^2) / input.data.nelement()

def run_epoch(label_predictor_train_data, adversary_train_data_generator, is_training, label_predictor, adversary,
              encoder_optimizer, domain_classifier_optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader_label_predictor = torch.utils.data.DataLoader(
        label_predictor_train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    dataset_size = len(label_predictor_train_data)
    adversary_train_data = adversary_train_data_generator.get_new_dataset(dataset_size)
    data_loader_train_adversary = torch.utils.data.DataLoader(
        adversary_train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    losses = []

    if is_training:
        label_predictor.train()
        adversary.train()
    else:
        label_predictor.eval()
        adversary.eval()

    encoder_loss_function = torch.nn.MultiMarginLoss(p=1, margin=args.margin, weight=None, size_average=True)
    domain_classifier_loss_function = torch.nn.BCELoss(weight=None, size_average=True)

    for label_predictor_batch, adversary_batch in tqdm(izip(data_loader_label_predictor, data_loader_train_adversary)):

        q_title_tensors = autograd.Variable(label_predictor_batch['qid_title_tensor'])
        q_body_tensors = autograd.Variable(label_predictor_batch['qid_body_tensor'])

        candidate_title_tensors = torch.stack(label_predictor_batch['candidate_title_tensors'])
        candidate_body_tensors = torch.stack(label_predictor_batch['candidate_body_tensors'])

        # if args.debug: misc_utils.print_shape_tensor('candidate_body_tensors', candidate_body_tensors)

        # For the adversary:
        title_tensors = autograd.Variable(torch.stack(adversary_batch['title_tensors']))
        body_tensors = autograd.Variable(torch.stack(adversary_batch['body_tensors']))

        # Generate random sampling of negative examples
        # We do - 1 because candidate_title_tensors includes the title tensor for the query itself (at position 0)
        num_available_candidates = candidate_title_tensors.shape[0] - 1
        num_negative = min(num_available_candidates, args.num_negative)
        inds3d = np.zeros((num_negative + 1, args.batch_size, args.len_query),dtype=np.long)
        for i in xrange(args.batch_size):
            # we do + 1 so we are only choosing among the negative examples (0 is reserved for the query itself)
            random_sample = np.random.choice(num_available_candidates, num_negative) + 1
            random_sample3d = np.expand_dims(random_sample,1)
            inds3d[1:,i,:] = random_sample3d.repeat(args.len_query, 1)
        inds3d = torch.LongTensor(inds3d)

        selected_candidate_title_tensors = autograd.Variable(candidate_title_tensors.gather(0,inds3d))
        selected_candidate_body_tensors = autograd.Variable(candidate_body_tensors.gather(0,inds3d))
        # if args.debug: misc_utils.print_shape_variable('selected_candidate_body_tensors', selected_candidate_body_tensors)

        MML_targets = autograd.Variable(torch.LongTensor([0]*args.batch_size))
        # if args.debug: misc_utils.print_shape_variable('MML_targets', MML_targets)

        BCE_targets = autograd.Variable(torch.stack(adversary_batch['labels']))
        # if args.debug: misc_utils.print_shape_variable('BCE_targets', BCE_targets)

        if args.cuda:
            q_title_tensors = q_title_tensors.cuda()
            q_body_tensors = q_body_tensors.cuda()
            selected_candidate_title_tensors = selected_candidate_title_tensors.cuda()
            selected_candidate_body_tensors = selected_candidate_body_tensors.cuda()

            title_tensors = title_tensors.cuda()
            body_tensors = body_tensors.cuda()
            MML_targets = MML_targets.cuda()
            BCE_targets = BCE_targets.cuda()

        if is_training:
            encoder_optimizer.zero_grad()
            domain_classifier_optimizer.zero_grad()

        cosine_similarities, reconstruction_loss = label_predictor(q_title_tensors, q_body_tensors,
                                    selected_candidate_title_tensors, selected_candidate_body_tensors)
        domain_labels = adversary(title_tensors, body_tensors)

        encoder_loss = encoder_loss_function(cosine_similarities, MML_targets)
        domain_classifier_loss = domain_classifier_loss_function(domain_labels, BCE_targets)

        loss = (args.reconstruction_lam * reconstruction_loss) + encoder_loss - (args.domain_classifier_lam * domain_classifier_loss)

        if is_training:
            loss.backward()
            encoder_optimizer.step()
            domain_classifier_optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss