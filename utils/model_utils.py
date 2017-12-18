import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from models.Encoders import CNN
from models.Encoders import LSTM
from models.DomainClassifier import DomainClassifier

# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        return CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    elif args.model_name == 'adt-lstm':
        return LSTM(embeddings, args), DomainClassifier(args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))
