import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from models.Encoders import CNN
from models.Encoders import CNN_all
from models.Encoders import LSTM
from models.DomainClassifier import DomainClassifier
from models.LabelPredictor import LabelPredictor

# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        encoder = CNN(embeddings, args)
        return LabelPredictor(encoder)
    elif args.model_name == 'lstm':
        encoder = LSTM(embeddings, args)
        return LabelPredictor(encoder)
    elif args.model_name == 'adt-lstm':
        return LSTM(embeddings, args), DomainClassifier(args)
    elif args.model_name == 'adt-cnn':
        return CNN(embeddings, args), DomainClassifier(args)
    elif args.model_name == 'exploration':
        return CNN_all(embeddings, args), DomainClassifier(args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))
