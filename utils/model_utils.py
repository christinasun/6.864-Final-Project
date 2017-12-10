import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import models.AskUbuntuModels as AskUbuntuModels
from models.Encoders import EncoderModel
from models.DomainClassifiers import DomainClassifierModel


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        return AskUbuntuModels.CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return AskUbuntuModels.LSTM(embeddings, args)
    elif args.model_name == 'dan':
        return AskUbuntuModels.DAN(embeddings, args)
    elif args.model_name == 'adt':
        return EncoderModel(embeddings, args), DomainClassifierModel(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))



