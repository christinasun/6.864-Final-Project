import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import models.AskUbuntuModels as AskUbuntuModels
from models.BaselineModel import BaselineModel


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        return AskUbuntuModels.CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return AskUbuntuModels.LSTM(embeddings, args)
    elif args.model_name == 'dan':
        return AskUbuntuModels.DAN(embeddings, args)
    elif args.model_name == 'bow':
        return BaselineModel()
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))



