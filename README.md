# 6.864-Final-Project

To train the models with reported parameters, run: 

CNN: 
python scripts/ubuntu_main.py --model_name cnn --hidden_dim 670 --lr .001 --dropout 0 --margin 0.5 --train --cuda 

LSTM: 
python scripts/ubuntu_main.py --model_name lstm --hidden_dim 240 --lr .0003 --dropout 0 --margin 0.5 --train --cuda 

TF-IDF: 
python scripts/baseline_tfidf_main.py

Direct Transfer: 
python scripts/baseline_transfer_main.py --hidden_dim 110 --lr .0003 --dropout 0.4 --margin 0.2 --train --cuda

Adversarial Domain Transfer: 
python scripts/transfer_main.py --hidden_dim 110 --encoder_lr .0003 --domain_classifier_lr .00001 --dropout 0.4 --margin 0.2 --lam .01 --train --cuda


The best performing models are saved in saved_models. To evaluate these models, run: 

CNN: 

LSTM: 

TF-IDF: 

Direct Transfer: 

Adversarial Domain Transfer: 
python scripts/transfer_main.py --encoder_snapshot saved_models/running_transfer2/encoder_epoch_2.pt --domain_classifier_snapshot saved_models/running_transfer2/domain_classifier_epoch_2.pt --eval --cuda