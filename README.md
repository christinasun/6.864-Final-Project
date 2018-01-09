# 6.864-Final-Project


To run the following commands, first download the pruned glove embeddings and saved models from here: https://www.dropbox.com/sh/yjps1opsa0drm9r/AAAlBByJgx6YTVR1z40UH6uNa?dl=0

Put pruned_glove.840B.300d.txt inside the data folder. Put the saved_models folder the main folder. 

Use PyTorch version 0.2.0_3

## To train the models with reported parameters, run: 

**CNN:** 
* python scripts/ubuntu_main.py --model_name cnn --hidden_dim 670 --lr .001 --dropout 0 --margin 0.5 --train --cuda

**LSTM:** 
* python scripts/ubuntu_main.py --model_name lstm --hidden_dim 240 --lr .0003 --dropout 0 --margin 0.5 --train --cuda

**TF-IDF:** 
* python scripts/tfidf_main.py

**Direct Transfer (CNN):**
* python scripts/direct_transfer_main.py TODO

**Direct Transfer (LSTM):** 
* python scripts/direct_transfer_main.py --hidden_dim 110 --lr .0003 --dropout 0.4 --margin 0.2 --train --cuda

**Adversarial Domain Transfer (CNN):**
* python scripts/transfer_main.py TODO

**Adversarial Domain Transfer (LSTM):** 
* python scripts/transfer_main.py --hidden_dim 110 --encoder_lr .0003 --domain_classifier_lr .00001 --dropout 0.4 --margin 0.2 --lam .01 --train --cuda

**Exploration (CNN):**
* python scripts/exploration_main.py TODO

**Exploration (LSTM):**
* python scripts/exploration_main.py TODO

## To evaluate the best performing models (in saved_models), run: 

**CNN:** 
* python scripts/ubuntu_main.py --snapshot saved_models/cnn_encoder.pt --eval --cuda

**LSTM:** 
* python scripts/ubuntu_main.py --snapshot saved_models/lstm_encoder.pt --eval --cuda

**TF-IDF:** 
* python scripts/tfidf_main.py

**Direct Transfer (CNN):**
* python scripts/direct_transfer_main.py --snapshot saved_models/direct_transfer_cnn_encoder.pt --eval --cuda

**Direct Transfer (LSTM):**
* python scripts/direct_transfer_main.py --snapshot saved_models/direct_transfer_lstm_encoder.pt --eval --cuda

**Adversarial Domain Transfer (CNN):**
* python scripts/transfer_main.py --encoder_snapshot saved_models/adt_cnn_encoder.pt --domain_classifier_snapshot saved_models/adt_cnn_domain_classifier.pt --eval --cuda

**Adversarial Domain Transfer (LSTM):**
* python scripts/transfer_main.py --encoder_snapshot saved_models/adt_lstm_encoder.pt --domain_classifier_snapshot saved_models/adt_lstm_domain_classifier.pt --eval --cuda

**Exploration (CNN):**
* python scripts/exploration_main.py --encoder_snapshot saved_models/exploration_cnn_encoder.pt --domain_classifier_snapshot saved_models/exploration_cnn_domain_classifier.pt --eval --cuda

**Exploration (LSTM):**
* python scripts/exploration_main.py --encoder_snapshot saved_models/exploration_lstm_encoder.pt --domain_classifier_snapshot saved_models/exploration_lstm_domain_classifier.pt --eval --cuda
