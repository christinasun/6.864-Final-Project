# 6.864-Final-Project


To run the following commands, first download the pruned glove embeddings and saved models from here: https://www.dropbox.com/sh/yjps1opsa0drm9r/AAAlBByJgx6YTVR1z40UH6uNa?dl=0

Put pruned_glove.840B.300d.txt inside the data folder. Put the saved_models folder the main folder. The final directory structure should look like this:

* 6.864-Final-Project
    * saved_models
        * cnn_encoder.pt
        * lstm_encoder.pt
        * cnn_adt_dc.pt
        * lstm_adt_dc.pt
        * cnn_adt_encoder.pt
        * lstm_adt_encoder.pt
        * cnn_adt_recon_dc.pt
        * lstm_adt_recon_dc.pt
        * cnn_adt_recon_encoder.pt
        * lstm_adt_recon_encoder.pt
        * cnn_direct_transfer.pt
        * lstm_direct_transfer.pt
        * cnn_encoder.pt
        * lstm_encoder.pt
    * data
        * pruned_glove.840B.300d.txt



Use PyTorch version 0.2.0_3

## To train the models with reported parameters, run: 

**CNN:** 
* python scripts/ubuntu_main.py --model_name cnn --hidden_dim 670 --lr .001 --dropout 0 --margin 0.5 --train --cuda

**LSTM:** 
* python scripts/ubuntu_main.py --model_name lstm --hidden_dim 240 --lr .0003 --dropout 0 --margin 0.5 --train --cuda

**TF-IDF:** 
* python scripts/tfidf_main.py

**Direct Transfer (CNN):**
* python scripts/direct_transfer_main.py --train --cuda --hidden_dim 110 --lr 0.0001 --dropout 0.3 --margin 0.2

**Direct Transfer (LSTM):** 
* python scripts/direct_transfer_main.py --train --cuda --model_name lstm --hidden_dim 110 --lr .0003 --dropout 0.2 --margin 0.2

**Adversarial Domain Transfer (CNN):**
* python scripts/transfer_main.py --train --cuda --hidden_dim 110 --encoder_lr .0001 --dropout 0.3 --margin .2 --domain_classifier_lr .001 --lam .01

**Adversarial Domain Transfer (LSTM):** 
* python scripts/transfer_main.py --train --cuda --model_name 'adt-lstm' --hidden_dim 110 --encoder_lr .0003 --dropout 0.2 --margin 0.2 --domain_classifier_lr .000015 --lam .01

**Exploration (CNN):**
* python scripts/exploration_main.py --train --cuda --hidden_dim 110 --encoder_lr .0001 --dropout 0.3 --margin 0.2 --domain_classifier_lr .001 --domain_classifier_lam .01 --reconstructor_lr .0001 --reconstruction_lam .001

**Exploration (LSTM):**
* python scripts/exploration_main.py --train --cuda --model_name adt-lstm-recon --hidden_dim 110 --encoder_lr .0003 --dropout 0.2 --margin 0.2 --domain_classifier_lr .000015 --domain_classifier_lam .01 --reconstructor_lr .001 --reconstruction_lam .0001


## To evaluate the best performing models (in saved_models), run:

**CNN:** 
* python scripts/ubuntu_main.py --eval --cuda --snapshot saved_models/cnn_encoder.pt

**LSTM:** 
* python scripts/ubuntu_main.py --eval --cuda --snapshot saved_models/lstm_encoder.pt

**TF-IDF:** 
* python scripts/tfidf_main.py

**Direct Transfer (CNN):**
* python scripts/direct_transfer_main.py --eval --cuda --snapshot saved_models/cnn_direct_transfer.pt

**Direct Transfer (LSTM):**
* python scripts/direct_transfer_main.py --eval --cuda --snapshot saved_models/lstm_direct_transfer.pt

**Adversarial Domain Transfer (CNN):**
* python scripts/transfer_main.py --eval --cuda --encoder_snapshot saved_models/cnn_adt_encoder.pt --domain_classifier_snapshot saved_models/cnn_adt_dc.pt

**Adversarial Domain Transfer (LSTM):**
* python scripts/transfer_main.py --eval --cuda --encoder_snapshot saved_models/lstm_adt_encoder.pt --domain_classifier_snapshot saved_models/lstm_adt_dc.pt

**Exploration (CNN):**
* python scripts/exploration_main.py --eval --cuda --encoder_snapshot saved_models/cnn_adt_recon_encoder.pt --domain_classifier_snapshot saved_models/cnn_adt_recon_dc.pt

**Exploration (LSTM):**
* python scripts/exploration_main.py --eval --cuda --encoder_snapshot saved_models/lstm_adt_recon_encoder.pt --domain_classifier_snapshot saved_models/lstm_adt_recon_dc.pt
