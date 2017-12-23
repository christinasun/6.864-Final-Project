# 6.864-Final-Project

## To train the models with reported parameters, run: 

**CNN:** 
* python scripts/ubuntu_main.py --model_name cnn --hidden_dim 670 --lr .001 --dropout 0 --margin 0.5 --train --cuda --seed 5 
* (reported model at epoch 4)

**LSTM:** 
* python scripts/ubuntu_main.py --model_name lstm --hidden_dim 240 --lr .0003 --dropout 0 --margin 0.5 --train --cuda --seed 3
* (reported model at epoch 1)

**TF-IDF:** 
* python scripts/tfidf_main.py

**Direct Transfer:** 
* python scripts/direct_transfer_main.py --hidden_dim 110 --lr .0003 --dropout 0.4 --margin 0.2 --train --cuda --seed 5
* (reported model at epoch 4)

**Adversarial Domain Transfer:** 
* python scripts/transfer_main.py --hidden_dim 110 --encoder_lr .0003 --domain_classifier_lr .00001 --dropout 0.4 --margin 0.2 --lam .01 --train --cuda


## To evaluate the best performing models (in saved_models), run: 

**CNN:** 
* python scripts/ubuntu_main.py --snapshot saved_models/final_models/cnn_encoder.pt --eval --cuda

**LSTM:** 
* python scripts/ubuntu_main.py --snapshot saved_models/final_models/lstm_encoder.pt --eval --cuda

**TF-IDF:** 
* python scripts/tfidf_main.py

**Direct Transfer:**
* python scripts/direct_transfer_main.py --snapshot saved_models/final_models/direct_transfer_encoder.pt --eval --cuda

**Adversarial Domain Transfer:** 
* python scripts/transfer_main.py --encoder_snapshot saved_models/final_models/adt_encoder.pt --domain_classifier_snapshot saved_models/final_models/adt_domain_classifier.pt --eval --cuda
