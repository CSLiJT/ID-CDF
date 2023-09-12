# The Code of Identifiable Cognitive Diagnosis Model (ID-CDM)

## Configuring Running Environment
> pip install -r requirements.txt

## Running ID-CDM
- Start training and testing ID-CDM:
> python run.py --train_file .data/math1_train_0.8_0.2.csv --valid_file .data/math1_valid_0.8_0.2.csv --test_file .data/math1_test_0.8_0.2.csv --Q_matrix data/math1_Q_matrix.npy --save_path ./result/ID-CDM-Math1 --eta 1.0 --n_user 4209 --n_item 20 --n_know 11 --user_dim 32 --item_dim 32 --batch_size 32 --lr 0.0005 --epoch 5 --device cpu

- If you have any question about the configuration of the running program, please use:
> python run.py --help