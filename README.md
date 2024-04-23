# Seq2Seq Model

This is an implementation of a Seq2Seq model on NLG task —— Question Generation. The basic structure is an encoder with two-layer Bidirectional LSTM and a decoder with two-layer LSTM.

Our work is based on [the work of jiangqn](https://github.com/jiangqn/natural-question-generation/tree/master) and we make other contribution to improve the code quality and augment the code function.

Besides, this code has been submitted to a certain course in 2024. It is a part of group project and we also try large language models like BART and T5. You can find these jupyter notebooks in other repository.

To run the code, please install related python packages first. Then, download `en_core_web_sm` with `python -m spacy download en_core_web_sm`, download `suqad nqg dataset` including `train.json, dev.json and test.json` to `./data/squadnqg`and download `glove.840B.300d.txt` to `./data/vocab/` for word embedding. 

+ Run `initial_squad.py` to generate raw data from the SQuAD dataset
+ Run `initial_squad_nqg.py` to generate raw data from the SQUAD NQG dataset
+ Run `preprocess.py` to preprocess raw data
+ Run `train.py` to train the model with train set and dev set
+ Run `evaluation.py` to evaluate the model with test set

`Note: Please modify the training parameters, like vocab size, rnn-type and attention-type, on both train.py and evaluation.py.` 

`Make sure the vocab size is corresponding to data_log.txt file.`

# Code Organization

    ├── data             
    ├───── checkpoints -- Store the model parameters for each epoch
    ├───── log -- Store the dataset information
    ├───── output -- Store the generated questions based on dev set and test set
    ├───── processed -- Store the preprocessed data
    ├───── raw -- Store the raw data generated from dataset
    ├───── squadnqg -- squad nqg dataset
    ├───── vocab -- Store the glove.840B.300d.txt
    ├────────────────────────────────────────────────
    ├── model
    ├───── attention.py -- Attention class
    ├───── beam_search.py -- Beam search class
    ├───── bridge.py -- Bridge class 
    ├───── criterion.py -- Criterion for word crossentropy or sentence crossentropy
    ├───── decoder.py -- Decoder class
    ├───── encoder.py -- Encoder class
    ├───── multi_layer_rnn_cell.py -- Multilayer LSTM class or GRU class
    ├───── seq2seq.py -- Seq2seq class
    ├───── utils.py -- Other functions such as tokenize and load_word_embeddings
    ├────────────────────────────────────────────────
    ├── dataset.py -- Make the vocab and dataset
    ├── evaluation.py -- Evaluate the model with test set
    ├── initial_squad.py -- Generate raw data from squad dataset
    ├── initial_squad_nqg.py -- Generate raw data from squad nqg dataset
    ├── logger.py -- Logger class to log the information
    ├── preprocess.py -- Preprocess raw data
    ├── train.py -- Train the model with train set and dev set
    ├── README.md



