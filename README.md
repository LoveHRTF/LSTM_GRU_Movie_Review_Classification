# LSTM_GRU_Movie_Review_Classification
Provides script for train LSTM or GRU model for sentence classification.


# Usage
## Train
$ python nlp_model.py

## Test
### General Test
$ python rate_movie.py

### Test on New Youk Times article on Avangers: EndGame
$ python rete_endgame.py

## Dependency
tensorflow-gpu was recommended

## Run Time
### GPU
Training LSTM model on a Nvidia 2080Ti cost less than 2 minutes under default settings and parameters. Where batch_size = 1024, epoch = 20

### CPU
Training LSTM model on a 6 core Intel i7-8850H cost around 10 minutes under default settings and parameters. Where batch_size = 1024, epoch = 20
