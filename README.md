# LSTM GRU Movie Review Classification
Provides script for train LSTM or GRU model for sentence classification.

## Training Data
The entry for training dataset was a sentence of movie review on IMDB, and value was either 0 representing a negative review, or 1 for podsitive review. 

## Testing Data
The program provides a function predice(), that takes a list of sentence for review, and the movie name. It returns a turple contains the movie name, and a list of float rating from 0 to 1 for each sentence in the list of sentence.


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
