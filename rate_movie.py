"""
    This script provides an example for using the trained model
"""
# Import dependency
from nlp_model import predict 

# Define parameters
movie = "Snow White and 7 Assholes"
review = ["This is the worst movie I have ever seen!"]

model = 'lstm_movie_model.h5'

# Perform Prediction
result = predict(movie,review,model)

print("Movie Nmae: ", movie)
print("Review: ", review)

print(result)