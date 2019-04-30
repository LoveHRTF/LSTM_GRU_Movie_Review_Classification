"""
    This script provides an example for using the trained model
"""
# Import dependency
from nlp_model import predict 

# Define parameters
movie = "End Game"

review1 =  "Sri Lanka tries to move forward."
review2 =  "The superhero film broke box-office records, taking in $1.2 billion worldwide, and led to gridlock at the multiplexes in a fragmented era of streaming."
review3 =  "Fans around the globe packed movie theaters for the debut of Avengers: Endgame over the weekend, pushing total ticket sales for the Walt Disney Co superhero spectacle to a stunning $1.2 billion and crushing records in dozens of countries."
review4 =  "The universe belongs to Marvel. Avengers: Endgame shattered the record for biggest opening weekend with an estimated $350 million in ticket sales domestically and $1.2 billion globally, reaching a new pinnacle in the blockbuster era that the com..."
review5 =  "Here’s what you need to know about the week’s top stories."
review6 =  "Avengers: Endgame is crushing the competition by setting multiple records at the box office a day after its release."
review7 =  "A jeweler’s dark secret. Missing the old Penn Station. Genealogy heats up cold cases. Tom Ford, family man. Finland’s hobbyhorse girls. And more."
review8 =  "Thanos in a T-shirt? Thicc Thor? And what about Captain Marvel’s new haircut? Let’s dive in."
review9 =  "Here’s what you need to know at the end of the day."
review10 =  "Some theaters overseas are providing an intermission for the blockbuster “Avengers: Endgame,” but American moviegoers who want to step away must strategize."
review11 =  "Avengers: Endgame has gotten off to a mighty start at the box office, earning a record $60 million from Thursday night preview showings in North America, according to the Walt Disney Co."
review12 =  "“21 Bridges” looks like a throwback to New York City cop movies like “The French Connection. Other interesting clips include a teaser for “Gemini Man.”"
review13 =  "Marvel superhero spectacle Avengers: Endgame hauled in a record $60 million at U.S. and Canadian box offices on Thursday night, and distributor Walt Disney Co cautiously predicted an unprecedented $300 million weekend debut."
review14 =  "The actress, who played Princess Shuri in “Black Panther” and the “Avengers” movies, knows how to keep a secret. She didn’t say a word about “Guava Island.”"
review15 = "A surprisingly strong report card on the U.S. economy helped power the benchmark S&amp;P 500 and Nasdaq Composite indexes to record high closes on Friday, capping a week of gains for stocks that came largely on the back of resilient corporate prof..."


review = [review1, review2,review3,review4,review5,review6,review7,review8,review9,review10,review11,review12,review13,review14,review15]
model = 'lstm_movie_model.h5'

# Perform Prediction
result = predict(movie,review,model)

print("Movie Nmae: ", movie)
for item in review:
    print("Review: ", item)

print(result)