import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#importing data from csv to dataframe
movie_data = pd.read_csv('Cine_genius/Recommender/movies.csv')
selected_attr = ['budget','genres','keywords','original_language','original_title','popularity','vote_average','vote_count','cast','director']
#We fill in any null values present in the data
for attr in selected_attr:
  movie_data[attr] = movie_data[attr].fillna('NaN')
 combined_feat = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + movie_data['original_language']+ ' ' + movie_data['original_title'] + ' ' + movie_data['cast'] + ' ' + movie_data['director'] 

#Used to convert string data into int
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_feat)
feature_vectors_df = pd.DataFrame(feature_vectors.toarray(), columns=vectorizer.get_feature_names_out())

#Gives us a correlation between each pair of movies
similarity = cosine_similarity(feature_vectors)

#standin for input movies from user(get using API)
input_movies = ["Iron man","Transformers","Avatar","Titanic","Dark Knight rises"]
#Create a list with all movie names
listofmovies = movie_data['title'].tolist()

#For the whole input data, we convert to closest matches
for i in range(5):
  input_movies[i] = difflib.get_close_matches(input_movies[i],listofmovies)[0]

 #We need the index of the movie in our data for further operations on the data
indexofmovies=[]
for i in range(5):
  idx_mv = movie_data[movie_data.title == input_movies[i]]['index'].values[0]
   indexofmovies.append(idx_mv)

#Now to get the similarity scores for all movies with our current i/p movies
similarity_scores = []
for i in range(5):
  similarity_scores.append(list(enumerate(similarity[indexofmovies[i]])))
 
#we sort movies in descending order of similarity to the input movies
sorted_similar_movies = []
for i in range(5):
  sorted_similar_movies.append(sorted(similarity_scores,key = lambda x:x[1],reverse = True))

#to_send contains all the recommendations on the basis of the original movies provided
to_send = []
for i in range(5):
  temp = [row[0] for row in sorted_similar_movies[:6]]
  to_send.append(temp)
