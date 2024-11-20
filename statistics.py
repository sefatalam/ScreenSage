import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# read in movies file
movies = pd.read_csv("./movies.csv", sep="\t")

######################################################
### print some simple statistics about the dataset ###
######################################################

# statistics for averageRating
print("Count avgRating: ", movies["averageRating"].count())
print("Mean avgRating: ", movies["averageRating"].mean())
print("Max avgRating: ", movies["averageRating"].max())
print("Min avgRating: ", movies["averageRating"].min())
print("Std avgRating: ", movies["averageRating"].std())
print("Q1 avgRating: ", movies["averageRating"].quantile(q=0.25))
print("Q2 avgRating: ", movies["averageRating"].quantile(q=0.5))
print("Q3 avgRating: ", movies["averageRating"].quantile(q=0.75))

# statistics for runtimeMinutes
print("Count runtimeMinutes: ", movies["runtimeMinutes"].count())
print("Mean runtimeMinutes: ", movies["runtimeMinutes"].mean())
print("Max runtimeMinutes: ", movies["runtimeMinutes"].max())
print("Min runtimeMinutes: ", movies["runtimeMinutes"].min())
print("Std runtimeMinutes: ", movies["runtimeMinutes"].std())
print("Q1 runtimeMinutes: ", movies["runtimeMinutes"].quantile(q=0.25))
print("Q2 runtimeMinutes: ", movies["runtimeMinutes"].quantile(q=0.5))
print("Q3 runtimeMinutes: ", movies["runtimeMinutes"].quantile(q=0.75))

# statistics for numVotes
print("Count numVotes: ", movies["numVotes"].count())
print("Mean numVotes: ", movies["numVotes"].mean())
print("Max numVotes: ", movies["numVotes"].max())
print("Min numVotes: ", movies["numVotes"].min())
print("Std numVotes: ", movies["numVotes"].std())
print("Q1 numVotes: ", movies["numVotes"].quantile(q=0.25))
print("Q2 numVotes: ", movies["numVotes"].quantile(q=0.5))
print("Q3 numVotes: ", movies["numVotes"].quantile(q=0.75))

# count adult movies
adult = movies[movies["isAdult"] == 1]
print("There are ", len(adult), " adult movies.")

# print and count all genres
genre_lists = movies['genres'].str.split(',')
unique_genres = set(genre for genres in genre_lists for genre in genres)
print(unique_genres)
print('There are ', len(unique_genres), 'unique genres in the dataset.')

# count unique directors
directors_lists = movies['directors'].str.split(',')
unique_directors = set(director for directors in directors_lists for director in directors)
print('There are ', len(unique_directors), 'unique directors in the dataset.')

# count unique writers
writers_lists = movies['writers'].str.split(',')
unique_writers = set(writer for writers in writers_lists for writer in writers)
print('There are ', len(unique_writers), 'unique writers in the dataset.')


#########################
### create some plots ###
#########################

# prepare genre list for word cloud
flattened_genres = [genre for sublist in genre_lists for genre in sublist]
genre_counts = Counter(flattened_genres)

# create a word cloud with genre frequencies
wordcloud = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(genre_counts)

# display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.show()

# plot distribution of average movie ratings
counts, edges, bars = plt.hist(movies['averageRating'])
plt.title("distribution of average movie ratings")
plt.ylabel("counts rating")
plt.xlabel("rating")
plt.bar_label(bars, labels=[int(v) if v > 0 else '' for v in bars.datavalues], label_type='edge')

plt.show()

###################################################################################
### check if distribution of movie rating in training and test dataset is equal ###
###################################################################################

# read in test and training dataset
test_data = pd.read_csv("./testData.csv", sep="\t")
training_data = pd.read_csv("./trainingData.csv", sep="\t")

# plot distribution of average rating in test dataset
counts, edges, bars = plt.hist(test_data['averageRating'])
plt.title("distribution of average movie ratings in test dataset")
plt.ylabel("counts rating")
plt.xlabel("rating")
plt.bar_label(bars, labels=[int(v) if v > 0 else '' for v in bars.datavalues], label_type='edge')

plt.show()

# plot distribution of average rating in training dataset
counts, edges, bars = plt.hist(training_data['averageRating'])
plt.title("distribution of average movie ratings in training dataset")
plt.ylabel("counts rating")
plt.xlabel("rating")
plt.bar_label(bars, labels=[int(v) if v > 0 else '' for v in bars.datavalues], label_type='edge')

plt.show()

# count how many films a director has made and print top 20
director_counts = Counter(director for directors in directors_lists for director in directors)
print(director_counts.most_common(20))

# plot distribution for top 100 directors
director_counts = movies['directors'].value_counts()[1:101]
plt.figure(figsize=(10, 6))
director_counts.plot(kind='bar')
plt.title('Number of films per director (top 100)')
plt.xlabel('director')
plt.ylabel('number of films')
plt.xticks(rotation=45)
plt.xticks(fontsize=5)
plt.tight_layout()

plt.show()

##########################################
### create plots for our first results ###
##########################################

# open and read the text file containing the predicted ratings
with open('predicted_ratings.txt', 'r') as file:
    lines = file.readlines()

# initialization of lists for tconst and predicted ratings
tconst_list = []
predicted_list = []

# going through each line in the file
for line in lines:
    # split the line after the colon to get the tconst and score
    parts = line.strip().split(': ')
    if len(parts) == 2:
        tconst = parts[0].split()[-1]  # the tconst value should be the last part after spaces
        rating = float(parts[1].replace(',', '.'))  # for evaluation, if necessary convert comma to a point

        # adding the tconst and rating to the lists
        tconst_list.append(tconst)
        predicted_list.append(rating)

# initialization of an empty list
predicted_list_rounded = []

# loop over the indices of the tconst_list and their associated predicted scores
for i in range(len(tconst_list)):
    tconst = tconst_list[i]
    rating = predicted_list[i]

    # round the rating to 0 decimal place
    rounded_rating = round(rating, 0)
    predicted_list_rounded.append(rounded_rating)

# extract the actual ratings for the movies contained in tconst_list
actual_ratings = movies[movies['tconst'].isin(tconst_list)]['averageRating']
actual_list = actual_ratings.tolist()  # convert the column to a list
actual_list_rounded = [round(value, 0) for value in actual_list]  # round the values in the list

# creating the Confusion Matrix
confusion_mat = confusion_matrix(actual_list_rounded, predicted_list_rounded, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('predicted ratings')
plt.ylabel('actual ratings')
plt.title('Confusion Matrix for prediction with cosine similarity only')

plt.show()

##########################
### plot final results ###
##########################

# open and read the text file containing the predicted ratings
with open('predicted_ratings3.txt', 'r') as file:
    # read the line in the file
    line = file.readline()

# remove the outer brackets and separate the triples
triples = line.strip('()').split(', ')

# initialize the lists
tconst_list = []
predicted_ratings_list = []
time_list = []

# iterate over the triples and separate the values
for triple in triples:
    # Remove the brackets from the triplets and split the values
    values = triple.strip('()').split(',')
    tconst_list.append(values[0].strip())
    predicted_ratings_list.append(float(values[1]))
    time_list.append(float(values[2]))

# print the lists for review
print("tconst_list:", tconst_list)
print("predicted_ratings_list:", predicted_ratings_list)
print("time_list:", time_list)

predicted_list_rounded = []

# loop over the indices of the tconst_list and their associated predicted scores
for i in range(len(tconst_list)):
    tconst = tconst_list[i]
    rating = predicted_ratings_list[i]

    # round the rating to 0 decimal place
    rounded_rating = round(rating, 0)
    predicted_list_rounded.append(rounded_rating)

# extract the actual ratings for the movies contained in tconst_list
actual_ratings = movies[movies['tconst'].isin(tconst_list)]['averageRating']
actual_list = actual_ratings.tolist()  # convert the column to a list
actual_list_rounded = [round(value, 0) for value in actual_list]  # round the values in the list

# creating the Confusion Matrix
confusion_mat = confusion_matrix(actual_list_rounded, predicted_list_rounded, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('predicted ratings')
plt.ylabel('actual ratings')
plt.title('Confusion Matrix for prediction with cosine similarity and LSH')

plt.show()

# print average time for film rating prediction
average_time = sum(time_list) / len(time_list)
print(f"It takes on average {average_time} seconds to predict a film rating.")

# convert the lists to NumPy arrays for mathematical calculations
actual_ratings = np.array(actual_list)
predicted_ratings = np.array(predicted_ratings_list)

# calculate Squared Error (MSE)
mse = ((actual_ratings - predicted_ratings) ** 2).mean()

# calculate the RMSE by taking the square root of the MSE
rmse = np.sqrt(mse)
print("RMSE:", rmse)
