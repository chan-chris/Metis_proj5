README 
Metis Project 5

author: chris chan
date: Mar 4, 2021



## Building a Modern Jazz Music Discovery Tool

**Description:**<br>
The purpose of this project was to build a recommendation system to allow users to discover modern jazz music. Users can explore connections to music by filtering on genres they already enjoy or search by artists they already love.
<br>
<br>
**Data:**<br>
	. I scraped jazz album review data from Pitchfork magazines website:<br>
	. https://pitchfork.com/<br>
	. Between 1999-2021 we had roughly 20,000 total reviews, 700 of which were primarily jazz album reviews. Each document in our data was a jazz album review.<br>
	. There were a number of exploratory analyses that were done that didn't make it into the final presentation. Data sources used for these are as follows:<br>
	. For sentiment/emotion analysis, I used the NRC word-emotion association lexicon which is also publicly available and widely used in research:<br>
	http://web.stanford.edu/class/cs124/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt
	. I obtained audio features for all jazz albums through Spotify's API. Features include danceability, tempo, liveness, speechiness, etc.<br>
<br>
<br>
**Features and Target Variables:**<br>
This was primarily an unsupervised learning project so there was no target feature. The data used was text data and general processes included the following:<br>
<br>
**Data cleaning and pre-processing:**<br>
		. count vectorize<br>
		. TF IDF vectorize<br>
		. lemmatize<br>
		. cosine similarity and pairwise distances<br>
		. EDA <br>
		. topic modeling<br>
		. sentiment/emotion analysis  <br>
<br>
**Programs:**<br>
<br>
**. pitchfork-bsoup.ipynb**<br>
		. This program was used to scrape reviews and other essential data from pitchfork.com<br>
**. 01_pitchfork_review_analysis.ipynb**<br>
		. This is the primary program that includes data cleaning as well as EDA and recommendation analysis<br>
		. This program exports .csv files that are eventually used for the streamlit app<br>
		. Since the program was exploratory in nature, I decided to split out some sections and create separate programs to minimize program length<br>
		. Visualizations for EDA (subgenre mentions, artists mentions) were included in this program<br>
		. TFIDF vectorization and cosine similarity were performed within this program <br>
		. Topic Modeling was also performed in this program to extract insights about instruments used for albums <br>
	<br>
**. 02_pitchfork_review_emo.ipynb**<br>
		. This program analyzes emotion and sentiment of the reviews using the NRC lexicon mentioned above<br>
		. I also create the network graph between artists in this program<br>
		. I also create spotify audio feature analysis in this program	<br>
    <br>
**. 03_radar_graphs.ipynb**<br>
		. This program was primarily used to create separate spotify audio feature graphs<br>
		. I decided not to include this information in the analysis since we eventually obtained audio clips for users to listen to. This info would have been redundant<br>
**. pitchfork_streamlit5.py**<br>
		. This is the program to build the streamlit app<br>
		. This incorporates .csv output mostly from programs 01 and visuals from all programs<br>
**. cosine_sim.npy**<br>
		. this is the cosine similarity matrix which is used in the streamlit app for obtaining top pairwise distances when getting recommended albums<br>
**. def_circbar.py**<br>
		. this is a function to create the circle bar chart for the most artist mentions from reviews<br>
**. def_radar.py**<br>
		. this function creates the spotify audio feature graphs for albums<br>
<br>
**Summary:**<br>
<br>
The final product is a streamlit app which allows users to do the following:<br>
		. explore the pitchfork data - meaning looking at top genres of music mentioned in jazz reviews<br>
		. explore the top artists mentioned in jazz reviews<br>
		. see how artists are connected to Miles Davis<br>
		. obtain recommended jazz albums to listen to based on users preference of music genres<br>
		. obtain recommended jazz albums to listen to based on searching for non-jazz artists<br>
<br>
**Future Improvements:**<br>
I performed quite a few QC checks of the data to make sure recommendations and matches made sense by comparing review text themselves and seeing the word counts and matches between some examples.<br>
There is still future work to be done on this project as this app is currently still in beta mode. These are the proposed future improvements:<br>
		. link to spotify audio for streaming full albums<br>
		. obtain more reviews through other sites (i.e. Stereogum, jazztimes, paste, bandcamp, etc)<br>
		. use non-jazz reviews for more interesting insights and connections for search capabilities<br>
		. incorporate collaborative filtering using "like" button<br>
		. use spotify audio features as part of recommendation algorithm