'''
pitchfork_streamlit6.py

The purpose of this program is to build a streamlit app recommender for modern jazz albums.
The data used if from Pitchfork.com (jazz album reviews)
There are 2 primary datasources with the same info. One file is a long format, one is a wide format.
THe long format contains multiple lines per album with different subgenres assigned. This is for the 
genre search parameters for the tool.

Need to update the exploratory page with more functionality
Need to incorporate collaborative filtering with "like" button

'''


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import def_circbar as cb
from IPython.display import Audio
import streamlit.components.v1 as components

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.sidebar.title("What do you want to do?")
    app_mode = st.sidebar.selectbox("Choose the mode", ["Explore the data", "Jazz Discovery"])


    # 2 options - exploratory and recommender
    if app_mode == "Explore the data":
        
        st.title("The Modern Jazz Album Discovery Tool")
        st.image('heroku/heroku_img/jazzcoverstring.png',width=690)
        st.markdown("---")
        st.markdown("**Hello music lovers** and welcome to my **Modern Jazz Album Discovery Recommender**. On this page, feel free to take a peek at some data insights from **Pitchfork Jazz** album reviews. Alternatively, on the **Jazz Discovery** page, search by Artists or Sub-Genres you know and enjoy and we'll introduce you to a variety of jazz albums that may be connected or similar in some way. Enjoy the jazz discovery journey!")

        st.title("Get a feel for the data")
        st.markdown("Choose below to see some insights into the data.")
        with st.spinner('Gathering the data...'):
            run_the_data()

    elif app_mode == "Jazz Discovery":
        
        st.title("The Modern Jazz Album Discovery Tool")
        st.image("heroku/heroku_img/drummer1.png",width=690)
        st.markdown("---")                
        st.markdown("**Welcome to the Jazz Discovery page**. See if an artist you enjoy was referenced in a jazz album review by Pitchfork. Use the **'Search by Artist'** button below for this. Alternatively look for an album based on a genre and subgenre you love and it will return albums that are most similar based on your musical preferences. For this, use the **'Search by Genre'** button below.")
        st.title("Get some Jazz Recs")
        with st.spinner("Setting the stage..."):
            run_the_app()
           
## This is called in the main first
            
def run_the_data():
    st.subheader('The data at a glance:')

    @st.cache
    def load_full_data(desti):
        data = pd.read_csv(desti, index_col=0)
        return data

    # cc: trying out diff dataframe formats    
    sm_df_full  = load_full_data('heroku/heroku_data/df_emo_top_long_nm_spfy.csv')    
    sm_df       = load_full_data('heroku/heroku_data/df_emo_top_wide_nm_spuri.csv')
    gentime_df  = load_full_data('heroku/heroku_data/df_genre_time.csv')
    circbar_df  = load_full_data('heroku/heroku_data/df_top_mentions120.csv')
    #spot_df = load_full_data('../data/df_emo_top_long_nm_spfy.csv')
    
    # cc: check how much data we have
    nreview     = len(sm_df)
    nartists    = len(pd.unique(sm_df['artist']))
   
    st.write(f"**{nreview}** Unique Albums and Reviews")
    st.write(f"**{nartists}** Artists")
    st.write("225 Sub-Genres")
    #st.write("---")
    
    st.title("Explore the data:")
    
    # cc: sub genres
    st.subheader("Here are Some Visuals based on the Reviews:")
    
    viz_disp = st.checkbox("Look at the sub-genres in reviews across time")
    
    if viz_disp:
        st.write("This is the trend of jazz sub-genres over the years:")
        st.image(['heroku/heroku_img/heatmap_subgenre_10.png'])
# cc: remove function since it's less important        
#         st.write("Take a closer look at some artists per sub-genre and year:")
#         sg_select1 = st.selectbox('Choose a sub-genre:', sm_df_full['subgenre2'].unique().tolist())
#         sg_select2 = st.selectbox('Choose a year:', sm_df_full.loc[sm_df_full['subgenre2']==sg_select1]['year'].unique().tolist())        
#         sg_select3 = sm_df_full.loc[(sm_df_full['subgenre2']==sg_select1) & (sm_df_full['year']==sg_select2),'album'].unique().tolist()
#         st.write(f"Here are teh results: {sg_select3}")
        
                
    # cc: top artist mentions
    viz_disp2 = st.checkbox("Look at the top-artist mentions of all-time")
    if viz_disp2:        
        nart=st.slider("Choose Number of Artists:",20,120,1)
        st.write(f"These are the top {nart} artists mentioned in reviews over the years:")
        cb.circbar(circbar_df,nart)
        st.pyplot()
        
        #st.image(['../img/circbar_top_mentions.png'])
    
    # cc: miles connections
    viz_disp3 = st.checkbox("Checkout how Miles Davis has influenced music:")
    if viz_disp3:
        st.write("Here's a network graph of different artists Miles has influenced:")
        st.image(['heroku/heroku_img/network_map_lbl.png'])
        
## This is the primary function that begins and calls other functions

def run_the_app():
    
    st.subheader("Let's get started!")

    # cc: trying out different dataframe formats    
    @st.cache
    def load_sm_df():
        data = pd.read_csv('heroku/heroku_data/df_emo_top_wide_nm_spuri.csv', index_col=0)
        return data
    
    # cc: change to long
    @st.cache
    def load_full_df():
        #data = pd.read_csv('../data/df_genre_stream_sub_long.csv', index_col=0)
        data = pd.read_csv('heroku/heroku_data/df_emo_top_long_nm_spfy.csv', index_col=0)
        return data

    # cc: change to long
    @st.cache
    def load_spot_df():
        #data = pd.read_csv('../data/df_genre_stream_sub_long.csv', index_col=0)
        data = pd.read_csv('heroku/data/df_emo_top_long_nm_spfy.csv', index_col=0)
        return data
        
    
    with st.spinner("Tuning the instruments..."):
        sm_df = load_sm_df()

    with st.spinner("Taking the stage..."):
        sm_df_full = load_full_df()
        
    
    ## FILTERING Happens here ##
    
    filter_by = st.radio('Do you want to search by Artist name OR by Genre?', ['Search by Artist', 'Search by Genre'])
    if filter_by == 'Search by Artist':
        
        # cc: try a text string    

        # Condition 1 (text search)
        user_input  = st.text_input("OPTION 1: Type in a favorite artist and see if there's a connection (lowercase):")
        user_list   = list(user_input.split(" "))
        
        
        # Condition 2 (integer - how many matches would you like to see)
        user_match  = st.number_input("Type in the number of matches (if any) you'd like to see (Max 10):",min_value=0, max_value=10, step=1) #, format=None, key=None))
        #user_match_list = list(user_match_int)
    
        disco_button=st.button('Discover by Artist Mentions!')    

        # Turn off for now
        user_sims   =0
#         user_sims = st.number_input("Type in the number of similar albums to each match (Max 3):",min_value=0,max_value=3,step=1)
        
    
    elif filter_by == 'Search by Genre':
        # cc: change all genres to keyword
        # Condition 2 (genre search)
        sm_df_full.sort_values(by=['subgenre','subgenre2','count','artist'],inplace=True)
        chosen_genre = st.selectbox('OPTION 2: Choose a genre:', sm_df_full['subgenre'].unique().tolist())

        #cc: dropdown:
        # Condition 3 (subgenre search)
        chosen_subsub = st.selectbox('Choose a subgenre:', sm_df_full.loc[sm_df_full['subgenre'] == chosen_genre]['subgenre2'].unique().tolist())

        # cc: add an o-meter by looking at the number of subgenre mentions
        # Condition 4 (how often keywords popup)
        sub_ometer = st.selectbox('Subgenre Frequency:', sm_df_full.loc[sm_df_full['subgenre2'] == chosen_subsub]['count'].unique().tolist()) 
        
        # Based on above conditions here are your lists
        desired_artist  = st.selectbox("Here's a list of artists:", sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["count"] == sub_ometer), 'artist'].unique().tolist()) #.iloc[0]
        desired_album   = sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["artist"] == desired_artist) & (sm_df_full["count"]==sub_ometer),'album'].unique().tolist()


    @st.cache
    def load_matrix():
        cosine_sim = np.load('cosine_sim.npy')
        return cosine_sim

    cosine_sim = load_matrix()

    indices = pd.Series(sm_df.artist)
    
    #st.write(f"indices: {indices}")

    def load_image(url):
        if type(url) == str:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                #st.write(f"IMG Bytes IO {img}")
                st.image(img,width=200)
            
            except:
                st.image(['heroku/heroku_img/sax.jpeg'])
                #None 

    def load_audio(url):
        if type(url) == str:
            try:
                response = requests.get(url)
                audio = url
                
                st.image([audio])
            
            except:
                #st.write("Sorry, Album artwork not available")
                audio=url
                st.image([audio])
                
    def load_audio2(url):
        if type(url) == str:
            try:
                audio2=url 
                components.iframe(audio2 , width=600, height=200)
            except:
                st.image(['heroku/heroku_img/sax.jpeg'])
#                 audio2=None
#                 st.write(f"Sorry, no playlist found in Spotify for this album")
    
    
########################################################################################
### Content Based Recommendation (by SEARCH)
########################################################################################

    #if st.button('button'):
    if (filter_by == 'Search by Artist') and (disco_button):

        similarities = {}

        for i in range(len(cosine_sim)):

            # Sort each element in cosine_similarities and get the indexes of the artists
            similar_indices = cosine_sim[i].argsort()[:-50:-1] 

            # Store in similarities each name of the 10 most similar artists
            # Keep the first one as the main search item - and then the remaining similar artists 
            similarities[sm_df['album'].iloc[i]] = [(cosine_sim[i][x], sm_df['album'][x], sm_df['artist'][x]) for x in similar_indices]

        class ContentBasedRecommender:
            def __init__(self, matrix):
                self.matrix_similar = matrix
            
            # this goes third (3)
            def _print_message(self,album,recom_album,artist,image,audio2,topic_label):
                rec_items = len(recom_album)        

                for i in range(rec_items):
                    # this will return the first artist match
                    if i==0:
                        
                        st.write(f"**{recom_album[i][1].title()}** *by* **{artist.title()}**") # with {round(recom_album[i][0],3)} similarity score")  # if can add artist sim score
                        # load the album cover
                        
                        st.write(f"**Album Texture:** {topic_label.title()}") # with {round(recom_album[i][0],3)} 
                        
                        #print(f"Matched Album Number {i+1}:")
                        load_image(image)
                        
                        load_audio2(audio2)
                        st.button(f"Love it!: {recom_album[i][1]}")
                        st.button(f"Not into it! {recom_album[i][1]}")
                        
                        
                    else:    
                        st.subheader(f"Recommended album based on matched (above) {i}:")            
                        st.write(f"**{recom_album[i][1].title()}** *by* **{recom_album[i][2].title()}** with **{round(recom_album[i][0], 3)}** similarity score") 
                        # load the album cover
                        
            # This goes second (2)
            def recommend(self, recommendation):
                artist  = recommendation['artist']
                # Get album to find recommendations for
                album   = recommendation['album']                
                image   = recommendation['image']
                audio2  = recommendation['audio2']
                topic_label = recommendation['topic_label']
                # Get number of albums to recommend
                similar_albums = recommendation['similar_albums']
                # Get the number of albums most similars from matrix similarities
                recom_album = self.matrix_similar[album][:similar_albums]
                # print each item (cc test
                self._print_message(album=album,recom_album=recom_album,
                                    artist=artist,image=image,audio2=audio2,topic_label=topic_label)

        ##instantiate rec function

        recommedations = ContentBasedRecommender(similarities)

        ### Primary function for album recs based on REVIEW CLEAN (lowercase, remove punc, no lem)
        # This goes first (1)
        def artsimrec(artmatch,simalb=0): # cc: simalb
            # get list of indices containing the chosen word that's in the review text            
            _a= sm_df.index[sm_df['review_clean2'].str.contains(user_input,na=False)]
            # get only # of keyword matches requested by user
            a=_a[:artmatch]

            if a.empty:
                return st.write("Sorry, No Matches")
                #return 'not found'

            # if the keyword shows up in more than one review get requested # of recs for each artist that is mentioned    
            else: # len(a)>1:
                #print(f"The number of reviews containing {art} are: {len(_a)}") 
                st.write(f"The number of reviews containing {user_list} are: {len(_a)}") 
                st.write(f"User requested {len(a)} Matches")
                #st.write(f"User requested {simalb} Recommended albums per Match") # cc:simalb
         
                # loops through the function by number of desired matches
                for k,j in enumerate(a):
                    st.markdown("---")
                    st.subheader(f"Matched Album Number {k+1}:")
                                        
                    recommendation = {
                       "artist": sm_df['artist'].iloc[j],
                       "album": sm_df['album'].iloc[j],                    
                       "topic_label": sm_df['topic_label'].iloc[j], 
                       "image": sm_df['img_link'].iloc[j],
                       "audio2": sm_df['album_uri_link'].iloc[j], # cc testing
                       "similar_albums": simalb+1  # add +1 to obtain the right number of similar albums
                    }
                    recommedations.recommend(recommendation)


        ### RUN RECOMMENDER - based on cosine sim

        #if st.button("Discover some Jazz!"):
        st.write(artsimrec(artmatch=user_match,simalb=user_sims)) # cc: user_sims
#         else:
#             st.write(artsimrec(artmatch=user_match,simalb=user_sims))
    
    
########################################################################################
### Content Based Recommendation (BY GENRE)
########################################################################################
    
    elif filter_by=='Search by Genre':

        
        def full_recommendations(cosine_sim = cosine_sim):
            
            
            title = desired_artist # index artist based on full data
        #   if filter_by == 'Yes':
        #       location = best_reviewed
            # initializing the empty list of recommended brs
            recommended_albums = []

            # gettin the index of the artist that matches the name
            idx = indices[indices == title].index[0]

            # creating a Series with the similarity scores in descending order
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
            
            # getting the indexes of the 100 most similar albums
            top_20_indexes = list(score_series.iloc[0:100].index)

                # populating the list with the titles of the best 10 matching albums

            #if filter_by == 'No':

            for i in top_20_indexes:
                rec_alb_dict = {}
                # cc : change .index to artist                                
                #full_ind = sm_df_full.index[sm_df_full['artist'] == list(sm_df.artist)[i]].tolist()[0]
                full_ind = sm_df_full.index[sm_df_full['revID'] == list(sm_df.revID)[i]].tolist()[0]

                rec_alb_dict['album'] = list(sm_df.album)[i]
                rec_alb_dict['artist'] = sm_df_full.loc[full_ind]['artist']
                rec_alb_dict['score'] = sm_df_full.loc[full_ind]['score']
                rec_alb_dict['subgenre2'] = sm_df_full.loc[full_ind]['subgenre2']
                rec_alb_dict['Image'] = sm_df_full.loc[full_ind]['img_link']
                rec_alb_dict['topic_label'] = sm_df_full.loc[full_ind]['topic_label']
                # testing
                rec_alb_dict['audio2'] = sm_df_full.loc[full_ind]['album_uri_link']
                
                recommended_albums.append(rec_alb_dict)
                
            st.subheader("These are the albums you should check out:")
            
            # Print to the App
            for i in range(0,5):
                #st.write("***")
                st.markdown("---")
                st.subheader(f"Recommended Album # {i+1}:")
                
                st.write(f"**{recommended_albums[i]['album'].title()}** *by* **{recommended_albums[i]['artist'].title()}** with **{round(score_series.iloc[i],3)}** similarity score") # in the {recommended_albums[i]['subgenre2'].title()} sub-genre")
                st.write(f"**Album Texture:** {recommended_albums[i]['topic_label'].title()}")
                
                img_url = recommended_albums[i]['Image']
                img_url = img_url.strip('\"')                
                load_image(img_url)
                
                audio_url2 = recommended_albums[i]['audio2']
                audio_url2 = audio_url2.strip('\"')
                load_audio2(audio_url2)
                st.button(f"Love it! {recommended_albums[i]['album']}")
                st.button(f"Not into it! {recommended_albums[i]['album']}")

        if st.button("Discover by Genre!"):
            st.write(full_recommendations())
    

if __name__ == "__main__":
    main()