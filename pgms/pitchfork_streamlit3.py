import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pyLDAvis.gensim
import def_circbar as cb


st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Hello Music Lovers!")
    st.markdown("Welcome to my Jazz Album Discovery site. On this page, feel free to take a peek at some general jazz highlights from Jazz album reviews. Alternatively, on the Jazz Discovery page, you can simply choose a music sub-genre or other filters and it will introduce you to a variety of jazz albums based on your musical tastes. Enjoy the jazz discovery journey!")
    
    st.sidebar.title("What do you want to do?")
    app_mode = st.sidebar.selectbox("Choose the mode", ["Look at the data", "Jazz Discovery"])

    if app_mode == "Look at the data":
        st.title("Get a feel for the data")
        st.markdown("Choose below to see some insights into the data.")
        with st.spinner('Gathering the data...'):
            run_the_data()

    elif app_mode == "Jazz Discovery":
        st.title("Get some Jazz Recs")
        st.markdown("Below you can choose from over x jazz albums reviewed by Pitchfork and Jazztimes and it will return the albums that are the most similar based on your musical preferences")
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
    sm_df_full = load_full_data('../data/df_genre_stream_sub_long.csv')
    stats_df = load_full_data('../data/df_genre_stream_long.csv')
    gentime_df = load_full_data('../data/df_genre_time.csv')
    circbar_df = load_full_data('../data/df_top_mentions.csv')
    
    st.write("X Unique albums")
    st.write("X Reviews")
    st.write("X Artists")
    st.write("X Sub-Genres")
    st.write("---")
    st.subheader("Here is a sample of the data:")
    sm_df_full.sort_values(by=['artist'],inplace=True)
    st.dataframe(sm_df_full.sample(10))
    
    # cc: descriptive stats
    stats_vis = st.checkbox("Look at the descriptive statistics")
    if stats_vis:        
        st.dataframe(stats_dfsort.describe())
        
    st.subheader("Here are some visualisations of the data:")
    
    # cc: sub genres
    
    #st.plotly_chart(fig)
    viz_disp = st.checkbox("Look at the sub-genres in reviews across time")
    if viz_disp:
        st.write("This is the trend of jazz sub-genres over the years:")                
        fig, ax = plt.subplots(figsize=(11, 8))
        #cmap= sns.cubehelix_palette()
        kwargs = {'alpha':.9,'linewidth':1, 'linestyle':'-', 'linecolor':'k','rasterized':False, 'edgecolor':'w', "capstyle":'projecting',}
        sns.heatmap(gentime_df, cmap='cubehelix', **kwargs )
        plt.tight_layout()
        st.pyplot() 
    
#     viz_disp = st.checkbox("Look at the sub-genres in reviews across time")
#     if viz_disp:
#         st.write("This is the trend of jazz sub-genres over the years:")
#         st.image(['../img/heatmap_subgenre.png'])
    
    # cc: top artist mentions
    viz_disp2 = st.checkbox("Look at the top-artist mentions of all-time")
    if viz_disp2:
        st.write("These are the top 30 artists mentioned in reviews over the years:")
        cb.circbar(circbar_df)
        st.pyplot()
        
        #st.image(['../img/circbar_top_mentions.png'])
    
        
## This is the primary function that begins and calls other functions

def run_the_app():
    
    st.subheader("Let's get started!")

    # cc: trying out different dataframe formats
    # Keep as wide small for now
    @st.cache
    def load_sm_df():
        #data = pd.read_csv('../data/df_stream_wide_sm.csv', index_col=0)
        data = pd.read_csv('../data/df_genre_stream_wide.csv', index_col=0)
        return data
    
    # cc: change to long
    @st.cache
    def load_full_df():
        data = pd.read_csv('../data/df_genre_stream_sub_long.csv', index_col=0)
        return data

    with st.spinner("Tuning the instruments..."):
        sm_df = load_sm_df()

    with st.spinner("Taking the stage..."):
        sm_df_full = load_full_df()
    
    
    ## FILTERING Happens here ##
    
    filter_by = st.radio('Do you want to search by a non-jazz artist name OR by genre?', ['Search', 'Genre'])
    if filter_by == 'Search':
        
        # cc: try a text string    

        # Condition 1 (text search)
        user_input = st.text_input("OPTION 1: Type in a favorite artist and see if there's a connection:")
        user_list = list(user_input.split(" "))

        # filtered_df = sm_df_full[sm_df_full['review_clean2'].isin(user_list)]
        st.write(f"Your artist search term: {user_list}")

    #st.write("OR")
    
    elif filter_by == 'Genre':
        # cc: change all genres to keyword
        # Condition 2 (genre search)
        chosen_genre = st.selectbox('OPTION 2: Choose a genre:', sm_df_full['subgenre'].unique().tolist())

        #cc: dropdown:
        # Condition 3 (subgenre search)
        chosen_subsub = st.selectbox('Choose a subgenre:', sm_df_full.loc[sm_df_full['subgenre'] == chosen_genre]['subgenre2'].unique().tolist())

        # cc: add an o-meter by looking at the number of subgenre mentions
        # Condition 4 (how often keywords popup)
        sub_ometer = st.selectbox('Subgenre-ometer:', sm_df_full.loc[sm_df_full['subgenre2'] == chosen_subsub]['count'].unique().tolist()) 

    
    #dropdown:
    # cc: add subgenre
#         desired_artist = st.selectbox("Here's a list of artists:", sm_df_full.loc[sm_df_full['subgenre2'] == chosen_subsub]['artist'].unique().tolist())

        # enter button here first to generate results
        
        # Based on above conditions here are your lists
        desired_artist = st.selectbox("Here's a list of artists:", sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["count"] == sub_ometer), 'artist'].unique().tolist()) #.iloc[0]

        desired_album = sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["artist"] == desired_artist) & (sm_df_full["count"]==sub_ometer),'album'].unique().tolist()

        
        #st.write(f"album: {desired_album}") 

        # cc add
        # st.write(f"Album and Artist Recs: {desired_album} by {desired_artist}")

#             artist_df = sm_df_full[sm_df_full.artist == desired_artist]

    #     filter_by = st.radio('Do you want to filter by Best reviewed?', ['No', 'Yes'])
    #     if filter_by == 'Yes':
    #         best_reviewed = st.selectbox('Choose Best reviewed:', sm_df_full['best'].unique())

    @st.cache
    def load_matrix():
        cosine_sim = np.load('cosine_sim.npy')
        return cosine_sim

    cosine_sim = load_matrix()

    #indices = pd.Series(sm_df.index)
    indices = pd.Series(sm_df.artist)
    
    #st.write(f"indices: {indices}")

    def load_image(url):
        if type(url) == str:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                st.image(img)
            except:
                None 


    ### RECOMMENDING Happens here ###
    
    
########################################################################################
### Content Based Recommendation (by SEARCH)
########################################################################################

    if filter_by == 'Search':

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

            def _print_message(self, album, recom_album,artist):
                rec_items = len(recom_album)        

                #print(f'The {rec_items} recommended albums for {album} are:')
                #print(f"{album} by {artist} ") # with {album[0][0]} similarity score")  # if can add artist sim score        
                #print(f'The {rec_items} recommended albums are:') # if can add artist search title

                for i in range(rec_items):
                    if i==0:
                        #print(f"Matched Album Number {i+1}:")
                        st.write(f"{recom_album[i][1]} by {artist}") # with {round(recom_album[i][0],3)} similarity score")  # if can add artist sim score
                        st.write("--------------------")
                    else:    
                        st.write(f"Recommended album based on matched (above) {i}:")            
                        st.write(f"{recom_album[i][1]} by {recom_album[i][2]} with {round(recom_album[i][0], 3)} similarity score") 
                        st.write("--------------------")

            def recommend(self, recommendation):
                artist = recommendation['artist']
                # Get album to find recommendations for
                album = recommendation['album']
                # Get number of albums to recommend
                similar_albums = recommendation['similar_albums']
                # Get the number of albums most similars from matrix similarities
                recom_album = self.matrix_similar[album][:similar_albums]
                # print each item
                self._print_message(album=album, recom_album=recom_album,artist=artist)

        ##instantiate rec function

        recommedations = ContentBasedRecommender(similarities)

        ### Primary function for album recs based on REVIEW CLEAN (lowercase, remove punc, no lem)

        def artsimrec(artmatch=1,simalb=0):
            # get list of indeces containing the chosen word that's in the review text
            #_a= sm_df_full.index[sm_df_full['review_clean2'].str.contains(art,na=False)]
            _a= sm_df.index[sm_df['review_clean2'].str.contains(user_input,na=False)]
            # get only # of keyword matches requested by user
            a=_a[:artmatch]

            if a.empty:
                return 'not found'

            # if the keyword shows up in more than one review get requested # of recs for each artist that is mentioned    
            else: # len(a)>1:
                #print(f"The number of reviews containing {art} are: {len(_a)}") 
                st.write(f"The number of reviews containing {user_list} are: {len(_a)}") 
                st.write(f"User requested {len(a)} Matches")
                st.write(f"User requested {simalb} Recommended albums per Match")
                st.write("--------------------")

                for k,j in enumerate(a):
                    st.write(f"Matched Album Number {k+1}:")
                    recommendation = {
                       "artist": sm_df['artist'].iloc[j],
                       "album": sm_df['album'].iloc[j],
                       "similar_albums": simalb+1 # add +1 to obtain the right number of similar albums
                    }
                    recommedations.recommend(recommendation)


        ### RUN RECOMMENDER - based on cosine sim

        # select text input, # of matches based on text, # of recs per match
    #    artsimrec('kanye',10,3)
        #st.write(artsimrec('kanye'))

        if st.button("Discover some Jazz!"):
            st.write(artsimrec())
        else:
            st.write(artsimrec())
    
    
########################################################################################
### Content Based Recommendation (BY GENRE)
########################################################################################
    
    elif filter_by=='Genre':

        
        def full_recommendations(cosine_sim = cosine_sim):
            
            
            title = desired_artist # index artist based on full data
        #   if filter_by == 'Yes':
        #       location = best_reviewed
            # initializing the empty list of recommended brs
            recommended_albums = []

            # gettin the index of the br that matches the name
            idx = indices[indices == title].index[0]

            # creating a Series with the similarity scores in descending order
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

            # getting the indexes of the 100 most similar brs
            top_20_indexes = list(score_series.iloc[1:100].index)

                # populating the list with the titles of the best 10 matching brs

            #if filter_by == 'No':

            for i in top_20_indexes:
                rec_alb_dict = {}
                # cc : change .index to artist                                
                full_ind = sm_df_full.index[sm_df_full['artist'] == list(sm_df.artist)[i]].tolist()[0]

                rec_alb_dict['album'] = list(sm_df.album)[i]
                rec_alb_dict['artist'] = sm_df_full.loc[full_ind]['artist']
                rec_alb_dict['score'] = sm_df_full.loc[full_ind]['score']
                rec_alb_dict['subgenre2'] = sm_df_full.loc[full_ind]['subgenre2']

                recommended_albums.append(rec_alb_dict)
            st.write("These are the albums you should check out:")

            for i in range(0,3):
                st.write(i+1)
        #         img_url = recommended_albums[i]['Image']
        #         load_image(img_url)
                st.write(f"{recommended_albums[i]['album']} from {recommended_albums[i]['artist']} in the {recommended_albums[i]['subgenre2']} sub-genre")
                st.write("The stats:")

        #             st.write(f"Rating: {recommended_albums[i]['Rating']}")
        #             st.write(f"ABV: {recommended_albums[i]['abv']}")
        #             st.write(f"Availability: {recommended_albums[i]['avail']}")
        #             st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")

                #return recommended_albums[:3]

    # cc: given we don't use filter by here we can skip
    #             elif filter_by == 'Yes':

    #                 for i in top_20_indexes:
    #                     rec_alb_dict = {}
    #                     # cc: changed .index to artist
    #                     full_ind = sm_df_full.index[sm_df_full['artist'] == list(sm_df.artist)[i]].tolist()[0]
    #                     rec_alb_dict['album'] = list(sm_df.artist)[i]
    #                     rec_alb_dict['score'] = sm_df_full.loc[full_ind]['score']
    #                     rec_alb_dict['subgenre2'] = sm_df_full.loc[full_ind]['subgenre2']
    #                     if sm_df_full.loc[full_ind]['subgenre2'] == location:
    #                         recommended_albums.append(rec_alb_dict)

    #                 if len(recommended_albums) > 3:
    #                     for i in range(0,3):
    #                         st.write(i+1)
    #     #                     img_url = recommended_albums[i]['Image']
    #     #                     load_image(img_url)
    #                         st.write(f"{recommended_albums[i]['album']} from {recommended_albums[i]['artist']} in {recommended_albums[i]['subgenre2']}")
    #                         st.write("The stats:")
    #                         st.write(f"Rating: {recommended_albums[i]['Rating']}")
    #                         st.write(f"ABV: {recommended_albums[i]['abv']}")
    #                         st.write(f"Availability: {recommended_albums[i]['avail']}")
    #                         st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")

    #                 elif len(recommended_albums) >= 1:
    #                     for i in range(len(recommended_albums)):
    #                         st.write(i+1)
    #                         img_url = recommended_albums[i]['Image']
    #                         load_image(img_url)
    #                         st.write(f"{recommended_albums[i]['album']} from {recommended_albums[i]['artist']} in {recommended_albums[i]['subgenre2']}")
    #                         st.write("The stats:")
        #                     st.write(f"Rating: {recommended_albums[i]['Rating']}")
        #                     st.write(f"ABV: {recommended_albums[i]['abv']}")
        #                     st.write(f"Availability: {recommended_albums[i]['avail']}")
        #                     st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")


    #                 else:
    #                     st.write("Sorry, there are no similar albums in that subgenre")

        if st.button("Discover!"):
            st.write(full_recommendations())
#         else:
#             st.write(full_recommendations())
            #st.image("images/br_meme.jpeg")
    

if __name__ == "__main__":
    main()