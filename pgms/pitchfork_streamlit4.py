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
import xml.etree.ElementTree as ET

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
            
            
def render_svg(svg_file):

    with open(svg_file, "r") as f:
        lines = f.readlines()
        svg = "".join(lines)

        """Renders the given svg string."""
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        return html            


## This is called in the main first
            
def run_the_data():
    st.subheader('The data at a glance:')

    @st.cache
    def load_full_data(desti):
        data = pd.read_csv(desti, index_col=0)
        return data

    # cc: trying out diff dataframe formats
    #sm_df_full = load_full_data('../data/df_genre_stream_sub_long.csv')
    sm_df_full = load_full_data('../data/df_emo_top_long_nm.csv')
    #stats_df = load_full_data('../data/df_genre_stream_long.csv')
    gentime_df = load_full_data('../data/df_genre_time.csv')
    circbar_df = load_full_data('../data/df_top_mentions.csv')
    sm_df = load_full_data('../data/df_emo_top_wide_nm.csv')
    
    st.write("1,025 Unique Albums and Reviews")    
    st.write("693 Artists")
    st.write("225 Sub-Genres")
    #st.write("---")
    
#     st.subheader("Here is a sample of the data:")
#     sm_df_full.sort_values(by=['artist'],inplace=True)
#     st.dataframe(sm_df_full.sample(10))
    
    # cc: descriptive stats
    
    st.title("Some Visuals and Descriptives:")
    #st.write("---------------------------------------------------------------")
#     st.subheader("Here are Some Descriptives:")
#     stats_vis = st.checkbox("Look at the descriptive statistics")
#     if stats_vis:        
#         st.dataframe(sm_df.describe())
    
    # cc: sub genres
    st.subheader("Here are Some Visuals based on the Reviews:")
    #st.plotly_chart(fig)
    #viz_disp = st.checkbox("Look at the sub-genres in reviews across time")
    
    viz_disp = st.checkbox("Look at the sub-genres in reviews across time")
    
    if viz_disp:
        st.write("This is the trend of jazz sub-genres over the years:")
        st.image(['../img/heatmap_subgenre_10.png'])

#     if viz_disp:
#         st.write("This is the trend of jazz sub-genres over the years:")                
#         fig, ax = plt.subplots(figsize=(10, 6))
#         cmap=sns.color_palette("YlOrBr", as_cmap=True)
#         kwargs = {'alpha':.9,'linewidth':1, 'linestyle':'-', 'linecolor':'k','rasterized':False, 'edgecolor':'w', "capstyle":'projecting',} #'linewidth':1, 'linestyle':'-', 'linecolor':'k',

#         gentime_df_rmv = gentime_df[1:]
#         sns.heatmap(gentime_df_rmv, cmap=cmap, **kwargs )
#         plt.tight_layout()
#         st.pyplot() 
    
        # cc: pull top 3 artists based on genre (mentions) and year (long file)
        st.write("Take a closer look at some artists per sub-genre and year:")

        sg_select1 = st.selectbox('Choose a sub-genre:', sm_df_full['subgenre2'].unique().tolist())
        sg_select2 = st.selectbox('Choose a year:', sm_df_full.loc[sm_df_full['subgenre2']==sg_select1]['year'].unique().tolist())
        
        sg_select3 = sm_df_full.loc[(sm_df_full['subgenre2']==sg_select1) & (sm_df_full['year']==sg_select2),'album'].unique().tolist()
                
    
    # cc: top artist mentions
    viz_disp2 = st.checkbox("Look at the top-artist mentions of all-time")
    if viz_disp2:
        st.write("These are the top 60 artists mentioned in reviews over the years:")
        cb.circbar(circbar_df)
        st.pyplot()
        
        #st.image(['../img/circbar_top_mentions.png'])
    
    # cc: miles connections
    viz_disp3 = st.checkbox("Checkout how Miles Davis has influenced music:")
    if viz_disp3:
        st.write("Here's a network graph of different artists Miles has influenced:")
        st.image(['../img/network_map_lbl.png'])
        
    st.subheader("Here are Some Visuals based on the Audio Features:")        
    
    # cc: Spotify Audio
    viz_disp4 = st.checkbox("Let's see how different sub-genres compare in audio features:")
    if viz_disp4:
        st.image(['../img/radar_hip_hop.png'])
        st.image(['../img/radar_rock.png'])
        st.image(['../img/radar_electronic.png'])
#         f = open("../img/radar_hip_hop.svg","r")        
#         lines = f.readlines()
#         line_string=''.join(lines)
#         render_svg(line_string)

        
        
#         st.write("Here are the top 3 sub-genres:")
#         st.image([test])    
        
    
        
## This is the primary function that begins and calls other functions

def run_the_app():
    
    st.subheader("Let's get started!")

    # cc: trying out different dataframe formats
    # Keep as wide small for now
    @st.cache
    def load_sm_df():
        data = pd.read_csv('../data/df_emo_top_wide_nm.csv', index_col=0)
        return data
    
    # cc: change to long
    @st.cache
    def load_full_df():
        #data = pd.read_csv('../data/df_genre_stream_sub_long.csv', index_col=0)
        data = pd.read_csv('../data/df_emo_top_long_nm.csv', index_col=0)
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
        
        
        # Condition 2 (integer - how many matches would you like to see)
        user_match = st.number_input("Type in the number of matches (if any) you'd like to see (Max 10):",min_value=0, max_value=10, step=1) #, format=None, key=None))
        #user_match_list = list(user_match_int)

        disco_button=st.button('Discover by Artist Mentions!')
    
        # Condition 3 (integer - how many album recs would you like to see)
        user_sims = st.number_input("Type in the number of similar albums to each match (Max 3):",min_value=0,max_value=3,step=1)
        # filtered_df = sm_df_full[sm_df_full['review_clean2'].isin(user_list)]
        #st.write(f"Your artist search term: {user_list}")

    
    
    
    elif filter_by == 'Genre':
        # cc: change all genres to keyword
        # Condition 2 (genre search)
        sm_df_full.sort_values(by=['genre','subgenre','count','artist'],inplace=True)
        chosen_genre = st.selectbox('OPTION 2: Choose a genre:', sm_df_full['subgenre'].unique().tolist())

        #cc: dropdown:
        # Condition 3 (subgenre search)
        chosen_subsub = st.selectbox('Choose a subgenre:', sm_df_full.loc[sm_df_full['subgenre'] == chosen_genre]['subgenre2'].unique().tolist())

        # cc: add an o-meter by looking at the number of subgenre mentions
        # Condition 4 (how often keywords popup)
        #sm_df_full.sort_values(by=['count'],inplace=True)
        sub_ometer = st.selectbox('Subgenre-ometer:', sm_df_full.loc[sm_df_full['subgenre2'] == chosen_subsub]['count'].unique().tolist()) 
        
        # Based on above conditions here are your lists
        #sm_df_full.sort_values(by=['artist'],inplace=True)
        desired_artist = st.selectbox("Here's a list of artists:", sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["count"] == sub_ometer), 'artist'].unique().tolist()) #.iloc[0]

        desired_album = sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["artist"] == desired_artist) & (sm_df_full["count"]==sub_ometer),'album'].unique().tolist()


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
                st.image(img)
            except:
                None 


    ### RECOMMENDING Happens here ###
    
    
########################################################################################
### Content Based Recommendation (by SEARCH)
########################################################################################

    #if st.button('button'):
    if (filter_by == 'Search') and (disco_button):

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
                        st.write(f"{recom_album[i][1].title()} by {artist.title()}") # with {round(recom_album[i][0],3)} similarity score")  # if can add artist sim score
                        #st.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    else:    
                        st.subheader(f"Recommended album based on matched (above) {i}:")            
                        st.write(f"{recom_album[i][1].title()} by {recom_album[i][2].title()} with {round(recom_album[i][0], 3)} similarity score") 
                        #st.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

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

        def artsimrec(artmatch,simalb):
            # get list of indeces containing the chosen word that's in the review text            
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
                

                for k,j in enumerate(a):
                    st.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    st.subheader(f"Matched Album Number {k+1}:")
                    recommendation = {
                       "artist": sm_df['artist'].iloc[j],
                       "album": sm_df['album'].iloc[j],
                       "similar_albums": simalb+1 # add +1 to obtain the right number of similar albums
                    }
                    recommedations.recommend(recommendation)


        ### RUN RECOMMENDER - based on cosine sim

        #if st.button("Discover some Jazz!"):
        st.write(artsimrec(artmatch=user_match,simalb=user_sims))
#         else:
#             st.write(artsimrec(artmatch=user_match,simalb=user_sims))
    
    
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
            st.subheader("These are the albums you should check out:")
            #st.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            for i in range(0,3):
                st.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                st.write(i+1)
        #         img_url = recommended_albums[i]['Image']
        #         load_image(img_url)
                #st.write(f"{recommended_albums[i]['album'].title()} by {recommended_albums[i]['artist'].title()} in the {recommended_albums[i]['subgenre2'].title()} sub-genre")
                st.write(f"{recommended_albums[i]['album'].title()} by {recommended_albums[i]['artist'].title()} in the {recommended_albums[i]['subgenre2'].title()} sub-genre")
                #st.image(['../img/soundmirrors_coldcut.jpg'])
            
                #st.write("The stats:")

        #             st.write(f"Rating: {recommended_albums[i]['Rating']}")
        #             st.write(f"ABV: {recommended_albums[i]['abv']}")
        #             st.write(f"Availability: {recommended_albums[i]['avail']}")
        #             st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")

                #return recommended_albums[:3]

        if st.button("Discover by Genre!"):
            st.write(full_recommendations())
    

if __name__ == "__main__":
    main()