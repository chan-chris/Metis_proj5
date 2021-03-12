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

def main():
    #st.title("Hello Music Lovers")
    #st.markdown("Welcome to my jazz album discovery site. Simply choose a music sub-genre or other filters and it will introduce you to x jazz albums based on your musical tastes. Enjoy the jazz discovery journey!")
    
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


def run_the_data():
    st.subheader('The data at a glance:')

    @st.cache
    def load_full_data(desti):
        data = pd.read_csv(desti, index_col=0)
        return data

    # cc: trying out diff dataframe formats
    sm_df_full = load_full_data('../data/df_genre_stream_sub_long.csv')
    stats_df = load_full_data('../data/df_genre_stream_long.csv')
    
    st.write("X Unique albums")
    st.write("X Reviews")
    st.write("X Artists")
    st.write("X Sub-Genres")
    st.write("---")
    st.subheader("Here is a sample of the data:")
    st.dataframe(sm_df_full.sample(10))
    stats_vis = st.checkbox("Look at the descriptive statistics")
    if stats_vis:
        st.dataframe(stats_df.describe())
        
    
#     plt.style.use('seaborn')

#     st.subheader("Here are some visualisations of the data:")
#     display = st.checkbox("Show the rating distributions")
#     if display:
#         fig = go.Figure()
#         fig.add_trace(go.Histogram(x=stats_df['avg_score'], name='Overall Distribution'))
#         fig.add_trace(go.Histogram(x=stats_df.loc[stats_df['style'] == 'New England IPA']['avg_score'], name='New England IPA'))
#         fig.add_trace(go.Histogram(x=stats_df.loc[stats_df['style'] == 'American Light Lager']['avg_score'],name='American Light Lager'))
#         fig.add_trace(go.Histogram(x=stats_df.loc[stats_df['style'] == 'American Stout']['avg_score'], name='American Stout'))
#         fig.add_trace(go.Histogram(x=stats_df.loc[stats_df['style'] == 'Bohemian Pilsener']['avg_score'], name='Bohemian Pilsener'))
#         fig.add_trace(go.Histogram(x=stats_df.loc[stats_df['style'] == 'Russian Imperial Stout']['avg_score'],name='Russian Imperial Stout'))
#         fig.add_trace(go.Histogram(x=stats_df.loc[stats_df['style'] == 'Belgian Saison']['avg_score'], name='Belgian Saison'))

#         # Overlay both histograms
#         fig.update_layout(title='Distribution of User Ratings', barmode='overlay')
#         # Reduce opacity to see both histograms
#         fig.update_traces(opacity=0.7)
#         st.plotly_chart(fig)


#         #plt.figure()
#         #plt.hist(stats_df['avg_score'], bins=18, color='indigo')
#         #plt.title("Distribution of User Ratings")
#         #st.pyplot()

#     pl_display = st.checkbox("Show where the breweries are")
#     if pl_display:
#         labels = ['US', 'Canada', 'England', 'Germany', 'Belgium', 'Australia', 'Spain', 'RestOfWorld']
#         values = [77616, (2235+1734+1578), 4458, 2652, 1881, 1141, 1015, (109508 - (77616+2235+1734+1578+4458+2652+1881+1141+1015))]
#         fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
#         st.plotly_chart(fig)

#     lda_display = st.checkbox("Look at the topic modelling.")
#     if lda_display:
#         st.markdown("Click [here](file:///Users/sarah/Documents/DataScience/final_project/br_recommender/lda_vis.html) to check out the interactive graph!", unsafe_allow_html=True)
        
#     #hist_data = [stats_df['avg_score'], stats_df['taste_avg'].dropna() , stats_df['look_avg'].dropna(), stats_df['smell_avg'].dropna(), stats_df['feel_avg'].dropna()]
#     #groups = ['Overall', 'Taste', 'Look', 'Smell', 'Feel']

#     #fig = ff.create_distplot(hist_data, groups, bin_size=[20, 15, 15, 15, 15])
    
#     #st.plotly_chart(fig)
#     viz_disp = st.checkbox("Look at the topics in the br reviews")
#     if viz_disp:
#         st.write("These are the words most commonly used in each of the 12 topics:")
#         st.image(['images/topic_0.png','images/topic_1.png','images/topic_2.png','images/topic_3.png','images/topic_4.png','images/topic_5.png','images/topic_6.png','images/topic_7.png','images/topic_8.png','images/topic_9.png','images/topic_10.png','images/topic_11.png'])



def run_the_app():
    st.subheader("Let's get started!")

    # cc: trying out different dataframe formats
    # Keep as wide small for now
    @st.cache
    def load_sm_df():
        data = pd.read_csv('../data/df_stream_wide_sm.csv', index_col=0)
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
    
    #working code
    #flavors = st.radio('Do you want to choose some key words?', ['No', 'Yes'])
    #if filter_by == 'Yes':
    #    br_location = st.selectbox('Choose a location:', sm_df_full['location'].unique()) 

    # cc: try a text string    
    user_input = st.text_input("OPTION 1: Type in a favorite artist and see if there's a connection:")
    user_list = list(user_input.split(" "))
    #user_list = list(user_input) 
    filtered_df = sm_df_full[sm_df_full['review_clean2'].isin(user_list)]
    st.write(f"users input list: {user_list}")
    
    st.write("OR")
    
    # cc: change all genres to keyword
  
    chosen_genre = st.selectbox('OPTION 2: Choose a genre:', sm_df_full['subgenre'].unique().tolist())
    
    #cc: dropdown:
    
    chosen_subsub = st.selectbox('Choose a subgenre:', sm_df_full.loc[sm_df_full['subgenre'] == chosen_genre]['subgenre2'].unique().tolist())
    
    # cc: add an o-meter by looking at the number of subgenre mentions
    
    sub_ometer = st.selectbox('Subgenre-ometer:', sm_df_full.loc[sm_df_full['subgenre2'] == chosen_subsub]['count'].unique().tolist()) 

    
    #dropdown:
    # cc: add subgenre
#         desired_artist = st.selectbox("Here's a list of artists:", sm_df_full.loc[sm_df_full['subgenre2'] == chosen_subsub]['artist'].unique().tolist())

    desired_artist = st.selectbox("Here's a list of artists:", sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["count"] == sub_ometer), 'artist'].unique().tolist()) #.iloc[0]
    
    desired_album = sm_df_full.loc[(sm_df_full["subgenre2"] == chosen_subsub) & (sm_df_full["artist"] == desired_artist) & (sm_df_full["count"]==sub_ometer),'album'].unique().tolist()
    
    #st.write(f"album: {desired_album}") 
    
    # cc add
    st.write(f"Album and Artist Recs: {desired_album} by {desired_artist}")
    
    # slider test
#     sliders = {
#     "count": st.sidebar.slider(
#         "Filter A", min_value=1.0, max_value=20.0, value=(1, 20.0), step=1
#     ),
#     "B": st.sidebar.slider(
#         "Filter B", min_value=1.0, max_value=20.0, value=(1, 20.0), step=1
#     ),
#     }
#     n_rows=1000
#     filter = np.full(n_rows, True)  # Initialize filter as only True
        
#     for feature_name, slider in sliders.items():
#     # Here we update the filter to take into account the value of each slider
#         filter = (
#             filter
#             & (sm_df_full[feature_name] >= slider[1])
#             & (sm_df_full[feature_name] <= slider[20])
#         )
        
        
    artist_df = sm_df_full[sm_df_full.artist == desired_artist]

    # cc: count is capped
    # cc: change chose brew to chosen subsub
    #topic = int(artist_df[artist_df.count == sub_ometer]['count'].values[0])
    #topic = int(artist_df[artist_df.subgenre2 == chosen_subsub]['count'].values[0])
#    st.write("Here is a look at the dominant words used to describe your choosen artist:")
#     st.image(f'images/topic_{topic}.png')
#    st.write(f'Style: {sm_df_full[sm_df_full.artist == desired_artist]["album"].values[0]}')
    #if I want to show the df for the br, uncomment this:
    #st.write(artist_df[artist_df.brewery == chosen_genre])

    filter_by = st.radio('Do you want to filter by Best reviewed?', ['No', 'Yes'])
    if filter_by == 'Yes':
        best_reviewed = st.selectbox('Choose Best reviewed:', sm_df_full['best'].unique())

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

    
    def full_recommendations(cosine_sim = cosine_sim):
        title = desired_artist # index artist based on full data
        if filter_by == 'Yes':
            location = best_reviewed
        # initializing the empty list of recommended brs
        recommended_albums = []
    
        # gettin the index of the br that matches the name
        idx = indices[indices == title].index[0]

        # creating a Series with the similarity scores in descending order
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

        # getting the indexes of the 100 most similar brs
        top_20_indexes = list(score_series.iloc[1:100].index)
    
            # populating the list with the titles of the best 10 matching brs
    
    
        if filter_by == 'No':
        
            for i in top_20_indexes:
                rec_alb_dict = {}
                # cc : change .index to artist                                
                full_ind = sm_df_full.index[sm_df_full['artist'] == list(sm_df.artist)[i]].tolist()[0]
                
                rec_alb_dict['album'] = list(sm_df.album)[i]
                rec_alb_dict['artist'] = sm_df_full.loc[full_ind]['artist']
                rec_alb_dict['score'] = sm_df_full.loc[full_ind]['score']
                rec_alb_dict['subgenre2'] = sm_df_full.loc[full_ind]['subgenre2']
#                 rec_alb_dict['Rating'] = sm_df_full.loc[full_ind]['avg_score']
#                 rec_alb_dict['url'] = sm_df_full.loc[full_ind]['url']
#                 rec_alb_dict['abv'] = sm_df_full.loc[full_ind]['abv']
#                 rec_alb_dict['avail'] = sm_df_full.loc[full_ind]['avail']
#                 #if not sm_df_full.loc[full_ind]['img'] == None:
#                 rec_alb_dict['Image'] = sm_df_full.loc[full_ind]['img']
                #else:
                    #rec_alb_dict['Image'] = 'no'

                recommended_albums.append(rec_alb_dict)
            st.write("These are the albums you should check out:")
            for i in range(0,3):
                st.write(i+1)
#                 img_url = recommended_albums[i]['Image']
#                 load_image(img_url)
                st.write(f"{recommended_albums[i]['album']} from {recommended_albums[i]['artist']} in the {recommended_albums[i]['subgenre2']} sub-genre")
                st.write("The stats:")
#                 st.write(f"Rating: {recommended_albums[i]['Rating']}")
#                 st.write(f"ABV: {recommended_albums[i]['abv']}")
#                 st.write(f"Availability: {recommended_albums[i]['avail']}")
#                 st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")
            
            #return recommended_albums[:3]
    
        elif filter_by == 'Yes':

            for i in top_20_indexes:
                rec_alb_dict = {}
                # cc: changed .index to artist
                full_ind = sm_df_full.index[sm_df_full['artist'] == list(sm_df.artist)[i]].tolist()[0]
                rec_alb_dict['album'] = list(sm_df.artist)[i]
                rec_alb_dict['score'] = sm_df_full.loc[full_ind]['score']
                rec_alb_dict['subgenre2'] = sm_df_full.loc[full_ind]['subgenre2']
#                 rec_alb_dict['Rating'] = sm_df_full.loc[full_ind]['avg_score']
#                 rec_alb_dict['url'] = sm_df_full.loc[full_ind]['url']
#                 rec_alb_dict['abv'] = sm_df_full.loc[full_ind]['abv']
#                 rec_alb_dict['avail'] = sm_df_full.loc[full_ind]['avail']
#                 rec_alb_dict['Image'] = sm_df_full.loc[full_ind]['img']
                if sm_df_full.loc[full_ind]['subgenre2'] == location:
                    recommended_albums.append(rec_alb_dict)
    
            if len(recommended_albums) > 3:
                for i in range(0,3):
                    st.write(i+1)
#                     img_url = recommended_albums[i]['Image']
#                     load_image(img_url)
                    st.write(f"{recommended_albums[i]['album']} from {recommended_albums[i]['artist']} in {recommended_albums[i]['subgenre2']}")
                    st.write("The stats:")
                    st.write(f"Rating: {recommended_albums[i]['Rating']}")
                    st.write(f"ABV: {recommended_albums[i]['abv']}")
                    st.write(f"Availability: {recommended_albums[i]['avail']}")
                    st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")
            
            elif len(recommended_albums) >= 1:
                for i in range(len(recommended_albums)):
                    st.write(i+1)
                    img_url = recommended_albums[i]['Image']
                    load_image(img_url)
                    st.write(f"{recommended_albums[i]['album']} from {recommended_albums[i]['artist']} in {recommended_albums[i]['subgenre2']}")
                    st.write("The stats:")
#                     st.write(f"Rating: {recommended_albums[i]['Rating']}")
#                     st.write(f"ABV: {recommended_albums[i]['abv']}")
#                     st.write(f"Availability: {recommended_albums[i]['avail']}")
#                     st.write(f"[Click here]({recommended_albums[i]['url']}) to check out the reviews!")

            
            else:
                st.write("Sorry, there are no similar albums in that subgenre")

    if st.button("Discover some Jazz!"):
        st.write(full_recommendations())
    else:
        st.write(full_recommendations())
#         st.image("images/br_meme.jpeg")
    

if __name__ == "__main__":
    main()