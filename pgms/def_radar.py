# import pandas for data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
%matplotlib inline



def spotfeats(data,alb):
    df_spot2 = pd.DataFrame()
    df_spot2 = data.loc[data['album'] == alb] # ) & (data['topic']==topics)]    
    #df_spot2 = data.loc[data['subgenre'] == alb] # ) & (data['topic']==topics)]    
    #df_spot3 = df_spot2.loc[df_spot2['year']<=2018]
    df_spot2 = df_spot2[['acousticness',   'instrumentalness', 'tempo', 'speechiness', 'valence','energy','liveness','danceability','loudness']]
    
    # get mean of scores
    df_spot2 = df_spot2[['acousticness',   'instrumentalness', 'tempo', 'speechiness', 'valence','energy','liveness','danceability','loudness']].mean()    
    df_spot3 = pd.DataFrame(df_spot2)
    df_spot3_T = df_spot3.T
    df_spot3_T['loudness'] = abs(df_spot3_T['loudness'])
       
    
    return df_spot3_T



def radar(data,gen,colors):

    # convert column names into a list
    categories=list(data.columns)
    # number of categories
    N=len(categories)

    # create a list with the average of all features
    value = list(data.mean())
    
    # tempo - scaled
    value[2] = value[2]/220
    
    # speech - scaled
    value[3] = value[3]*1.5
    
    # inst - scaled
    value[1] = value[1] # *2
    
    # loudness - scaled
    value[8] = value[8]/25
    
    # repeat first value to close the circle
    # the plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    value+=value[:1]
    
    # calculate angle for each category
    angles=[n/float(N)*2*math.pi for n in range(N)]
    angles+=angles[:1]

    # plot
    fig=plt.figure(figsize = (16,16))

    ax = fig.add_subplot(221, polar=True)

    #plot 1 hits
    ax.plot(angles, value,  linewidth=2, label = gen, color= colors)
    ax.fill(angles, value, alpha=0.35, facecolor=colors)
    ax.set_ylim(0, 1)
    
    ax.grid(True)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),fontsize='large')

    plt.xticks(angles[:-1],categories, size=16,color='black')
    plt.yticks(color='grey',size=16)

    # Create a color palette:
    plt.cm.get_cmap("Set2", len(data.index))
    
    plt.show()