# import pandas for data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def circbar(data,maxarts=40):

    my_cmap=sns.cm.rocket_r  #cmap= sns.cubehelix_palette()    
    #my_cmap = plt.get_cmap("viridis") # viridis
    # Reorder the dataframe
    data = data[0:maxarts]
    data = data.sort_values(by=['Count'])

    # initialize the figure
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111, polar=True)

    # fig=plt.figure()
    # fig.set_figheight(20)
    # fig.set_figwidth(20)

    plt.axis('off')

    # Constants = parameters controling the plot layout:
    upperLimit = 200
    lowerLimit = 15
    labelPadding = 4

    # Compute max and min in the dataset
    max = data['Count'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * data.Count + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(data.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(data.index)+1))
    angles = [element * width for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=3, 
        edgecolor="white",
        #color="#61a4b2",
        color=my_cmap.colors,
    )

    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, data["Word"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
        
    #plt.tight_layout()
    