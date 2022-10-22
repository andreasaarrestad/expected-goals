#we are going to create a contour plot to see how the heaviside function classifies onto a pitch
#first we must simulate the data above and we can do that by initializing 
#some numpy arrays what have certain properties
from mplsoccer import Pitch, VerticalPitch, Sbopen
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
img = Image.open('pitch.png')

def xg_scatter(test):

    fig  = go.Figure()
    test['xG_display'] = test['xG'].map('{:,.2f}'.format)
    test['posx'] = test[['posx', 'side']].apply(lambda x: 100-x['posx'] if x['side']=="home" else x['posx'], axis=1)
    test = test[test['posx']<50]
    test['posx'] = (test['posx'])*120/100
    test['posy'] = test['posy']*80/100

    fig.add_trace(
        go.Scatter(x = test['posy'],
                y = test['posx'],
                mode='markers',
                marker=dict(size=15,
                showscale=True, opacity=0.7),
                text = test['xG_display'],
                customdata = test[['posx', 'posy', 'xG_display', 'goal', 'side', 'minutes']],
                hovertemplate = 'x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}<br>xG: %{customdata[2]}<br>goal: %{customdata[3]}<br>side: %{customdata[4]}<br>minutes: %{customdata[5]}<extra></extra>',
                marker_color = test['xG'],
    ))

    # axis hide、yaxis reversed
    fig.update_layout(
        autosize=False,
        width=500*1.5,
        height=600*1.2,
        xaxis=dict(visible=True,autorange=False, range = [0, 80]),
        yaxis=dict(visible=True, range = [50,0])
    )

    # background image add
    fig.add_layout_image(
        dict(source=img,
            xref='x',
            yref='y',
            x=-4,
            y=-3,
            sizex=88,
            sizey=53,
            sizing='stretch',
            opacity=1,
            layer='below')
    )

    # Set templates
    fig.update_layout(template="plotly_white")

    return fig