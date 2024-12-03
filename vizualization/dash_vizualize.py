import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import requests
import plotly.graph_objects as go
from collections import Counter

API_KEY = 'AIzaSyCxA3J87dODnkaVWeQ_bMmMp8q6wPMQb8s'
CATEGORY_ID = '28'
BASE_URL = 'https://www.googleapis.com/youtube/v3'

def get_popular_videos(category_id, api_key):
    url = f'{BASE_URL}/videos'
    params = {
        'part': 'id,snippet',
        'chart': 'mostPopular',
        'videoCategoryId': category_id,
        'maxResults': 50,
        'regionCode': 'RU',
        'key': api_key
    }

    response = requests.get(url, params=params)
    return response.json()

def get_video_tags(video_id, api_key):
    url = f'{BASE_URL}/videos'
    params = {
        'part': 'contentDetails,snippet',
        'id': video_id,
        'key': api_key
    }

    response = requests.get(url, params=params)
    video_data = response.json()

    if 'items' in video_data and len(video_data['items']) > 0:
        tags = video_data['items'][0]['snippet'].get('tags', [])
        return tags
    return []


def get_tags_list():
    popular_videos = get_popular_videos(CATEGORY_ID, API_KEY)

    videos_with_tags = []
    tags_list = []

    if 'items' in popular_videos:
        for video in popular_videos['items']:
            video_id = video['id']
            title = video['snippet']['title']
            tags = get_video_tags(video_id, API_KEY)

            if tags:
                videos_with_tags.append((title, tags))
                for tag in tags:
                    tags_list.append(tag.lower())
    return tags_list


tags_list = get_tags_list()

app = dash.Dash(__name__)

tags_count = Counter(tags_list)

popular_tags = tags_count.most_common(10)

popular_tags = [(tag, count) for tag, count in popular_tags if count > 1]

if popular_tags:  
    tags, counts = zip(*popular_tags)

    fig = go.Figure(data=[go.Bar(x=tags, y=counts)])

    fig.update_layout(
        title='Частота тегов',
        xaxis_title='Теги',
        yaxis_title='Частота',
        xaxis_tickangle=-45
    )

    fig.show()

    print("Most_popular:")
    for tag, count in popular_tags:
        print(f"{tag}: {count}")
else:
    print("Not found")

app.layout = html.Div(children=[
   html.H1(children='Hello Dash'),  # Create a title with H1 tag

   html.Div(children='''
       Dash: A web application framework for your data.
   '''),  # Display some text

   dcc.Graph(
       id='example-graph',
       figure=fig
   )  # Display the Plotly figure
])

if __name__ == '__main__':
   app.run_server(debug=True) 
