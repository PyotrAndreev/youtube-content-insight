import requests
import os
import logging
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from app.models_module import work_with_models
from googleapiclient.errors import HttpError


load_dotenv()
logger = logging.getLogger(__name__)
# API_KEY = os.getenv("API_KEY")
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
# youtube = build('youtube', 'v3', developerKey=API_KEY)

MAX_REQUESTS_PER_DAY = 10000
MAX_COMMENTS_PER_REQUEST = 100

def get_video_details(video_id: str, api_key):
    url = f'{YOUTUBE_API_URL}videos'
    params = {
        'part': 'snippet,contentDetails,status,statistics,paidProductPlacementDetails',
        'id': video_id,
        'key': api_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        video_info = response.json()['items'][0]
        channel_id = video_info['snippet']['channelId']
        if not work_with_models.check_exists_channel_by_id(channel_id):
            get_channel_info(channel_id, api_key)

        urlApi = 'https://returnyoutubedislikeapi.com/votes'
        params = {
            'videoId': video_id,
        }
        response2 = requests.get(urlApi, params=params, headers={
             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
             "Pragma": "no-cache", "Cache-Control": "no-cache",
             "Connection": "keep-alive"})

        if response2.status_code == 200:
            videoApi = response2.json()
            work_with_models.save_video_info(video_info, videoApi, channel_id, video_id)
    else:
        print(f'Error: {response.status_code}')


def get_channel_info(channel_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics,topicDetails,status,brandingSettings,contentOwnerDetails,localizations',
        id=channel_id)

    response = request.execute()
    channel_info = response['items'][0]
    work_with_models.save_channel_info(channel_info, channel_id)

def fetch_comments(video_id: str, api_key):
    counter = 0
    requests_made = 0

    youtube = build('youtube', 'v3', developerKey=api_key)
    request_comments = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        textFormat='plainText',
        maxResults=MAX_COMMENTS_PER_REQUEST
    )
    try:
        response = request_comments.execute()
    except HttpError:
        return False, False, False

    while True:
        for item in response['items']:
            comment_id = item['snippet']['topLevelComment']['id']
            comment = item['snippet']['topLevelComment']['snippet']
            print(comment_id, comment)
            work_with_models.save_comments(comment, comment_id)
            counter += 1

            if 'replies' in item:
                for reply in item['replies']['comments']:
                    reply_comment_id = reply['id']
                    reply_comment = reply['snippet']
                    work_with_models.save_comments(reply_comment, reply_comment_id)
                    print(comment_id, comment)
                    counter += 1

        logger.info('Parsed {counter} comments for video_id - {video_id}'.format(counter=counter, video_id=video_id))

        requests_made += 1

        if requests_made >= MAX_REQUESTS_PER_DAY:
            logger.info('Reached daily limit of API requests. Waiting for the next day.')
            return False, False, True

        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                textFormat='plainText',
                maxResults=MAX_COMMENTS_PER_REQUEST,
                pageToken=response['nextPageToken']
            ).execute()
        else:
            break

    return True, response, True


def get_transcript(video_id: str) -> list[dict]:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


def get_latest_videos(api_key, order_by):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'id',
        'maxResults': 30,
        'order': order_by,
        'videoCategoryId': 26,
        'type': 'video',
        'key': api_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        videos = response.json().get('items', [])
        video_ids = [video['id']['videoId'] for video in videos]
        return video_ids
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


