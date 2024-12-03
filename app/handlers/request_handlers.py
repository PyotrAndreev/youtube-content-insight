import os
import re

from dotenv import load_dotenv
from googleapiclient.discovery import build
import requests
from ..parsing_module import get_info

load_dotenv()

API_KEY = os.getenv("API_KEY")
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
youtube = build('youtube', 'v3', developerKey=API_KEY)


def get_latest_videos(channel_id, max_results):
    request = youtube.search().list(part='id', channelId=channel_id, order='date', maxResults=max_results)
    response = request.execute()
    if max_results == 0 or response['pageInfo']['totalResults'] < max_results:
        max_results = response['pageInfo']['totalResults']
    video_ids = [item['id']['videoId'] for item in response['items'] if item['id']['kind'] == 'youtube#video']
    next_page_token = response.get('nextPageToken', None)

    while len(video_ids) < max_results and next_page_token:
        request = youtube.search().list(part='id', channelId=channel_id, order='date', maxResults=max_results,
                                        pageToken=next_page_token)
        response = request.execute()
        video_ids += [item['id']['videoId'] for item in response['items'] if item['id']['kind'] == 'youtube#video']
        next_page_token = response.get('nextPageToken', None)
    return video_ids


def get_channel_handle_by_url(channel_url: str) -> str | None:
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/@([a-zA-Z0-9_-]+)',
    ]

    for pattern in patterns:
        match = re.match(pattern, channel_url)
        if match:
            return match.group(1)
    return None


def get_channel_id(channel_handle: str):
    api_url = 'https://www.googleapis.com/youtube/v3/channels'
    params = {'part': 'contentDetails',
              'forHandle': '@' + channel_handle,
              'key': API_KEY}
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['id']
        else:
            print("Канал не найден.")
            return None
    else:
        print(f"Ошибка запроса: {response.status_code}")
        return None


def get_info_from_last_videos_in_channel(channel_url: str, video_count: int):
    channel_handle = get_channel_handle_by_url(channel_url)
    channel_id = get_channel_id(channel_handle)
    get_info.get_channel_info(channel_id)

    video_ids = get_latest_videos(channel_id, video_count)
    print(*video_ids)
    print(len(video_ids))

    for video_id in video_ids:
        get_info.get_video_details(video_id)
        get_info.fetch_comments(video_id)


def get_video_info(video_id):
    get_info.get_video_details(video_id)
    get_info.fetch_comments(video_id)
