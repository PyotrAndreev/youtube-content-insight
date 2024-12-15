import os
import re

from dotenv import load_dotenv
from googleapiclient.discovery import build
import logging
import requests
from ..parsing_module import get_info
from ..models_module import work_with_models
from ..analytics.comments_analytics import comments_emotional_analytics_in_video
from ..analytics.comments_analytics import comments_emotional_analytics_video_to_video
from ..analytics.comments_clustering import clustering

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("API_KEY")
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
youtube = build('youtube', 'v3', developerKey=API_KEY)


def get_latest_videos(channel_id, max_results):
    """
    Retrieve the most recent videos from a YouTube channel.

    Args:
        channel_id (str): The unique ID of the YouTube channel.
        max_results (int): The maximum number of video IDs to fetch.

    Behavior:
        - Fetches video IDs of the latest uploads from the specified channel, using the YouTube Data API.
        - Handles pagination to ensure up to `max_results` videos are retrieved if available.
        - Logs the process of fetching videos and the total number retrieved.

    Returns:
        list[str]: A list of video IDs corresponding to the latest uploads.
    """
    logging.info(f'Start getting last videos from {channel_id}')
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
    logging.info(f'End getting last videos from {channel_id}, finally find - {len(video_ids)} videos')
    return video_ids


def get_channel_handle_by_url(channel_url: str) -> str | None:
    """
    Extract the YouTube channel handle from a given channel URL.

    Args:
        channel_url (str): The URL of the YouTube channel.

    Behavior:
        - Matches the provided URL against a set of regular expression patterns to extract the channel handle.
        - Logs the successful extraction of the channel handle.
        - Raises a `ValueError` if the handle cannot be extracted.

    Returns:
        str: The extracted channel handle if successful.
    """
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/@([a-zA-Z0-9_-]+)',
    ]

    for pattern in patterns:
        match = re.match(pattern, channel_url)
        if match:
            logging.info(f'Get channel handle from URL: {match.group(1)}')
            return match.group(1)
    raise ValueError(f'Could not find channel handle from URL: {channel_url}')


def get_channel_id(channel_handle: str):
    """
    Retrieve the channel ID for a given channel handle using the YouTube Data API.

    Args:
        channel_handle (str): The handle of the YouTube channel, prefixed with "@".

    Behavior:
        - Sends a request to the YouTube Data API to fetch the channel ID.
        - Logs the fetched channel ID if the request is successful.
        - Raises a `ValueError` if the channel ID cannot be retrieved or the API request fails.

    Returns:
        str: The unique ID of the channel.
    """
    api_url = 'https://www.googleapis.com/youtube/v3/channels'
    params = {'part': 'contentDetails',
              'forHandle': '@' + channel_handle,
              'key': API_KEY}
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:

            logging.info('Get channel id from URL: ' + str(data['items'][0]['id']))
            return data['items'][0]['id']
        else:

            raise ValueError(f'Could not get channel id from URL: {channel_handle}')
    else:
        logging.error("Request to get channel id failed with status code: " + str(response.status_code))
        raise ValueError(f'Request to get channel id failed with status code: {response.status_code}')


def get_video_id(url):
    video_id = None
    match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
    if match:
        video_id = match.group(1)
    return video_id


def get_info_from_last_videos_in_channel(channel_url: str, video_count: int):
    """
    Fetch information from the latest videos of a specified YouTube channel.

    Args:
        channel_url (str): The URL of the YouTube channel.
        video_count (int): The number of latest videos to process.

    Behavior:
        - Extracts the channel handle from the URL and retrieves the channel ID.
        - Collects metadata and statistics about the specified number of recent videos.
        - For each video:
            - Creates a video parsing context.
            - Fetches video details and associated comments.

    Returns:
        None
    """
    channel_handle = get_channel_handle_by_url(channel_url)
    channel_id = get_channel_id(channel_handle)
    get_info.get_channel_info(channel_id)

    video_ids = get_latest_videos(channel_id, video_count)

    for video_id in video_ids:
        work_with_models.create_video_context(video_id)

    for video_id in video_ids:
        get_info.get_video_details(video_id)
        get_info.fetch_comments(video_id)


def get_video_info(video_id):
    """
    Retrieve information and comments for a specific video.

    Args:
        video_id (str): The unique ID of the YouTube video.

    Behavior:
        - Creates a parsing context for the video.
        - Fetches video metadata and statistics using the YouTube Data API.
        - Fetches and processes comments associated with the video.

    Returns:
        None
    """
    work_with_models.create_video_context(video_id)
    get_info.get_video_details(video_id)
    get_info.fetch_comments(video_id)


def get_video_analytics(video_url: str):
    video_id = get_video_id(video_url)
    get_video_info(video_id)
    comments_emotional_analytics_in_video(video_id)


def get_videos_analytics(video_urls: [str]):
    video_ids = []
    for video_url in video_urls:
        video_id = get_video_id(video_url)
        video_ids.append(video_id)
        get_video_info(video_id)
    comments_emotional_analytics_video_to_video(video_ids)
