import requests
import os
import time
import logging
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from ..models_module import work_with_models

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = os.getenv("API_KEY")
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
youtube = build('youtube', 'v3', developerKey=API_KEY)


class RetryableRequestError(Exception):
    def __init__(self, request, *args):
        super().__init__(*args)
        self.request = request

    def __str__(self):
        return super().__str__() + "\n" + repr(self.request)


def get_info_from_dislike_api(video_id: str):
    """
    Retrieve dislike information for a YouTube video using the Return YouTube Dislike API.

    Args:
        video_id (str): The unique ID of the YouTube video.

    Behavior:
        - Sends a GET request to the API with the provided video ID.
        - Parses the JSON response if the request is successful.
        - Logs the process of fetching dislike information.
        - Handles retryable errors (status codes 5xx) by raising a `RetryableRequestError`.
        - Raises an exception for non-retryable errors.

    Returns:
        dict: The JSON response containing dislike data for the video.
    """
    url_api = 'https://returnyoutubedislikeapi.com/votes'
    params = {
        'videoId': video_id,
    }
    response = requests.get(url_api, params=params, headers={
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Pragma": "no-cache", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    if response.status_code == 200:
        video_api = response.json()
        logger.info(f'Get info from dislike API for {video_id}')
        return video_api
    elif response.status_code % 100 == 5:
        raise RetryableRequestError([url_api, params], response.status_code, response.text)
    else:
        raise Exception(response.status_code, response.text)


def get_video_info_from_youtube(video_id: str):
    """
    Fetch detailed information for a YouTube video using the YouTube Data API.

    Args:
        video_id (str): The unique ID of the YouTube video.

    Behavior:
        - Sends a GET request to the YouTube Data API with the video ID.
        - Logs the process of fetching video details.
        - Handles retryable errors (status codes 5xx) by raising a `RetryableRequestError`.
        - Raises an exception for other errors.
        - Extracts the video information and associated channel ID from the response.

    Returns:
        tuple: A tuple containing:
            - video_info (dict): The metadata and statistics of the video.
            - channel_id (str): The ID of the channel that uploaded the video.
    """
    url = f'{YOUTUBE_API_URL}videos'
    params = {
        'part': 'snippet,contentDetails,status,statistics,paidProductPlacementDetails',
        'id': video_id,
        'key': API_KEY
    }
    logger.info(f'Start getting video details for {video_id}')

    response = requests.get(url, params=params)
    if response.status_code == 200:
        video_info = response.json()['items'][0]
        channel_id = video_info['snippet']['channelId']
        logger.info(f'Get video details for {video_id}')
        return (video_info, channel_id)
    elif response.status_code % 100 == 5:
        raise RetryableRequestError([url, params], response.status_code, response.text)
    else:
        raise Exception("Youtube API: Status code " + str(response.status_code))


def with_retry(func, retry_cnt):
    """
    Execute a function with retry logic for handling transient errors.

    Args:
        func (callable): The function to execute.
        retry_cnt (int): The maximum number of retries for retryable errors.

    Behavior:
        - Executes the provided function.
        - Retries the function call if a `RetryableRequestError` is raised.
        - Logs any unexpected errors and raises them immediately.
        - Waits for 1 second between retries.

    Returns:
        Any: The result of the successfully executed function.

    Raises:
        Exception: If all retry attempts fail.
    """
    exception = None
    retry_cnt = max(retry_cnt, 0)
    for i in range(retry_cnt + 1):
        try:
            res = func()
            return res
        except RetryableRequestError as e:
            exception = e
            time.sleep(1)
            continue
        except Exception as e:
            logger.error("Unexpected error \t" + str(e))
            raise
    logger.error(f"After {retry_cnt} tries got \t" + str(exception))


def get_video_details(video_id: str):
    """
    Fetch and store detailed information for a YouTube video.

    Args:
        video_id (str): The unique ID of the YouTube video.

    Behavior:
        - Retrieves video metadata and statistics using `get_video_info_from_youtube` with retries.
        - Checks if the video's channel exists in the database, and fetches its information if not.
        - Fetches dislike data using `get_info_from_dislike_api` with retries.
        - Saves the video's metadata, dislike data, and context to the database.

    Returns:
        None
    """
    video_info, channel_id = with_retry(lambda: get_video_info_from_youtube(video_id), 5)
    if not work_with_models.check_exists_channel_by_id(channel_id):
        get_channel_info(channel_id)

    video_api = with_retry(lambda: get_info_from_dislike_api(video_id), 5)
    work_with_models.save_video_info(video_info, video_api, channel_id, video_id)
    work_with_models.finish_video_context(video_id)


def get_channel_info(channel_id):
    """
    Fetch and store detailed information for a YouTube channel.

    Args:
        channel_id (str): The unique ID of the YouTube channel.

    Behavior:
        - Sends a request to the YouTube Data API to fetch detailed channel information.
        - Logs the process of fetching channel details.
        - Saves the channel's metadata and context to the database.

    Returns:
        None
    """
    work_with_models.create_channel_context(channel_id)
    logger.info(f'Start getting channel details for {channel_id}')
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics,topicDetails,status,brandingSettings,contentOwnerDetails,localizations',
        id=channel_id)

    # auditDetails - doesn't have permission;
    # defaultLanguage, selfDeclaredMadeForKids, trackingAnalyticsAccountId, contentOwner, timeLinked - None

    response = request.execute()
    logger.info('Get channel details for {}'.format(channel_id))
    channel_info = response['items'][0]
    work_with_models.save_channel_info(channel_info, channel_id)
    work_with_models.finish_channel_context(channel_id)


def fetch_comments(video_id: str):
    """
    Retrieve and store comments for a YouTube video.

    Args:
        video_id (str): The unique ID of the YouTube video.

    Behavior:
        - Checks if comments are available for the video.
        - Fetches top-level comments and their replies using the YouTube Data API.
        - Handles pagination to retrieve all available comments.
        - Logs the process of fetching and saving comments.
        - Updates the comments context in the database.

    Returns:
        None
    """
    counter = 0
    if work_with_models.check_is_comments_available(video_id):
        response = youtube.commentThreads().list(
            part='snippet, replies',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100
        )
        try:
            response = response.execute()
            logging.info('Get first comments for video {}'.format(video_id))
        except Exception as e:
            # video has no comments
            return
        while response:
            for item in response['items']:
                comment_id = item['snippet']['topLevelComment']['id']
                comment = item['snippet']['topLevelComment']['snippet']
                work_with_models.save_comments(comment, comment_id)
                counter += 1
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_comment_id = reply['id']
                        reply_comment = reply['snippet']
                        work_with_models.save_comments(reply_comment, reply_comment_id)
                        counter += 1
            logging.info(' Parsing successfully {counter} comments for video_id - {video_id}'.format(
                counter=counter, video_id=video_id))
            if 'nextPageToken' in response:
                work_with_models.update_comments_context(video_id, response['nextPageToken'])
                logging.info('Parsing next page token - ' + response['nextPageToken'] + 'for video ' + video_id)
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=100,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break
        work_with_models.finish_comment_context(video_id)
    else:
        logging.info(f'Video {video_id} has no comments available.')
        work_with_models.finish_comment_context(video_id)


def get_transcript(video_id: str) -> list[dict]:
    """
    Retrieve the transcript for a YouTube video.

    Args:
        video_id (str): The unique ID of the YouTube video.

    Behavior:
        - Uses the YouTubeTranscriptApi library to fetch the video's transcript.
        - Returns the transcript as a list of dictionaries.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - `text`: The transcript text.
            - `start`: The start time of the segment.
            - `duration`: The duration of the segment.
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript
