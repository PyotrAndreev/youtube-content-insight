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
    video_info, channel_id = with_retry(lambda: get_video_info_from_youtube(video_id), 5)
    if not work_with_models.check_exists_channel_by_id(channel_id):
        get_channel_info(channel_id)

    video_api = with_retry(lambda: get_info_from_dislike_api(video_id), 5)
    work_with_models.save_video_info(video_info, video_api, channel_id, video_id)
    work_with_models.finish_video_context(video_id)


def get_channel_info(channel_id):
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
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript
