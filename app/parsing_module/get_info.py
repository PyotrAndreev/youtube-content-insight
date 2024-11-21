import requests
import os
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


def get_video_details(video_id: str):
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
        if not work_with_models.check_exists_channel_by_id(channel_id):
            get_channel_info(channel_id)

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
            raise Exception("Dislike API: Status code " + str(response2.status_code))
    else:
        raise Exception("Youtube API: Status code " + str(response.status_code))


def get_channel_info(channel_id):
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
                logging.info('Parsing next page token - ' + response['nextPageToken'] + 'for video '+ video_id)
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=100,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break
    else:
        logging.info(f'Video {video_id} has no comments available.')


def get_transcript(video_id: str) -> list[dict]:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript
