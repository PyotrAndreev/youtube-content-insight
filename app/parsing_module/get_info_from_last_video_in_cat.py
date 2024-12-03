import requests
import os
import logging
from dotenv import load_dotenv
from ai_analyzer.app.parsing_module import get_info

load_dotenv()

API_KEY = os.getenv("API_KEY")
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
logger = logging.getLogger(__name__)


def search_last_videos_by_category(category_id: int, max_results: int) -> dict | None:
    url = f'{YOUTUBE_API_URL}search'
    params = {
        'part': 'snippet',
        'type': 'video',
        'videoCategoryId': str(category_id),
        'maxResults': max_results,
        'order': 'date',
        'key': API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Error: {response.status_code}')
        return None


def get_info_from_last_videos_in_category(category_id: int, max_results=10) -> None:
    list_of_videos = search_last_videos_by_category(category_id, max_results)

    if list_of_videos:
        for idx, item in enumerate(list_of_videos['items'], start=1):
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            print(f'{idx}. Video ID: {video_id}, Title: {title}')
            video_details = get_info.get_video_details(video_id)
            print(video_details)

            print("Comments")
            video_comments = get_info.fetch_comments(video_id)
            print(video_comments)

            print("Transcript")
            transcript = get_info.get_transcript(video_id)
            print(transcript)


if __name__ == '__main__':
    get_info_from_last_videos_in_category(28, 1)
