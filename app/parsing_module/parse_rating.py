from app.parsing_module.automatized_parsing import get_latest_videos, fetch_comments
import os
import time

rating_key = os.getenv("API_KEY_RATING")


def parse_relevance(api_key):
    VIDEO_IDS = get_latest_videos(api_key, "rating")
    while VIDEO_IDS:
        video_id = VIDEO_IDS[0]
        while True:
            success, response, can_get = fetch_comments(video_id, api_key)
            if not can_get:
                break
            if not success:
                print('waiting...')
                time.sleep(24 * 60 * 60)
                break
            if not response.get('nextPageToken'):
                break
        VIDEO_IDS.pop(0)
        if len(VIDEO_IDS) == 0:
            VIDEO_IDS = get_latest_videos(api_key, "rating")


def main():
    parse_relevance(rating_key)


if __name__ == "__main__":
    main()