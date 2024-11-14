from fastapi import FastAPI
from .handlers.request_handlers import get_info_from_last_videos_in_channel
from .handlers.request_handlers import get_video_info

app = FastAPI()


@app.get("/channel/")
async def root(youtube_channel_url: str, video_count: int = 0):
    get_info_from_last_videos_in_channel(youtube_channel_url, video_count)


@app.get("/video_id/")
async def root(video_id: str):
    get_video_info(video_id)
