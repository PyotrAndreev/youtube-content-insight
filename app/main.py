from fastapi import FastAPI, HTTPException
from .handlers.request_handlers import get_info_from_last_videos_in_channel
from .handlers.request_handlers import get_video_info

app = FastAPI()


@app.get("/channel/", status_code=200)
async def root(youtube_channel_url: str, video_count: int = 0):
    try:
        get_info_from_last_videos_in_channel(youtube_channel_url, video_count)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/video_id/", status_code=200)
async def root(video_id: str):
    get_video_info(video_id)
