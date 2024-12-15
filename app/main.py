from fastapi import FastAPI
from .handlers.request_handlers import get_video_info, get_videos_info
from .analytics.comments_analytics import comments_emotional_analytics_in_video
from .analytics.comments_analytics import comments_emotional_analytics_video_to_video
from .analytics.comments_clustering import clustering
from typing import List

app = FastAPI()


@app.get("/video_analytics/", status_code=200)
async def root(video_id: str):
    get_video_info(video_id)
    comments_emotional_analytics_in_video(video_id)


@app.post("/videos_analytics/", status_code=200)
def read_items(video_ids: List[str]):
    get_videos_info(video_ids)
    comments_emotional_analytics_video_to_video(video_ids)


@app.get("/comment_clustering/", status_code=200)
async def root(video_id: str):
    get_video_info(video_id)
    clustering(video_id)
