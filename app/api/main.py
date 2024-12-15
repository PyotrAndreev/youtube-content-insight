from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.handlers.request_handlers import get_video_analytics, get_videos_analytics
from app.handlers.request_handlers import get_video_info
from app.analytics.comments_clustering import clustering
from typing import List

app = FastAPI()


@app.post("/video_analytics/", status_code=200)
async def root(video_url: str):
    get_video_analytics(video_url)
    image_path = "app/content/in_video.png"
    return FileResponse(image_path, media_type="image/png")


@app.post("/videos_analytics/", status_code=200)
def read_items(video_urls: List[str]):
    get_videos_analytics(video_urls)
    image_path = "app/content/video_to_video.png"
    return FileResponse(image_path, media_type="image/png")


# todo replace video_id to video_url
@app.post("/comment_clustering/", status_code=200)
async def root(video_id: str):
    get_video_info(video_id)
    clustering(video_id)
