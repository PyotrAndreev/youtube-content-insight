import logging

import pandas as pd
from sqlalchemy import exists, and_
from .db_architecture import VideoStatsLast, ChannelStatsLast, Context, Video
from ..models_module import db_architecture
from ..models_module import db_sessions
from datetime import datetime, timezone
from .db_architecture import Source, Status

logging.basicConfig(level=logging.INFO)


def save_channel_info(channel_info: dict, channel_id: str):
    if not check_exists_channel_by_id(channel_id):
        channel_imp = db_architecture.Channel(
            channelId=channel_id,
            title=channel_info.get('snippet', {}).get('title'),
            description=channel_info.get('snippet', {}).get('description', None),
            customUrl=channel_info.get('snippet', {}).get('customUrl', None),
            publishedAt=channel_info.get('snippet', {}).get('publishedAt', None),
            thumbnail=channel_info.get('snippet', {}).get('thumbnails', {}).get('default', {}).get('url', None),
            localizedTitle=channel_info.get('snippet', {}).get('localized', {}).get('title', None),
            localizedDescription=channel_info.get('snippet', {}).get('localized', {}).get('description', None),
            country=channel_info.get('snippet', {}).get('country', None),
            relatedPlaylistsLikes=channel_info.get('contentDetails', {}).get('relatedPlaylists', {}).get('likes', None),
            relatedPlaylistsUploads=channel_info.get('contentDetails', {}).get('relatedPlaylists', {}).get('uploads',
                                                                                                             None),
            privacyStatus=channel_info.get('status', {}).get('privacyStatus', None),
            isLinked=channel_info.get('status', {}).get('isLinked', None),
            longUploadsStatus=channel_info.get('status', {}).get('longUploadsStatus', None),
            madeForKids=channel_info.get('status', {}).get('madeForKids', None),
            brandingSettingsChannelTitle=channel_info.get('brandingSettings', {}).get('channel', {}).get('title', None),
            brandingSettingsChannelDescription=channel_info.get('brandingSettings', {}).get('channel', {}).get(
                'description', None),
            brandingSettingsChannelKeywords=channel_info.get('brandingSettings', {}).get('channel', {}).get('keywords',
                                                                                                               None),
            brandingSettingsChannelUnsubscribedTrailer=channel_info.get('brandingSettings', {}).get('channel', {}).get(
                'unsubscribedTrailer', None))
        channel_stats_imp = db_architecture.ChannelStatsLast(
            channelId=channel_id,
            viewCount=channel_info.get('statistics', {}).get('viewCount', None),
            subscribersCount=channel_info.get('statistics', {}).get('subscriberCount', None),
            hiddenSubscriberCount=channel_info.get('statistics', {}).get('hiddenSubscriberCount', None),
            videoCount=channel_info.get('statistics', {}).get('videoCount', None),
            topicCategories=channel_info.get('topicDetails', {}).get('topicCategories', None),
            parsingDate=datetime.now(timezone.utc).replace(microsecond=0)
        )
        db_sessions.session.add(channel_imp)
        db_sessions.session.add(channel_stats_imp)
        db_sessions.session.commit()
        logging.info(f'Saved channel info for {channel_id} successfully')

    else:
        q = db_sessions.session.query(ChannelStatsLast)
        q = q.filter(ChannelStatsLast.channelId == channel_id)
        record = q.one()
        record.viewCount = channel_info.get('statistics', {}).get('viewCount', None)
        record.subscribersCount = channel_info.get('statistics', {}).get('subscriberCount', None)
        record.hiddenSubscriberCount = channel_info.get('statistics', {}).get('hiddenSubscriberCount', None)
        record.videoCount = channel_info.get('statistics', {}).get('videoCount', None)
        record.topicCategories = channel_info.get('topicDetails', {}).get('topicCategories', None)
        record.parsingDate = datetime.now(timezone.utc).replace(microsecond=0)
        db_sessions.session.commit()
        logging.info(f'Update channel stats for {channel_id} successfully')


def save_video_info(video_info: dict, video_api_info: dict, channel_id: str, video_id: str):
    if not check_exists_video_by_id(video_id):
        video_imp = db_architecture.Video(
            channelId=channel_id,
            videoId=video_id,
            publishedAt=video_info.get('snippet', {}).get('publishedAt'),
            title=video_info.get('snippet', {}).get('title'),
            description=video_info.get('snippet', {}).get('description'),
            thumbnail=video_info.get('snippet', {}).get('thumbnails', {}).get('default', {}).get('url', None),
            channelTitle=video_info.get('snippet', {}).get('channelTitle', None),
            tags=video_info.get('snippet', {}).get('tags', None),
            defaultLanguage=video_info.get('snippet', {}).get('defaultLanguage', None),
            defaultAudioLanguage=video_info.get('snippet', {}).get('defaultAudioLanguage', None),
            categoryId=video_info.get('snippet', {}).get('categoryId', None),
            duration=video_info.get('contentDetails', {}).get('duration', None),
            dimension=video_info.get('contentDetails', {}).get('dimension', None),
            definition=video_info.get('contentDetails', {}).get('definition', None),
            caption=video_info.get('contentDetails', {}).get('caption', None),
            licensedContent=video_info.get('contentDetails', {}).get('licensedContent', None),
            uploadStatus=video_info.get('status', {}).get('uploadStatus', None),
            privacyStatus=video_info.get('status', {}).get('privacyStatus', None),
            license=video_info.get('status', {}).get('license', None),
            embeddable=video_info.get('status', {}).get('embeddable', None),
            publicStatsViewable=video_info.get('status', {}).get('publicStatsViewable', None),
            madeForKids=video_info.get('status', {}).get('madeForKids', None))
        db_sessions.session.add(video_imp)

        video_stats_imp = db_architecture.VideoStatsLast(
            videoId=video_id,
            liveBroadcastContent=video_info.get('snippet', {}).get('liveBroadcastContent', None),
            viewsCount=video_info.get('statistics', {}).get('viewCount', None),
            likesCount=video_info.get('statistics', {}).get('likeCount', None),
            likesFromApi=video_api_info.get('likes', None),
            dislikesFromApi=video_api_info.get('dislikes', None),
            ratingFromApi=video_api_info.get('rating', None),
            favoriteCount=video_info.get('statistics', {}).get('favoriteCount', None),
            commentCount=video_info.get('statistics', {}).get('commentCount', None),
            parsingDate=datetime.now(timezone.utc).replace(microsecond=0)
        )
        db_sessions.session.add(video_stats_imp)
        db_sessions.session.commit()
        logging.info(f'Save video info for {video_id} successfully')
    else:
        q = db_sessions.session.query(VideoStatsLast)
        q = q.filter(VideoStatsLast.videoId == video_id)
        record = q.one()
        record.liveBroadcastContent = video_info.get('snippet', {}).get('liveBroadcastContent', None),
        record.viewsCount = video_info.get('statistics', {}).get('viewCount', None),
        record.likesCount = video_info.get('statistics', {}).get('likeCount', None),
        record.likesFromApi = video_api_info.get('likes', None),
        record.dislikesFromApi = video_api_info.get('dislikes', None),
        record.ratingFromApi = video_api_info.get('rating', None),
        record.favoriteCount = video_info.get('statistics', {}).get('favoriteCount', None),
        record.commentCount = video_info.get('statistics', {}).get('commentCount', None),
        record.parsingDate = datetime.now(timezone.utc).replace(microsecond=0)
        db_sessions.session.commit()
        logging.info(f'Update video stats for {video_id} successfully')


def save_comments(comment: dict, comment_id: str):
    if not check_exists_comment_by_id(comment_id):
        comment_imp = db_architecture.Comment(
            commentId=comment_id,
            videoId=comment.get('videoId', None),
            authorDisplayName=comment.get('authorDisplayName', None),
            authorProfileImageUrl=comment.get('authorProfileImageUrl', None),
            authorChannelUrl=comment.get('authorChannelUrl', None),
            authorChannelId=comment.get('authorChannelId', {}).get('value', None),
            textDisplay=comment.get('textDisplay', None),
            textOriginal=comment.get('textOriginal', None),
            parentId=comment.get('parentId', None),
            canRate=comment.get('canRate', None),
            viewerRating=comment.get('viewerRating', None),
            likeCount=comment.get('likeCount', None),
            publishedAt=comment.get('publishedAt', None),
            updatedAt=comment.get('updatedAt', None),
            gotFrom=Source.query
        )
        db_sessions.session.add(comment_imp)
        db_sessions.session.commit()
        logging.info(f'Save comment for {comment_id} successfully')


def create_channel_context(channel_id: str):
    context_imp = db_architecture.Context(
        channelId=channel_id,
        status=Status.parsing_channel,
        date=datetime.now(timezone.utc).replace(microsecond=0)
    )
    db_sessions.session.add(context_imp)
    db_sessions.session.commit()
    logging.info(f'Create channel context for {channel_id} successfully')


def finish_channel_context(channel_id: str):
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.channelId == channel_id, Context.status != Status.finish))
    record = q.one()
    record.status = Status.finish
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Finish channel context for {channel_id} successfully')


def create_video_context(video_id: str):
    context_imp = db_architecture.Context(
        videoId=video_id,
        status=Status.parsing_video,
        date=datetime.now(timezone.utc).replace(microsecond=0)
    )
    db_sessions.session.add(context_imp)
    db_sessions.session.commit()
    logging.info(f'Create video context for {video_id} successfully')


def finish_video_context(video_id: str):
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.videoId == video_id, Context.status != Status.finish))
    record = q.one()
    record.status = Status.parsing_comments
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Finish video context for {video_id} successfully')


def update_comments_context(video_id: str, comment_page_id: str):
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.videoId == video_id, Context.status != Status.finish))
    record = q.one()
    record.commentPageId = comment_page_id
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Update comment context for {video_id} starts parsing from {comment_page_id} page')


def finish_comment_context(video_id: str):
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.videoId == video_id, Context.status != Status.finish))
    record = q.one()
    record.status = Status.finish
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Finish comment context for {video_id} successfully')


def check_exists_video_by_id(video_id: str):
    exists_query = db_sessions.session.query(exists().where(db_architecture.Video.videoId == video_id)).scalar()

    if exists_query:
        return True
    else:
        return False


def check_exists_channel_by_id(channel_id: str):
    exists_query = db_sessions.session.query(exists().where(db_architecture.Channel.channelId == channel_id)).scalar()

    if exists_query:
        return True
    else:
        return False


def check_exists_comment_by_id(comment_id: str):
    exists_query = db_sessions.session.query(exists().where(db_architecture.Comment.commentId == comment_id)).scalar()

    if exists_query:
        return True
    else:
        return False


def check_is_comments_available(video_id: str):
    video_el = db_sessions.session.query(db_architecture.VideoStatsLast).filter(db_architecture.VideoStatsLast.videoId
                                                                                == video_id).first()
    if not video_el.commentCount:
        return False
    if video_el.commentCount > 0:
        return True
    return False


def get_comments_df(video_id: str):
    df = pd.read_sql(f"""SELECT * FROM comments WHERE comments."videoId" = '{video_id}'""", con=db_sessions.engine)
    return df


def video_published_time(video_id: str):
    q = db_sessions.session.query(Video)
    q = q.filter(Video.videoId == video_id)
    record = q.one()
    return record.publishedAt.strftime("%Y-%m-%d")
