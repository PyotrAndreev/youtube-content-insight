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
    """
    Save or update channel information in the database.

    Args:
        channel_info (dict): The dictionary containing channel data, including snippets, statistics, and settings.
        channel_id (str): The unique ID of the channel.

    Behavior:
        - If the channel does not already exist in the database:
            - Creates a new record in the `Channel` table.
            - Creates a corresponding record in the `ChannelStatsLast` table.
        - If the channel exists:
            - Updates the `ChannelStatsLast` record with the latest statistics.

    Returns:
        None
    """
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
    """
    Save or update video information in the database.

    Args:
        video_info (dict): The dictionary containing video details from the primary API response.
        video_api_info (dict): Additional video statistics fetched from a secondary API.
        channel_id (str): The unique ID of the channel to which the video belongs.
        video_id (str): The unique ID of the video.

    Behavior:
        - If the video does not already exist in the database:
            - Creates a new record in the `Video` table.
            - Creates a corresponding record in the `VideoStatsLast` table.
        - If the video exists:
            - Updates the `VideoStatsLast` record with the latest statistics.

    Returns:
        None
    """
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
    """
    Save a new comment in the database.

    Args:
        comment (dict): The dictionary containing comment details.
        comment_id (str): The unique ID of the comment.

    Behavior:
        - If the comment does not already exist in the database:
            - Creates a new record in the `Comment` table.
        - Logs a success message upon successful saving.

    Returns:
        None
    """
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
    """
    Create a parsing context for a channel.

    Args:
        channel_id (str): The unique ID of the channel.

    Behavior:
        - Creates a new record in the `Context` table with `status` set to `parsing_channel`.
        - Logs a success message upon successful creation.

    Returns:
        None
    """
    context_imp = db_architecture.Context(
        channelId=channel_id,
        status=Status.parsing_channel,
        date=datetime.now(timezone.utc).replace(microsecond=0)
    )
    db_sessions.session.add(context_imp)
    db_sessions.session.commit()
    logging.info(f'Create channel context for {channel_id} successfully')


def finish_channel_context(channel_id: str):
    """
    Mark a channel's parsing context as finished.

    Args:
        channel_id (str): The unique ID of the channel.

    Behavior:
        - Updates the `Context` table record for the given channel ID, setting its `status` to `finish`.
        - Logs a success message upon successful update.

    Returns:
        None
    """
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.channelId == channel_id, Context.status != Status.finish))
    record = q.one()
    record.status = Status.finish
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Finish channel context for {channel_id} successfully')


def create_video_context(video_id: str):
    """
    Create a parsing context for a video.

    Args:
        video_id (str): The unique ID of the video.

    Behavior:
        - Creates a new record in the `Context` table with `status` set to `parsing_video`.
        - Logs a success message upon successful creation.

    Returns:
        None
    """
    context_imp = db_architecture.Context(
        videoId=video_id,
        status=Status.parsing_video,
        date=datetime.now(timezone.utc).replace(microsecond=0)
    )
    db_sessions.session.add(context_imp)
    db_sessions.session.commit()
    logging.info(f'Create video context for {video_id} successfully')


def finish_video_context(video_id: str):
    """
    Mark a video's parsing context as parsing comments.

    Args:
        video_id (str): The unique ID of the video.

    Behavior:
        - Updates the `Context` table record for the given video ID, setting its `status` to `parsing_comments`.
        - Logs a success message upon successful update.

    Returns:
        None
    """
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.videoId == video_id, Context.status != Status.finish))
    record = q.one()
    record.status = Status.parsing_comments
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Finish video context for {video_id} successfully')


def update_comments_context(video_id: str, comment_page_id: str):
    """
    Update the context of comments parsing for a video.

    Args:
        video_id (str): The unique ID of the video.
        comment_page_id (str): The ID of the comment page from which parsing should resume.

    Behavior:
        - Updates the `Context` table record for the given video ID with the new `commentPageId`.
        - Logs a success message with details of the updated page.

    Returns:
        None
    """
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.videoId == video_id, Context.status != Status.finish))
    record = q.one()
    record.commentPageId = comment_page_id
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Update comment context for {video_id} starts parsing from {comment_page_id} page')


def finish_comment_context(video_id: str):
    """
    Mark a video's comment parsing context as finished.

    Args:
        video_id (str): The unique ID of the video.

    Behavior:
        - Updates the `Context` table record for the given video ID, setting its `status` to `finish`.
        - Logs a success message upon successful update.

    Returns:
        None
    """
    q = db_sessions.session.query(Context)
    q = q.filter(and_(Context.videoId == video_id, Context.status != Status.finish))
    record = q.one()
    record.status = Status.finish
    record.date = datetime.now(timezone.utc).replace(microsecond=0)
    db_sessions.session.commit()
    logging.info(f'Finish comment context for {video_id} successfully')


def check_exists_video_by_id(video_id: str):
    """
    Check whether a video exists in the database by its ID.

    Args:
        video_id (str): The unique ID of the video.

    Returns:
        bool: True if the video exists, False otherwise.
    """
    exists_query = db_sessions.session.query(exists().where(db_architecture.Video.videoId == video_id)).scalar()

    if exists_query:
        return True
    else:
        return False


def check_exists_channel_by_id(channel_id: str):
    """
    Check whether a channel exists in the database by its ID.

    Args:
        channel_id (str): The unique ID of the channel.

    Returns:
        bool: True if the channel exists, False otherwise.
    """
    exists_query = db_sessions.session.query(exists().where(db_architecture.Channel.channelId == channel_id)).scalar()

    if exists_query:
        return True
    else:
        return False


def check_exists_comment_by_id(comment_id: str):
    """
    Check whether a comment exists in the database by its ID.

    Args:
        comment_id (str): The unique ID of the comment.

    Returns:
        bool: True if the comment exists, False otherwise.
    """
    exists_query = db_sessions.session.query(exists().where(db_architecture.Comment.commentId == comment_id)).scalar()

    if exists_query:
        return True
    else:
        return False


def check_is_comments_available(video_id: str):
    """
    Check if comments are available for a video.

    Args:
        video_id (str): The unique ID of the video.

    Returns:
        bool: True if comments exist and the count is greater than zero, False otherwise.
    """
    video_el = db_sessions.session.query(db_architecture.VideoStatsLast).filter(db_architecture.VideoStatsLast.videoId
                                                                                == video_id).first()
    if not video_el.commentCount:
        return False
    if video_el.commentCount > 0:
        return True
    return False


def get_comments_df(video_id: str):
    """
    Fetch comments for a given video as a pandas DataFrame.

    Args:
        video_id (str): The unique ID of the video.

    Returns:
        pd.DataFrame: A DataFrame containing comments for the specified video.
    """
    df = pd.read_sql(f"""SELECT * FROM comments WHERE comments."videoId" = '{video_id}'""", con=db_sessions.engine)
    return df


def video_published_time(video_id: str):
    """
    Get the published date of a video as a string.

    Args:
        video_id (str): The unique ID of the video.

    Returns:
        str: The published date of the video in "YYYY-MM-DD" format.
    """
    q = db_sessions.session.query(Video)
    q = q.filter(Video.videoId == video_id)
    record = q.one()
    return record.publishedAt.strftime("%Y-%m-%d")
