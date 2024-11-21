import os

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger, Column, ForeignKey, Boolean, String, Time, Double, DateTime, ARRAY, DDL, Enum, create_engine
)
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base
from enum import Enum as PyEnum
Base = declarative_base()


class Source(PyEnum):
    """
    Enum class representing the source of data extraction.

    Attributes:
        relevance (str): Indicates data extracted by relevance.
        rating (str): Indicates data extracted based on ratings.
        date (str): Indicates data extracted by date.
        query (str): Indicates data extracted based on a specific query.
    """
    relevance = 'relevance'
    raing = 'rating'
    date = 'date'
    query = 'query'


class Status(PyEnum):
    """
    Enum class representing the status of data parsing.

    Attributes:
        wait (str): Indicates the data is in the queue waiting for processing.
        parsing_channel (str): Indicates the channel metadata is being parsed.
        parsing_video (str): Indicates the video metadata is being parsed.
        parsing_comments (str): Indicates the comments are being parsed.
        finish (str): Indicates the parsing process is complete.
    """
    wait = 'wait'
    parsing_channel = 'parsing_channel'
    parsing_video = 'parsing_video'
    parsing_comments = 'parsing_comment'
    finish = 'finish'


class Channel(Base):
    """
    Represents a YouTube channel with associated metadata.

    Attributes:
        id (int): Unique identifier for the channel.
        channelId (str): YouTube channel ID.
        title (str): Title of the channel.
        description (str): Description of the channel content.
        customUrl (str): Custom URL associated with the channel.
        publishedAt (datetime): Timestamp of the channel's creation.
        thumbnail (str): URL to the channel's thumbnail image.
        localizedTitle (str): Localized version of the channel title.
        localizedDescription (str): Localized version of the channel description.
        country (str): Country of origin for the channel.
        relatedPlaylistsLikes (str): Playlist ID for liked videos.
        relatedPlaylistsUploads (str): Playlist ID for uploaded videos.
        privacyStatus (str): Privacy status of the channel.
        isLinked (bool): Indicates if the channel is linked to a Google account.
        longUploadsStatus (str): Status of long video uploads for the channel.
        madeForKids (bool): Specifies if the channel is made for kids.
        brandingSettingsChannelTitle (str): Title from branding settings.
        brandingSettingsChannelDescription (str): Description from branding settings.
        brandingSettingsChannelKeywords (str): Keywords from branding settings.
        brandingSettingsChannelUnsubscribedTrailer (str): Trailer for unsubscribed viewers.

    Relationships:
        videos (relationship): List of videos associated with the channel.
        received_comments (relationship): Comments directed to this channel.
        channel_stats_last (relationship): Latest statistics for the channel.
        channel_stats_hist (relationship): Historical statistics for the channel.
    """

    __tablename__ = 'channels'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    channelId = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    customUrl = Column(String, nullable=True)
    publishedAt = Column(DateTime, nullable=True)
    thumbnail = Column(String, nullable=True)
    localizedTitle = Column(String, nullable=True)
    localizedDescription = Column(String, nullable=True)
    country = Column(String, nullable=True)
    relatedPlaylistsLikes = Column(String, nullable=True)
    relatedPlaylistsUploads = Column(String, nullable=True)
    privacyStatus = Column(String, nullable=True)
    isLinked = Column(Boolean, nullable=True)
    longUploadsStatus = Column(String, nullable=True)
    madeForKids = Column(Boolean, nullable=True)
    brandingSettingsChannelTitle = Column(String, nullable=True)
    brandingSettingsChannelDescription = Column(String, nullable=True)
    brandingSettingsChannelKeywords = Column(String, nullable=True)
    brandingSettingsChannelUnsubscribedTrailer = Column(String, nullable=True)

    # Relationships
    videos = relationship('Video', back_populates='channel')
    # authored_comments = relationship('Comment', back_populates='author_channel', foreign_keys='Comment.authorChannelId')
    received_comments = relationship('Comment', back_populates='target_channel', foreign_keys='Comment.channelId')
    channel_stats_last = relationship('ChannelStatsLast', back_populates='channel')
    channel_stats_hist = relationship('ChannelStatsHist', back_populates='channel')

    def __repr__(self):
        return (f"<Channel(id={self.id}, title='{self.title}', description='{self.description}', "
                f"publishedAt='{self.publishedAt}'>")


class ChannelStatsLast(Base):
    """
    Represents the latest statistics for a YouTube channel.

    Attributes:
        id (int): Unique identifier for the record.
        channelId (str): Foreign key referencing the channel ID.
        viewCount (int): Total view count of the channel.
        subscribersCount (int): Total subscriber count of the channel.
        hiddenSubscriberCount (bool): Indicates if the subscriber count is hidden.
        videoCount (int): Total number of videos uploaded to the channel.
        topicCategories (list): List of topic categories for the channel.
        parsingDate (datetime): Date when the data was parsed.

    Relationships:
        channel (relationship): Reference to the associated channel.
    """

    __tablename__ = 'channels_stats_last'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    channelId = Column(String, ForeignKey('channels.channelId'), nullable=False)
    viewCount = Column(BigInteger, nullable=True)
    subscribersCount = Column(BigInteger, nullable=True)
    hiddenSubscriberCount = Column(Boolean, nullable=True)
    videoCount = Column(BigInteger, nullable=True)
    topicCategories = Column(ARRAY(String), nullable=True)
    parsingDate = Column(DateTime, nullable=False)

    channel = relationship('Channel', back_populates='channel_stats_last')


class ChannelStatsHist(Base):
    """
    Represents historical statistics for a YouTube channel.

    Attributes:
        id (int): Unique identifier for the record.
        channelId (str): Foreign key referencing the channel ID.
        viewCount (int): Total view count of the channel at the time of parsing.
        subscribersCount (int): Total subscriber count at the time of parsing.
        hiddenSubscriberCount (bool): Indicates if the subscriber count was hidden.
        videoCount (int): Total number of videos uploaded at the time of parsing.
        topicCategories (list): List of topic categories at the time of parsing.
        parsingDate (datetime): Date when the data was parsed.

    Relationships:
        channel (relationship): Reference to the associated channel.
    """

    __tablename__ = 'channels_stats_hist'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    channelId = Column(String, ForeignKey('channels.channelId'))
    viewCount = Column(BigInteger, nullable=True)
    subscribersCount = Column(BigInteger, nullable=True)
    hiddenSubscriberCount = Column(Boolean, nullable=True)
    videoCount = Column(BigInteger, nullable=True)
    topicCategories = Column(ARRAY(String), nullable=True)
    parsingDate = Column(DateTime, nullable=False)

    channel = relationship('Channel', back_populates='channel_stats_hist')


class Video(Base):
    """
    Represents a YouTube video with associated metadata.

    Attributes:
        id (int): Unique identifier for the video.
        videoId (str): YouTube video ID.
        publishedAt (datetime): Timestamp of the video's publication.
        channelId (str): Foreign key referencing the associated channel ID.
        title (str): Title of the video.
        description (str): Description of the video's content.
        thumbnail (str): URL to the video's thumbnail image.
        channelTitle (str): Title of the channel that uploaded the video.
        tags (list): List of tags associated with the video.
        defaultLanguage (str): Default language of the video.
        defaultAudioLanguage (str): Default audio language of the video.
        categoryId (str): Category ID of the video.
        duration (str): Duration of the video in ISO 8601 format.
        dimension (str): Video dimensions (e.g., 2D, 3D).
        definition (str): Quality of the video (e.g., HD, SD).
        caption (str): Indicates if captions are available.
        licensedContent (bool): Indicates if the video is licensed.
        uploadStatus (str): Upload status of the video.
        privacyStatus (str): Privacy status of the video.
        license (str): License type of the video.
        embeddable (bool): Specifies if the video can be embedded.
        publicStatsViewable (bool): Indicates if public statistics are viewable.
        madeForKids (bool): Specifies if the video is made for kids.

    Relationships:
        channel (relationship): Reference to the associated channel.
        subtitles (relationship): Subtitles linked to the video.
        comments (relationship): Comments on the video.
        video_stats_last (relationship): Latest statistics for the video.
        video_stats_hist (relationship): Historical statistics for the video.
    """

    __tablename__ = 'videos'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    videoId = Column(String, nullable=False, unique=True)
    publishedAt = Column(DateTime, nullable=True)
    channelId = Column(String, ForeignKey('channels.channelId'), nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    thumbnail = Column(String, nullable=True)
    channelTitle = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    defaultLanguage = Column(String, nullable=True)
    defaultAudioLanguage = Column(String, nullable=True)
    categoryId = Column(String, nullable=True)
    duration = Column(String, nullable=True)
    dimension = Column(String, nullable=True)
    definition = Column(String, nullable=True)
    caption = Column(String, nullable=True)
    licensedContent = Column(Boolean, nullable=True)
    uploadStatus = Column(String, nullable=True)
    privacyStatus = Column(String, nullable=True)
    license = Column(String, nullable=True)
    embeddable = Column(Boolean, nullable=True)
    publicStatsViewable = Column(Boolean, nullable=True)
    madeForKids = Column(Boolean, nullable=True)

    # Relationships
    channel = relationship('Channel', back_populates='videos')
    subtitles = relationship('Subtitle', back_populates='video')
    comments = relationship('Comment', back_populates='video')
    video_stats_last = relationship('VideoStatsLast', back_populates='video')
    video_stats_hist = relationship('VideoStatsHist', back_populates='video')

    def __repr__(self):
        return (f"<Video(id={self.id}, publishedAt='{self.publishedAt}', channelId={self.channelId}, "
                f"title='{self.title}', description='{self.description}', channelTitle='{self.channelTitle}', "
                f"tags={self.tags}, categoryId={self.categoryId}, defaultLanguage='{self.defaultLanguage}', "
                f"duration='{self.duration}')>")


class VideoStatsLast(Base):
    """
    Represents the latest statistics for a YouTube video.

    Attributes:
        id (int): Unique identifier for the record.
        videoId (str): Foreign key referencing the video ID.
        liveBroadcastContent (str): Indicates if the video is a live broadcast.
        viewsCount (int): Total view count of the video.
        likesCount (int): Total like count of the video.
        likesFromApi (int): Likes retrieved from the API.
        dislikesFromApi (int): Dislikes retrieved from the API.
        ratingFromApi (float): Rating retrieved from the API.
        favoriteCount (int): Total favorite count of the video.
        commentCount (int): Total number of comments on the video.
        parsingDate (datetime): Date when the data was parsed.

    Relationships:
        video (relationship): Reference to the associated video.
    """

    __tablename__ = 'videos_stats_last'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    videoId = Column(String, ForeignKey('videos.videoId'), nullable=False, unique=True)
    liveBroadcastContent = Column(String, nullable=True)
    viewsCount = Column(BigInteger, nullable=True)
    likesCount = Column(BigInteger, nullable=True)
    likesFromApi = Column(BigInteger, nullable=True)
    dislikesFromApi = Column(BigInteger, nullable=True)
    ratingFromApi = Column(Double, nullable=True)
    favoriteCount = Column(BigInteger, nullable=True)
    commentCount = Column(BigInteger, nullable=True)
    parsingDate = Column(DateTime, nullable=False)

    video = relationship('Video', back_populates='video_stats_last')


class VideoStatsHist(Base):
    """
    Represents historical statistics for a YouTube video.

    Attributes:
        id (int): Unique identifier for the record.
        videoId (str): Foreign key referencing the video ID.
        liveBroadcastContent (str): Indicates if the video was a live broadcast.
        viewsCount (int): Total view count at the time of parsing.
        likesCount (int): Total like count at the time of parsing.
        likesFromApi (int): Likes retrieved from the API at the time of parsing.
        dislikesFromApi (int): Dislikes retrieved from the API at the time of parsing.
        ratingFromApi (float): Rating retrieved from the API at the time of parsing.
        favoriteCount (int): Total favorite count at the time of parsing.
        commentCount (int): Total comment count at the time of parsing.
        parsingDate (datetime): Date when the data was parsed.

    Relationships:
        video (relationship): Reference to the associated video.
    """

    __tablename__ = 'videos_stats_hist'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    videoId = Column(String, ForeignKey('videos.videoId'), nullable=False)
    liveBroadcastContent = Column(String, nullable=True)
    viewsCount = Column(BigInteger, nullable=True)
    likesCount = Column(BigInteger, nullable=True)
    likesFromApi = Column(BigInteger, nullable=True)
    dislikesFromApi = Column(BigInteger, nullable=True)
    ratingFromApi = Column(Double, nullable=True)
    favoriteCount = Column(BigInteger, nullable=True)
    commentCount = Column(BigInteger, nullable=True)
    parsingDate = Column(DateTime, nullable=False)

    video = relationship('Video', back_populates='video_stats_hist')


class Subtitle(Base):
    """
    Represents a subtitle associated with a YouTube video.

    Attributes:
        id (int): Unique identifier for the subtitle.
        videoId (str): Foreign key referencing the video ID.
        text (str): Text content of the subtitle.
        start (float): Start time of the subtitle in seconds.
        duration (float): Duration of the subtitle in seconds.

    Relationships:
        video (relationship): Reference to the associated video.
    """

    __tablename__ = 'subtitles'

    id = Column(BigInteger, primary_key=True)
    videoId = Column(String, ForeignKey('videos.videoId'), nullable=False)
    text = Column(String)
    start = Column(Double)
    duration = Column(Double)

    # Relationships
    video = relationship('Video', back_populates='subtitles')

    def __repr__(self):
        return f"<Subtitle(id={self.id}, videoId={self.videoId}, text='{self.text}')>"


class Comment(Base):
    """
    Represents a comment on a YouTube video.

    Attributes:
        id (int): Unique identifier for the comment.
        commentId (str): YouTube comment ID.
        videoId (str): Foreign key referencing the video ID.
        authorDisplayName (str): Display name of the comment's author.
        authorProfileImageUrl (str): URL to the author's profile image.
        authorChannelUrl (str): URL to the author's YouTube channel.
        authorChannelId (str): ID of the author's YouTube channel.
        channelId (str): Foreign key referencing the channel receiving the comment.
        textDisplay (str): Displayed text of the comment.
        textOriginal (str): Original text of the comment.
        parentId (str): Foreign key referencing the parent comment ID (if it is a reply).
        canRate (bool): Indicates if the comment can be rated.
        viewerRating (str): Viewer’s rating of the comment.
        likeCount (int): Total number of likes on the comment.
        moderationStatus (str): Moderation status of the comment.
        publishedAt (datetime): Timestamp of the comment's publication.
        updatedAt (datetime): Timestamp of the comment's last update.
        gotFrom (Source): Enum value indicating the source of the comment.

    Relationships:
        video (relationship): Reference to the associated video.
        target_channel (relationship): Reference to the channel receiving the comment.
        parent_comment (relationship): Reference to the parent comment (if applicable).
    """

    __tablename__ = 'comments'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    commentId = Column(String, nullable=False, unique=True)
    videoId = Column(String, ForeignKey('videos.videoId'), nullable=False)
    authorDisplayName = Column(String)
    authorProfileImageUrl = Column(String)
    authorChannelUrl = Column(String)
    authorChannelId = Column(String)
    channelId = Column(String, ForeignKey('channels.channelId'))
    textDisplay = Column(String)
    textOriginal = Column(String)
    parentId = Column(String, ForeignKey('comments.commentId'), nullable=True)
    canRate = Column(Boolean)
    viewerRating = Column(String)
    likeCount = Column(BigInteger)
    publishedAt = Column(DateTime)
    updatedAt = Column(DateTime)
    gotFrom = Column(Enum(Source))

    # Relationships
    video = relationship('Video', back_populates='comments')
    # author_channel = relationship('Channel', back_populates='authored_comments', foreign_keys=[authorChannelId])
    target_channel = relationship('Channel', back_populates='received_comments', foreign_keys=[channelId])
    parent_comment = relationship('Comment', remote_side=[commentId], uselist=False)

    def __repr__(self):
        return (f"<Comment(id={self.id}, videoId={self.videoId}, authorDisplayName='{self.authorDisplayName}', "
                f"authorChannelUrl='{self.authorChannelUrl}', authorChannelId={self.authorChannelId}, "
                f"channelId={self.channelId}, textDisplay='{self.textDisplay}', textOriginal='{self.textOriginal}', "
                f"parentId={self.parentId}, canRate={self.canRate}, viewerRating='{self.viewerRating}', "
                f"likeCount={self.likeCount}', publishedAt='{self.publishedAt}', "
                f"updatedAt='{self.updatedAt}')>")


class Context(Base):
    """
    Represents the parsing context of a YouTube operation.

    Attributes:
        id (int): Unique identifier for the context.
        channelId (str): Channel ID.
        videoId (str): Video ID.
        commentPageId (str): Comment Page ID.
        current_status (Status): Current status of the parsing operation.
        parsingDate (datetime): Date when the context was logged.
     """

    __tablename__ = 'context'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    channelId = Column(String, nullable=False)
    videoId = Column(String, nullable=True)
    commentPageId = Column(String, nullable=True)
    status = Column(String, nullable=True)
    date = Column(Enum(Status))


load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("DB_NAME")


engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{DB_NAME}')

Base.metadata.create_all(engine)

with engine.connect() as connection:
    trigger_for_channel_info = DDL("""CREATE OR REPLACE FUNCTION channels_stats_last_to_hist()
                RETURNS TRIGGER AS $$
                BEGIN
                    INSERT INTO channels_stats_hist ("videoCount", "parsingDate", "channelId", "topicCategories", "viewCount", "subscribersCount", "hiddenSubscriberCount")
                    VALUES (OLD."videoCount", OLD."parsingDate", OLD."channelId", OLD."topicCategories", OLD."viewCount", OLD."subscribersCount", OLD."hiddenSubscriberCount");
                    RETURN NEW;
                END;
                $$ LANGUAGE 'plpgsql';
                
                CREATE TRIGGER channels_stats_last_to_hist_table
                BEFORE UPDATE ON channels_stats_last
                FOR EACH ROW
                EXECUTE FUNCTION channels_stats_last_to_hist();""")
    trigger_for_video_info = DDL("""CREATE OR REPLACE FUNCTION videos_stats_last_to_hist()
                RETURNS TRIGGER AS $$
                BEGIN
                    INSERT INTO videos_stats_hist ("favoriteCount", "dislikesFromApi", "videoId", "parsingDate", "likesCount", "liveBroadcastContent", "viewsCount", "commentCount", "likesFromApi", "ratingFromApi")
                    VALUES (OLD."favoriteCount", OLD."dislikesFromApi", OLD."videoId", OLD."parsingDate", OLD."likesCount", OLD."liveBroadcastContent", OLD."viewsCount", OLD."commentCount", OLD."likesFromApi", OLD."ratingFromApi");
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                CREATE TRIGGER videos_stats_last_to_hist_table
                BEFORE UPDATE ON videos_stats_last
                FOR EACH ROW
                EXECUTE FUNCTION videos_stats_last_to_hist();""")
    connection.execute(trigger_for_channel_info)
    connection.execute(trigger_for_video_info)
