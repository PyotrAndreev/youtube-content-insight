import os

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger, Column, ForeignKey, Boolean, String, Time, Double, DateTime, ARRAY, DDL, Enum, create_engine
)
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base
from enum import Enum as PyEnum
from sqlalchemy.sql import text
Base = declarative_base()


class Source(PyEnum):
    relevance = 'relevance'
    raing = 'rating'
    date = 'date'
    query = 'query'


class Channel(Base):
    """
    Represents a YouTube channel with metadata including title, description, and various settings.

    Attributes:
        id (BigInteger): Unique identifier for the channel.
        title (str): The title of the YouTube channel.
        description (str): Detailed description of the channel's content.
        customUrl (str): Custom URL for the channel.
        publishedAt (DateTime): Date and time of the channel's publication.
        thumbnail (str): URL to the channel's thumbnail image.
        localizedTitle (str): Localized version of the channel's title.
        localizedDescription (str): Localized version of the channel's description.
        county (str): Country associated with the channel.
        relatedPlaylistsLikes (str): Playlist ID for videos liked by the channel.
        relatedPlaylistsUploads (str): Playlist ID for videos uploaded by the channel.
        viewCount (BigInteger): Total view count of the channel.
        subscribersCount (BigInteger): Total subscriber count of the channel.
        hiddenSubscriberCount (bool): Indicates if subscriber count is hidden.
        videoCount (BigInteger): Total number of videos on the channel.
        topicCategories (ARRAY of str): List of topic categories associated with the channel.
        privacyStatus (str): Privacy status of the channel.
        isLinked (bool): Indicates if the channel is linked to a Google account.
        longUploadsStatus (str): Status of long uploads for the channel.
        madeForKids (bool): Indicates if the channel is designated for kids.
        brandingSettingsChannelTitle (str): Custom title in branding settings.
        brandingSettingsChannelDescription (str): Custom description in branding settings.
        brandingSettingsChannelKeywords (str): Custom keywords in branding settings.
        brandingSettingsChannelUnsubscribedTrailer (str): Trailer URL shown to unsubscribed viewers.

    Relationships:
        videos (Video): List of `Video` objects related to the channel.
        authored_comments (Comment): List of `Comment` objects authored by this channel.
        received_comments (Comment): List of `Comment` objects directed to this channel.
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
    Represents a video on a YouTube channel, with metadata about publication, content, and categorization.

    Attributes:
        id (BigInteger): Unique identifier for the video.
        publishedAt (DateTime): Publication timestamp for the video.
        channelId (BigInteger): Foreign key reference to the associated channel's ID.
        title (str): Title of the video.
        description (str): Detailed description of the video's content.
        thumbnail (str): URL to the video's thumbnail image.
        channelTitle (str): Name of the channel that uploaded the video.
        tags (ARRAY of str): Tags associated with the video.
        liveBroadcastContent (str): Indicates if the video is a live broadcast.
        defaultLanguage (str): Default language of the video's content.
        defaultAudioLanguage (str): Default audio language of the video's content.
        categoryId (str): ID of the video's category.
        duration (str): Duration of the video in ISO 8601 format.
        dimension (str): Dimension of the video (e.g., 2D, 3D).
        definition (str): Quality definition (e.g., HD, SD).
        caption (str): Indicates if captions are available.
        licensedContent (bool): Indicates if the video is licensed content.
        uploadStatus (str): Upload status of the video.
        privacyStatus (str): Privacy status of the video.
        license (str): License type for the video.
        embeddable (bool): Indicates if the video is embeddable on other sites.
        publicStatsViewable (bool): Indicates if public statistics are viewable.
        madeForKids (bool): Indicates if the video is designated for kids.
        viewsCount (BigInteger): View count of the video.
        likesCount (BigInteger): Like count for the video.
        favoriteCount (BigInteger): Favorite count for the video.
        comment_count (BigInteger): Comment count for the video.

    Relationships:
        channel (Channel): The `Channel` object associated with the video.
        subtitles (Subtitle): List of `Subtitle` objects associated with the video.
        comments (Comment): List of `Comment` objects associated with the video.
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
    Represents subtitles linked to a YouTube video, containing textual content and timing information.

    Attributes:
        id (BigInteger): Unique identifier for the subtitle.
        videoId (BigInteger): Foreign key reference to the associated video's ID.
        text (str): Subtitle text.
        start (Double): Start time of the subtitle in seconds.
        duration (Double): Duration of the subtitle in seconds.

    Relationships:
        video (Video): The `Video` object associated with the subtitle.
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
       Represents a comment on a video, including author information and comment metadata.

       Attributes:
           id (BigInteger): Primary key identifier for the comment.
           videoId (BigInteger): Foreign key reference to the video's ID the comment is associated with.
           authorDisplayName (str): Display name of the comment's author.
           authorChannelUrl (str): URL of the author's channel.
           authorChannelId (BigInteger): Foreign key reference to the author's channel ID.
           channelId (BigInteger): Foreign key reference to the channel receiving the comment.
           textDisplay (str): Display text of the comment.
           textOriginal (str): Original text of the comment.
           parentId (BigInteger): Foreign key reference to the parent comment ID, if it is a reply.
           canRate (bool): Indicates if the comment can be rated.
           viewerRating (str): The viewer's rating on the comment.
           likeCount (BigInteger): Number of likes on the comment.
           moderationStatus (str): Moderation status of the comment.
           publishedAt (Time): Timestamp for when the comment was published.
           updatedAt (Time): Timestamp for when the comment was last updated.

       Relationships:
           video (Video): Relationship with the `Video` class.
           author_channel (Channel): Relationship with the channel authoring the comment.
           target_channel (Channel): Relationship with the channel receiving the comment.
           parent_comment (Comment): Self-referential relationship for nested comments.
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
