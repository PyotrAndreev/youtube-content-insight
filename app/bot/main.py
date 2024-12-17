import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from dotenv import load_dotenv

from app.bot.create_scenario import generate_scenario
import requests
import re
import os
from vizualization.dash_vizualize import get_tags_list
from app.handlers.request_handlers import get_video_analytics
from app.handlers.request_handlers import get_videos_analytics
from app.handlers.request_handlers import comment_clustering

load_dotenv()
logging.basicConfig(level=logging.INFO)
bot = Bot(token=os.getenv("TG_BOT_API_KEY"))
dp = Dispatcher()
API_KEY = os.getenv("API_KEY")


class Form(StatesGroup):
    waiting_for_topic = State()


class GetChannelLink(StatesGroup):
    waiting_for_channel = State()


class GetVideoLink(StatesGroup):
    waiting_for_video = State()


class GetTopicId(StatesGroup):
    waiting_for_id = State()


class GetCategoryId(StatesGroup):
    waiting = State()


class GetVideosList(StatesGroup):
    waiting_for_videos_list = State()


def is_valid_youtube_link(url):
    pattern = r"https://www\.youtube\.com/@[\w\d_-]+"
    if re.match(pattern, url):
        return True
    else:
        return False


def is_valid_video_link(url):
    pattern = r'^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|embed\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})(&.*)?$'
    if re.match(pattern, url):
        return True
    else:
        return False


def get_latest_videos(api_key, id):
    logging.info(f"Start parsing last videos for {id} category")
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'id',
        'maxResults': 5,
        'order': 'rating',
        'videoCategoryId': id,
        'type': 'video',
        'key': api_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        videos = response.json().get('items', [])
        video_ids = [video['id']['videoId'] for video in videos]
        return video_ids
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


@dp.message(Command("start"))
async def send_welcome(message: types.Message, state: FSMContext):
    logging.info("Printing start menu")
    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await state.set_state(None)

    await message.reply("Привет!\nЯ - бот-анализатор контента на Ютубе! \nВыберите, что вам интересно узнать",
                        reply_markup=keyboard)


# @dp.message(lambda message: message.text == "Аналитика канала")
# async def analyze_video(message: types.Message, state: FSMContext):
#     kb = [
#         [
#             types.KeyboardButton(text="Назад"),
#         ],
#     ]
#     keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
#     await message.answer("Вы выбрали анализ канала. Пожалуйста, отправьте ссылку на канал.", reply_markup=keyboard)
#     await state.set_state(GetChannelLink.waiting_for_channel)

# @dp.message(GetChannelLink.waiting_for_channel)
# async def process_topic(message: types.Message, state: FSMContext):
#     print("im here")
#     link = message.text
#     print(link)
#     if is_valid_youtube_link(link):
#         await message.answer("Ссылка получена")
#         chan_id = get_channel_id(link)
#         info = get_channel_info_by_id(chan_id)
#         items = [f"{key}: {value}" for key, value in info.items()]
#         res = '\n'.join(items)
#         await message.answer(res)
#         await state.set_state(None)
#
#     else:
#         await message.answer("Введена неверная ссылка, повторите попытку")
#         await state.set_state(GetChannelLink.waiting_for_channel)


@dp.message(lambda message: message.text == "Назад")
async def back(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.answer("Возвращаемся в главное меню", reply_markup=keyboard)
    await state.set_state(None)


@dp.message(lambda message: message.text == "Аналитика видео")
async def channel_statistics(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Назад"),
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.answer("Вы выбрали анализ видео. Пожалуйста, отправьте ссылку на видео.", reply_markup=keyboard)
    await state.set_state(GetVideoLink.waiting_for_video)


@dp.message(GetVideoLink.waiting_for_video)
async def process_topic(message: types.Message, state: FSMContext):
    link = message.text
    if is_valid_video_link(link):
        await message.answer("Ссылка получена")
        await message.answer("Скоро вы увидите график эмоциональной окраски комментариев для этого видео")
        get_video_analytics(link)
        kb = [
            [
                types.KeyboardButton(text="Аналитика видео"),
                types.KeyboardButton(text="Динамика видео"),
                types.KeyboardButton(text="Популярные теги"),
                types.KeyboardButton(text="Популярные видео"),
                types.KeyboardButton(text="Сценарий для видео")
            ],
        ]
        keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
        photo_path = "app/content/in_video.png"
        await message.answer_photo(
            types.FSInputFile(path=photo_path), caption=f"Аналитика комментариев для видео {link}"
        )
        await message.answer("А теперь на очереди кластеризация комментариев данного видео")
        result = comment_clustering(link)
        for i in range(len(result)):
            if i == len(result) - 1:
                await message.answer(result[i], reply_markup=keyboard)
            else:
                await message.answer(result[i])
        await state.set_state(None)
    else:
        await message.answer("Введена неверная ссылка, повторите попытку")
        await state.set_state(GetVideoLink.waiting_for_video)


@dp.message(lambda message: message.text == "Динамика видео")
async def channel_statistics(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Назад"),
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.answer("Вы выбрали обзор динамики видео. Пожалуйста, отправьте список из ссылок на видео через "
                         "пробел, которые хотите отследить.", reply_markup=keyboard)
    await state.set_state(GetVideosList.waiting_for_videos_list)


@dp.message(GetVideosList.waiting_for_videos_list)
async def process_topic(message: types.Message, state: FSMContext):
    videos = message.text.split()
    res = []
    cnt = 0
    for link in videos:
        cnt += 1
        if not is_valid_video_link(link):
            await message.answer("Введена неверная ссылка под номером " + str(cnt) + ", повторите попытку")
            await state.set_state(GetVideoLink.waiting_for_video)
            return
        else:
            res.append(link)
    await message.answer("Ссылки на видео получены")
    await message.answer("Скоро вы увидите график эмоциональной окраски комментариев для этих видео")
    get_videos_analytics(res)
    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    photo_path = "app/content/video_to_video.png"
    await message.answer_photo(
        types.FSInputFile(path=photo_path), reply_markup=keyboard
    )
    await state.set_state(None)


@dp.message(lambda message: message.text == "Популярные теги")
async def channel_statistics(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Назад"),
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.answer("Вы выбрали список популярных тегов. Введите ID категории, теги по которой вам интересны",
                         reply_markup=keyboard)
    await state.set_state(GetCategoryId.waiting)


@dp.message(GetCategoryId.waiting)
async def process_topic(message: types.Message, state: FSMContext):
    topic_id = message.text
    tags = get_tags_list(topic_id)
    await message.answer("Стартовал поиск самых популярных тегов для выбранной категории")
    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    res = ""
    for tag in tags:
        res += (tag[0] + "\n")
    await message.answer("Вот список самых популярных тегов:" + "\n" + res, reply_markup=keyboard)
    await state.set_state(None)


@dp.message(GetTopicId.waiting_for_id)
async def process_topic(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    topic_id = message.text
    await message.answer("Стартовал поиск самых популярных видео из категории")
    videos = get_latest_videos(API_KEY, topic_id)
    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    base_url = "https://www.youtube.com/watch?v="
    for i in range(0, len(videos)):
        if i == len(videos) - 1:
            await message.answer(f"{base_url}{videos[i]}", reply_markup=keyboard)
        else:
            await message.answer(f"{base_url}{videos[i]}")

    await state.set_state(None)


@dp.message(lambda message: message.text == "Популярные видео")
async def channel_statistics(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Назад"),
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.answer("Вы выбрали список популярных видео. Введите ID категории, видео по которой вам "
                         "интересны", reply_markup=keyboard)
    await state.set_state(GetTopicId.waiting_for_id)


@dp.message(lambda message: message.text == "Сценарий для видео")
async def channel_statistics(message: types.Message, state: FSMContext):
    kb = [
        [
            types.KeyboardButton(text="Назад"),
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.answer("Вы выбрали написание сценария для видео. Введите тему для видео", reply_markup=keyboard)
    await state.set_state(Form.waiting_for_topic)


@dp.message(Form.waiting_for_topic)
async def process_topic(message: types.Message, state: FSMContext):
    topic = message.text
    await message.answer("Началась генерация сценария для видео")

    scenario = generate_scenario(topic)

    kb = [
        [
            types.KeyboardButton(text="Аналитика видео"),
            types.KeyboardButton(text="Динамика видео"),
            types.KeyboardButton(text="Популярные теги"),
            types.KeyboardButton(text="Популярные видео"),
            types.KeyboardButton(text="Сценарий для видео")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)

    await message.answer(scenario, reply_markup=keyboard)
    await state.set_state(None)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())