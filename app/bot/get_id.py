import os

import requests
import re

api_key = os.getenv("API_KEY")

def get_channel_id(channel_url):
  match = re.search(r'https?://youtube.com/@([a-zA-Z0-9_-]+)', channel_url)
  if not match:
    print("Некорректный формат URL.")
    return None
  channel_name = match.group(1)

  api_url = f"https://www.googleapis.com/youtube/v3/channels?forUsername={channel_name}&key={api_key}"

  response = requests.get(api_url)

  if response.status_code == 200:
      data = response.json()
      if 'items' in data and len(data['items']) > 0:
        return data['items'][0]['id']
      else:
        return 1
  else:
    print(f"Ошибка запроса: {response.status_code}")
    return 2