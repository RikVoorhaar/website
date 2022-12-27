import asyncio
# import json
import os
from pathlib import Path
from requests import HTTPError

import aiohttp
import dotenv
import yaml
from tqdm import tqdm

dotenv.load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
YT_CHANNELS_FILE = "_data/yt-channels.yml"
OUTPUT_FILE = "_data/channels.yaml"
IMAGE_FOLDER = Path("assets/images/yt_thumbnails")
IMAGE_FOLDER.mkdir(exist_ok=True)


async def get_channel_data(handle):
    youtube_api_url = "https://youtube.googleapis.com/youtube/v3/"
    direct = f"{youtube_api_url}channels?part=snippet&forUsername={handle}&key={API_KEY}"
    search_based = f"{youtube_api_url}search?part=snippet&maxResults=1&q={handle}&type=channel&key={API_KEY}"

    async with aiohttp.ClientSession() as session:
        # async with session.get(direct) as resp:
        #     resp_data = await resp.json()
        # if resp.status != 200 or resp_data["pageInfo"]["totalResults"] == 0:
        async with session.get(search_based) as resp:
            resp_data = await resp.json()
            resp_kind = "search"
        if resp.status != 200:
            raise HTTPError("Error while fetching data from YouTube API")
        # else:
        #     resp_kind = "direct"

        snippet = resp_data["items"][0]["snippet"]

        channel_title = snippet["title"]
        channel_thumbnail_url = snippet["thumbnails"]["default"]["url"]
        channel_url = f"https://www.youtube.com/@{handle}"

        if resp_kind == "search":
            channel_id = resp_data["items"][0]["id"]["channelId"]
        else:
            channel_id = resp_data["items"][0]["id"]

        async with session.get(channel_thumbnail_url) as response:
            image_filename = f"{handle}.jpg"
            IMAGE_FOLDER.mkdir(exist_ok=True)
            image_path = IMAGE_FOLDER / image_filename
            with open(image_path, "wb") as f:
                f.write(await response.read())

    channel_data = {
        "channel_handle": handle,
        "channel_title": channel_title,
        "channel_url": channel_url,
        "channel_thumbnail_url": channel_thumbnail_url,
        "channel_thumbnail_path": str(image_path),
        "channel_id": channel_id,
    }

    return channel_data


async def main():

    with open(YT_CHANNELS_FILE, "r") as f:
        channels = yaml.load(f, Loader=yaml.FullLoader)

    channels_dict = {}
    tasks = []
    for channel_handle in channels:
        task = asyncio.create_task(get_channel_data(channel_handle))
        tasks.append(task)

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        channel_data = await task
        channel_handle = channel_data.pop("channel_handle")
        channels_dict[channel_handle] = channel_data

    with open(OUTPUT_FILE, "w") as f:
        yaml.dump({"channels": list(channels_dict.values())}, f)
        # json.dump(channels_dict, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
