"""
Fetches data from the YouTube API and saves it to a YAML file.

To use set a valid YouTube API key in the environment variable YOUTUBE_API_KEY. Pip
requirements can be found in ../requirements.txt

This loads a list of channels from `jekyll/_data/yt-channels.yml`, processes it, and
outputs the result in `jekyll/_data/channels.yml`, while storing the thumbnail images in
`jekyll/assets/images/yt_thumbnails`.

Channels that had been processed before are skipped. This is because each channel search
uses up a significant number of api calls, and only around 100 channels can be processed
for free in one day.
"""
import asyncio
import os
from pathlib import Path
from typing import Any

import aiohttp
import dotenv
import yaml
from tqdm import tqdm

dotenv.load_dotenv()


class YoutubeAPIError(Exception):
    def __init__(
        self, message: str, status: int, data: Any, url: str, handle: str
    ) -> None:
        super().__init__(message)
        self.status = status
        self.data = data
        self.url = url
        self.handle = handle


def get_git_root() -> Path:
    """Get the root directory of the git repo."""
    path = Path.cwd()
    while path != path.parent and not (path / ".git").is_dir():
        path = path.parent

    if path == path.parent:
        raise RuntimeError("Not in a git repository")
    return path


root = get_git_root()
JEKYLL_FOLDER = root / "jekyll"

API_KEY = os.getenv("YOUTUBE_API_KEY", None)

YT_CHANNELS_FILE = str(JEKYLL_FOLDER / "_data/yt-channels.yml")
OUTPUT_FILE = str(JEKYLL_FOLDER / "_data/channels.yaml")
IMAGE_FOLDER_RELATIVE = Path("assets/images/yt_thumbnails")
(JEKYLL_FOLDER / IMAGE_FOLDER_RELATIVE).mkdir(exist_ok=True)


async def get_channel_data(handle):
    youtube_api_url = "https://youtube.googleapis.com/youtube/v3/"
    api_request = f"{youtube_api_url}search?part=snippet&maxResults=1&q={handle}&type=channel&key={API_KEY}"

    async with aiohttp.ClientSession() as session:
        async with session.get(api_request) as resp:
            resp_data = await resp.json()
            resp_kind = "search"
        if resp.status != 200:
            raise YoutubeAPIError(
                "Error while fetching data from YouTube API",
                resp.status,
                resp_data,
                api_request,
                handle,
            )

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
            image_path_relative = IMAGE_FOLDER_RELATIVE / image_filename
            image_path_absolute = JEKYLL_FOLDER / image_path_relative
            with open(image_path_absolute, "wb") as f:
                f.write(await response.read())

    channel_data = {
        "channel_handle": handle,
        "channel_title": channel_title,
        "channel_url": channel_url,
        "channel_thumbnail_url": channel_thumbnail_url,
        "channel_thumbnail_path": str(image_path_relative),
        "channel_id": channel_id,
    }

    return channel_data


async def main():
    with open(YT_CHANNELS_FILE, "r") as f:
        channels = yaml.load(f, Loader=yaml.FullLoader)

    try:
        with open(OUTPUT_FILE, "r") as f:
            channels_yaml = yaml.load(f, Loader=yaml.FullLoader)["channels"]
    except FileNotFoundError:
        channels_yaml = []

    already_processed = {channel["channel_handle"] for channel in channels_yaml}
    not_processed = list(set(channels) - already_processed)
    to_be_removed = set(already_processed) - set(channels)
    total_num = len(already_processed) + len(not_processed)

    channel_data_dict = {}
    for channel_data in channels_yaml:
        handle = channel_data["channel_handle"]
        if handle in to_be_removed:
            continue
        channel_data_dict[handle] = channel_data

    print(f"Skipping {len(already_processed)}/{total_num} channels")
    print(f"Removed data for {len(to_be_removed)} channels")
    print(f"Fetching data for {len(not_processed)} channels...")
    tasks = []

    if API_KEY is None:
        raise ValueError("YOUTUBE_API_KEY not found in environment variables")

    for channel_handle in not_processed:
        task = asyncio.create_task(get_channel_data(channel_handle))
        tasks.append(task)

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            channel_data = await task
            channel_handle = channel_data["channel_handle"]
            channel_data_dict[channel_handle] = channel_data
        except YoutubeAPIError as e:
            print(
                f"Error with response code {e.status} while fetching data for '{e.handle}'"
            )
            print(f"API request was '{e.url}'\n\n")

    with open(OUTPUT_FILE, "w") as f:
        channel_data_list_sorted = sorted(
            list(channel_data_dict.values()), key=lambda x: x["channel_title"].lower()
        )
        yaml.dump({"channels": channel_data_list_sorted}, f)


if __name__ == "__main__":
    asyncio.run(main())
