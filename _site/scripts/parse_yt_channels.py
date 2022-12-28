# %%
"""This is a web-scraping approach to getting youtube channel name and avatars. This is however, not very practical, and it's probably better to just use the Youtube API"""
import asyncio
import json
from pathlib import Path

import urllib3
import yaml
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

CHANNEL_JSON = "channel_data.json"

handle = "@Wendoverproductions"


def get_render_hmtl(url, http=None, browser=None):

    output_filename = Path("output.html").resolve()
    output_filename = str(output_filename)

    if http is None:
        http = urllib3.PoolManager()
    response = http.request("GET", url)
    html = response.data.decode("utf-8")

    with open(output_filename, "w") as f:
        f.write(html)

    if browser is None:
        browser = webdriver.Chrome()
    browser.get("file:///" + output_filename)
    html = browser.page_source
    browser.quit()
    return html


def parse_channel_data(handle, http=None, browser=None):
    if http is None:
        http = urllib3.PoolManager()
    url = f"https://www.youtube.com/c/{handle}"
    html = get_render_hmtl(url, http=http, browser=browser)
    soup = BeautifulSoup(html, "html.parser")

    channel_header_container = soup.find("div", id="channel-header-container")
    channel_image = channel_header_container.find("img").get("src")
    channel_name = channel_header_container.find("yt-formatted-string").text

    response = http.request("GET", channel_image)
    image_filename = f"{handle}.jpg"
    image_folder = Path("images")
    image_folder.mkdir(exist_ok=True)
    image_path = image_folder / image_filename
    with open(image_path, "wb") as f:
        f.write(response.data)

    return {
        "name": channel_name,
        "image": str(image_path),
        "url": url,
    }

def update_channel_data(handle, data):

    try:
        with open(CHANNEL_JSON, "r") as f:
            channel_data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        channel_data = {}

    channel_data[handle] = data

    with open(CHANNEL_JSON, "w") as f:
        json.dump(channel_data, f, indent=4)


YT_CHANNELS_FILE = "../_data/yt-channels.yml"
if __name__ == "__main__":
    with open(YT_CHANNELS_FILE, "r") as f:
        channels = yaml.load(f, Loader=yaml.FullLoader)

    http = urllib3.PoolManager()
    browser = webdriver.Chrome()

    for channel in tqdm(channels):
        handle = f"@{channel}"
        try:
            data = parse_channel_data(handle, http=http, browser=browser)
        except AttributeError:
            print(f"Failed to parse {handle}")
            continue
        update_channel_data(handle, data)


