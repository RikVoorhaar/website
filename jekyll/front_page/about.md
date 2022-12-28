---
layout: single
title: Personal interests and hobbies
permalink: /about/
---

<style>
/* left algin an image and scale to 50% */
    img[src*="#nice"] {
        margin: 0 1em 1em 0;
        width: 600px;
        margin-left: auto;
        margin-right: auto;
    }

    .yt_chan {
        text-align: center;
    }
    .yt_chan img {
        width: 1.5em;
        box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5);
        margin-right: .1em;
        vertical-align: text-bottom;
    }
    .yt_chan a {
        font-size: 1.2em;
        font-weight: bold;
        text-shadow: 0 0 1px #fff;
    }
    .yt_chan_item {
        display: inline-block;
        margin-right: 1em;
        margin-bottom: .3em;
        margin-top: .3em;
        vertical-align: text-bottom;
        line-height: 1.5em;
    }
</style>

I am a person of many interests, some more geeky than the rest.

Also check out [the website of my lovely wife Arina](https://arina-voorhaar.com/)

## Music

![Music](/assets/images/about/music.png#nice)


Music is a big part of my life! I like almost any kind of rock music, but I tend to gravitate towards heavier and noisier genres. Check out my [last.fm page](https://www.last.fm/user/Tilpo) and my [rym page](https://rateyourmusic.com/~Tilpo)! Usually I find new music by looking at recent album charts on RYM, and I strongly suggest this method for anyone into music.

I like to play electric guitar (though admittedly I haven't been doing it much lately...). I have Japanese made Fender "Aerodyne" strat, and an Epiphone Les Paul model (tuned to drop B♭!). I also have more pairs of headphones than I can count...

## Gaming

![Music](/assets/images/about/gaming.png#nice)

I really enjoy playing open world (both RPG and survivalcraft) games, the feeling of freedom, exploration  and creativity is wonderful. I also really enjoy replaying old (Nintendo) games from my youth. Sandbox games and grand strategy games are also amazing. I'm very enthusiastic about speedrunning and its wonderful community. It feels like taking the scientific approach to gaming.  Check out my [Steam account](https://steamcommunity.com/profiles/76561197996562422)!

## Cooking
![Music](/assets/images/about/cooking.png#nice)

I love cooking and the sense of satisfaction that comes with eating food you made yourself. I don't shy away from any cuisine, and I'm constantly trying to improve and learn new things. I have a nice collection of pans and knives.

## Reading
![Music](/assets/images/about/reading.png#nice)

I like to read hard sci-fi books, some of my favorite series are The Expanse (Corey), Foundation (Asimov), The Bobiverse (Taylor) and Old Man's War (Scalzi).  I'm also a big fan of the books of Murakami. Recently I've been on  a spree of reading long fantasy series. I have read the Lord of the Rings, Stormlight Archive, and Song of Ice and Fire, Mistborn, Narnia and Harry Potter. Check out [my goodreads page](https://www.goodreads.com/user/show/62542056-tilpo) for more!

## Gear
![Music](/assets/images/about/gear.png#nice)

I'm a bit of a technophile, and I have accumulated a lot of gear over the years. Here's a list of some of my cool stuff (that I actually use):

**PC:**  
- <ins>CPU</ins>: AMD Ryzen 7 2700X (8 core / 16 threads)
- <ins>GPU:</ins> NVidia GTX 1080
- <ins>RAM:</ins> 32GB DDR4-3200 CL16
- <ins>Cooler:</ins> Noctua NH-U12S
- <ins>Case:</ins> InWin Midi Tower 101C with 6 Arctic F12 120mm fans
- <ins>PSU:</ins> 750W Corsair
- <ins>Monitor:</ins> 2x 1080p Dell UltraSharp, 1x 1440p Dell UltraSharp
- <ins>OS:</ins> Manjaro Linux with KDE Plasma

**Peripherals:**
- <ins>Mouse:</ins> Logitech G502
- <ins>Trackpad:</ins> Apple Magic Trackpad 2 (using a trackpad instead of a mouse helps immensely with ergonomics for me)
- <ins>Keyboard:</ins>  Keychron Q6 with lubed 60g silent tactile Gatoron Aliaz switches and Drop ``/dev/tty`` keycaps.
- <ins>Wacom O</ins>ne 10" tablet for taking notes on the go
- <ins>Wacom O</ins>ne 1080p pen display for taking notes at home
- <ins>Controllers:</ins> Nintendo Pro Controller, 8bitdo SN30 PRO+ (because Pro Controller has a terrible d-pad)

**Laptop:**
- <ins>Model:</ins> Lenovo ThinkPad E14 gen 3.
- <ins>CPU:</ins> AMD Ryzen 7 5700U (8 core / 16 threads)
- <ins>RAM:</ins> 40GB DDR4-3200 CL22

**Headphones:**
- Beyerdynamic DT880 (at home)
- Beyerdynamic DT990 (at work)
- Sony LinkBuds (for the gym, or taking walks)
- Sony WH-1000XM3 (noise cancelling)

## Engineering
![Music](/assets/images/about/engineering.png#nice)

I am fascinated with engineering of many kinds. For example I am fascinated by logistics, civil
engineering, rocket science, semiconductors, and computing. Rather than academic research in these
areas I tend to watch YouTube channels and read wikipedia articles. There is so much fascinating
thought behind everything we experience around us. Here are some of my favorite YouTube channels:

<div class="yt_chan">
    {% for item in site.data.channels.channels %}
    <div class="yt_chan_item">
        <a href="{{ item.channel_url }}">
        <img src="/{{ item.channel_thumbnail_path}}"> <span class="use-serif">{{ item.channel_title }}</span></a>
    </div>
    {% endfor %}
</div>