---
layout: posts
title: "Dev log: interactive website dashboard"
date: 2023-05-01
categories: website data-science tools
excerpt: "I made a dashboard to explore the access logs of this website."
header:
  teaser: "/imgs/teasers/dashboard.webp"
---

## TODO:

What do I want to write in this article? 
1. Rough thought process and dev process
2. New technologies I learned in the process
3. Closing thoughts, and ideas on what to do with this. 


## Back story
This has been a long time in the making. At first, I was using Github pages to host this website, so there was no way for me to get access to usage statistics.  Since I'm using Google domains, I did get some analytics, but that was just about Google searches. Since I like data science projects I thought it would be really cool to write my own analytics tools for this website. In the meantime, I also got a VPS back in September so that I can host my own cloud storage and have good offsite backups among other things. 

This set the stage to finally get to a self-hosted solution back around the new year. Doing this as well as other sysasdmin things on my VPS made me get comfortable with using Docker to deploy things, and has overall been a great learning experience. Honestly, this was very frustrating at times, and there were multiple times I just gave up. For example, to get familiar with AWS, I wanted to use `rdiff-backup` to make backups to an S3 bucket, but after probably spending around 10 hours on this, I just gave up. Nevertheless, this website has been running great on my VPS, and switching to running everything from a Docker container has also really improved the development process. 

Right now this website is running in a docker container running `Jekyll`, which is behind an `nginx` server, which is in turn behind a `traefik` reverse proxy. Back in early January, I got `nginx` to spit out HTTP access logs of my website, but I didn't touch these logs for a few months, until I was finally able to dedicate some of my free time to this project.

## Initial idea

In early March I started on this project. From work I gained some experience in Typescript, particularly to build frontends for Python applications. I also got acquainted with `plotly` as an alternative plotting framework, which is particularly nice if you want to display it in a browser. My initial idea was therefore to do the following:
- Write some scripts to ingest the logs into a database
- Clean the data to make it suitable for analysis
- Do data exploration and find some interesting things to plot
- Make robust, fast functions that can output these plots
- Make a web frontend to display these plots

This started off relatively painless. While I took some online courses on SQL databases, I never actually used them in a project, but it was still quite simple to use `sqlalchemy` to ingest the logs into a `sqllite` database. Since there are never any concurrent users, it didn't make much sense to me to go for anything more advanced. I then wrote code using `pandas` to clean the data and get interesting data out such as the web page the user connects to or the geographic location of each user. I then did data exploration and made some nice time series plots among other things, and I started on making a dashboard frontend using `dash`. 

I did some benchmarking and found that even on two months of access data most of the plot functions took a couple hundred milliseconds to compute. Since I planned from the start to make the dashboard publicly accessible, I really needed to keep performance in mind. I don't want people to bog down my VPS if there are several concurrent users. That's why I decided to switch from `pandas` to `polars`; a dataframe library written in `rust` more geared towards performance. I found this switch to be very interesting because `polars` forces you into a different way of thinking about the data transformations. While this is less intuitive, it does guide you towards solutions that are inherently more performant than what I would come up with when attempting the same thing in `pandas`. More on performance later. 

## Making it interactive

One thing I really wanted to understand from the data was the differences in geographic regions. But this seems to be very difficult to capture in one or two plots. Instead, I realized that to explore and convey this information properly we need to have some kind of interactivity. I came up with the idea of allowing a user of the dashboard to select different subsets of the dataset which are then displayed in the same plot. For example, suppose I want to know at what time users in Denmark typically connect to my website versus people in Switzerland. I could then make one subset of the dataset containing only the people connecting from Denmark, and another subset of people connecting from Switzerland. After proper normalization, I can then just make a time series of the time of day when users from both subsets of the data connect, and I've got my answer.

#TODO: explain why this kind of interactivity is actually hard to code. Why I had to switch from `dash` to just using `flask` with `typescript`. 

