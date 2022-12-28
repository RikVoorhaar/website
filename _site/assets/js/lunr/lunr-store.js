var store = [{
        "title": "First post",
        "excerpt":"Ah, the classical “first post”, often the only post in the blog. Let us hope this is not the case. I feel like I should write down some things about my side projects. I tried using Medium, but it has two significant problems: the platform feels too monetized, and it...","categories": ["jekyll"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/first-post/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/first-post.jpg"
      },{
        "title": "Bias in figure skating judging",
        "excerpt":"My wife is very enthuisiastic about figure skating. She often mentions that the judging is biased, in the sense that many judges give higher scores to athletes from their own country, and lower scores to athletes from other countries. This doesn’t sound too surprising, but I wondered if it is...","categories": ["data-science"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/figure-skating/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/figure-skating.jpg"
      },{
        "title": "Is my data normal?",
        "excerpt":"Normally distributed data is great. It is easy to interpet, and many statistical and machine learning method work much better on normally distributed data. But how do we know if our data is actually normally distributed? Let’s start with the well-known MNIST digit dataset. This is a very famous dataset....","categories": ["data-science","statistics"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/normal-data/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/normal-data.png"
      },{
        "title": "How do my music preferences evolve?",
        "excerpt":"Last.fm is a great tool to keep track of the music you listen to. It’s also great for analysis! In this article I will discuss how to download your listening history, and to annotate the data with genre information from discogs. We can then use this to track how music...","categories": ["data-science","music"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/lastfm/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/lastfm.jpg"
      },{
        "title": "How big should my validation set be?",
        "excerpt":"In machine learning it is important to split your data into a training, validation and test set. People often use a heuristic like suggesting to split your data into sizes of 50/25/25 or 70/20/10 or something like that. Can we do better than a heuristic? We can also do such...","categories": ["data-science","statistics"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/validation-size/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/validation-data.jpg"
      },{
        "title": "Modeling uncertainty in exam scores",
        "excerpt":"We widely use exams in education to gauge the level of students. The result of an exam is really only indicator of the students actual level, and has a certainly level of uncertainty. In this article we will try to model the uncertainty in the grades of an exam through...","categories": ["data-science","statistics","education"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/bayes_exam/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/bayes-exam.jpg"
      },{
        "title": "2020 in music",
        "excerpt":"It goes without saying that 2020 is a special year in a number of ways. I want to look back at what 2020 meant for me in terms of music. As a side effect of staying home almost all year, I have also listened to more music than previous years....","categories": ["music"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/music_2020/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/music-2020.jpg"
      },{
        "title": "Time series analysis of my email traffic",
        "excerpt":"I’ve been using gmail since back 2006 – when it was still an invite-only beta. In these last 15 years I have received a lot of emails. I wondered if I’m actually receiving more emails now than back then, or if there are any interesting trends. I want to see...","categories": ["data-science","statistics"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/email-time-series/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/email-time-series.jpg"
      },{
        "title": "Blind Deconvolution #1: Non-blind Deconvolution",
        "excerpt":"I recently became interested in blind deconvolution. Initially I didn’t even know the proper name for this, I simply wondered if it’s possible to automatically sharpen images given we have some limited information about how they are blurred. Then I went on to do some actual research, and I started...","categories": ["machine-learning","signal-processing","computer-vision"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/deconvolution-part1/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/st-vitus-blur.png"
      },{
        "title": "Blind Deconvolution #2: Image Priors",
        "excerpt":"This is part two in a series on blind deconvolution of images. In the previous part we looked at non-blind deconvolution, where we have an image and we know exactly how it was distorted. While this situation may seem unrealistic, it does occur in cases where we have excellent understanding...","categories": ["machine-learning","signal-processing","computer-vision"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/deconvolution-part2/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/st-vitus-laplace.png"
      },{
        "title": "Blind Deconvolution #3: More about non-blind deconvolution",
        "excerpt":"In part 1 we saw how to do non-blind image deconvolution. In part 2 we saw a couple good image priors and we saw how they can be used for simple blind deconvolution. This worked well for deconvolution of Gaussian point spread functions, but it gave bad artifacts for motion...","categories": ["machine-learning","signal-processing","computer-vision"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/deconvolution-part3/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/cow-weird-blur.png"
      },{
        "title": "Blind deconvolution #4: Blind deconvolution",
        "excerpt":"In this final part on the deconvolution series, we will look at blind deconvolution. That is, we want to remove blur from images while having only partial knowledge about how the image was blurred. First of all we will develop a simple method to generate somewhat realistic forms of combined...","categories": ["machine-learning","computer-vision"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/deconvolution-part4/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/st-vitus-deblurred.png"
      },{
        "title": "How to edit Microsoft Word documents in Python",
        "excerpt":"In preparation for the job market, I started polishing my CV. I try to keep the CV on my website as up-to-date as possible, but many recruiters and companies prefer a single-page neat CV in a Microsoft Word document. I used to always make my CV’s in LaTeX, but it...","categories": ["data-mining","code"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/python-docx/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/python_docx/doc_comparison.png"
      },{
        "title": "Low-rank matrices: using structure to recover missing data",
        "excerpt":"Tensor networks are probably the most important tool in my research, and I want explain them. Before I can do this however, I should first talk about low-rank matrix decompositions, and why they’re so incredibly useful. At the same time I will illustrate everything using examples in Python code, using...","categories": ["machine-learning","mathematics","linear-algebra","code"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/low-rank-matrix/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/st-vitus-rank-10.png"
      },{
        "title": "Machine learning with discretized functions and tensors",
        "excerpt":"In my new paper together with my supervisor, we explain how to use discretized functions and tensors to do supervised machine learning. A discretized function is just a function defined on some grid, taking a constant value on each grid cell. We can describe such a function using a multi-dimensional...","categories": ["machine-learning","mathematics","linear-algebra","code"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/discrete-function-tensor/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/discrete-function-tensor.png"
      },{
        "title": "GMRES: or how to do fast linear algebra",
        "excerpt":"Linear algebra is the foundation of modern science, and the fact that computers can do linear algebra very fast is one of the primary reasons modern algorithms work so well in practice. In this blog post we will dive into some of the principles of fast numerical linear algebra, and...","categories": ["mathematics","linear-algebra","code"],
        "tags": [],
        "url": "https://blog.rikvoorhaar.com/gmres/",
        "teaser": "https://blog.rikvoorhaar.com/imgs/teasers/gmres-teaser.png"
      }]
