---
layout: single
title: Curriculum Vitae
permalink: /cv/
toc: true
---


<link rel="stylesheet" href="/assets/css/cv.css"/>

<br>
<div class="container">
  <div class="leftbox" style="margin-right: 10px; width: 50px;">
    <!-- <span style="display:inline-block; width: 10px;"></span> -->
    <a href="https://github.com/RikVoorhaar/RikVoorhaar.github.io/raw/master/_data/cv.pdf" target="_blank"> <i class="fa fa-file-pdf fa-3x"></i></a>
  </div>
  <div class="rightbox">
    Below is a detailed CV. If you want to <strong>download a traditional CV</strong>, click the icon on the left.
  </div>
</div>

## Work experience

{% for item in site.data.cv.work %}
<div class="listWithDescription" markdown="1">
  <div class="container">
    <div class="leftbox" style="max-width: 15%">
      {% if item.img %}
      <img src="{{item.img}}" style="width: 100%;">
      {% endif %}
    </div>
    <div class="rightbox">
      <em>{{item.date}}</em><br>  
      <div class="experience-header">
        {{item.name}}{% if item.location %} at {% if item.url %}<a href="{{item.url}}">{{item.location}}</a>{% else %}{{item.location}}{%endif%}{% endif %}.
      </div><br>
      <div class="small">{{item.description}}</div>
    </div>
  </div>
</div>
<br>
{% endfor %}
 

## Education

{% for item in site.data.cv.education %}
<div class="listWithDescription" markdown="1">
{% if item.date %}{% unless item.firstitem %}<br>{% endunless %}<em>{{item.date}}</em><br>{% endif %}
<div class="block-line">
<span class="experience-header">
{{item.name}}</span> | <em>{{item.location}}{% if item.url %}<a href="{{item.url}}"> (Coursera certificate)</a>{%endif%}{% if item.note %} {{item.note}}{% endif %}</em>.
</div></div>
{% endfor %}

### Online courses
{% for item in site.data.cv.online-courses %}
<div class="listWithDescription" markdown="1">
{% if item.date %}{% unless item.firstitem %}<br>{% endunless %}<em>{{item.date}}</em><br>{% endif %}
<div class="block-line">
<span class="experience-header">
{{item.name}}</span> | <em>{{item.location}}{% if item.url %}<a href="{{item.url}}"> (Coursera certificate)</a>{%endif%}{% if item.note %} {{item.note}}{% endif %}</em>.
</div></div>
{% endfor %}
 
## Skills

### Programming languages

{% for item in site.data.cv.programming-languages %}
<div class="listWithDescription" markdown="1">
<div class="experience-header">{{item.name}}</div>
<div class="block-line" style="margin-bottom: 15px;">
{%- for lang in item.items -%}
{% if item.istools %}{{lang}}{% if forloop.last == false %}, {% endif %}{% else %} {{lang}}<br>{% endif %}
{% endfor %}
</div>
</div>
{% endfor %}


### Languages

{% for item in site.data.cv.languages %}
<div class="experience-header">{{item.name}}</div>
<div class="block-line" style="margin-bottom: 15px;">
{%- for lang in item.items -%}
{{lang}}<br> 
{% endfor %}
</div>
{% endfor %}

## Publications and preprints
{% for item in site.data.cv.publications %}

<div class="listWithDescription" markdown="1">
<em>{{item.date}}</em><br>
<div class="block-line" style="margin-bottom: 15px">
<span class="experience-header">
<a href="{{item.url}}">{{item.name}}{% if item.publisher %},{% endif %}</a></span>{% if item.publisher %} <em>published in {{item.publisher}}</em>{% endif %}{% if item.coauthor %}<br><em>Joint work with</em> {{item.coauthor}} {% endif %}
{% if item.description %}<br><description><details><summary>More info</summary>{{item.description}}</details></description>{% endif %}
<br>
</div>
</div>
{% endfor %}
 
## Open source contributions

{% for item in site.data.cv.open-source-contribs %}
<div class="listWithDescription" markdown="1">
  <div class="block-line" style="margin-bottom: 15px">
    <div class="experience-header">
      <a href="{{item.url}}"><i class="fab fa-fw fa-github"></i> {{item.name}}</a>
    </div><br>
    <description><details><summary>More info</summary>{{item.description}}</details></description>
  </div>
</div>
{% endfor %}





<!-- ### Mathematical expertise

I have a wide background in pure and applied mathematics, and I feel comfortable with research-level
mathematics in the following areas:

{% for item in site.data.cv.math %}
**{{item.title}}:**<br>
{%- for skill in item.items -%}
-- {{skill}}  
{% endfor %}
{% endfor %} -->
