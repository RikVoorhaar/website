---
layout: single
title: Curriculum Vitae
permalink: /cv/
toc: true
---


<style> 
.listWithDescription p{
  margin: 0.25em
}
</style> 

Below is a detailed CV. If you want to **download a traditional CV**, click one of the two icons below.

<span style="display:inline-block; width: 50px;"></span>
<a href="https://github.com/RikVoorhaar/RikVoorhaar.github.io/raw/master/_data/resume.pdf" target="_blank"> <i class="fa fa-file-pdf fa-3x"></i></a>
<span style="display:inline-block; width: 20px;"></span>
<a href="https://github.com/RikVoorhaar/RikVoorhaar.github.io/raw/master/_data/resume.docx"> <i class="fa fa-file-word fa-3x"></i> </a>

## Research

_Research interests_:
{%- for item in site.data.cv.research-interests-summary -%}
<br>-- {{item}}
{% endfor %}

{% for item in site.data.cv.research-interests-text %}
{{item}}
{% endfor %}
<!-- {{site.data.cv.research-interests}} -->

## Publications and preprints
{% for item in site.data.cv.publications %}

<div class="listWithDescription" markdown="1">
[{{item.name}}]({{item.url}}) {{item.date}}{% if item.publisher %}, _published in {{item.publisher}}_{% endif %}
{% if item.description %}<description>{{item.description}}</description>{% endif %}
<br>
</div>
{% endfor %}

## Open source contributions

{% for item in site.data.cv.open-source-contribs %}
<div class="listWithDescription" markdown="1">
[{{item.name}}]({{item.url}})
<description>{{item.description}}</description>
<br>
</div>
{% endfor %}

## Work experience

{% for item in site.data.cv.work %}
<div class="listWithDescription" markdown="1">
**{{item.date}}**:  
{% if item.img %}<div style="padding-right:10px; float: left"><img src="{{item.img}}"></div>{% endif %}
 {{item.name}}{% if item.location %} at _{% if item.url %}[{{item.location}}]({{item.url}}){% else %}{{item.location}}{%endif%}_{% endif %}.  
<small>{{item.description}}</small>
</div>
<br>
{% endfor %}


## Education

{% for item in site.data.cv.education %}
<div class="listWithDescription" markdown="1">
{%if item.date %}<br>**{{item.date}}**<br>{% endif %}
-- {{item.name}}, at _{% if item.url %}[{{item.location}}]({{item.url}}){% else %}{{item.location}}{%endif%}{% if item.note %} {{item.note}}{% endif %}_.
</div>
{% endfor %}

## Skills

### Programming languages

{% for item in site.data.cv.programming-languages %}
<div class="listWithDescription" markdown="1">
**{{item.name}}**<br>
{%- for lang in item.items -%}
{% if item.istools %}{{lang}}{% if forloop.last == false %}, {% endif %}{% else %}-- {{lang}}<br>{% endif %}
{% endfor %}<br>
</div>
{% endfor %}


### Languages

{% for item in site.data.cv.languages %}
**{{item.name}}**<br>
{%- for lang in item.items -%}
-- {{lang}}  
{% endfor %}
{% endfor %}


### Mathematical expertise

I have a wide background in pure and applied mathematics, and I feel comfortable with research-level
mathematics in the following areas:

{% for item in site.data.cv.math %}
**{{item.title}}:**<br>
{%- for skill in item.items -%}
-- {{skill}}  
{% endfor %}
{% endfor %}
