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

## Research
{{site.data.cv.research-interests}}

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
-- {{item.name}}{% if item.location %} at _{{item.location}}_{% endif %}.
<description>{{item.description}}</description>
</div>
<br>
{% endfor %}


## Education

{% for item in site.data.cv.education %}
<div class="listWithDescription" markdown="1">
{%if item.date %}<br>**{{item.date}}**<br>{% endif %}
-- {{item.name}}, at _{{item.location}}{% if item.note %} {{item.note}}{% endif %}_.
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
