---
layout: posts
permalink: /machine-learning/
title: "Machine Learning projects"
author_profile: true
---

{% capture written_label %}'None'{% endcapture %}

{% for post in site.posts %}
  {% unless post.output == false or post.label == "posts" %}
    {% capture label %}{{ post.label }}{% endcapture %}
    {% if label != written_label %}
      <h2 id="{{ label | slugify }}" class="archive__subtitle">{{ label }}</h2>
      {% capture written_label %}{{ label }}{% endcapture %}
    {% endif %}
  {% endunless %}
  {% for post in post.docs %}
    {% unless post.output == false or post.label == "posts" %}
      {% include archive-single.html %}
    {% endunless %}
  {% endfor %}
{% endfor %}