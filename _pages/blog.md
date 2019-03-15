---
layout: posts
permalink: /blog/
title: "Machine Learning projects"
author_profile: true
---

{% capture written_label %}'None'{% endcapture %}

{% for blog in site.blogs %}
  {% unless blog.output == false or blog.label == "blogs" %}
    {% capture label %}{{ blog.label }}{% endcapture %}
    {% if label != written_label %}
      <h2 id="{{ label | slugify }}" class="archive__subtitle">{{ label }}</h2>
      {% capture written_label %}{{ label }}{% endcapture %}
    {% endif %}
  {% endunless %}
  {% for blog in blog.docs %}
    {% unless blog.output == false or blog.label == "blogs" %}
      {% include archive-single.html %}
    {% endunless %}
  {% endfor %}
{% endfor %}