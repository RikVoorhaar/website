---
layout: single
title: Test playground
permalink: /test/
---

<link rel="stylesheet" href="cv_stuff.css">
<script>
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
        content.style.display = "none";
        } else {
        content.style.display = "block";
        }
    });
    }
</script>

<button type="button" class="collapsible">Open Collapsible</button>
<div class="content">
  <p>Lorem ipsum...</p>
</div>

<details>
    <summary>Click to expand!</summary>
    <p>Here is some hidden text.</p>
</details>