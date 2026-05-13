# MSXF Notes

A clean, modern, zero-dependency static blog for notes, essays, and article interpretations.

## Write a post

Create a Markdown or HTML file under `content/posts/` with front matter. Markdown is best for normal notes; HTML is useful when you need richer layout, embedded media, custom cards, or small interactive elements.

Markdown example:

```md
---
title: Your title
description: Short summary
date: 2026-05-13
tags: [Reading, Notes]
---

Your content...
```

HTML example:

```html
---
title: Interactive reading note
description: A richer article layout
date: 2026-05-13
tags: [HTML, Notes]
---
<section class="note-hero">
  <p class="note-kicker">HTML Article</p>
  <h2>A more expressive note</h2>
  <p>Write any HTML fragment here.</p>
</section>
```

## Local build

```bash
node scripts/build.mjs
node scripts/serve.mjs
```

The generated site is in `dist/`.

## Deploy

This site is designed for GitHub Pages user sites. Build locally with `node scripts/build.mjs`, then publish the contents of `dist/` to the repository default branch root.
