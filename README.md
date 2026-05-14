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
npm run build
npm run serve
```

The generated site is in `dist/`.

## Deploy

Source files live on `main`. The live GitHub Pages site is deployed from the `master` branch root, which contains the built `dist/` output. Pushing `main` alone does not update the live site.

```bash
npm run deploy
```

The deploy script builds the site, syncs `dist/` into a temporary worktree based on `origin/master`, commits deployment changes if needed, and pushes `HEAD:master`.
