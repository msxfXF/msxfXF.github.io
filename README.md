# MSXF Notes

A clean, modern, zero-dependency static blog for notes, essays, and article interpretations.

## Write a post

Create a Markdown file under `content/posts/` with front matter:

```md
---
title: Your title
description: Short summary
date: 2026-05-13
tags: [Reading, Notes]
---

Your content...
```

## Local build

```bash
node scripts/build.mjs
node scripts/serve.mjs
```

The generated site is in `dist/`.

## Deploy

This site is designed for GitHub Pages user sites. Build locally with `node scripts/build.mjs`, then publish the contents of `dist/` to the repository default branch root.
