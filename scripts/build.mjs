import { mkdir, readFile, readdir, rm, writeFile, cp } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const root = process.cwd();
const dist = path.join(root, 'dist');
const contentDir = path.join(root, 'content');
const postsDir = path.join(contentDir, 'posts');
const publicDir = path.join(root, 'public');
const srcDir = path.join(root, 'src');
const config = JSON.parse(await readFile(path.join(root, 'site.config.json'), 'utf8'));

const isWatch = process.argv.includes('--watch');

function escapeHtml(value = '') {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function slugify(input) {
  return String(input)
    .trim()
    .toLowerCase()
    .replace(/\.md$/i, '')
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '') || 'post';
}

function parseFrontMatter(source) {
  if (!source.startsWith('---')) return [{}, source];
  const end = source.indexOf('\n---', 3);
  if (end === -1) return [{}, source];
  const raw = source.slice(3, end).trim();
  const body = source.slice(source.indexOf('\n', end + 4) + 1);
  const data = {};
  for (const line of raw.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    const [, key, value] = match;
    if (value.startsWith('[') && value.endsWith(']')) {
      data[key] = value.slice(1, -1).split(',').map((item) => item.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean);
    } else {
      data[key] = value.replace(/^['"]|['"]$/g, '');
    }
  }
  return [data, body];
}

function inlineMarkdown(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return out;
}

function renderMarkdown(md) {
  const lines = md.replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let paragraph = [];
  let list = [];
  let code = [];
  let inCode = false;

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${inlineMarkdown(paragraph.join(' '))}</p>`);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      html.push(`<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join('')}</ul>`);
      list = [];
    }
  };

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
        code = [];
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = slugify(text);
      html.push(`<h${level} id="${id}">${inlineMarkdown(text)}</h${level}>`);
      continue;
    }
    const listItem = line.match(/^[-*]\s+(.+)$/);
    if (listItem) {
      flushParagraph();
      list.push(listItem[1]);
      continue;
    }
    paragraph.push(line.trim());
  }
  flushParagraph();
  flushList();
  if (inCode) html.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
  return html.join('\n');
}

function formatDate(date) {
  return new Intl.DateTimeFormat(config.language || 'zh-CN', { dateStyle: 'medium' }).format(new Date(date));
}

function pageShell({ title, description, body, current = '/', type = 'website' }) {
  const pageTitle = title === config.title ? config.title : `${title} | ${config.title}`;
  const nav = config.nav.map((item) => {
    const active = current === item.href || (item.href !== '/' && current.startsWith(item.href));
    return `<a class="nav-link${active ? ' active' : ''}" href="${item.href}">${escapeHtml(item.label)}</a>`;
  }).join('');

  return `<!doctype html>
<html lang="${escapeHtml(config.language || 'zh-CN')}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(pageTitle)}</title>
  <meta name="description" content="${escapeHtml(description || config.description)}">
  <meta property="og:title" content="${escapeHtml(pageTitle)}">
  <meta property="og:description" content="${escapeHtml(description || config.description)}">
  <meta property="og:type" content="${type}">
  <meta name="theme-color" content="#111827">
  <link rel="alternate" type="application/rss+xml" title="${escapeHtml(config.title)}" href="/rss.xml">
  <link rel="stylesheet" href="/assets/styles.css">
</head>
<body>
  <div class="site-bg"></div>
  <header class="site-header">
    <a class="brand" href="/" aria-label="${escapeHtml(config.title)} home">
      <span class="brand-mark">M</span>
      <span>${escapeHtml(config.title)}</span>
    </a>
    <nav class="nav" aria-label="Main navigation">${nav}</nav>
  </header>
  <main>${body}</main>
  <footer class="site-footer">
    <p>© ${new Date().getFullYear()} ${escapeHtml(config.title)}</p>
    <p>Notes, readings, and considered interpretation.</p>
  </footer>
</body>
</html>`;
}

async function readPosts() {
  if (!existsSync(postsDir)) return [];
  const files = (await readdir(postsDir)).filter((file) => file.endsWith('.md'));
  const posts = [];
  for (const file of files) {
    const source = await readFile(path.join(postsDir, file), 'utf8');
    const [frontMatter, body] = parseFrontMatter(source);
    const slug = slugify(frontMatter.slug || file);
    const date = frontMatter.date || new Date().toISOString().slice(0, 10);
    posts.push({
      ...frontMatter,
      title: frontMatter.title || slug,
      description: frontMatter.description || '',
      tags: Array.isArray(frontMatter.tags) ? frontMatter.tags : [],
      date,
      slug,
      url: `/posts/${slug}/`,
      body,
      html: renderMarkdown(body),
      readingTime: Math.max(1, Math.ceil(body.replace(/\s+/g, '').length / 500))
    });
  }
  return posts.sort((a, b) => new Date(b.date) - new Date(a.date));
}

function postCard(post) {
  const tags = post.tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join('');
  return `<article class="post-card">
    <div class="post-meta"><time datetime="${escapeHtml(post.date)}">${formatDate(post.date)}</time><span>${post.readingTime} min read</span></div>
    <h2><a href="${post.url}">${escapeHtml(post.title)}</a></h2>
    <p>${escapeHtml(post.description)}</p>
    <div class="card-footer">
      <div class="tags">${tags}</div>
      <a class="read-more" href="${post.url}" aria-label="Read ${escapeHtml(post.title)}">阅读全文</a>
    </div>
  </article>`;
}

async function writePage(route, html) {
  const dir = path.join(dist, route);
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, 'index.html'), html);
}

function renderHome(posts) {
  const featured = posts.slice(0, 3).map(postCard).join('');
  return pageShell({
    title: config.title,
    description: config.description,
    current: '/',
    body: `<section class="hero">
      <div class="hero-content">
        <p class="eyebrow">Reading / Notes / Interpretation</p>
        <h1>把值得反复阅读的内容，整理成可复用的思考。</h1>
        <p class="hero-copy">${escapeHtml(config.description)}</p>
        <div class="hero-actions">
          <a class="button primary" href="/posts/">浏览文章</a>
          <a class="button ghost" href="/about/">关于本站</a>
        </div>
      </div>
      <aside class="hero-panel" aria-label="Site focus">
        <span>Focus</span>
        <strong>Article Interpretation</strong>
        <p>用摘要、判断和追问，把碎片阅读沉淀为长期笔记。</p>
      </aside>
    </section>
    <section class="section-head">
      <div>
        <p class="eyebrow">Latest</p>
        <h2>最新文章</h2>
      </div>
      <a href="/posts/">全部文章 →</a>
    </section>
    <section class="post-grid">${featured}</section>`
  });
}

function renderPosts(posts) {
  return pageShell({
    title: '文章',
    description: 'MSXF Notes 的全部文章。',
    current: '/posts/',
    body: `<section class="page-hero compact">
      <p class="eyebrow">Archive</p>
      <h1>文章</h1>
      <p>分享、阅读摘记、文章解读和持续生长的想法。</p>
    </section>
    <section class="post-list">${posts.map(postCard).join('')}</section>`
  });
}

function renderPost(post) {
  return pageShell({
    title: post.title,
    description: post.description,
    current: '/posts/',
    type: 'article',
    body: `<article class="article">
      <header class="article-header">
        <a class="back-link" href="/posts/">← 返回文章</a>
        <div class="post-meta"><time datetime="${escapeHtml(post.date)}">${formatDate(post.date)}</time><span>${post.readingTime} min read</span></div>
        <h1>${escapeHtml(post.title)}</h1>
        <p>${escapeHtml(post.description)}</p>
        <div class="tags">${post.tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</div>
      </header>
      <div class="prose">${post.html}</div>
    </article>`
  });
}

async function renderAbout() {
  const source = await readFile(path.join(contentDir, 'about.md'), 'utf8');
  const [frontMatter, body] = parseFrontMatter(source);
  return pageShell({
    title: frontMatter.title || '关于',
    description: frontMatter.description || config.description,
    current: '/about/',
    body: `<section class="page-hero compact">
      <p class="eyebrow">About</p>
      <h1>${escapeHtml(frontMatter.title || '关于')}</h1>
      <p>${escapeHtml(frontMatter.description || '')}</p>
    </section>
    <article class="article narrow"><div class="prose">${renderMarkdown(body)}</div></article>`
  });
}

function renderRss(posts) {
  const items = posts.map((post) => `<item>
    <title>${escapeHtml(post.title)}</title>
    <link>${escapeHtml(config.url)}${post.url}</link>
    <guid>${escapeHtml(config.url)}${post.url}</guid>
    <pubDate>${new Date(post.date).toUTCString()}</pubDate>
    <description>${escapeHtml(post.description)}</description>
  </item>`).join('\n');
  return `<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>${escapeHtml(config.title)}</title>
  <link>${escapeHtml(config.url)}</link>
  <description>${escapeHtml(config.description)}</description>
  ${items}
</channel>
</rss>`;
}

async function build() {
  await rm(dist, { recursive: true, force: true });
  await mkdir(dist, { recursive: true });
  if (existsSync(publicDir)) await cp(publicDir, dist, { recursive: true });
  await mkdir(path.join(dist, 'assets'), { recursive: true });
  const css = await readFile(path.join(srcDir, 'styles.css'), 'utf8');
  const hash = createHash('sha256').update(css).digest('hex').slice(0, 8);
  await writeFile(path.join(dist, 'assets', 'styles.css'), `/* build:${hash} */\n${css}`);

  const posts = await readPosts();
  await writePage('', renderHome(posts));
  await writePage('posts', renderPosts(posts));
  for (const post of posts) await writePage(path.join('posts', post.slug), renderPost(post));
  await writePage('about', await renderAbout());
  await writeFile(path.join(dist, 'rss.xml'), renderRss(posts));
  await writeFile(path.join(dist, '404.html'), pageShell({ title: '页面不存在', description: '页面不存在', body: `<section class="page-hero compact"><h1>页面不存在</h1><p>你访问的页面可能已经移动或删除。</p><a class="button primary" href="/">回到首页</a></section>` }));
  console.log(`Built ${posts.length} posts into dist/`);
}

await build();

if (isWatch) {
  const { watch } = await import('node:fs');
  console.log('Watching for changes...');
  let timer;
  for (const dir of [contentDir, srcDir]) {
    watch(dir, { recursive: true }, () => {
      clearTimeout(timer);
      timer = setTimeout(() => build().catch(console.error), 150);
    });
  }
}
