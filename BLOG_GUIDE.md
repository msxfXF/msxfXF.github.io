# MSXF Notes 写作与发布指南

这份文档说明如何在本地新增 blog、如何预览和发布，以及写作时建议遵循的规范。

## 目录结构

```text
/Users/bytedance/blog
├── content/
│   ├── about.md                 # 关于页内容
│   └── posts/                   # 所有文章放这里
│       ├── welcome.md
│       ├── article-interpretation-template.md
│       └── html-playground.html
├── scripts/
│   ├── build.mjs                # 构建脚本
│   └── serve.mjs                # 本地预览服务
├── src/
│   └── styles.css               # 全站样式
├── site.config.json             # 站点配置
├── README.md
└── BLOG_GUIDE.md                # 本文档
```

## 支持的文章格式

目前支持两种文章格式：

1. Markdown：适合常规笔记、文章解读、读书笔记、技术分享。
2. HTML：适合需要更丰富视觉表现、嵌入媒体、分栏、卡片、图表或交互的小专题。

所有文章都放在：

```text
/Users/bytedance/blog/content/posts/
```

文件扩展名可以是：

```text
.md
.html
```

## 新增一篇 Markdown 文章

在 `content/posts/` 下新增一个 `.md` 文件，例如：

```text
content/posts/my-reading-note.md
```

内容示例：

```md
---
title: 一篇文章的解读
description: 用一句话说明这篇文章主要讲什么，以及你为什么推荐读。
date: 2026-05-13
tags: [Reading, Notes]
---

## 一句话结论

这里写你最想让读者带走的判断。

## 原文背景

- 原文主题：
- 作者/机构：
- 发布时间：
- 阅读原因：

## 核心观点

1. 第一个关键观点。
2. 第二个关键观点。
3. 第三个关键观点。

## 我的判断

这里写你的理解、反驳、疑问或延伸。
```

## 新增一篇 HTML 文章

在 `content/posts/` 下新增一个 `.html` 文件，例如：

```text
content/posts/interactive-note.html
```

内容示例：

```html
---
title: HTML 文章标题
description: 这是一篇使用 HTML 编写的富表现力文章。
date: 2026-05-13
tags: [HTML, Notes]
---
<section class="note-hero">
  <p class="note-kicker">HTML Article</p>
  <h2>这里可以写更自由的版式</h2>
  <p>HTML 文章正文会作为原生 HTML 片段渲染。</p>
</section>

<div class="insight-grid">
  <article class="insight-card">
    <span>01</span>
    <h3>第一个洞察</h3>
    <p>这里写解释。</p>
  </article>
  <article class="insight-card">
    <span>02</span>
    <h3>第二个洞察</h3>
    <p>这里写解释。</p>
  </article>
</div>
```

HTML 文章可以复用的内置 class：

- `note-hero`：文章内的大型说明区块。
- `note-kicker`：小号英文/分类标签。
- `insight-grid`：两列洞察卡片容器。
- `insight-card`：洞察卡片。
- `pull-quote`：强调引用。
- `callout-panel`：提示或总结面板。

这些样式定义在：

```text
/Users/bytedance/blog/src/styles.css
```

## Front matter 规范

每篇文章顶部都需要 front matter，用 `---` 包起来。

推荐字段：

```yaml
---
title: 文章标题
description: 一句话摘要
date: 2026-05-13
tags: [Reading, Notes]
---
```

字段说明：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `title` | 建议必填 | 文章标题，会展示在首页、文章列表和详情页。 |
| `description` | 建议必填 | 文章摘要，用于卡片、SEO 描述和社交分享。 |
| `date` | 建议必填 | 日期格式使用 `YYYY-MM-DD`。 |
| `tags` | 可选 | 标签数组，建议 1 到 4 个。 |
| `slug` | 可选 | 自定义 URL。如果不写，会根据文件名生成。 |

## 文件命名规范

建议使用英文小写、数字和连字符：

```text
good-reading-note.md
html-playground.html
ai-product-observation.md
```

不推荐：

```text
我的文章.md
New Post 1.md
final最终版.html
```

原因：英文 slug 更适合 URL、分享和长期维护。

## 写作规范

### 标题

- `title` 尽量清楚，不要太泛。
- 好标题示例：`如何读懂一份行业报告`
- 弱标题示例：`一些想法`

### 摘要

`description` 建议控制在 40 到 100 个中文字符内。

它会显示在文章卡片里，所以应该回答：

- 这篇文章讲什么？
- 读者为什么应该点进去？

### 标签

标签建议保持稳定，不要每篇文章都创造新标签。

推荐标签示例：

- `Reading`
- `Notes`
- `AI`
- `Product`
- `Essay`
- `HTML`
- `Template`

### 正文结构

文章解读类内容建议使用这个结构：

```md
## 一句话结论

## 原文背景

## 核心观点

## 我的判断

## 可以继续追问的问题
```

普通分享类内容可以使用：

```md
## 背景

## 问题

## 方法

## 结果

## 复盘
```

## HTML 使用规范

HTML 能提供更多可玩性，但也需要更克制。

建议：

- 优先写语义化标签：`section`、`article`、`blockquote`、`figure`、`aside`。
- 不要在文章里写大段内联样式，优先把样式加到 `src/styles.css`。
- 自定义 class 命名尽量语义化，例如 `reading-map`、`idea-card`。
- 尽量避免复杂 JavaScript，保证 GitHub Pages 上长期稳定。
- 如果嵌入第三方 iframe，要确认来源可信。

不建议：

- 在文章里直接粘贴大量杂乱的导出 HTML。
- 使用会自动加载未知脚本的第三方组件。
- 依赖远程不稳定资源导致页面打开很慢。

## 本地构建

在项目根目录运行：

```bash
cd /Users/bytedance/blog
node scripts/build.mjs
```

构建完成后，生成文件会在：

```text
/Users/bytedance/blog/dist/
```

## 本地预览

先构建：

```bash
node scripts/build.mjs
```

再启动预览服务：

```bash
node scripts/serve.mjs
```

浏览器打开：

```text
http://localhost:4321
```

如果你改了文章或样式，需要重新运行：

```bash
node scripts/build.mjs
```

然后刷新浏览器。

## 发布流程

当前线上地址：

```text
https://msxfxf.github.io/
```

当前 GitHub 仓库：

```text
https://github.com/msxfXF/msxfXF.github.io
```

### 方式一：让 Codex 帮你发布

你可以直接说：

```text
帮我发布博客
```

发布前建议先说明是否要提交当前改动。

### 方式二：手动发布

当前站点发布方式是：把 `dist/` 目录内容推送到远程仓库的 `master` 分支根目录。

常规步骤：

```bash
cd /Users/bytedance/blog
node scripts/build.mjs
```

然后将 `dist/` 的内容发布到 `msxfXF/msxfXF.github.io` 的 `master` 分支。

注意：不要把源码目录直接发布到 GitHub Pages 的 `master` 分支。源码保留在本地 `main` 分支，线上页面使用构建后的静态文件。

## Git 提交规范

本地源码建议每次重要修改后提交一次：

```bash
git status --short
git add .
git commit -m "Add new reading note"
```

提交信息建议使用英文短句，例如：

```text
Add reading note about AI agents
Update article card styles
Support HTML posts
Fix mobile layout
```

## 发布前检查清单

发布前建议检查：

- [ ] 新文章文件是否放在 `content/posts/`。
- [ ] 文件扩展名是否是 `.md` 或 `.html`。
- [ ] front matter 是否包含 `title`、`description`、`date`。
- [ ] 日期是否是 `YYYY-MM-DD`。
- [ ] 标签数量是否克制。
- [ ] HTML 文章是否没有引入不可信脚本。
- [ ] 本地是否能成功运行 `node scripts/build.mjs`。
- [ ] 本地预览是否正常。

## 常见问题

### 文章没有出现在首页？

检查：

1. 文件是否在 `content/posts/` 下。
2. 文件扩展名是否是 `.md` 或 `.html`。
3. front matter 是否用 `---` 正确包裹。
4. 是否已经重新运行 `node scripts/build.mjs`。

### URL 是怎么生成的？

默认根据文件名生成。

例如：

```text
content/posts/my-reading-note.md
```

会生成：

```text
/posts/my-reading-note/
```

如果想自定义，可以在 front matter 写：

```yaml
slug: custom-url
```

### HTML 文章里的样式没生效？

检查：

1. class 名是否写对。
2. 样式是否已经添加到 `src/styles.css`。
3. 是否重新运行了 `node scripts/build.mjs`。
4. 浏览器是否需要强制刷新。

### 发布后线上还是旧内容？

可能是 GitHub Pages 构建或 CDN 缓存延迟。

建议等待 1 到 2 分钟，然后强制刷新页面。

## 安全注意事项

不要把 GitHub token、密码、私钥、Cookie 等敏感信息写进：

- 文章文件；
- README；
- 提交记录；
- 聊天窗口；
- `site.config.json`。

如果 token 已经暴露，应该立即在 GitHub 中撤销并重新生成。
