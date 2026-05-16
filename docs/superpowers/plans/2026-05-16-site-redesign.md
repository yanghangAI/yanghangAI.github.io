# Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace academicpages-derived multi-page site with a content-first single-page academic CV plus a `/tools/` directory, per `docs/superpowers/specs/2026-05-16-site-redesign-design.md`.

**Architecture:** Jekyll static site. Four layouts (`default`, `home`, `archive`, `single`), four includes (`masthead`, `footer`, `archive-single`, `tool-single`), one stylesheet (`assets/css/site.css`, system Georgia, ~720px measure, classic web-blue links). Home page is the single-page CV; `/tools/` lists items from a new `_tools` collection.

**Tech Stack:** Jekyll on GitHub Pages, hand-written HTML/Liquid/CSS. No SCSS, no web fonts, no JS frameworks. Cluster page keeps its existing vanilla-JS fetch-and-render logic.

**Engineer notes:**
- No local Ruby/Jekyll/Docker on the Unity HPC node — final build verification happens via GitHub Pages after push (Task 14). Until then, "verify" means file-content checks, `git status`, and `grep` for stale references.
- This is a personal static site, not a tested codebase. There is no test framework. Each task's "verify" step is the substitute for a test run: it confirms the change exists and that no other file still references something that was removed.
- Commit after every task. Push only at Task 14 unless explicitly noted.

---

## Task 1: Add `_tools` collection (config + two initial tool files)

**Files:**
- Modify: `_config.yml` (add `tools` to the `collections:` block)
- Create: `_tools/cluster-dashboard.md`
- Create: `_tools/what2do.md`

- [ ] **Step 1: Find the existing `collections:` block in `_config.yml`**

Run: `grep -n "^collections:" _config.yml`
Expected: a line number for the `collections:` key. Read 30 lines around it to see the existing collection entries (publications, teaching, portfolio, talks).

- [ ] **Step 2: Add `tools` to the `collections:` block**

After the last existing collection entry (likely `talks`), insert (preserve the YAML indentation used by the other entries):

```yaml
  tools:
    output: true
    permalink: /:collection/:path/
```

- [ ] **Step 3: Create the cluster-dashboard tool file**

Write to `_tools/cluster-dashboard.md`:

```markdown
---
title: Cluster dashboard
collection: tools
url: /cluster/
date: 2026-05-16
status: live
summary: Live snapshot of my UMass Unity HPC queue (squeue, sacct, sinfo, storage), refreshed every 15 minutes from a cron job.
---
```

- [ ] **Step 4: Create the what2do tool file**

Write to `_tools/what2do.md`:

```markdown
---
title: What2Do
collection: tools
url: https://yanghangAI.github.io/what2do/
date: 2026-05-13
status: live
summary: Daily-updated operating hours for UMass Amherst swim, climbing, ice skating, and fitness facilities. Separate static site with a GitHub Action scraper.
---
```

- [ ] **Step 5: Verify**

Run: `ls _tools/ && grep -A2 "^  tools:" _config.yml`
Expected: both `.md` files listed and the `tools:` collection block printed.

- [ ] **Step 6: Commit**

```bash
git add _config.yml _tools/
git commit -m "feat(tools): add _tools collection with initial two entries"
```

---

## Task 2: Tools listing page and entry include

**Files:**
- Create: `_includes/tool-single.html`
- Create: `_pages/tools.html`

- [ ] **Step 1: Create the tool entry include**

Write to `_includes/tool-single.html`:

```liquid
<li class="entry">
  <div class="entry__date">
    {% if post.date %}{{ post.date | date: "%Y-%m" }}{% else %}—{% endif %}
  </div>
  <div class="entry__main">
    <h3 class="entry__title">
      {% if post.url contains '://' %}
        <a href="{{ post.url }}">{{ post.title }}</a>
      {% else %}
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      {% endif %}
      {% if post.status and post.status != 'live' %}
        <span class="entry__meta"><em>{{ post.status }}</em></span>
      {% endif %}
    </h3>
    {% if post.summary %}
      <p class="entry__excerpt">{{ post.summary }}</p>
    {% endif %}
  </div>
</li>
```

- [ ] **Step 2: Create the tools listing page**

Write to `_pages/tools.html`:

```liquid
---
layout: archive
title: "Tools"
permalink: /tools/
lede: "Small things I built for myself, listed here in case they're useful to anyone else."
---

<ul class="entries">
{% assign sorted = site.tools | sort: "date" | reverse %}
{% for post in sorted %}
  {% include tool-single.html %}
{% endfor %}
</ul>
```

- [ ] **Step 3: Verify**

Run: `cat _pages/tools.html && cat _includes/tool-single.html`
Expected: both files exist with the content above.

- [ ] **Step 4: Commit**

```bash
git add _pages/tools.html _includes/tool-single.html
git commit -m "feat(tools): add /tools/ listing page and entry include"
```

---

## Task 3: Rewrite `_layouts/home.html` as fully self-contained single-page CV

**Files:**
- Modify (full replace): `_layouts/home.html`

- [ ] **Step 1: Read current file to satisfy editor tool**

Run: `head -3 _layouts/home.html`
Expected: current frontmatter (`layout: default`).

- [ ] **Step 2: Replace the file entirely**

Write to `_layouts/home.html`:

```html
---
layout: default
---
<article class="page">
  <h1>Hang Yang</h1>
  <p>PhD candidate, Department of Mathematics &amp; Statistics, University of Massachusetts Amherst.</p>
  <p>
    <a href="mailto:hangyang@umass.edu">hangyang@umass.edu</a> ·
    <a href="https://scholar.google.com/citations?user=usu8YBAAAAAJ">Google Scholar</a> ·
    <a href="https://github.com/yanghangAI">GitHub</a>
  </p>

  <h2>Research</h2>
  <p>I work on machine learning, invertible neural networks, and robotics.</p>
  <ul>
    <li><strong>Mobile motion capture.</strong> Using a quadruped robot as a mobile platform to record human motion outside the constraints of a fixed studio.</li>
    <li><strong>Invertible neural networks</strong> for lossless information recovery, applied to deep-learning image steganography (<a href="https://arxiv.org/abs/2309.13620">PRIS</a>, <a href="https://arxiv.org/abs/2311.18243">DKiS</a>).</li>
    <li><strong>Medical imaging.</strong> Vision transformers for automated disease detection — <a href="https://link.springer.com/article/10.1007/s13042-022-01676-7">CovidViT</a> reaches 98% on chest X-rays.</li>
    <li><strong>Robotics.</strong> Legged-robot perception, autonomy, and human–robot interaction.</li>
  </ul>

  {{ content }}

  <h2>Publications</h2>
  <ul class="entries">
    {% for post in site.publications reversed %}
      {% include archive-single.html %}
    {% endfor %}
  </ul>

  <h2>Teaching</h2>
  <ul class="entries">
    {% for post in site.teaching reversed %}
      {% include archive-single.html %}
    {% endfor %}
  </ul>

  <h2>Projects</h2>
  <ul class="entries">
    {% for post in site.portfolio %}
      {% include archive-single.html %}
    {% endfor %}
  </ul>

  <h2>Education</h2>
  <div class="register">
    <dl>
      <dt>2024 — present</dt><dd><strong>Ph.D., Mathematics</strong>, University of Massachusetts Amherst</dd>
      <dt>2021 — 2024</dt><dd><strong>M.Sc., Mathematics</strong>, China Agricultural University</dd>
      <dt>2017 — 2021</dt><dd><strong>B.Sc., Mathematics</strong>, China Agricultural University</dd>
    </dl>
  </div>
</article>
```

- [ ] **Step 3: Verify**

Run: `grep -c "<h2>" _layouts/home.html`
Expected: `5` (Research, Publications, Teaching, Projects, Education).

- [ ] **Step 4: Commit**

```bash
git add _layouts/home.html
git commit -m "refactor(home): self-contained single-page CV layout"
```

---

## Task 4: Strip `_pages/about.md` to frontmatter-only

The home layout now owns all home page content. `about.md` only needs to declare the permalink and layout. Any future free-form paragraph would go in its body.

**Files:**
- Modify (full replace): `_pages/about.md`

- [ ] **Step 1: Read current file to satisfy editor tool**

Run: `head -3 _pages/about.md`
Expected: current frontmatter.

- [ ] **Step 2: Replace the file entirely**

Write to `_pages/about.md`:

```markdown
---
layout: home
permalink: /
title: "Hang Yang"
redirect_from:
  - /about/
  - /about.html
---
```

(No body. The home layout renders everything.)

- [ ] **Step 3: Verify**

Run: `wc -l _pages/about.md`
Expected: 7 lines.

- [ ] **Step 4: Commit**

```bash
git add _pages/about.md
git commit -m "refactor(home): strip about.md to frontmatter-only"
```

---

## Task 5: Update `_data/navigation.yml` to a single `Tools` entry

**Files:**
- Modify: `_data/navigation.yml`

- [ ] **Step 1: Read current file**

Run: `cat _data/navigation.yml`
Expected: current content with Publications, Teaching, Portfolio, CV, Cluster, What2Do entries.

- [ ] **Step 2: Replace the file entirely**

Write to `_data/navigation.yml`:

```yaml
# Header navigation. The brand "Hang Yang" already links to /, so Home is implicit.
main:
  - title: "Tools"
    url: /tools/
```

- [ ] **Step 3: Verify**

Run: `grep -c "^  - title:" _data/navigation.yml`
Expected: `1`.

- [ ] **Step 4: Commit**

```bash
git add _data/navigation.yml
git commit -m "refactor(nav): collapse top-level nav to single Tools entry"
```

---

## Task 6: Rename `brutal.css` → `site.css`

The current filename was a leftover from an earlier aesthetic direction. Renaming for clarity. Same content.

**Files:**
- Rename: `assets/css/brutal.css` → `assets/css/site.css`
- Modify: `_layouts/default.html` (update `<link>` reference)

- [ ] **Step 1: Move the file**

Run: `git mv assets/css/brutal.css assets/css/site.css`
Expected: no output, exit 0.

- [ ] **Step 2: Update the stylesheet reference in `_layouts/default.html`**

In `_layouts/default.html`, replace:

```html
  <link rel="stylesheet" href="{{ '/assets/css/brutal.css' | relative_url }}">
```

with:

```html
  <link rel="stylesheet" href="{{ '/assets/css/site.css' | relative_url }}">
```

- [ ] **Step 3: Verify no other references to the old name**

Run: `grep -rn "brutal\.css" . --include="*.html" --include="*.md" --include="*.yml" --include="*.scss" 2>/dev/null | grep -v "^./docs/"`
Expected: no output (the spec/plan under `docs/` may still mention it; ignore those).

- [ ] **Step 4: Commit**

```bash
git add assets/css/site.css _layouts/default.html
git commit -m "refactor(css): rename brutal.css to site.css"
```

---

## Task 7: Delete demo content (collections)

Removes academicpages placeholder posts, talks, and the two demo portfolio entries. Real teaching entries and the real AmazingHand portfolio entry stay.

**Files:**
- Delete: all files under `_posts/`
- Delete: all files under `_talks/`
- Delete: `_portfolio/portfolio-1.md`, `_portfolio/portfolio-2.html`
- Keep: `_portfolio/amazinghand.md`, all of `_teaching/`, all of `_publications/`

- [ ] **Step 1: Inspect what will be removed**

Run: `ls _posts/ _talks/ _portfolio/`
Expected: 5 posts, 4 talks, 3 portfolio files (one real: `amazinghand.md`).

- [ ] **Step 2: Delete**

```bash
git rm -r _posts _talks
git rm _portfolio/portfolio-1.md _portfolio/portfolio-2.html
```

- [ ] **Step 3: Verify**

Run: `ls _portfolio/ && ls _posts _talks 2>&1`
Expected: `_portfolio/` shows only `amazinghand.md`; `_posts` and `_talks` print "No such file or directory".

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: delete academicpages demo posts, talks, and portfolio entries"
```

---

## Task 8: Delete demo pages and obsolete listing pages

Removes academicpages demo and listing pages whose collections have been folded into the home page or eliminated.

**Files (delete all of):**
- `_pages/cv.md`, `_pages/cv-json.md`
- `_pages/portfolio.html`, `_pages/talks.html`, `_pages/teaching.html`
- `_pages/sitemap.md`, `_pages/terms.md`
- `_pages/markdown.md`, `_pages/non-menu-page.md`, `_pages/archive-layout-with-content.md`
- `_pages/page-archive.html`, `_pages/category-archive.html`, `_pages/tag-archive.html`, `_pages/year-archive.html`, `_pages/collection-archive.html`
- `_pages/talkmap.html`

**Files to keep in `_pages/`:**
- `about.md`, `404.md`, `cluster.html`, `tools.html`

- [ ] **Step 1: Delete**

```bash
git rm _pages/cv.md _pages/cv-json.md \
       _pages/portfolio.html _pages/talks.html _pages/teaching.html \
       _pages/sitemap.md _pages/terms.md \
       _pages/markdown.md _pages/non-menu-page.md _pages/archive-layout-with-content.md \
       _pages/page-archive.html _pages/category-archive.html _pages/tag-archive.html \
       _pages/year-archive.html _pages/collection-archive.html \
       _pages/talkmap.html
```

- [ ] **Step 2: Verify**

Run: `ls _pages/`
Expected: exactly four files: `404.md`, `about.md`, `cluster.html`, `tools.html`.

- [ ] **Step 3: Verify nothing in remaining HTML/Liquid still links to a deleted page**

Run:
```bash
grep -rE "/(cv|portfolio|talks|teaching|sitemap|terms|markdown|talkmap|cv-json)/" \
  _layouts/ _includes/ _pages/ _config.yml 2>/dev/null | grep -v "^docs/"
```
Expected: no output. (If the home layout still links to `/publications/` from the "All publications" line, that's fine — `/publications/` is not a page being removed; it was removed in Task 8 above, so we DO need to remove that link too. Re-check.) Actually `_pages/publications.html` is NOT in the deletion list above — confirm with `ls _pages/`. It should NOT exist after Task 8 because we removed listings. **However** publication detail pages at `/publication/<slug>/` still exist (those are auto-rendered from `_publications` via the `single` layout, not from a `_pages/publications.html`).

If the home layout (`_layouts/home.html` from Task 3) still references `/publications/`, remove that line — the home page now lists all publications inline so the "All publications →" link is redundant. Task 3 already omits that link, but re-confirm:

Run: `grep -n "publications/" _layouts/home.html _layouts/single.html`
Expected: no references to `/publications/` listing page; references to individual publication pages via `post.url` are fine.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: delete demo pages and obsolete listing pages"
```

---

## Task 9: Delete academicpages chrome (includes, layouts, sass, data, fonts, old assets)

**Files (delete all of):**

`_includes/` (keep only `masthead.html`, `footer.html`, `archive-single.html`, `tool-single.html`):
- `sidebar.html`, `page__hero.html`, `breadcrumbs.html`, `comments.html`, `comment.html`, `comments-providers/`, `analytics.html`, `analytics-providers/`, `seo.html`, `head.html`, `head/`, `footer/`, `feature_row/`, `gallery/`, `nav_list`, `toc`, `group-by-array`, `read-time.html`, `social-share.html`, `post_pagination.html`, `paginator.html`, `page__taxonomy.html`, `tag-list.html`, `category-list.html`, `archive-single-cv.html`, `archive-single-talk-cv.html`, `archive-single-talk.html`, `cv-template.html`, `scripts.html`, `browser-upgrade.html`, `base_path`, `author-profile.html`

`_layouts/` (keep only `default.html`, `home.html`, `archive.html`, `single.html`):
- `cv-layout.html`, `talk.html`, `splash.html`, `archive-taxonomy.html`, `compress.html`

`_data/`:
- `authors.yml`, `cv.json`, `comments/`, `ui-text.yml`

`_sass/` (entire directory)

`assets/css/`: keep only `site.css`
- delete: `academicons.css`, `academicons.min.css`, `collapse.css`, `main.scss`

`assets/js/` (entire directory)

`assets/fonts/` and `assets/webfonts/` (entire directories — academicons + Font Awesome, no longer referenced)

- [ ] **Step 1: Delete includes**

```bash
git rm _includes/sidebar.html _includes/page__hero.html _includes/breadcrumbs.html \
       _includes/comments.html _includes/comment.html _includes/analytics.html \
       _includes/seo.html _includes/head.html _includes/nav_list _includes/toc \
       _includes/group-by-array _includes/read-time.html _includes/social-share.html \
       _includes/post_pagination.html _includes/paginator.html _includes/page__taxonomy.html \
       _includes/tag-list.html _includes/category-list.html _includes/archive-single-cv.html \
       _includes/archive-single-talk-cv.html _includes/archive-single-talk.html \
       _includes/cv-template.html _includes/scripts.html _includes/browser-upgrade.html \
       _includes/base_path _includes/author-profile.html
git rm -r _includes/comments-providers _includes/analytics-providers \
          _includes/head _includes/footer/ _includes/feature_row _includes/gallery
```

- [ ] **Step 2: Verify `_includes/` contents**

Run: `ls _includes/`
Expected: exactly four files — `archive-single.html`, `footer.html`, `masthead.html`, `tool-single.html`.

- [ ] **Step 3: Delete layouts**

```bash
git rm _layouts/cv-layout.html _layouts/talk.html _layouts/splash.html \
       _layouts/archive-taxonomy.html _layouts/compress.html
```

Run: `ls _layouts/`
Expected: `archive.html`, `default.html`, `home.html`, `single.html`.

- [ ] **Step 4: Delete `_data/` leftovers**

```bash
git rm _data/authors.yml _data/cv.json _data/ui-text.yml
git rm -r _data/comments
```

Run: `ls _data/`
Expected: only `navigation.yml`.

- [ ] **Step 5: Delete `_sass/` entirely**

```bash
git rm -r _sass
```

- [ ] **Step 6: Delete old CSS / JS / fonts**

```bash
git rm assets/css/academicons.css assets/css/academicons.min.css \
       assets/css/collapse.css assets/css/main.scss
git rm -r assets/js assets/fonts assets/webfonts
```

Run: `find assets -type f | sort`
Expected: only `assets/css/site.css` and `assets/cluster-data.json`.

- [ ] **Step 7: Verify nothing remaining references a deleted file**

Run:
```bash
grep -rE "include (sidebar|page__hero|breadcrumbs|comments|analytics|seo|head\.html|head/|footer/|feature_row|gallery|nav_list|toc|read-time|social-share|paginator|category-list|tag-list|archive-single-cv|archive-single-talk|cv-template|scripts\.html|browser-upgrade|base_path|author-profile)" \
  _layouts/ _includes/ _pages/ 2>/dev/null
```
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git commit -m "chore: delete academicpages chrome (includes, layouts, sass, data, js, fonts)"
```

---

## Task 10: Delete dev tooling and unused scripts

**Files (delete all of):**
- `Dockerfile`, `docker-compose.yaml`, `_config_docker.yml`
- `.devcontainer/` (entire directory)
- `package.json`
- `markdown_generator/` (entire directory)
- `talkmap/`, `talkmap.ipynb`, `talkmap_out.ipynb`, `talkmap.py`
- Everything under `scripts/` **except** `update-cluster-dashboard.sh`

**Keep:**
- `Gemfile`, `Gemfile.lock` (GitHub Pages build resolution)
- `scripts/update-cluster-dashboard.sh`
- `_config.yml`, `.gitignore`, `LICENSE`

- [ ] **Step 1: Inventory `scripts/` so we know what we're removing**

Run: `ls scripts/`

- [ ] **Step 2: Delete dev tooling**

```bash
git rm Dockerfile docker-compose.yaml _config_docker.yml package.json
git rm -r .devcontainer
```

- [ ] **Step 3: Delete markdown generator and talkmap artifacts**

```bash
git rm -r markdown_generator talkmap
git rm talkmap.ipynb talkmap_out.ipynb talkmap.py
```

- [ ] **Step 4: Delete `scripts/` except `update-cluster-dashboard.sh`**

```bash
# List first, in case there are no other files:
ls scripts/
# Then remove everything except the dashboard updater:
find scripts -mindepth 1 -maxdepth 1 -type f ! -name 'update-cluster-dashboard.sh' -print -exec git rm {} \;
find scripts -mindepth 1 -maxdepth 1 -type d -print -exec git rm -r {} \;
```

- [ ] **Step 5: Verify**

Run: `ls scripts/ && ls -la | grep -E "Dockerfile|docker-compose|_config_docker|devcontainer|package\.json|markdown_generator|talkmap"`
Expected: `scripts/` contains exactly `update-cluster-dashboard.sh`; the second `ls` produces no output.

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: delete dev tooling and unused scripts (keep cluster dashboard updater)"
```

---

## Task 11: Audit and delete unused images

**Keep:**
- `images/IMG_0037.jpeg` (referenced by `_portfolio/amazinghand.md`)
- `images/profile.png` (user content; may be used later)
- `images/favicon.ico`, `images/favicon.svg`, `images/favicon-32x32.png`, `images/favicon-192x192.png`, `images/favicon-512x512.png`, `images/apple-touch-icon-180x180.png`, `images/manifest.json` (browser tab / PWA functionality)

**Delete:**
- `images/500x300.png`, `images/bio-photo.jpg`, `images/bio-photo-2.jpg`, `images/editing-talk.png`
- `images/themes/` (entire directory — academicpages theme preview screenshots)

- [ ] **Step 1: Confirm reference map before deleting**

Run:
```bash
grep -rohE "['\"](/?images/[^'\"]+)['\"]" _publications/ _teaching/ _portfolio/ _pages/ _layouts/ _includes/ _config.yml 2>/dev/null | sort -u
```
Expected: only references to `IMG_0037.jpeg`. (If anything else appears, do NOT delete the corresponding file — investigate first.)

- [ ] **Step 2: Delete**

```bash
git rm images/500x300.png images/bio-photo.jpg images/bio-photo-2.jpg images/editing-talk.png
git rm -r images/themes
```

- [ ] **Step 3: Verify**

Run: `ls images/`
Expected: `IMG_0037.jpeg`, `apple-touch-icon-180x180.png`, `favicon-192x192.png`, `favicon-32x32.png`, `favicon-512x512.png`, `favicon.ico`, `favicon.svg`, `manifest.json`, `profile.png`.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: prune academicpages demo images"
```

---

## Task 12: Clean `_config.yml` of obsolete theme settings

Strip out all academicpages-specific settings: site_theme, site_theme_dark, sidebar config, breadcrumbs, comments providers, analytics providers, search, atom_feed UI, mathjax, paginate, and all author social-media fields the user has not filled in (most of them are empty placeholders). Keep: site identity (title, name, description, url, baseurl), repository, the `collections:` block, the `defaults:` block, plugins, markdown engine, `permalink`, and any timezone/locale settings. Remove `publication_category` since the home page lists all publications inline without category grouping.

**Files:**
- Modify: `_config.yml`

- [ ] **Step 1: Back up current config (for diffing during review)**

Run: `cp _config.yml /tmp/_config.yml.before`

- [ ] **Step 2: Read the current `_config.yml`**

Use the Read tool on `_config.yml`. Identify which sections are academicpages-specific (anything related to themes, sidebar, breadcrumbs, search, atom_feed UI options, comments, analytics, the `author:` block beyond name/email/scholar/github, mathjax, paginate, `publication_category`).

- [ ] **Step 3: Write the cleaned `_config.yml`**

Replace `_config.yml` with a minimal version preserving only what Jekyll/GitHub Pages and the four remaining layouts actually use. Reference template:

```yaml
# Site
locale: "en-US"
title: "Hang Yang"
name: "Hang Yang"
description: "Hang Yang — PhD candidate in Mathematics at UMass Amherst. Machine learning, invertible neural networks, robotics."
url: https://yanghangAI.github.io
baseurl: ""
repository: "yanghangAI/yanghangAI.github.io"
timezone: America/New_York

# Author — used only by the footer
author:
  name: "Hang Yang"
  email: "hangyang@umass.edu"
  github: "yanghangAI"
  googlescholar: "https://scholar.google.com/citations?user=usu8YBAAAAAJ"

# Build
markdown: kramdown
highlighter: rouge
permalink: /:categories/:year/:month/:day/:title/

include:
  - _pages

# Plugins (subset GitHub Pages supports)
plugins:
  - jekyll-feed
  - jekyll-sitemap
  - jekyll-redirect-from

# Collections
collections:
  publications:
    output: true
    permalink: /:collection/:path/
  teaching:
    output: true
    permalink: /:collection/:path/
  portfolio:
    output: true
    permalink: /:collection/:path/
  tools:
    output: true
    permalink: /:collection/:path/

# Defaults
defaults:
  - scope:
      path: ""
      type: pages
    values:
      layout: archive
  - scope:
      path: ""
      type: publications
    values:
      layout: single
  - scope:
      path: ""
      type: teaching
    values:
      layout: single
  - scope:
      path: ""
      type: portfolio
    values:
      layout: single
  - scope:
      path: ""
      type: tools
    values:
      layout: single

# Exclude from build
exclude:
  - Gemfile
  - Gemfile.lock
  - LICENSE
  - README.md
  - docs/
  - scripts/
  - vendor/
  - node_modules/
```

- [ ] **Step 4: Diff against the backup to be sure nothing critical was lost**

Run: `diff /tmp/_config.yml.before _config.yml | head -80`
Visually scan: anything you removed should be a theme setting, a placeholder social-media URL, or `publication_category`. If you accidentally dropped `url`, `baseurl`, `repository`, the `collections:` block, or one of the `defaults:` scopes — restore it.

- [ ] **Step 5: Verify the collections still match what the home layout iterates**

Run: `grep -E "site\.(publications|teaching|portfolio|tools)" _layouts/home.html _pages/tools.html`
Expected: each `site.*` reference matches a collection in `_config.yml`.

- [ ] **Step 6: Commit**

```bash
git add _config.yml
git commit -m "chore(config): strip academicpages settings; keep only what's used"
```

---

## Task 13: Replace `README.md`, delete `CONTRIBUTING.md`

The current README is the academicpages template instructions. Replace with a one-paragraph project README describing what this repo is.

**Files:**
- Modify (full replace): `README.md`
- Delete: `CONTRIBUTING.md`

- [ ] **Step 1: Replace `README.md`**

Write to `README.md`:

```markdown
# yanghangAI.github.io

Personal academic site for Hang Yang, PhD candidate in Mathematics at UMass Amherst. Built with Jekyll on GitHub Pages. The home page is a single-scrolling CV; `/tools/` lists personal projects (currently the live cluster dashboard at `/cluster/` and an external link to [What2Do](https://yanghangAI.github.io/what2do/)).

The cluster dashboard is fed by `scripts/update-cluster-dashboard.sh`, which runs via cron on the UMass Unity HPC cluster and pushes a refreshed `assets/cluster-data.json` every 15 minutes.
```

- [ ] **Step 2: Delete `CONTRIBUTING.md`**

```bash
git rm CONTRIBUTING.md
```

- [ ] **Step 3: Verify**

Run: `cat README.md && ls CONTRIBUTING.md 2>&1`
Expected: new README printed; `ls` reports "No such file or directory".

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: replace academicpages README with project README; remove CONTRIBUTING"
```

---

## Task 14: Push and verify GitHub Pages build

This is the only point at which the redesign actually goes live. Push, then watch the Pages build, then smoke-test the URLs.

- [ ] **Step 1: Final local sanity check**

```bash
git status
git log --oneline origin/master..HEAD
```
Expected: clean working tree; commits 1–13 listed.

- [ ] **Step 2: Verify no Liquid reference loose ends**

Run:
```bash
grep -rE "\{% include " _layouts/ _includes/ _pages/ 2>/dev/null | awk -F'include ' '{print $2}' | awk '{print $1}' | sort -u
```
Expected: every include name printed should correspond to a file in `_includes/` (exactly: `archive-single.html`, `footer.html`, `masthead.html`, `tool-single.html`).

For each name in the output, verify with:
```bash
ls _includes/<name>
```

If any printed name doesn't resolve to a file, fix the referencing layout/page before pushing.

- [ ] **Step 3: Push**

```bash
git push origin master
```

- [ ] **Step 4: Wait for and check the Pages build**

```bash
gh run list --branch master --limit 3
```
If the most recent run is in progress, watch it:
```bash
gh run watch
```
Expected: build status `success`. If `failure`, run `gh run view <id> --log-failed` and fix the error.

- [ ] **Step 5: Smoke-test the live URLs**

```bash
for path in / /tools/ /cluster/ /publication/2025-01-01-dkis/; do
  printf "%-40s " "$path"
  curl -sI "https://yanghangAI.github.io$path" | head -1
done
```
Expected: each line ends in `HTTP/2 200`.

- [ ] **Step 6: Spot-check rendered content**

```bash
curl -s https://yanghangAI.github.io/ | grep -E "<h1>|<h2>" | head -10
```
Expected: `<h1>Hang Yang</h1>` and the five `<h2>` section headings (Research, Publications, Teaching, Projects, Education).

```bash
curl -s https://yanghangAI.github.io/tools/ | grep -E "Cluster dashboard|What2Do"
```
Expected: both tool titles appear.

- [ ] **Step 7: Commit any fixups, if needed**

If Step 4, 5, or 6 surfaces an issue, fix and push. Each fix is its own commit: `fix(build): <what>`.

---

## Self-review checklist

(Already performed by the planner; recorded here for the executor's reference.)

- **Spec coverage:** every section of the spec maps to a task. URLs (T1–T8), home page (T3), tools (T1, T2), cluster (no change needed; T9 leaves it intact), publication detail (no change needed; `single` layout already implements it), deletions (T7–T11), config (T12), README (T13), risks (verifications in T6/T8/T9/T14).
- **Placeholder scan:** none — every step has concrete file paths, code, or commands.
- **Type consistency:** include names match (`archive-single.html`, `tool-single.html`); layout names match (`default`, `home`, `archive`, `single`); collection names match between `_config.yml`, the home layout, and the tools page.
- **Risk #3 (sass deletion):** Task 9 Step 6 deletes `main.scss` together with `_sass/`. Task 6 already verified no other `.scss` file imports from `_sass/`.
