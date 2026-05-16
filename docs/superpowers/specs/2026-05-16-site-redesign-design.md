# Site Redesign — Two-Tier Academic + Tools

**Date:** 2026-05-16
**Repo:** `yanghangAI/yanghangAI.github.io`
**Goal:** Replace the academicpages-based multi-page site with a content-first single-page academic CV plus a small tools directory. Keep what's real; delete what's placeholder.

## Audience & purpose

Two audiences served by one URL:

1. **Academic visitors** (faculty, collaborators) — default landing experience. Want to skim research areas, find publications, see teaching, scan credentials.
2. **Me / power users** — personal tools (HPC dashboard, etc.) that need a stable home.

Design priority: content and logic first, design second. Plain academic aesthetic (system Georgia, classic blue links, ~720px measure). No web fonts, no hero treatments, no accent flourishes.

## Site shape

```
/                       single-page CV (home)
/tools/                 directory of personal tools
/cluster/               cluster dashboard (kept at top level for bookmarking)
/publication/<slug>/    one page per paper (permalinks for citations)
```

**Nav (top of every page):** brand `Hang Yang` (links home) · `Tools`. Two items total. Cluster and What2Do are one click deeper, listed on `/tools/`. The cluster URL stays at top level so external bookmarks survive.

## Home page (`/`)

Single scroll, sections separated by h2 only (no rules, no counters). Order:

1. **Name** — `<h1>Hang Yang</h1>`
2. **Role line** — one sentence: *PhD candidate, Department of Mathematics & Statistics, University of Massachusetts Amherst.*
3. **Contact** — inline single line: `hangyang@umass.edu · Google Scholar · GitHub`
4. **Research** — one short paragraph + bulleted list of 3–4 interests (mobile mocap, invertible nets, medical AI, robotics), one sentence each
5. **Publications** — full reverse-chron list (currently 3), rendered from `_publications` collection. Format per entry: `Year. Title (link). Venue. [Paper] [Code]`. No "selected" qualifier.
6. **Teaching** — list of TA entries, rendered from `_teaching` collection. Format: `Term Year. Course code, name. Institution.`
7. **Projects** — short list (currently just AmazingHand, from `_portfolio/amazinghand.md`). Each entry: title + 1–2 sentence description + link.
8. **Education** — three lines: degree, institution, years.

No "Talks" section (current entries are all placeholders). No "Blog/News" section. No CV PDF link.

## Tools page (`/tools/`)

Rendered from a new `_tools/` collection. Each tool is one markdown file:

```yaml
---
title: Cluster dashboard
collection: tools
url: /cluster/        # internal path or external URL
date: 2026-05-16
status: live          # live | wip | archived
summary: Live snapshot of my UMass Unity HPC queue.
---
```

Page renders reverse-chron, one entry per line: `YYYY-MM. Title (link). Summary text.` `wip` and `archived` get a small italic tag after the title. Adding tool #6 is a one-file drop with no template changes.

**Initial entries:**
- `_tools/cluster-dashboard.md` → url `/cluster/`
- `_tools/what2do.md` → url `https://yanghangAI.github.io/what2do/` (external)

## Cluster page (`/cluster/`)

Unchanged in structure from current implementation. Continues to fetch `assets/cluster-data.json` (updated by the cron-driven `scripts/update-cluster-dashboard.sh`) and render four h2-led tables: Active jobs, Recent jobs (24h), Storage, Partitions. Plain academic styling from the shared stylesheet.

## Publication detail pages

Each `_publications/*.md` still renders at `/publication/<slug>/` via the `single` layout. Format: title, kicker (collection · year), venue/date subline, paper/slides/bibtex/code link row, abstract/body, citation block at bottom.

## Layouts

Only four layouts after pruning:

- `_layouts/default.html` — HTML shell, masthead, footer, stylesheet
- `_layouts/home.html` — single-page CV; loops over `_publications`, `_teaching`, `_portfolio` for inline sections
- `_layouts/archive.html` — generic listing page (used by `/tools/` and `/cluster/`)
- `_layouts/single.html` — individual publication pages

Deleted layouts: `_layouts/cv-layout.html`, `_layouts/talk.html`, `_layouts/splash.html`, `_layouts/archive-taxonomy.html`, `_layouts/compress.html`. The current `home.html` is rewritten in place to render the single-page CV described above (no longer mixes a header hero with `{{ content }}` from `about.md`).

## Includes

Only four includes after pruning:

- `_includes/masthead.html` — brand + 1-item nav (Tools)
- `_includes/footer.html` — copyright + contact line
- `_includes/archive-single.html` — one entry for `_publications` and `_teaching`
- `_includes/tool-single.html` — one entry for `_tools` (new)

All other academicpages includes deleted: `sidebar.html`, `page__hero.html`, `breadcrumbs.html`, `comments*.html`, `analytics*.html`, `seo.html`, `head/`, `footer/`, `feature_row/`, `gallery/`, `comments-providers/`, `analytics-providers/`, `nav_list`, `toc`, `group-by-array`, `read-time.html`, `social-share.html`, `post_pagination.html`, `paginator.html`, `page__taxonomy.html`, `tag-list.html`, `category-list.html`, `archive-single-cv.html`, `archive-single-talk-cv.html`, `archive-single-talk.html`, `cv-template.html`, `scripts.html`, `browser-upgrade.html`, `comment.html`, `base_path`, `head.html`.

## Stylesheet

Single hand-written file: `assets/css/site.css` (rename from current `brutal.css`). System Georgia, ~720px measure, classic web-blue links (#0645ad), 1px hairline rules only where needed (under headings of section borders). No external font loads. Dashboard tables use system monospace.

## Deletions

**Pages (`_pages/`):**
- `cv.md`, `cv-json.md`, `portfolio.html`, `talks.html`, `teaching.html`, `sitemap.md`, `terms.md`, `markdown.md`, `non-menu-page.md`, `archive-layout-with-content.md`, `page-archive.html`, `category-archive.html`, `tag-archive.html`, `year-archive.html`, `talkmap.html`, `collection-archive.html`

**Collections content:**
- All `_posts/*` (5 demo posts)
- All `_talks/*` (4 demo talks)
- `_portfolio/portfolio-1.md`, `_portfolio/portfolio-2.html` (keep `amazinghand.md`)

**Data:**
- `_data/authors.yml`, `_data/cv.json`, `_data/comments/`, `_data/ui-text.yml`
- `_data/navigation.yml` rewritten to contain only the Tools link

**Theme & assets:**
- `_sass/` directory (entire old theme)
- `assets/css/*` except the new `site.css`
- `assets/js/` entirely
- `assets/fonts/`, `assets/webfonts/` — unless needed by remaining markdown content; audit and delete the unused

**Other:**
- `talkmap.ipynb`, `talkmap_out.ipynb`, `talkmap.py`, `talkmap/`
- `markdown_generator/`
- `scripts/` everything except `update-cluster-dashboard.sh`
- `Dockerfile`, `docker-compose.yaml`, `_config_docker.yml`, `.devcontainer/`, `package.json`
- **Keep `Gemfile`** (used by GitHub Pages build resolution) and `Gemfile.lock` if present
- `CONTRIBUTING.md`, `README.md` — replace with one-paragraph README of the user's own
- `images/` — audit, delete academicpages stock imagery. Keep `profile.png`, `IMG_0037.jpeg`, anything else referenced by real content.

**Collections config (`_config.yml`):**
- Remove `posts`, `talks` collection definitions (or set `output: false` if Jekyll complains)
- Remove `publication_category` (only 3 papers, all the same category; flat list is cleaner)
- Remove all theme-specific config (site_theme, sidebar, breadcrumbs, comments, search, analytics, atom_feed, etc.)
- Add `tools` collection definition

## Risks & gotchas

1. **GitHub Pages build.** No local Ruby/Jekyll/Docker on the Unity HPC node where edits happen; first verification is when GH Pages rebuilds. Spec is structured so removed includes are also removed from any layout that referenced them.
2. **Permalink stability.** `/publication/<slug>/` URLs are preserved (papers are already cited externally). `/cluster/` is preserved. `/portfolio/`, `/talks/`, `/teaching/`, `/cv/` URLs go away — acceptable, low/no external link risk.
3. **`_sass/` is referenced by Jekyll only if a SCSS file `@import`s from it.** With the old `assets/css/main.scss` deleted, the directory becomes unused and safe to remove. Confirm no remaining `.scss` file imports from `_sass/` before deleting.
4. **`_data/navigation.yml`.** Currently has 4 entries; new shape is 1. Masthead already iterates it, so just edit the data file — no template change required.
5. **`_data/ui-text.yml`.** Possibly referenced by remaining academicpages partials being deleted. Should be safe to remove with them; if any remaining template errors on a missing key, replace the reference.

## Out of scope

- Visual polish beyond "looks like a normal academic page"
- Mobile menu drawer (nav is 2 items, will wrap)
- Search, comments, analytics, RSS UI (Jekyll still emits `feed.xml` from the plugin; no UI link)
- Dark mode
- i18n
- New tools beyond the 2 existing
- Talks, blog, CV PDF
