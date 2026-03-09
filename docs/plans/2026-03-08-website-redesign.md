# Website Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Completely replace the current Jekyll Minimal Mistakes site with an al-folio-based high-caliber research website using a dark theme and cyan/teal accent.

**Architecture:** Download al-folio from GitHub, replace all existing theme files, customize SCSS for dark palette, populate all content (About, Publications, CV, Projects, Blog), and configure GitHub Actions for deployment since al-folio uses `jekyll-scholar` which is not in GitHub Pages' safe plugin list.

**Tech Stack:** Jekyll, al-folio theme, jekyll-scholar (BibTeX publications), GitHub Actions (deploy), Ruby/Bundler

---

## Pre-flight: Understand the repo

The repo lives at `/home/hassam/hassamsheikh.github.io` and is a GitHub Pages user page (served from `master` branch). After this plan, GitHub Pages will be configured to serve from the `gh-pages` branch (built by GitHub Actions). The user must change their repo's Pages source to `gh-pages` after Task 2 completes.

Key files to preserve:
- `assets/images/bio-photo.jpg` (profile photo)
- `HassamSheikh_Resume_Feb2025.pdf` (linked from CV page)
- `.git/` directory

---

### Task 1: Download al-folio and replace repo content

**Goal:** Get a clean al-folio base into the repo.

**Files:**
- Replace: everything except `.git/`, `assets/images/bio-photo.jpg`, `HassamSheikh_Resume_Feb2025.pdf`, `docs/`

**Step 1: Clone al-folio into a temp directory**

```bash
git clone --depth 1 https://github.com/alshedivat/al-folio.git /tmp/al-folio
```

**Step 2: Preserve files we need to keep**

```bash
cp /home/hassam/hassamsheikh.github.io/assets/images/bio-photo.jpg /tmp/bio-photo.jpg
cp /home/hassam/hassamsheikh.github.io/HassamSheikh_Resume_Feb2025.pdf /tmp/HassamSheikh_Resume_Feb2025.pdf
```

**Step 3: Remove all existing tracked files (except .git and docs)**

```bash
cd /home/hassam/hassamsheikh.github.io
find . -not -path './.git/*' -not -path './docs/*' -not -name '.git' -maxdepth 1 -type f -delete
find . -not -path './.git/*' -not -path './docs/*' -not -name '.git' -not -name 'docs' -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
```

**Step 4: Copy al-folio files into the repo**

```bash
cp -r /tmp/al-folio/. /home/hassam/hassamsheikh.github.io/
rm -rf /home/hassam/hassamsheikh.github.io/.git/config  # don't overwrite git config
# restore git config from backup (al-folio clone may have overwritten assets)
```

Actually, use rsync to be safe:

```bash
rsync -av --exclude='.git' /tmp/al-folio/ /home/hassam/hassamsheikh.github.io/
```

**Step 5: Restore preserved files**

```bash
mkdir -p /home/hassam/hassamsheikh.github.io/assets/img
cp /tmp/bio-photo.jpg /home/hassam/hassamsheikh.github.io/assets/img/prof_pic.jpg
cp /tmp/bio-photo.jpg /home/hassam/hassamsheikh.github.io/assets/images/bio-photo.jpg
cp /tmp/HassamSheikh_Resume_Feb2025.pdf /home/hassam/hassamsheikh.github.io/assets/pdf/HassamSheikh_Resume_Feb2025.pdf
mkdir -p /home/hassam/hassamsheikh.github.io/assets/pdf
cp /tmp/HassamSheikh_Resume_Feb2025.pdf /home/hassam/hassamsheikh.github.io/assets/pdf/HassamSheikh_Resume_Feb2025.pdf
```

**Step 6: Install Ruby dependencies**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle install
```

Expected: All gems install without error. If `jekyll-scholar` fails, run `gem install jekyll-scholar` first.

**Step 7: Verify Jekyll builds**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | tail -20
```

Expected: `done in X seconds` with no fatal errors.

**Step 8: Commit**

```bash
cd /home/hassam/hassamsheikh.github.io
git add -A
git commit -m "feat: install al-folio theme as base

Replace Minimal Mistakes with al-folio for high-caliber academic site.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Configure GitHub Actions deployment

**Goal:** Set up al-folio's deployment workflow so GitHub Actions builds and pushes to `gh-pages`.

**Files:**
- Modify: `.github/workflows/deploy.yml` (already exists from al-folio copy)

**Step 1: Verify the deploy workflow exists**

```bash
cat /home/hassam/hassamsheikh.github.io/.github/workflows/deploy.yml
```

Expected: A workflow that builds Jekyll and deploys to `gh-pages` branch.

**Step 2: Update deploy workflow to use gh-pages branch for user pages**

The al-folio deploy.yml already handles this. Verify the deploy action target is `gh-pages`. Open `.github/workflows/deploy.yml` and confirm the `JamesIves/github-pages-deploy-action` step has:

```yaml
with:
  branch: gh-pages
  folder: _site
```

If not present, add it.

**Step 3: Commit workflow**

```bash
cd /home/hassam/hassamsheikh.github.io
git add .github/
git commit -m "ci: configure GitHub Actions deployment to gh-pages branch

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

**Step 4: Manual action required (inform user)**

After pushing, the user must go to their GitHub repo Settings > Pages > Source and change it from `master` to `gh-pages` branch.

---

### Task 3: Configure `_config.yml`

**Goal:** Replace all placeholder config values with Hassam's real information and enable dark mode.

**Files:**
- Modify: `_config.yml`

**Step 1: Replace `_config.yml` with the following content**

```yaml
# -----------------------------------------------------------------------------
# Site settings
# -----------------------------------------------------------------------------

title: blank
first_name: Hassam
middle_name: U.
last_name: Sheikh
email: hassamsheikh1@gmail.com
description: >
  Software Engineer – RL at Anyscale. Research in distributed RL,
  multi-agent systems, and ensemble methods. PhD from UCF.
footer_text: >
  Powered by <a href="https://jekyllrb.com/" target="_blank">Jekyll</a>
  with <a href="https://github.com/alshedivat/al-folio">al-folio</a> theme.

icon: ⚡️

url: https://hassamsheikh.github.io
baseurl: ""
last_updated: true
impressum_path:

# Theme settings
repo_theme_light: default
repo_theme_dark: dark
repo_trophies:
  enabled: true
  theme_light: flat
  theme_dark: gitdimmed

# -----------------------------------------------------------------------------
# RSS Feed
# -----------------------------------------------------------------------------

rss_icon: true

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------

navbar_fixed: true
footer_fixed: false
search_enabled: true
search:
  highlight: true
bib_search:
  enabled: true
  weight: 10

max_width: 950px

# -----------------------------------------------------------------------------
# Open Graph & Schema.org
# -----------------------------------------------------------------------------

serve_og_meta: false
serve_schema_org: false
og_image:

# -----------------------------------------------------------------------------
# Social integration
# -----------------------------------------------------------------------------

github_username: HassamSheikh
gitlab_username:
x_username:
linkedin_username: hassam-ullah-sheikh
scholar_userid: QTCAAGQAAAAJ
semanticscholar_id:
whatsapp_number:
orcid_id:
medium_username:
quora_username:
publons_id:
lattes_id:
osf_id:
research_gate_profile:
scopus_id:
blogger_url:
work_url:
keybase_username:
wikidata_id:
wikipedia_id:
dblp_url: https://dblp.uni-trier.de/pid/234/8668.html
stackoverflow_id:
kaggle_id:
lastfm_id:
spotify_id:
pinterest_username:
unsplash_username:
instagram_id:
facebook_id:
discord_id:
zotero_username:
wechat_qr:

contact_note:

# -----------------------------------------------------------------------------
# Analytics and verifications
# -----------------------------------------------------------------------------

google_analytics:
cronitor_analytics:
pirsch_analytics:
openpanel_analytics:

google_site_verification:
bing_site_verification:

# -----------------------------------------------------------------------------
# Blog
# -----------------------------------------------------------------------------

blog_name: Notes
blog_description: Thoughts on RL, systems, and engineering
permalink: /blog/:year/:title/
lsi: false
future: false
markdown: kramdown

pagination:
  enabled: true
  per_page: 5
  permalink: "/page/:num/"
  title: ":title - page :num"
  sort_field: "date"
  sort_reverse: true
  trail:
    before: 1
    after: 1

related_blog_posts:
  enabled: true
  max_related: 5

imagemagick:
  enabled: false

media_insertion:
  enabled: true

# -----------------------------------------------------------------------------
# Collections
# -----------------------------------------------------------------------------

collections:
  news:
    defaults:
      layout: post
    output: false
    permalink: /news/:path/
  projects:
    output: true
    permalink: /projects/:path/

announcements:
  enabled: true
  scrollable: true
  limit: 5

latest_posts:
  enabled: true
  scrollable: false
  limit: 3

# -----------------------------------------------------------------------------
# Jekyll settings
# -----------------------------------------------------------------------------

whitelist:
  - jekyll-paginate-v2
  - jekyll-scholar
  - jekyll-jupyter-notebook

plugins:
  - jekyll-archives
  - jekyll-email-protect
  - jekyll-feed
  - jekyll-get-json
  - jekyll-imagemagick
  - jekyll-jupyter-notebook
  - jekyll-link-attributes
  - jekyll-minifier
  - jekyll-paginate-v2
  - jekyll/scholar
  - jekyll-sitemap
  - jekyll-toc
  - jekyll-twitter-plugin
  - jemoji

# Markdown and syntax highlight
highlighter: rouge
markdown: kramdown
kramdown:
  input: GFM
  syntax_highlighter_opts:
    css_class: "highlight"
    span:
      line_numbers: false
    block:
      line_numbers: false
      start_line: 1

sass:
  style: compressed

# -----------------------------------------------------------------------------
# Jekyll Archives
# -----------------------------------------------------------------------------

jekyll-archives:
  enabled:
    - year
    - tags
    - categories
  layout: archive
  permalinks:
    year: "/blog/:year/"
    tag: "/blog/tag/:name/"
    category: "/blog/category/:name/"

display_tags: ["reinforcement-learning", "multi-agent", "systems", "research"]
display_categories: []

# -----------------------------------------------------------------------------
# Jekyll Scholar
# -----------------------------------------------------------------------------

scholar:
  last_name: [Sheikh]
  first_name: [Hassam, H.]
  style: apa
  locale: en
  source: /_bibliography/
  bibliography: papers.bib
  bibliography_template: bib
  bibtex_filters:
    - superscript
    - html_escape
  replace_strings: true
  join_strings: true
  details_dir: bibliography
  details_layout: bibtex.html
  details_link: Details
  query: "@*"
  group_by: year
  group_order: descending

enable_publication_thumbnails: true
enable_publication_badges:
  altmetric: false
  dimensions: false
  google_scholar: false
enable_tooltips: true

# -----------------------------------------------------------------------------
# Optional Features
# -----------------------------------------------------------------------------

enable_google_analytics: false
enable_cronitor_analytics: false
enable_google_verification: false
enable_bing_verification: false
enable_masonry: true
enable_math: true
enable_tooltips: true
enable_darkmode: true
enable_navbar_social: false
enable_project_categories: true
enable_medium_zoom: true
enable_progressbar: true
enable_video_embedding: true

# -----------------------------------------------------------------------------
# Library versions
# -----------------------------------------------------------------------------

bootstrap-table:
  version: "1.22.1"
chartjs:
  version: "4.4.1"
d3:
  version: "7.8.5"
  integrity: "sha256-1rA678n2xEx7x4cTZ5x4wpDP3o6FiSgkSHdKCMvPHIk="
highlightjs:
  version: "11.9.0"
  theme: "github.min"
  theme_dark: "github-dark.min"
jquery:
  version: "3.6.0"
  integrity: "sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4="
leaflet:
  version: "1.9.4"
  integrity:
    css: "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    js: "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV/XHeaster="
mathjax:
  version: "3.2.0"
masonry:
  version: "4.2.2"
  integrity: "sha256-Nn1q/fx0H7SNLZMQ5Hw5JLAToi5qCFF0FIieABbFC0U="
mdb:
  version: "4.20.0"
  integrity:
    css: "sha256-jpjYvU3G3N6nrrBwxyxHQbSGY0E3SJ8crFVd2OQkZUE="
    js: "sha256-NdbiivsvWt7VYCt6hYNT3h/th9vSTL4EDWeGs5SN3DA="
medium_zoom:
  version: "1.0.8"
  integrity: "sha256-7PhEpEWEW0XXQ0k6kQrPKwuoIomz8R8IYyuU1Qew4BI="

# -----------------------------------------------------------------------------
# Get external JSON data
# -----------------------------------------------------------------------------

jekyll_get_json:
  - data: resume
    json: assets/json/resume.json
jsonresume:
  - basics
  - work
  - volunteer
  - education
  - certificates
  - publications
  - skills
  - languages
  - interests
  - references
  - projects

include:
  - _pages

exclude:
  - bin/
  - CHANGELOG.md
  - CODE_OF_CONDUCT.md
  - CONTRIBUTING.md
  - docs/
  - Gemfile
  - Gemfile.lock
  - LICENSE
  - purgecss.config.js
  - README.md
  - vendor
```

**Step 2: Verify Jekyll builds without error**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | tail -20
```

Expected: No fatal errors.

**Step 3: Commit**

```bash
git add _config.yml
git commit -m "feat: configure site with personal info and dark mode

Set name, email, social links, scholar ID, and enable dark mode.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Apply custom dark theme with teal accent

**Goal:** Override al-folio's default colors with `#0d1117` background and `#00d4aa` accent.

**Files:**
- Create: `_sass/_custom-theme.scss`
- Modify: `_sass/_themes.scss` (or `assets/css/main.scss` — check which exists after al-folio copy)

**Step 1: Find where al-folio defines theme CSS variables**

```bash
grep -r "background-color" /home/hassam/hassamsheikh.github.io/_sass/ | head -20
grep -r "\.dark" /home/hassam/hassamsheikh.github.io/_sass/ | head -10
```

**Step 2: Create `_sass/_custom-theme.scss`**

```scss
// Custom dark theme overrides for hassamsheikh.github.io
// Palette: #0d1117 bg, #161b22 surface, #00d4aa accent, #e6edf3 text

html[data-theme="dark"],
:root {
  // Backgrounds
  --global-bg-color: #0d1117;
  --global-code-bg-color: #161b22;
  --global-card-bg-color: #161b22;

  // Text
  --global-text-color: #e6edf3;
  --global-text-color-light: #8b949e;

  // Theme accent (teal)
  --global-theme-color: #00d4aa;
  --global-hover-color: #00d4aa;
  --global-hover-text-color: #0d1117;
  --global-footer-bg-color: #010409;
  --global-footer-text-color: #8b949e;
  --global-footer-link-color: #00d4aa;
  --global-distill-app-color: #00d4aa;
  --global-divider-color: #30363d;

  // Navbar
  --global-navbar-bg-color: #010409;
  --global-navbar-link-color: #e6edf3;
  --global-navbar-link-hover-color: #00d4aa;

  // Links
  a {
    color: #00d4aa;
    &:hover {
      color: darken(#00d4aa, 10%);
    }
  }

  // Badges / venue labels
  .badge {
    background-color: rgba(0, 212, 170, 0.15);
    color: #00d4aa;
    border: 1px solid rgba(0, 212, 170, 0.3);
  }

  // Code blocks
  pre, code {
    background-color: #161b22;
    border-color: #30363d;
  }

  // Cards
  .card {
    background-color: #161b22;
    border-color: #30363d;
    &:hover {
      border-color: #00d4aa;
      box-shadow: 0 4px 20px rgba(0, 212, 170, 0.15);
    }
  }

  // Publication entries
  .bibliography li {
    border-left: 3px solid transparent;
    padding-left: 1rem;
    transition: border-color 0.2s;
    &:hover {
      border-left-color: #00d4aa;
    }
  }

  // News items
  .news article {
    border-bottom: 1px solid #30363d;
  }

  // Profile
  .profile img {
    border: 3px solid #00d4aa;
    border-radius: 50%;
  }
}

// Force dark mode as default (overrides system preference toggle)
:root {
  color-scheme: dark;
}

// Typography improvements
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-weight: 600;
}

code, pre, .monospace {
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}

// Venue badges on publications page
.abbr .badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 0.25rem 0.5rem;
}
```

**Step 3: Import the custom theme in the main SCSS entry point**

Find the main SCSS file:

```bash
ls /home/hassam/hassamsheikh.github.io/assets/css/
```

Open `assets/css/main.scss` and add at the end:

```scss
@import "custom-theme";
```

**Step 4: Add Inter and JetBrains Mono fonts to `_includes/head/custom.html`**

If the file doesn't exist, create it:

```html
<!-- Custom fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

**Step 5: Set dark mode as default in `_config.yml`**

Confirm `enable_darkmode: true` is set (done in Task 3).

Also find the theme toggle default. In al-folio this is typically in `_includes/header.html` or a JS file. Search:

```bash
grep -r "light\|dark\|theme" /home/hassam/hassamsheikh.github.io/_includes/ --include="*.html" -l
```

In the relevant JS file or `_includes/scripts/theme.html`, ensure the default is `dark`:

```javascript
// If no preference stored, default to dark
if (!localStorage.getItem('theme')) {
  document.documentElement.setAttribute('data-theme', 'dark');
}
```

**Step 6: Verify build**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | tail -5
```

**Step 7: Commit**

```bash
git add _sass/_custom-theme.scss assets/css/main.scss _includes/head/custom.html
git commit -m "feat: apply custom dark theme with teal accent

Background #0d1117, accent #00d4aa, Inter + JetBrains Mono fonts.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Write the About page

**Goal:** Replace al-folio's placeholder about page with Hassam's real bio, current position, and news items.

**Files:**
- Modify: `_pages/about.md`
- Create: `_news/announcement_1.md` through `_news/announcement_5.md`

**Step 1: Replace `_pages/about.md`**

```markdown
---
layout: about
title: about
permalink: /
subtitle: >
  Software Engineer – Reinforcement Learning · <a href="https://www.anyscale.com">Anyscale</a>

profile:
  align: right
  image: prof_pic.jpg
  image_circular: true
  more_info: >
    <p>Orlando, Florida</p>
    <p><a href="mailto:hassamsheikh1@gmail.com">hassamsheikh1@gmail.com</a></p>

news: true
selected_papers: false
social: true
---

I am a Software Engineer on the Reinforcement Learning team at [Anyscale](https://www.anyscale.com), where I am the technical owner of [RLlib](https://docs.ray.io/en/latest/rllib/index.html) — the industry-standard distributed RL library. My work focuses on stability, performance, and correctness across RLlib's distributed training and inference stack.

Previously, I was a Research Scientist at **Intel Labs** (2020–2024), where I published at ICML, ICLR, IJCNN, AAMAS, and COMPSAC on multi-agent reinforcement learning, ensemble diversity, and intrinsic reward learning.

I hold a **Ph.D. in Computer Science** from the University of Central Florida, where I was advised by [Ladislau Bölöni](http://www.cs.ucf.edu/~lboloni/). My dissertation addressed stability challenges in multi-agent RL systems through the lens of defensive escort teams.

My research interests span **distributed RL systems**, **multi-agent learning**, **ensemble methods**, and **offline RL**.
```

**Step 2: Create news announcement files**

Create `_news/announcement_1.md`:

```markdown
---
layout: post
date: 2025-08-01
inline: true
related_posts: false
---

Joined [Anyscale](https://www.anyscale.com) as Software Engineer – RL. Now the technical owner of [RLlib](https://docs.ray.io/en/latest/rllib/index.html).
```

Create `_news/announcement_2.md`:

```markdown
---
layout: post
date: 2022-05-15
inline: true
related_posts: false
---

Paper accepted at **ICML 2022**: [DNS: Determinantal Point Process Based Neural Network Sampler for Ensemble RL](/publications/).
```

Create `_news/announcement_3.md`:

```markdown
---
layout: post
date: 2022-04-10
inline: true
related_posts: false
---

Paper accepted at **ICLR 2022**: [Maximizing Ensemble Diversity in Deep Reinforcement Learning](/publications/).
```

Create `_news/announcement_4.md`:

```markdown
---
layout: post
date: 2022-03-01
inline: true
related_posts: false
---

Paper accepted at **IJCNN 2022**: [Learning Intrinsic Symbolic Rewards in Reinforcement Learning](/publications/).
```

Create `_news/announcement_5.md`:

```markdown
---
layout: post
date: 2020-12-15
inline: true
related_posts: false
---

PhD conferred in Computer Science from University of Central Florida. Dissertation: *Multi-agent Reinforcement Learning for Defensive Escort Teams*.
```

**Step 3: Verify build and check home page renders**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll serve --livereload 2>&1 &
sleep 5
curl -s http://localhost:4000 | grep -c "Hassam"
```

Expected: Output >= 1 (name appears on page).

Kill the server after checking: `pkill -f jekyll`

**Step 4: Commit**

```bash
git add _pages/about.md _news/
git commit -m "feat: write about page with bio and news items

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Populate publications BibTeX file

**Goal:** Add all of Hassam's publications to `_bibliography/papers.bib` so jekyll-scholar auto-renders the publications page.

**Files:**
- Modify: `_bibliography/papers.bib`

**Step 1: Replace `_bibliography/papers.bib` with the following**

```bibtex
---
---

@inproceedings{sheikh2022dns,
  title     = {{DNS}: {D}eterminantal Point Process Based Neural Network Sampler for Ensemble Reinforcement Learning},
  author    = {Sheikh, Hassam and Frisbee, Kory and Phielipp, Mariano},
  booktitle = {International Conference on Machine Learning},
  series    = {ICML 2022},
  year      = {2022},
  abbr      = {ICML},
  selected  = {true},
  abstract  = {Ensemble methods are powerful for improving the stability and performance of deep RL agents, but training large ensembles is computationally expensive. We propose DNS, a Determinantal Point Process-based neural network sampler that reduces computation by 50\% while outperforming baselines by selecting maximally diverse subsets of networks for backpropagation at each training step.}
}

@inproceedings{sheikh2022medrl,
  title     = {Maximizing Ensemble Diversity in Deep Reinforcement Learning},
  author    = {Sheikh, Hassam and Phielipp, Mariano and B{\"o}l{\"o}ni, Ladislau},
  booktitle = {International Conference on Learning Representations},
  series    = {ICLR 2022},
  year      = {2022},
  abbr      = {ICLR},
  selected  = {true},
  abstract  = {We propose five regularization methods for ensemble RL that prevent value-function collapse and maximize representation diversity in parameter space, leading to significantly more stable and performant agents.}
}

@inproceedings{sheikh2022lisr,
  title     = {Learning Intrinsic Symbolic Rewards in Reinforcement Learning},
  author    = {Sheikh, Hassam and Khadka, Shauharda and Miret, Santiago and Majumdar, Somdeb and Phielipp, Mariano},
  booktitle = {International Joint Conference on Neural Networks},
  series    = {IJCNN 2022},
  year      = {2022},
  abbr      = {IJCNN},
  abstract  = {We propose an intrinsic reward generator based on learned symbolic trees, producing interpretable intrinsic rewards via arithmetic and logical operators that improve sample efficiency in sparse-reward settings.}
}

@inproceedings{mendula2021smarthomes,
  title     = {Interaction and Behaviour Evaluation for Smart Homes: {D}ata Collection and Analytics in the {ScaledHome} Project},
  author    = {Mendula, Matteo and Khodadadeh, Siavash and Bacanli, Salih Safa and Zehtabian, Sharare and Sheikh, Hassam and B{\"o}l{\"o}ni, Ladislau and Turgut, Damla and Bellavista, Paolo},
  booktitle = {International Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems},
  series    = {MSWiM 2021},
  year      = {2021},
  abbr      = {MSWiM}
}

@inproceedings{sheikh2020demaddpg,
  title     = {Multi-Agent Reinforcement Learning for Problems with Combined Individual and Team Reward},
  author    = {Sheikh, Hassam and B{\"o}l{\"o}ni, Ladislau},
  booktitle = {International Joint Conference on Neural Networks},
  series    = {IJCNN 2020},
  year      = {2020},
  abbr      = {IJCNN},
  abstract  = {We propose DE-MADDPG, a multi-critic cooperative MARL framework that simultaneously maximizes global team rewards and local agent rewards, reducing parametric growth from exponential to linear and improving performance by 97\% over baselines.}
}

@inproceedings{sheikh2019aamas,
  title     = {Emergence of Scenario-Appropriate Collaborative Behaviors for Teams of Robotic Bodyguards},
  author    = {Sheikh, Hassam and B{\"o}l{\"o}ni, Ladislau},
  booktitle = {International Conference on Autonomous Agents and Multiagent Systems},
  series    = {AAMAS 2019},
  year      = {2019},
  abbr      = {AAMAS},
  abstract  = {We demonstrate how multi-agent policy gradient algorithms can be adapted to learn collaborative robot behaviors for VIP protection in crowded public spaces, producing complex emergent bodyguard formations.}
}

@inproceedings{sheikh2019compsac,
  title     = {Learning Distributed Cooperative Policies for Security Games via Deep Reinforcement Learning},
  author    = {Sheikh, Hassam and Razghandi, Mina and B{\"o}l{\"o}ni, Ladislau},
  booktitle = {IEEE International Conference on Computer Software and Applications},
  series    = {COMPSAC 2019},
  year      = {2019},
  abbr      = {COMPSAC},
  abstract  = {We find optimal defender policies using policy gradient in a partially observable cooperative RL framework for security game scenarios.}
}

@article{qadir2014malware,
  title   = {Automatic Feature Extraction, Categorization and Detection of Malicious Code in {Android} Applications},
  author  = {Qadir, Muhammad Zuhair and Jilani, Atif Nisar and Sheikh, Hassam},
  journal = {International Journal of Information \& Network Security},
  volume  = {3},
  number  = {1},
  pages   = {12--17},
  year    = {2014},
  abbr    = {IJINS}
}
```

**Step 2: Verify publications page builds**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | grep -i "error\|warn" | head -10
```

Expected: No scholar-related errors.

**Step 3: Commit**

```bash
git add _bibliography/papers.bib
git commit -m "feat: add all publications to BibTeX bibliography

8 conference/journal papers: ICML, ICLR, IJCNN, AAMAS, COMPSAC, MSWiM.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Configure the publications page

**Goal:** Make the publications page show papers cleanly with venue badges and workshop section.

**Files:**
- Modify: `_pages/publications.md`
- Add workshops to `_bibliography/papers.bib`

**Step 1: Replace `_pages/publications.md`**

```markdown
---
layout: page
permalink: /publications/
title: publications
nav: true
nav_order: 2
---

<!-- Conference Papers -->
<div class="publications">

{% bibliography %}

</div>

---

### Workshops

- **V. Kumar, H. Sheikh**, S. Majumdar, M. Phielipp. "Minimizing Communication while Maximizing Performance in Multi-Agent Reinforcement Learning." *BayLearn 2021.*

- **H. Sheikh**, L. Bölöni. "Preventing Value Function Collapse in Ensemble Q-Learning by Maximizing Representation Diversity." *Workshop on Deep Reinforcement Learning, NeurIPS 2020.*

- **H. Sheikh**, L. Bölöni. "Designing a Multi-Objective Reward Function for Creating Teams of Robotic Bodyguards Using Deep Reinforcement Learning." *Workshop on Goal Specifications for Reinforcement Learning, ICML 2018.*

- **H. Sheikh**, L. Bölöni. "The Emergence of Complex Bodyguard Behavior Through Multi-Agent Reinforcement Learning." *Workshop on Autonomy in Teams, ICML 2018.*

---

### Patents

- **V. Kumar, H. Sheikh**, S. Majumdar, M. Phielipp. "System and Method for Controlling Inter-Agent Communication in Multi-Agent Systems." Patent application *17/544,718.*
```

**Step 2: Verify build**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | tail -5
```

**Step 3: Commit**

```bash
git add _pages/publications.md
git commit -m "feat: configure publications page with workshops and patent sections

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Build the CV page

**Goal:** Render an inline CV page from structured data, with a PDF download button.

**Files:**
- Modify: `_pages/cv.md`
- Create: `assets/json/resume.json`

**Step 1: Replace `_pages/cv.md`**

```markdown
---
layout: cv
permalink: /cv/
title: cv
nav: true
nav_order: 3
cv_pdf: HassamSheikh_Resume_Feb2025.pdf
description: PhD in Computer Science (UCF). Software Engineer at Anyscale. Research Scientist at Intel Labs.
---
```

**Step 2: Create `assets/json/resume.json`**

al-folio's CV layout reads from this file. Create it with:

```json
{
  "basics": {
    "name": "Hassam U. Sheikh",
    "label": "Software Engineer – Reinforcement Learning",
    "email": "hassamsheikh1@gmail.com",
    "location": { "city": "Orlando", "region": "Florida" },
    "profiles": [
      { "network": "GitHub", "username": "HassamSheikh", "url": "https://github.com/HassamSheikh" },
      { "network": "LinkedIn", "username": "hassam-ullah-sheikh", "url": "https://linkedin.com/in/hassam-ullah-sheikh" },
      { "network": "Google Scholar", "username": "QTCAAGQAAAAJ", "url": "https://scholar.google.com/citations?user=QTCAAGQAAAAJ" }
    ]
  },
  "work": [
    {
      "name": "Anyscale",
      "position": "Software Engineer – Reinforcement Learning",
      "startDate": "2025-08",
      "endDate": "Present",
      "highlights": [
        "Technical owner for stability, performance, and correctness across RLlib's distributed training and inference stack.",
        "Lead reliability hardening by diagnosing and eliminating high-impact failure modes (hangs, deadlocks, non-determinism, resource leaks) in large-scale RL workloads.",
        "Drive benchmark-driven performance engineering and regression-prevention guardrails (CI stress tests, determinism checks, perf gates)."
      ]
    },
    {
      "name": "Intel Labs",
      "position": "Research Scientist",
      "startDate": "2020-05",
      "endDate": "2024-11",
      "highlights": [
        "Blue-Agents: Built a modular RL research library implementing state-of-the-art algorithms, accelerating experimentation across the RL team.",
        "MatSci-LLM: Core developer of MatSci-LLM, enabling automated materials discovery workflows.",
        "DNS (ICML 2022): DPP-based neural network sampler for ensemble RL, reducing computation by 50%.",
        "MED-RL (ICLR 2022): Five regularization methods for ensemble RL to prevent value-function collapse.",
        "LISR (IJCNN 2022): Intrinsic reward generator based on learned symbolic trees.",
        "ECNet (BayLearn 2021): Learned communication gates reducing inter-agent communication cost by 75%."
      ]
    },
    {
      "name": "University of Central Florida",
      "position": "Graduate Research Assistant",
      "startDate": "2016-08",
      "endDate": "2020-12",
      "highlights": [
        "DE-MADDPG (IJCNN 2020): Multi-critic MARL for combined individual and team reward; 97% performance improvement over baselines.",
        "MAUPG (AAMAS 2019): Multi-agent policy gradient using UVFA for multi-scenario cooperative learning."
      ]
    },
    {
      "name": "EZOfficeInventory",
      "position": "Software Engineer",
      "startDate": "2015-09",
      "endDate": "2016-08",
      "highlights": [
        "Full-stack Ruby on Rails engineer. Shipped features for EZOfficeInventory and EZRentOut.",
        "Designed and implemented Zendesk integration with bidirectional ticketing and asset workflows.",
        "Developed and launched the EZOfficeInventory Zendesk Marketplace application end-to-end."
      ]
    }
  ],
  "education": [
    {
      "institution": "University of Central Florida",
      "area": "Computer Science",
      "studyType": "Ph.D.",
      "startDate": "2016-08",
      "endDate": "2020-12",
      "score": "",
      "courses": ["Dissertation: Multi-agent Reinforcement Learning for Defensive Escort Teams", "Advisor: Ladislau Bölöni"]
    },
    {
      "institution": "University of Manchester",
      "area": "Advanced Computer Science",
      "studyType": "M.S.",
      "startDate": "2012-09",
      "endDate": "2013-09",
      "courses": ["Thesis: Who is Speaking? Male or Female"]
    },
    {
      "institution": "University of Engineering & Technology, Lahore",
      "area": "Computer Engineering",
      "studyType": "B.S.",
      "startDate": "2008-08",
      "endDate": "2012-06",
      "courses": ["Senior Project: Speech-Controlled Android Robot with Vision"]
    }
  ],
  "publications": [
    { "name": "DNS: Determinantal Point Process Based Neural Network Sampler for Ensemble RL", "publisher": "ICML 2022", "releaseDate": "2022" },
    { "name": "Maximizing Ensemble Diversity in Deep Reinforcement Learning", "publisher": "ICLR 2022", "releaseDate": "2022" },
    { "name": "Learning Intrinsic Symbolic Rewards in Reinforcement Learning", "publisher": "IJCNN 2022", "releaseDate": "2022" },
    { "name": "Multi-Agent RL for Problems with Combined Individual and Team Reward", "publisher": "IJCNN 2020", "releaseDate": "2020" },
    { "name": "Emergence of Scenario-Appropriate Collaborative Behaviors for Teams of Robotic Bodyguards", "publisher": "AAMAS 2019", "releaseDate": "2019" },
    { "name": "Learning Distributed Cooperative Policies for Security Games via Deep RL", "publisher": "COMPSAC 2019", "releaseDate": "2019" }
  ],
  "skills": [
    { "name": "ML/RL", "keywords": ["PyTorch", "JAX", "TensorFlow", "Keras", "NumPy", "Pandas"] },
    { "name": "Systems", "keywords": ["Kubernetes", "Docker", "MySQL", "MS-SQL", "Redis", "Horovod"] },
    { "name": "Languages", "keywords": ["Python", "C++", "C", "C#", "JavaScript", "Ruby", "SQL"] }
  ]
}
```

**Step 3: Verify build**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | tail -5
```

**Step 4: Commit**

```bash
git add _pages/cv.md assets/json/resume.json
git commit -m "feat: add inline CV page with full work/education/skills data

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Create project pages

**Goal:** Add research project cards for key projects from work history.

**Files:**
- Create: `_projects/rllib.md`, `_projects/blue-agents.md`, `_projects/dns.md`, `_projects/med-rl.md`, `_projects/de-maddpg.md`, `_projects/ecnet.md`

**Step 1: Create `_projects/1_rllib.md`**

```markdown
---
layout: page
title: RLlib
description: Industry-standard distributed reinforcement learning library at Anyscale
img: assets/img/projects/rllib.png
importance: 1
category: systems
---

[RLlib](https://docs.ray.io/en/latest/rllib/index.html) is the industry-standard open-source library for reinforcement learning, built on top of [Ray](https://www.ray.io/). As technical owner at Anyscale, I lead stability, performance, and correctness across RLlib's distributed training and inference stack.

**Key contributions:**
- Diagnosing and eliminating high-impact failure modes (hangs, deadlocks, non-determinism, resource leaks) in large-scale RL workloads
- Benchmark-driven performance engineering and regression-prevention guardrails
- CI stress tests, determinism checks, and performance gates

**Technologies:** Python, Ray, PyTorch, Kubernetes, distributed systems
```

**Step 2: Create `_projects/2_blue-agents.md`**

```markdown
---
layout: page
title: Blue-Agents
description: Modular RL research library for standardizing experimentation at Intel Labs
img:
importance: 2
category: research
---

Blue-Agents is a modular reinforcement learning research library built at Intel Labs. It implements state-of-the-art RL algorithms in a unified framework, accelerating experimentation and standardizing research workflows across the RL team.

**Technologies:** Python, PyTorch, JAX
```

**Step 3: Create `_projects/3_dns.md`**

```markdown
---
layout: page
title: DNS
description: Determinantal Point Process Based Neural Network Sampler for Ensemble RL (ICML 2022)
img:
importance: 3
category: research
---

**Published at ICML 2022.**

Ensemble methods improve stability and performance of RL agents, but training large ensembles is computationally expensive. DNS uses a **k-Determinantal Point Process (k-DPP)** to sample a maximally diverse subset of neural networks for backpropagation at each training step.

**Results:** 50% reduction in computation cost while outperforming full-ensemble baselines.

**Technologies:** PyTorch, DPP sampling, ensemble RL
```

**Step 4: Create `_projects/4_med-rl.md`**

```markdown
---
layout: page
title: MED-RL
description: Maximizing Ensemble Diversity in Deep Reinforcement Learning (ICLR 2022)
img:
importance: 4
category: research
---

**Published at ICLR 2022.**

Ensemble RL suffers from value function collapse — where agents converge to similar representations, defeating the purpose of the ensemble. We propose **five regularization methods** that maximize representation diversity in parameter space, preventing collapse and significantly improving stability.

**Technologies:** PyTorch, deep Q-learning, ensemble methods
```

**Step 5: Create `_projects/5_de-maddpg.md`**

```markdown
---
layout: page
title: DE-MADDPG
description: Multi-critic MARL for combined individual and team reward (IJCNN 2020)
img:
importance: 5
category: research
---

**Published at IJCNN 2020.**

In cooperative multi-agent settings, agents must simultaneously optimize for individual tasks and collective group success. DE-MADDPG (Decomposed Multi-Agent DDPG) introduces a **multi-critic architecture** that disentangles global team reward from local agent rewards, reducing parametric growth from exponential to linear.

**Results:** 97% performance improvement over MADDPG baselines.

**Technologies:** Python, PyTorch, multi-agent RL
```

**Step 6: Create `_projects/6_ecnet.md`**

```markdown
---
layout: page
title: ECNet
description: Efficient communication in multi-agent RL via learned communication gates (BayLearn 2021)
img:
importance: 6
category: research
---

**Presented at BayLearn 2021.**

Communication between agents in MARL is expensive. ECNet learns **communication gates** that allow agents to selectively communicate only when necessary, optimizing the trade-off between task performance and communication cost.

**Results:** 75% reduction in inter-agent communication cost without sacrificing task performance.

**Technologies:** Python, PyTorch, multi-agent RL
```

**Step 7: Verify build**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | tail -5
```

**Step 8: Commit**

```bash
git add _projects/
git commit -m "feat: add research project pages (RLlib, DNS, MED-RL, DE-MADDPG, ECNet)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Configure navigation and finalize pages

**Goal:** Set up correct navigation links and clean up any placeholder pages.

**Files:**
- Check/modify: `_data/navigation.yml` or nav config in `_config.yml`
- Modify: `_pages/teaching.md` (placeholder)

**Step 1: Verify navigation**

In al-folio, navigation is driven by `nav: true` and `nav_order` in page front matter. Verify pages have correct nav settings:

- `_pages/about.md`: `nav: true`, `nav_order: 1`
- `_pages/publications.md`: `nav: true`, `nav_order: 2`
- `_pages/cv.md`: `nav: true`, `nav_order: 3`
- `_pages/projects.md` (if exists, else create): `nav: true`, `nav_order: 4`
- Blog is automatic.

**Step 2: Create/update `_pages/projects.md` if not present**

```markdown
---
layout: page
title: projects
permalink: /projects/
description: Research projects and engineering work.
nav: true
nav_order: 4
display_categories: [research, systems]
horizontal: false
---
```

**Step 3: Create placeholder `_pages/teaching.md`**

```markdown
---
layout: page
title: teaching
permalink: /teaching/
description:
nav: false
---

Teaching materials coming soon.
```

**Step 4: Final full build and local serve**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1
bundle exec jekyll serve --port 4000 &
sleep 5
# Manually verify in browser at http://localhost:4000
# Check: home, publications, cv, projects pages
pkill -f jekyll
```

**Step 5: Commit**

```bash
git add _pages/
git commit -m "feat: finalize navigation and page structure

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Migrate existing blog posts

**Goal:** Convert existing Minimal Mistakes posts to al-folio format.

**Files:**
- Read existing: `_posts/*.md`
- Modify front matter of each post to use `layout: post` (al-folio format)

**Step 1: Check existing posts**

```bash
ls /home/hassam/hassamsheikh.github.io/_posts/
```

**Step 2: For each post, update front matter**

al-folio posts use this front matter format:

```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD
description: Short description
tags: [tag1, tag2]
categories: category-name
---
```

Remove Minimal Mistakes-specific keys: `header`, `overlay_image`, `teaser`, `read_time`, `comments`, `share`, `related`.

**Step 3: Verify all posts build**

```bash
cd /home/hassam/hassamsheikh.github.io
bundle exec jekyll build 2>&1 | grep -i "error" | head -10
```

**Step 4: Commit**

```bash
git add _posts/
git commit -m "feat: migrate existing blog posts to al-folio format

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 12: Push and verify deployment

**Goal:** Push to GitHub, verify GitHub Actions builds and deploys to `gh-pages`.

**Step 1: Push to master**

```bash
cd /home/hassam/hassamsheikh.github.io
git push origin master
```

**Step 2: Monitor GitHub Actions**

```bash
gh run list --limit 5
gh run watch
```

Expected: Workflow completes with green checkmark.

**Step 3: Remind user to update Pages source**

The user must go to:
`https://github.com/HassamSheikh/hassamsheikh.github.io/settings/pages`

And change **Source** from `master` branch to `gh-pages` branch.

**Step 4: Verify live site**

Visit `https://hassamsheikh.github.io` and confirm:
- Dark theme loads by default
- Teal accent visible on links and hover states
- All 5 nav items present (About, Publications, CV, Projects, Blog)
- Profile photo displays
- News items visible on home page
- Publications page shows all 8 papers with venue badges
- CV page renders inline with PDF download button
