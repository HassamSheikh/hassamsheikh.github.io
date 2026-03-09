# Website Redesign Design Doc
**Date:** 2026-03-08
**Author:** Hassam U. Sheikh

## Overview

Complete rewrite of hassamsheikh.github.io from Jekyll Minimal Mistakes to al-folio — a high-caliber academic research website with a dark theme and cyan/teal accent.

## Goals

- Present a professional, high-caliber research identity
- Accurately reflect current position (SWE-RL at Anyscale) and full publication record
- Dark theme with distinctive cyan/teal accent
- Full section coverage: About, Publications, CV, Blog, Projects, News

## Visual Identity

### Color Palette
- Background: `#0d1117`
- Surface/cards: `#161b22`
- Accent: `#00d4aa` (cyan-teal)
- Text primary: `#e6edf3`
- Text muted: `#8b949e`

### Typography
- Headings + Body: Inter
- Code/labels: JetBrains Mono

## Tech Stack

- **Theme:** al-folio (Jekyll)
- **Hosting:** GitHub Pages (unchanged)
- **Publications:** BibTeX file (`_bibliography/papers.bib`)
- **CV:** Inline YAML-driven page + downloadable PDF

## Site Architecture

### Navigation
`About` · `Publications` · `CV` · `Blog` · `Projects`

### Pages

| Page | Purpose |
|------|---------|
| About (home) | Hero, bio, news feed, social links |
| Publications | Papers grouped by year + workshops section |
| CV | Inline rendered experience, education, skills + PDF download |
| Blog | Posts |
| Projects | Research project cards |

## Page Designs

### About (Home)

**Hero block:**
- Photo (left) + Name/Title/Company (right)
- Name: Hassam U. Sheikh
- Title: Software Engineer – Reinforcement Learning
- Company: Anyscale
- Research focus line: "Distributed RL systems, multi-agent learning, and ensemble methods"
- Social links: GitHub, LinkedIn, Google Scholar, Email

**Bio:**
> I build and maintain RLlib at Anyscale, the industry-standard distributed RL library. Previously a Research Scientist at Intel Labs, I published at ICML, ICLR, IJCNN, and AAMAS on multi-agent RL, ensemble diversity, and intrinsic reward learning. I hold a Ph.D. in Computer Science from the University of Central Florida.

**News items (5):**
- Aug 2025: Joined Anyscale as SWE-RL, technical owner of RLlib
- May 2022: Paper accepted at ICML 2022 (DNS)
- Apr 2022: Paper accepted at ICLR 2022 (MED-RL)
- Mar 2022: Paper accepted at IJCNN 2022 (LISR)
- Dec 2020: PhD conferred, UCF

### Publications

Grouped by year, newest first. Each entry:
- Bold title (linked to arXiv/PDF)
- Authors (Hassam's name bolded)
- Venue badge styled in teal

**Conference Papers (newest first):**
1. DNS: Determinantal Point Process Based Neural Network Sampler for Ensemble RL — ICML 2022
2. Maximizing Ensemble Diversity in Deep Reinforcement Learning — ICLR 2022
3. Learning Intrinsic Symbolic Rewards in Reinforcement Learning — IJCNN 2022
4. Interaction and Behaviour Evaluation for Smart Homes... — MSWIM 2021
5. Multi-Agent RL for Problems with Combined Individual and Team Reward — IJCNN 2020
6. Emergence of Scenario-Appropriate Collaborative Behaviors for Teams of Robotic Bodyguards — AAMAS 2019
7. Learning Distributed Cooperative Policies for Security Games via Deep RL — COMPSAC 2019
8. Automatic Feature Extraction, Categorization and Detection of Malicious Code in Android Applications — Journal 2014

**Workshops (collapsible section):**
1. Minimizing Communication while Maximizing Performance in MARL — BayLearn 2021
2. Preventing Value Function Collapse in Ensemble Q-Learning — NeurIPS 2020 Workshop
3. Designing a Multi-Objective Reward Function for Robotic Bodyguards — ICML 2018 Workshop
4. The Emergence of Complex Bodyguard Behavior Through MARL — ICML 2018 Workshop

### CV Page

Sections in order:
1. Experience: Anyscale → Intel Labs → UCF Grad Assistant → EZOfficeInventory
2. Education: PhD UCF → MS Manchester → BS UET Lahore
3. Publications (same list)
4. Workshops
5. Patents: System and Method for Controlling Inter-Agent Communication (17/544,718)
6. Skills: ML/RL · Systems · Languages

PDF download button at top linking to `HassamSheikh_Resume_Feb2025.pdf`

### Projects

Cards for key research projects:
- RLlib (Anyscale) — Distributed RL library
- Blue-Agents (Intel Labs) — Modular RL research library
- DNS — DPP-based neural network sampler for ensemble RL
- MED-RL — Ensemble diversity regularization methods
- DE-MADDPG — Multi-critic MARL for combined individual/team reward
- ECNet — Learned communication gates for MARL

## _config.yml Key Settings

```yaml
title: Hassam U. Sheikh
first_name: Hassam
last_name: Sheikh
email: hassamsheikh1@gmail.com
description: Software Engineer – RL at Anyscale. Research in distributed RL, multi-agent systems, and ensemble methods.
url: https://hassamsheikh.github.io
github_username: HassamSheikh
scholar_userid: QTCAAGQAAAAJ
theme_darkmode: true
```

## Implementation Notes

- Fork al-folio into the existing repo (replace all files except `.git`, `assets/images/bio-photo.jpg`, `HassamSheikh_Resume_Feb2025.pdf`)
- Override `_sass/al-folio/_themes.scss` for custom dark palette and teal accent
- Populate `_bibliography/papers.bib` with all publications
- Populate `_data/cv.yml` with full CV data
- Add news items to `_data/news.yml`
- Add project cards to `_projects/`
- Keep existing blog posts, migrate to al-folio post format
