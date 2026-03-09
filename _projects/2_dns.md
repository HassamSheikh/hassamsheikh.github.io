---
layout: page
title: DNS
description: Determinantal Point Process Based Neural Network Sampler for Ensemble RL
img: assets/img/projects/dns.jpg
importance: 2
category: research
related_publications: true
---

**Published at ICML 2022.**

Ensemble methods improve stability and performance of RL agents, but training large ensembles is computationally expensive. DNS uses a **k-Determinantal Point Process (k-DPP)** to sample a maximally diverse subset of neural networks for backpropagation at each training step.

**Key result:** 50% reduction in computation while outperforming full-ensemble baselines.

**Stack:** Python, PyTorch, DPP sampling, ensemble RL
