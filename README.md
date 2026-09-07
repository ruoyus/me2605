# ME2605: Training Neural Networks: Theory and Practice

**Course website (for students): https://ruoyus.github.io/me2605/**

Fall 2026 · School of Data Science, CUHK-Shenzhen · Instructor: Ruoyu Sun

This repository contains the **source files** of the course website. It is published
with GitHub Pages from the `main` branch.

## Layout

| Path | Purpose |
|---|---|
| `index.html` | Single-page site: overview, prerequisites, schedule, assignments, resources |
| `assets/style.css` | Styles |
| `assets/main.js` | Chinese / English language toggle |
| `slides/` | Lecture slides, named `lecture_01.pdf`, `lecture_02.pdf`, … |

## Updating

- **Add slides**: drop `lecture_XX.pdf` into `slides/`, then link it in the Schedule
  table on `index.html` (replace the `&mdash;` placeholder with `[slides](/files/...)`-style link).
- **Edit content**: edit `index.html`; each text has `<span class="lang-en">` and
  `<span class="lang-zh">` versions — update both to keep the bilingual toggle working.
- Changes go live ~1–2 minutes after pushing to `main`.
