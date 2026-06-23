# ReDesign Project Webpage

A responsive, scroll-driven project page for the ReDesign paper: "Recovering Editable Design Structures from Images via Agentic Decomposition".

<br>

## Features

- **Responsive Design**: Mobile, tablet, and desktop layouts
- **Scroll-Driven Animations**: Toss-style pinned sections, enter reveals, count-ups
- **Afterglow Design System**: Gradient text, pink accents, centered narrow column
- **All Figures Embedded**: 6 WebP-optimized figures (1.6 MB total)
- **Interactive Table 1**: Quantitative accuracy results with best scores highlighted
- **Accessibility**: `prefers-reduced-motion` support, keyboard navigation, proper alt text

<br>

## Contents

- `index.html` — single-file responsive project page (inline CSS + vanilla JS)
- `assets/figures/` — all 6 paper figures optimized to WebP
- `.nojekyll` — tells GitHub Pages to skip Jekyll processing

<br>

## Deploy to GitHub Pages

1. Go to your repo settings: https://github.com/sonjt00/ReDesign/settings/pages
2. Under **Source**, select:
   - Branch: `main`
   - Folder: `/docs`
3. Click **Save**

GitHub will build and deploy the site to: **https://sonjt00.github.io/ReDesign**

The URL will show "ReDesign" as requested.

<br>

## Sections

1. **Hero** — Title, subtitle, buttons (Paper/Code/Dataset)
2. **Teaser** — Figure 1: from flat image to editable design
3. **Problem** — Why design reconstruction matters
4. **Method** — Figure 2 + 5 tool-backed actions with pinned scroll animation
5. **Results: Accuracy** — Table 1 (quantitative) + Figure 6 (qualitative)
6. **Results: Editability** — Figure 4 (quantitative) + Figure 5 (qualitative) + stat cards
7. **Analysis** — Figure 8: vs. generative editing (Nano Banana 2)
8. **Benchmark Stats** — Figma-909 dataset details
9. **Footer** — Repeated buttons, license info

<br>

## Technical Details

- **Design System**: Afterglow gradient (#FFA7A6 → #FF6A8A → #5A78FF → #9D4FFF)
- **Scroll Engine**: One rAF loop + IntersectionObserver for enters
- **Accessibility**: All motion respects `prefers-reduced-motion: reduce`
- **Performance**: Transform/opacity only (GPU-friendly), lazy image loads
