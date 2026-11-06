# Team slide style guide

Use this for Google Slides / PowerPoint master so Parts 3–6 look consistent.

## Colours (from `src/constants.py`)

| Role | Hex | Use |
|------|-----|-----|
| Stated | `#4C78A8` | Bars, lines, labels for survey / self-explicated weights |
| Revealed | `#F58518` | Bars, lines, labels for choice-based / logit weights |
| Neutral text | `#526273` | Footnotes, subtitles |
| Axis / CI caps | `#263747` | Error bars |

## Attribute order (always)

1. Attractive  
2. Sincere  
3. Intelligent  
4. Fun  
5. Ambitious  
6. Shared interests  

## Chart rules

- Y-axis / labels: **plain English**, not column names (`attr`, `sinc`, …).
- Export figures at **300 dpi** from `outputs/figures/`.
- Significance for stated–revealed **gaps**: bootstrap **gap CI excludes zero** (Jiayi slide 5). Do not infer significance from overlapping bar CIs alone.

## Slide master layout

- **Title:** left-aligned, bold, ~28–32 pt.
- **Subtitle:** `#526273`, 12–14 pt (sample size, method one-liner).
- **Footer:** citation + speaker name; e.g. `Fisman et al. (2006) · BT4211 · [Name]`.
- **Margins:** leave space for figure footnotes (seed, bootstrap reps, clipping note on revealed importance).

## Figure inventory

| Slide | File |
|-------|------|
| 5 (Jiayi) | `core_model_stated_vs_revealed.png` |
| 5 (optional) | `core_model_gap_intervals.png` |
| 8a (Person 6) | `customer_value_topk_yes_rate.png` (pooled; show FAQ on same slide) |
| 8b (Person 6) | `customer_value_topk_different_top1.png` (#1 differs, n=138; payoff line) |
