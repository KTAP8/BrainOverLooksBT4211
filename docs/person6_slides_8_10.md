# Slides 8a–10 (Person 6) — draft content

Copy into team deck in **speak order**: **8a** (pooled + FAQ) → **8b** (segment impact) → **9** → **10**. Reuse Jiayi's `core_model_stated_vs_revealed.png` only if needed as recap (slide 5 is hers).

---

## Slide 8a — Pooled ranking + FAQ (show **before** slide 8b)

**Title:** Would ranking by revealed preferences improve engagement?

**Method (subtitle or 1–2 bullets):**

- Score each date: weighted sum of six 1–10 ratings; rank with **stated** `*1_1` vs **revealed** (Part 3) weights.
- **Yes rate** on top **1** and top **3**; **452 people**, same sample as core model.

**Visual:** `outputs/figures/customer_value_topk_yes_rate.png` — top-1 **87% → 90%** (**+3.1 pp**); top-3 **+2.0 pp**.

**FAQ bullets (on slide — address confusion right after Figure 1):**

- **~70% same #1** → top-1 lift **0 pp** (~91% yes either way)
- Only **re-rank ~15 dates** already met (not a full app catalog)
- Jiayi **~20 pp** = **importance (0–100)**; this chart = **yes rate (0/1)**

**Speaker note (8a):** Walk Figure 1 first, then FAQ **before** advancing. Overall top-1 bootstrap 95% CI ≈ +0.9 to +5.8 pp (`customer_value_bootstrap.csv`, `segment=all`). Tease: “Next: where #1 **changes**.”

### Slide 8a — FAQ detail (speaker notes / appendix)

**“If stated and revealed differ so much, why ~87% vs ~90%? Same #1?”**

- **~70%** same top partner (~315); **~30%** (#1 swaps, ~138) — see **slide 8b**.
- High yes on both rules: stated weights still pick strong dates; many **multi-yes** evenings.
- **Fixed menu:** no partner who wasn’t in the room; **yes/no ceiling** vs importance scale.
- Pooled **+3.1 pp** is **diluted** by the 70% with zero top-1 lift.

---

## Slide 8b — Where re-ranking changes #1 (impact)

**Title (optional):** When stated and revealed #1 disagree

**Visual:** `outputs/figures/customer_value_topk_different_top1.png`

**Bullets:**

- **n = 138 (~31%)** — stated #1 and revealed #1 are **different partners**
- Yes on **stated #1 pick** **~78%** vs **revealed #1 pick** **~88%** (**+10 pp**)
- Women **~+17 pp** · Men **~+4 pp** in this segment (chart footnote)
- Pooled **~3 pp** ≈ **31% × ~10 pp** (where ranking matters)

**Payoff line (bold on slide):** *When survey weights and choice weights disagree, ranking by **revealed** preferences changes **yes**—not just what people **say** they want.*

**Speaker note (8b):** Different #1 bootstrap 95% CI ≈ +2.2 to +18.1 pp (`segment=different_top1`). Tables: `customer_value_lift_by_segment.csv`, `customer_value_top1_agreement.csv`.

---

### Backup: what if we changed the data? (Q&A slide, optional)

Use for Q&A; not required on the main deck.

**We would expect a bigger yes-rate (or business) gap if:**

| Change (data or design) | Why the result would look larger |
|-------------------------|----------------------------------|
| **More #1 changes when re-weighting** | Lift is **zero** when stated and revealed agree on #1 (~70% today). Bigger **per-person** stated–revealed weight gaps → more re-ranking (~40% different #1 in the highest disagreement quartile vs ~23% in the lowest). |
| **More spread among dates on looks vs “brains”** | Re-ranking only bites when partners **differ** on attributes revealed weights emphasize (e.g. attractiveness). |
| **Pickier users / stricter outcome** | Lower baseline yes rate → putting the “wrong” person first **hurts more** (speed dating here is fairly yes-heavy, which compresses top-1 rates). |
| **More dates per person (larger choice set)** | More candidates → weights flip #1 more often; revealed top pick can beat stated by a full decision more often. |
| **Choice-based surveys (conjoint / forced trade-offs)** | New collection, not a tweak—usually **sharper** stated–revealed conflict than independent 100-point scales. |
| **Ranking against profiles they didn’t meet** | Apps rank from **thousands** of profiles; wrong stated weights can hide high-fit people who **never appear** in a 15-date replay. |
| **Different metrics (same data)** | **P(yes)** from the logit, **mutual match**, or lift **only when #1 differs** (~10 pp) show more headroom than pooled binary top-1 yes. |

**One line:** Large **importance** gaps do not automatically become equally large **yes-rate** gaps when you only **re-rank a short, fixed list** of dates people already met.

---

## Slide 9 — Recommendations

**Title:** What should marketers do differently?

**Market researchers**

- Treat 100-point self-explicated surveys as **hypotheses**, not ground truth.
- Follow with choice data, conjoint tasks, or A/B tests on **ranked** offers.
- In this sample, expect **under-reported importance of attractiveness** and **over-reported sincerity/intelligence**.

**Product / UX teams**

- Do not optimize profile prompts or onboarding copy from stated weights alone.
- Test designs that surface attributes that **drive choices** (visual presentation, fun, shared interests) versus what users **say** they want.

**Matching platforms**

- Prototype feed ranking using **revealed-style** weights; track **yes / match rate** and downstream retention as CLV proxies.
- Frame as better **fit surfacing**, not manipulation; a yes is not long-term compatibility.

**Citation footer:** Fisman et al. (2006); course framing — stated vs revealed conjoint.

---

## Slide 10 — Limitations and close

**Limitations**

- Columbia graduate students, 2002–2004; short heterosexual speed dates — limited external validity.
- Observational; person fixed effects do not remove all date-level confounding (halo — see Part 5 when ready).
- Revealed **importance** uses clipped standardized coefficients; men's ambition can show 0% importance when the coefficient is negative.
- Customer-value exercise **replays** historical dates (~15 per person); not a deployed recommender or full-catalog ranking experiment.
- Large **importance gaps** (slide 5) need not imply equally large **yes-rate** lifts: fixed menu, **~70% same #1**, binary **yes/no ceiling**, vs importance on 0–100 (see slides **8a–8b**).

**Close (one line):** *Customers tell you what they value. Their choices tell you the truth. The gap is large enough to change how we research, design, and rank.*
