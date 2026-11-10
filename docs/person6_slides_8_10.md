# Slides 8a–10 (Person 6) — draft content

## Canva deck version (pages 20–26, current)

These are built in the team Canva deck with one hero number per slide. Each layout is a full-slide illustration made by `python -m src.deck_visuals` (`outputs/figures/deck_full_*.png`). The title, hero number, headline and pill are the deck's own text boxes on top. The spoken script is in `person6_presentation_script.md`, and it matches the Canva speaker notes.

| Page | Slide | Pill | Title (serif) | Hero (serif) | Headline | Visual |
|---|---|---|---|---|---|---|
| 20 | 8a | The ranking test | Rank by what people do? | +3 pts | Top-pick yes rate 87% → 90% · best case | `deck_full_8a_ring.png`: ring, 69% same #1 vs 31% changes |
| 21 | 8b | The twist | When the top pick changes | +10 pts | when they disagree on who goes first, behaviour's pick wins | `deck_full_8b_faceoff.png`: phones, 78% vs 88% |
| 22 | new | The leak | Most yeses go nowhere | 61% | of yeses are never returned. A yes isn't a match, and only matches keep users. | `deck_full_8c0_wasted.png`: 100 hearts, fix box 38% → 28% |
| 23 | 8c | The fix | Rank for both sides | 34% | of top picks become matches when the app ranks for both people, vs 27% on survey weights | `deck_full_8c_race.png`: race lanes, 73% → 57% sticker, beats a direct match model by +4 pts |
| 24 | 8d | Surveys vs behaviour | Watch, don't ask | 3 > 40 | three speed dates explain 29% of who gets wanted later. 40 sign-up answers explain just 1–10%. | `deck_full_8d_seesaw.png`: see-saw, R² .29 vs .01–.10 |
| 25 | 9 | Four decisions | Matches pay the bills | — | Ship each as an A/B test | `deck_full_9_flywheel.png`: loop and four decision badges |
| 26 | 10 | Bottom line | the closing quote | — | Thank you. Questions? | `deck_full_10_poster.png`: "Do this" / "Know the limits" panel |

**Known slide wording issue:** the flywheel's "Users stay" step says "80% vs 90% came back". The measure is actually *answered the next-day survey*. The script says this correctly.

---|---|---|---|---|---|---|
| 19 | 8a | The ranking test | Rank by what people do? | +3 pts | more yeses on the top pick when the app ranks by behaviour, not the survey | `deck_8a_donut.png`: 1 in 3 see a different #1 |
| 20 | 8b | The twist | When the top pick changes | +10 pts | when the survey and behaviour disagree on who goes first, behaviour's pick wins | `deck_8b_phones.png`: 78% vs 88% profile cards |
| 21 | 8c | Swipes ≠ matches | Yeses aren't matches | 34% | of top picks become matches when the app ranks for both people, vs 27% on survey weights | `deck_8c_matches.png`: match-rate bars, 73% → 57% callout |
| 22 | 8d | Surveys vs behaviour | Watch, don't ask | 3 > 40 | three speed dates predict who gets wanted later about 3× better than 40 sign-up answers | `deck_8d_survey.png`: R² .29 vs .01–.10 |
| 23 | 9 | Four decisions | Matches pay the bills | — | Ship each as an A/B test | `deck_9_chain.png` + `deck_9_tiles.png` |
| 24 | 10 | Bottom line | the closing quote | — | Thank you. Questions? | Do this / Know the limits cards |

---

Original draft content follows. Copy into team deck in **speak order**: **8a** (pooled + FAQ) → **8b** (segment impact) → **8c** (pre-date test and matches) → **8d** (behaviour beats surveys) → **9** → **10**. Reuse Jiayi's `core_model_stated_vs_revealed.png` only if needed as recap (slide 5 is hers).

---

## Slide 8a — Pooled ranking + FAQ (show **before** slide 8b)

**Title:** Would ranking by revealed preferences improve engagement?

**Method (subtitle or 1–2 bullets):**

- Score each date: weighted sum of six 1–10 ratings; rank with **stated** `*1_1` vs **revealed** (Part 3) weights.
- **Yes rate** on top **1** and top **3**; **452 people**, same sample as core model.
- **Best-case replay: uses ratings made after each date.**

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

**Payoff line (bold on slide):** *When survey weights and choice weights disagree, in this replay, ranking by **revealed** preferences changes **yes**—not just what people **say** they want.*

**Decision:** rank on behaviour, not stated weights. A yes is engagement, not revenue, so the next test is whether it creates matches.

**Speaker note (8b):** Different #1 bootstrap 95% CI ≈ +2.2 to +18.1 pp (`segment=different_top1`). Tables: `customer_value_lift_by_segment.csv`, `customer_value_top1_agreement.csv`.

**Speaker note — what 78% vs 88% means (30% only):** Among **138** people where stated #1 and revealed #1 are **different partners**, **~78%** said yes to **stated’s** top pick and **~88%** to **revealed’s** top pick—not the same person before/after on one profile. Paired comparison: *which wrong-first pick gets more yeses when weights disagree?*

**Speaker note — Person 5's say-do gap groups (if asked):** the high-gap group states attractiveness at **24.8** but behaves closer to **31.6**; match rates don't step up or down across gap groups (**20% / 16% / 19%**). `04_robustness_extensions updated.ipynb`, section 6.

---

## Slide 8c — Pre-date test and mutual matches

**Title:** What if a platform only knows what it knows before the date?

**Visual:** `outputs/figures/ext_a5_yes_vs_match.png`

**Bullets:**

- Pre-date information only, tested on unseen events
- Top-3 yes: **stated 65%**, **revealed 67%** (**+2.3 pp**, CI 0.6 to 4.1); top-1 difference not clear
- Revealed ranking: match **+0.7 pp** (CI -1.0 to 2.3), not clear
- Ranking for mutual interest: match **+4.1 to +7.1 pp**; stricter test **+2.5 pp** (CI 1.2 to 3.9)
- Cost: **3.5 to 5.0 pp** fewer yeses
- Least-chosen third: no match in top 3 falls from **73% to 57%**

**Decision:** make mutual matches the ranking target, not swipes, and cap mass-likers.

**Footnote on slide:** Lift depends on partner yes rate, so a real product would need a cap on mass-likers.

**Speaker note (8c):** Quote the range, because +7.1 is the highest of five random splits. Without partner yes rate, the lift is -0.7 pp. Tables: `ext_a5_lifts.csv`, `ext_a5_seed_sensitivity.csv`, `ext_a5_history_design.csv`, `ext_a5_partner_model.csv`.

**New material for 8c (optional, not in the script yet):**
- **On-slide line:** "Half of all dates end one-sided: **61% of yeses are never returned.**" Optional visual: `ext_a5_endings.png`.
- **Suggested spoken line:** "Half of all dates end one-sided. Six in ten yeses are never returned. That's wasted effort, and ranking for both sides cuts it from 38 to 28 percent of the top three."
- **Q&A backup:** "Why not just predict matches?" One model trained on matches gets 30% top-3 matches, and our two-sided rule gets 34% (+4.0 pp, CI 2.4 to 5.5; +1.2 to +4.0 across five splits). Table `ext_a5_direct_vs_two_sided.csv`.

---

## Slide 8d — Behaviour beats surveys

**Title:** Should a company ask more questions, or watch what people do?

**Visual:** `outputs/figures/ext_a3_survey_vs_behaviour.png`

**Bullets:**

- Predicting later yeses on unseen events: first 3 speed dates **R² .29**; full sign-up survey **.01 to .10**
- Person 5: **AUC about .83** with or without stated weights

**Decision:** shorten sign-up and learn from early activity.

**Speaker note (8d):** "First 3" means the first three four-minute speed dates of the evening, not real dates. Tables: `ext_a3_prediction.csv`, `ext_a3_r2_gap.csv`; Person 5's section 7.

---

### Real-world product translation (report / slide 9 backup)

This lab decision happens at **first contact** (~4-minute meeting). `dec` = *I* said yes; `match` = **both** said yes. We do **not** observe second dates—retention would be a later product KPI.

| Product lever | What we model | Metric | What to do |
|---------------|---------------|--------|------------|
| **Feed / scroll ranking** (infinite list, “who is #1 for **me**”) | **Slides 8a–8b** (`05_customer_value`) | **Your** yes on top-1 / top-3 | Rank with **revealed-style** weights to lift **engagement** on first slots (post-date replay: ~+3 pp pooled, **~+10 pp** when #1 changes; pre-date: **+2.3 pp** at top 3). |
| **Scarce introductions** (one Daily Pick, one event table, no double-booking) | **Extension market simulator** (e.g. `06_extended_analysis` / slide on limited recommendations) | **Mutual** match on assigned pair | **Coordinate** assignments; **optimise bilateral** interest (e.g. maximise predicted **P(I yes) × P(they yes)**), not only each user’s solo ranking. |

**Why both can be true:** Revealed ranking can raise **my** yes on **my** #1 while **mutual** match on one-shot pairing need not rise as much—looks-heavy picks can be one-sided. In the team’s replay, optimised pairing beats random (**+6.3 pp**, CI 2.5 to 10.3); **optimised mutual** ≈ **24.1%** vs **stated 22.1%** vs **revealed 20.0%** mutual match, with **wide CIs** among structured rules—value is often **scarcity + coordination**, not picking stated vs revealed alone.

**Not “optimising second dates”:** “Optimised for both saying yes” targets **mutual like on the first assigned meeting** (app: both swipe right on a curated pair), not date #2.

**For the company:** Choose the KPI, then the policy—**revealed re-ranking** for feeds and first-slot engagement, with a small pre-date gain; **bilateral matching** when slots are exclusive; **stated surveys** as onboarding priors, not the sole ranker; **A/B test** yes, match, and retention in production (historical replay ≠ live lift).

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
| **Different metrics (same data)** | **P(yes)** from the logit or lift **only when #1 differs** (~10 pp) show more headroom than pooled binary top-1 yes. **Mutual match does not:** revealed ranking shows no clear match lift. |

**One line:** Large **importance** gaps do not automatically become equally large **yes-rate** gaps when you only **re-rank a short, fixed list** of dates people already met.

**Other Q&A facts:**

- **Order (Person 5):** no change in yes rate across the evening (p=.78 women, p=.86 men).
- **Event condition (Person 5):** extensive-choice events show lower yes odds (OR .82, p=.05), but this is confounded with event.
- **Previous partner:** a 1 SD more attractive previous partner lowers the next yes by **1.4 pp** (shuffled-order p=.008), clearest for men. Replicates Bhargava & Fisman (2014).
- **Funnel:** 15.2 speed dates, 6.4 yeses, 2.5 matches; **18%** get no match. Follow-up only covers 48% of people.

---

## Slide 9 — Recommendations

**Title:** Four decisions that drive revenue

**Subtitle:** A dating app earns when users stay and pay, and users stay when they match. **18%** get no match; they answered the next-day survey less often (**80% vs 90%**); each extra match adds **6.2 pp** to the chance of a real date.

| Decision | Why it makes money | Our evidence | KPI to test |
|---|---|---|---|
| **Rank for mutual matches**; cap mass-likers and popular profiles | Matches keep users; no-match users drift away | Match **+2.5 to +7.1 pp**; top 20% of users get **56-61%** of slots under ranking (30% random) | Match rate, 30-day retention |
| **Shorten onboarding**; learn from behaviour | Less sign-up friction, no loss in prediction | First 3 dates **R² .29** vs survey **.01 to .10**; Person 5: AUC unchanged | Sign-up completion |
| **Sell scarce, coordinated picks** (Daily Pick) | A premium feature users can pay for | Optimised vs random **+6.3 pp** (CI 2.5 to 10.3); structured rules can't be separated | Upgrade rate |
| **Market on what drives choice** | Ads and prompts that match real choices | Revealed weights: looks **38-47**, fun **20-24**, shared interests **20-22**, ambition **0-1** | Ad click-through, profile completion |

**Optional nuance for decision 4 (S4):** "Shared interests" means the *feeling* of common ground. Felt shared interests goes with match rates of **5% → 35%**, while actual hobby overlap only **16% → 20%** (borderline after correction). So use conversation prompts and icebreakers, not hobby filters.
- **Suggested spoken add-on:** "...and shared interests, but the *feeling* of them: actual hobby overlap barely moves matches, so build prompts, not filters."
- **Figure:** `ext_s4_common_ground.png`.

**Footnote on slide:** Historical replays, not live tests, so each decision ships as an A/B test. Frame it as better fit surfacing, not manipulation.

**Speaker note (9):** Revenue is never measured here. Matches, reported dates and survey response are stand-ins, and the 80% vs 90% link is the weakest one. Tables: `ext_a5_funnel.csv`, `ext_a5_exposure.csv`, `ext_s3_lifts.csv`, `core_model_results.csv`. Method detail: `person6_methods_explainer.md`.

**Citation footer:** Fisman et al. (2006); Pizzato et al. (2010); course framing — stated vs revealed conjoint.

---

## Slide 10 — Limitations and close

**Limitations**

- Columbia graduate students, 2002–2004; short heterosexual speed dates — limited external validity.
- Observational; person fixed effects do not remove all date-level confounding. Person 5's consensus check shows both peers' view and a person's own rating matter (p<.001); consensus-only attractiveness remains the top trait.
- Revealed **importance** uses clipped standardized coefficients; men's ambition can show 0% importance when the coefficient is negative.
- Customer-value exercise **replays** historical dates (~15 per person); not a deployed recommender or full-catalog ranking experiment.
- Mutual-match results are replays, not live tests.
- No revenue data: matches, reported dates and survey response stand in for retention and revenue.
- Person 5's order check shows no drop in yeses across the evening (p=.78, p=.86).
- Survey format differed in waves 6-9; the attractiveness gap is **18.4** and **17.3** on 100-point waves.
- Large **importance gaps** (slide 5) need not imply equally large **yes-rate** lifts: fixed menu, **~70% same #1**, binary **yes/no ceiling**, vs importance on 0–100 (see slides **8a–8b**).

**Close (one line):** *Customers tell you what they value. Their choices tell you the truth. Rank, onboard and market on choices, and optimise for matches.*
