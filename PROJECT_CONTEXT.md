# Project Context: "People Say They Want Brains, Then Choose Looks"

> Context file for AI coding/writing agents working in this repository.
> Read this before writing code, analysis or slide text. If anything here conflicts with a direct instruction from a team member, follow the team member and flag the conflict.

---

## 1. Assignment at a glance

- **Course:** Marketing analytics module, final group project (6 members).
- **Deliverable:** a **15-minute group presentation** covering the marketing problem, data, analysis, conclusions and recommendations.
- **Submission:** slides **and code**, before the start of the lecture on presentation day (date on Canvas).
- **Hard rules from the brief:**
  - Analysis must be **coded by the team** (no point-and-click tools such as Enginius). All code is submitted.
  - **AI usage must be declared.** Log AI use in `AI_USAGE.md` (see §10).
  - Any technique is allowed (OLS, logistic regression, ML, etc.).
  - Results must be **as robust and as free from bias as possible**. This is explicitly graded.

### Rubric (20 points total)

| #   | Criterion                                                | Points | What it means for us                                                      |
| --- | -------------------------------------------------------- | ------ | ------------------------------------------------------------------------- |
| 1   | Research question / business problem, and why it matters | 5      | Frame as a _marketing research_ problem, not a dating study               |
| 2   | Methodology: relevant tools, robust and unbiased results | 5      | Fixed effects, cluster bootstrap, leakage control, halo check, grouped CV |
| 3   | Conclusions and managerial relevance                     | 5      | Concrete recommendations for researchers, product teams, platforms        |
| 4   | Presentation: cogent argument, flow, sources cited       | 5      | One storyline, smooth handovers, citations on slides                      |

---

## 2. Problem statement

**Business problem.** Companies routinely design products, messaging and recommendation algorithms around what customers _say_ they value in surveys and interviews. Stated preferences are distorted by social desirability bias and limited self-insight, so firms may invest in attributes customers praise but don't actually choose on. Matching platforms (dating apps, job boards, marketplaces) face this directly: ranking by users' stated preferences may show the wrong options and hurt engagement.

**Research question.** How closely do people's stated attribute preferences match the weights revealed by their actual choices? Which attributes are most over- or under-stated, and does the gap differ across segments (gender, age, etc.)?

**Sub-questions.**

1. Which attributes are over-stated (e.g. intelligence, sincerity) and which under-stated (e.g. attractiveness)?
2. Do people trade attributes off (compensatory) or use dealbreakers (non-compensatory thresholds)?
3. Do choice-set size and date order change selectivity (choice overload, fatigue)?
4. Do people correct their stated preferences after seeing their own choices?

**Why it matters.** Surveys are among the most widely used market research tools. Quantifying how far stated preferences drift from real choices shows when survey data can be trusted and how platforms should weight revealed behaviour over self-reports.

**One-line pitch.** _Customers tell you what they value. Their choices tell you the truth. How big is the gap, and what should marketers do about it?_

---

## 3. Link to course topics

The presentation should make these links explicit (one mapping slide near the start).

| Project section                                                                                                                          | Course topic                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Survey vs behaviour problem; survey biases                                                                                               | Introduction to Marketing and Marketing Analytics        |
| Stated (self-explicated) vs revealed (choice-based) weights; part-worths and relative importance; ML prediction and a "market simulator" | Conjoint Analysis, Forecasting, and Miscellaneous Topics |
| Benefit segments from individual revealed weights; self-image (`attr3_1`) vs how others rate you (`attr_o`)                              | Segmentation, Targeting, and Positioning                 |
| Value of ranking partners by revealed vs stated weights (higher yes/match rate → engagement → retention/CLV)                             | Measuring Customer Value                                 |

**Key framing:** this is effectively a conjoint study that already happened. The 100-point survey is **self-explicated conjoint**; the yes/no decisions on rated partners are **choice-based**. We compare both on the same people.

---

## 4. Data

- **File:** `data/raw/Speed_Dating_Data.csv` (Columbia Speed Dating Experiment; Fisman, Iyengar, Kamenica & Simonson, 2006, _Quarterly Journal of Economics_).
- **Encoding:** `cp1252`. Load with `pd.read_csv(path, encoding="cp1252")`.
- **Size:** 8,378 rows × 195 columns; 551 participants; 21 events ("waves"), 2002–2004.

### Unit of observation

**One row = one person's view of one speed date.** Every date appears twice (once from each side). ~4,189 actual dates. Each person met ~15 partners on average (range 5–22). Dates lasted ~4 minutes.

### Key columns

**IDs and setup**
| Column | Meaning |
|---|---|
| `iid` | Unique person ID. **Use for fixed effects and grouped CV.** |
| `pid` | Partner's `iid` |
| `gender` | 0 = female, 1 = male |
| `wave` | Event number (1–21) |
| `round` | Number of people met that night |
| `order` | Position of this date in the evening (1st, 2nd, …) |
| `condtn` | 1 = limited choice, 2 = extensive choice (choice overload test) |
| `id`, `partner` | Within-wave numbers only. **Do not use as identifiers.** |

**Outcome**
| Column | Meaning |
|---|---|
| `dec` | **Main target.** This person said yes (1) / no (0). ~42% yes. |
| `dec_o` | Partner said yes to this person |
| `match` | Both said yes (~16%) |

**Ratings of the partner, made right after the date (1–10) — revealed-preference inputs**
`attr` (attractive), `sinc` (sincere), `intel` (intelligent), `fun`, `amb` (ambitious), `shar` (shared interests).
Also `like` (overall liking), `prob` (perceived chance partner says yes), `met` (met before).

**Partner-side columns:** suffix `_o` = "the other person's", e.g. `attr_o` = how the partner rated _this_ person; `pf_o_att` = partner's stated preference for attractiveness; `age_o`, `race_o`.

**Pair-level:** `samerace`, `int_corr` (correlation of hobby ratings).

**Background survey:** `age`, `race`, `field_cd`, `career_c`, `goal`, `date`, `go_out`, `imprace`, `imprelig`, 17 hobby ratings (`sports` … `yoga`), `exphappy`, `expnum`. `income`, `tuition`, `mn_sat` are strings with commas → convert to numeric if used. Prefer coded fields (`field_cd`, `career_c`) over free text (`field`, `career`, `from`).

### Preference-question naming code

Pattern: **`<attribute><question>_<time>`**, attributes = `attr, sinc, intel, fun, amb, shar`.

| Question | Meaning                                        | Scale                            |
| -------- | ---------------------------------------------- | -------------------------------- |
| 1        | What **I** look for in a partner               | 100 points across six attributes |
| 2        | What I think the opposite sex looks for        | 100 points                       |
| 3        | How I rate myself                              | 1–10                             |
| 4        | What I think others of my gender look for      | 100 points                       |
| 5        | How I think others see me                      | 1–10                             |
| 7        | Looking back, what actually drove my decisions | 100 points                       |

| Suffix | When asked                |
| ------ | ------------------------- |
| `_1`   | Sign-up, before any dates |
| `_s`   | Halfway through the event |
| `_2`   | Day after                 |
| `_3`   | 3–4 weeks later           |

**Stated preference = `attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1`** (constant across a person's rows).
`*7_2` vs `*1_1` = self-correction analysis.

### Data quirks (handle explicitly)

- Stated `_1` weights don't always sum to 100 (some waves average 89–98). **Rescale each person's six weights to sum to 100.**
- Some 1–10 ratings are **0**. Decide team-wide: treat as missing or keep, and document it.
- Ratings are ~98% complete for `attr`, weakest is `shar` (~87%). `amb` ~92%.
- Follow-up surveys (`_s`, `_2`, `_3`) have heavy dropout → secondary analyses only.

---

## 5. Methodology

### Core model: revealed weights via conditional logit

- Target: `dec`. Features: standardised `attr, sinc, intel, fun, amb, shar`.
- **Conditional logit grouped by `iid`** (person fixed effects): removes each person's general pickiness so coefficients reflect _trade-offs_ only.
- Run **separately by gender**.
- Convert coefficients to **relative importance out of 100**: clip negatives to 0, divide by sum, × 100. State this choice on the slide.
- Compare with mean stated weights (rescaled to 100) per gender.

### Inference

- **Cluster bootstrap**: resample _people_ (`iid`) with replacement, ~1,000 reps; refit; report 95% CIs for each revealed weight and each gap (revealed − stated). Gap is significant if its CI excludes 0.
- Fix random seed (`SEED = 42`).

### ML cross-check

- Gradient boosting (XGBoost or LightGBM) on the same features.
- **Grouped cross-validation by `iid`** (`GroupKFold`) so no person is in both train and test. Report AUC as a sanity check, not the headline.
- **SHAP** mean |value| shares as a second importance estimate; SHAP dependence plots to look for dealbreaker thresholds.
- Agreement between logit and SHAP rankings = robustness evidence.

### Robustness and extensions

- **Halo effect:** rerun using the partner's _consensus_ ratings (mean rating the partner received from everyone else they met, excluding the current rater).
- **Choice overload / order:** interact attributes or yes-rate with `condtn` and `order`.
- **Self-correction:** compare `*7_2` with `*1_1` and with revealed weights.
- **Segmentation:** mixed-effects logit (random slopes) or per-person estimates → cluster into benefit segments; profile segments by age, `field_cd`, `goal`.
- **Positioning:** self-rating (`attr3_1` …) vs how others rated them (mean `attr_o` …).

### Customer value simulation (person 6)

For each person, rank the partners they met using (a) stated weights and (b) revealed weights. Compare the actual yes-rate among top-k ranked partners under each. Report the lift as "value of ranking by revealed preferences."

### Pitfalls / leakage — do NOT do these

- **Never use `match`, `dec_o`, `like`, `prob` as predictors of `dec`.** They are outcomes or near-duplicates of the decision.
- Don't use random row-level train/test splits; always group by `iid`.
- Don't bootstrap rows; bootstrap people.
- Don't use `id`/`partner` as person identifiers.
- People who said yes to everyone or no to everyone carry no within-person variation and are dropped by conditional logit (~75 people). Report this.

---

## 6. Preliminary results (to be replaced by final numbers)

Conditional logit, standardised ratings, coefficients clipped at 0 and rescaled to 100. **Preliminary, no CIs yet.**

| Attribute        | Women stated | Women revealed | Men stated | Men revealed |
| ---------------- | ------------ | -------------- | ---------- | ------------ |
| Attractive       | 18           | **38**         | 27         | **48**       |
| Sincere          | 18           | 8              | 16         | 3            |
| Intelligent      | 21           | 10             | 19         | 6            |
| Fun              | 17           | 19             | 18         | 24           |
| Ambitious        | 13           | 1              | 9          | 0            |
| Shared interests | 13           | 23             | 11         | 19           |

- Women: 3,457 decisions, 268 people used. Men: 3,583 decisions, 269 people used.
- Ambition not significant for either gender (p ≈ 0.6). Sincerity not significant for men (p ≈ 0.21).
- Other facts: men say yes more often (47% vs 37%).

**Headline (provisional):** attractiveness counts roughly twice as much as people say; intelligence and sincerity count about half or less; ambition has no measurable effect.

---

## 7. Team roles

| #   | Role                     | Owns                                                                                                    | Notebook                  |
| --- | ------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------- |
| 1   | Story lead               | Problem statement, literature, storyline, opening                                                       | —                         |
| 2   | Data lead                | Cleaning, normalisation, EDA, data dictionary                                                           | `01_cleaning.ipynb`       |
| 3   | Core model               | Conditional logit, stated vs revealed, bootstrap CIs, headline chart                                    | `02_core_model.ipynb`     |
| 4   | ML model                 | Gradient boosting, grouped CV, SHAP, dealbreakers                                                       | `03_ml.ipynb`             |
| 5   | Robustness & extensions  | Halo check, overload/order, self-correction, segments, positioning                                      | `04_robustness.ipynb`     |
| 6   | Implications & packaging | Customer value simulation, recommendations, limitations, slide template, code packaging, AI declaration | `05_customer_value.ipynb` |

**Dependencies:** 2 → (3, 4, 5) → 6. Person 2's clean file is the single source of truth.

---

## 8. Repository conventions

```
.
├── PROJECT_CONTEXT.md        # this file
├── README.md                 # how to run, notebook order
├── AI_USAGE.md               # AI usage log (required)
├── requirements.txt
├── data/
│   ├── raw/Speed_Dating_Data.csv     # never modified
│   └── processed/clean_unscaled.csv  # output of 01_cleaning (interpretable scales)
├── notebooks/
│   ├── 00_audit_cleaning.ipynb
│   ├── 01_cleaning.ipynb
│   ├── 02_core_model.ipynb
│   ├── 03_ml.ipynb
│   ├── 04_robustness.ipynb
│   └── 05_customer_value.ipynb
├── src/                       # shared helpers (constants, loaders, plotting)
└── outputs/
    ├── figures/               # PNGs for slides
    └── tables/                # CSV result tables
```

**Rules for code (humans and agents):**

- Python 3; pandas, numpy, statsmodels, scikit-learn, xgboost or lightgbm, shap, matplotlib/seaborn.
- Relative paths only. Read raw data from `data/raw/`, write derived data to `data/processed/`.
- Never overwrite the raw CSV.
- `SEED = 42` everywhere randomness is involved.
- Every notebook must **run top to bottom without errors** from a fresh kernel.
- Shared constants in `src/`, e.g.:
  ```python
  ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
  STATED = [a + "1_1" for a in ATTRS]
  LEAKY = ["match", "dec_o", "like", "prob"]
  ```
- Save figures to `outputs/figures/` at 300 dpi; result tables to `outputs/tables/`.
- Keep code readable and commented; graders read it.

**Chart style (keep consistent across all slides):**

- One fixed colour for **stated**, one for **revealed**, used everywhere.
- Attribute order always: Attractive, Sincere, Intelligent, Fun, Ambitious, Shared interests.
- Label axes in plain English, not column names.

---

## 9. Presentation outline (15 minutes, 6 speakers ≈ 2.5 min each)

1. Hook and problem: "say vs do" in marketing (P1)
2. Course-topic mapping slide (P1)
3. Data: what one row is, what we can and can't infer (P2)
4. Method: stated (self-explicated) vs revealed (choice-based) (P3)
5. **Headline chart: stated vs revealed by gender, with CIs** (P3)
6. ML cross-check and dealbreakers (P4)
7. Robustness, segments, positioning (P5)
8. Customer value of revealed-preference ranking (P6)
9. Recommendations for researchers, product teams, platforms (P6)
10. Limitations and closing (P6)

Cite sources on the slides where claims are made.

---

## 10. Limitations to state

- Sample: Columbia graduate students, 2002–2004 → limited generalisability.
- Heterosexual speed dating only; 4-minute interactions.
- Ratings are subjective; halo effect possible (addressed by consensus-rating check).
- A "yes" is not a long-term relationship outcome.
- Always-yes / always-no participants are excluded from conditional logit.
- Stated weights collected on slightly inconsistent scales across waves (rescaled).

---

## 11. AI usage declaration

Every team member logs AI use in `AI_USAGE.md` as they go:

```
| Date | Member | Tool | Used for | What we changed / verified |
```

Agents: when you generate code or text that ends up in the submission, remind the user to log it.

---

## 12. Key reference

- Fisman, R., Iyengar, S. S., Kamenica, E., & Simonson, I. (2006). Gender differences in mate selection: Evidence from a speed dating experiment. _Quarterly Journal of Economics_, 121(2), 673–697.
