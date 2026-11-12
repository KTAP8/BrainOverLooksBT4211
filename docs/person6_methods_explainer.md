# Person 6: methods behind slides 8a–10

Reference notes for Person 6, not for the audience. For each slide: the question, how the test works, the result, the catch, and where the numbers come from. Lines cut from the script are at the end, word for word.

All numbers come from `outputs/tables/`. The full run uses 1,000 bootstrap draws and seed 42.

## 1. Terms

| Term | Meaning |
|---|---|
| **pp** | Percentage points. 87% to 90% is +3 pp |
| **Paired comparison** | Both rules rank the same person's dates. Each person gets a difference, and the differences are averaged, so each person is compared with themselves |
| **Bootstrap CI** | Resample people with replacement 1,000 times, recompute, keep the middle 95%. If it excludes 0, the effect is "clear" |
| **Held-out events** | Train on 20 of the 21 events, test on the one left out, and repeat for all 21. This is how an app meets new users |
| **R²** | Share of the variation a model predicts. Out of sample it can be negative, meaning worse than guessing the average |
| **AUC** | How well a model separates yeses from nos. 0.5 is chance, 1 is perfect |

## 2. Foundation: Jiayi's revealed weights (Part 3)

- **Stated weights:** at sign-up, each person splits 100 points across six traits (`*1_1`).
- **Revealed weights:**
  1. A conditional (fixed-effects) logit compares each person's yes dates with their no dates, using the six ratings they gave.
  2. Because it only compares within a person, someone who says yes to everyone doesn't distort it.
  3. The coefficients are standardised, negatives are set to 0, and the rest are rescaled to sum to 100.
  4. It's fitted separately for women and men.
- **Result** (`core_model_results.csv`), revealed vs stated:

  | Trait | Women | Men |
  |---|---|---|
  | Attractiveness | 37.9 vs 17.7 | 47.4 vs 26.7 |
  | Fun | 19.8 | 24.0 |
  | Shared interests | 22.4 | 20.4 |
  | Ambition | 1.4 | 0 |

  This table is the evidence for decision four on slide 9.

## 3. Slide 8a: post-date replay

**Question:** if a platform ranked your dates by revealed instead of stated weights, would you say yes to the top pick more often?

**Method** (`notebooks/05_customer_value.ipynb`, `src/customer_value.py`):
1. Take each of the 452 people and their roughly 15 dates.
2. Score each date as the sum of weight × the rating the person gave that partner after the date. Do it twice: their own stated weights, and the revealed weights for their gender.
3. Sort the dates and take the top 1 and the top 3.
4. Record whether they actually said yes. Average across people, take the paired difference, and bootstrap it.

**Result:** top-1 yes goes from 87% to 90%, +3.1 pp (CI +0.9 to +5.8). Top 3 is +2.0 pp.

**The catch:**
- It's a best case. The ratings were made after the date, so they already reflect how it went, and the weights were fitted on the same data.
- `add_match_scores` applies standardised weights to raw ratings. With the correct scaling the lift is +2.7 pp, so the story doesn't change.
- **Why not 20 points?** In about 70% of people both rules pick the same partner, so the lift there is 0. A yes or no also has a ceiling. Jiayi's 20 is an importance score from 0 to 100, while this is a yes rate.

**Files:** `customer_value_bootstrap.csv`, `customer_value_top1_agreement.csv`.

## 4. Slide 8b: where the top pick changes

**Method:** the same replay, split by whether the two rules pick the same top partner. 315 people agree. In 138 (31%) the rules pick different partners, and for them we compare the yes rate on each rule's pick.

**Result:** 78% vs 88%, +10 pp (CI +2.2 to +18.1). Women gain about 17 pp, men about 4. The arithmetic is 31% × 10 pp ≈ 3 pp, the pooled figure.

**The catch:** only 138 people, so the interval is wide. They were chosen *because* the rules disagree, so this is the lift where ranking matters, not for everyone.

**Files:** `customer_value_lift_by_segment.csv`.

## 5. Slide 8c: pre-date test and mutual matches (A5)

**Question:** using only what an app knows before the date, does revealed ranking still help, and does it raise matches?

**Step 1: replace your ratings with other people's** (`add_split_fold` in `src/ext_features.py`).
- An app doesn't know your rating before you meet, but it knows how other users rated the partner.
- Everyone is randomly put in group A or B. Your rating of partner J is replaced by J's average rating from raters in the other group.
- **Why not "everyone except you"?** That average secretly contains your answer. If you rated J low, the others' average looks relatively high, so ranking on it picks people who rejected you. That bug once made a good rule look worse than random. `tests/test_extended.py` checks the fix.

**Step 2: rank on held-out events** (`ranking_frame` in `src/ext_market.py`).

| Rule | Scores a partner by |
|---|---|
| Random | Random order |
| Stated | Your survey weights × others' ratings |
| Revealed | Conditional logit refitted on others' ratings |
| Mutual interest | log P(you say yes) + log P(they say yes) |

P(they say yes) comes from a logistic regression that uses:
- how often the partner says yes in general, measured in the other group only
- the partner's stated preferences
- shared interests, age gap and same race

**Step 3:** record the yes rate and match rate in the top 3 and the top 1, compared in pairs with a bootstrap.

**Result (top 3):**

| Rule | Yes | Match |
|---|---:|---:|
| Random | 43% | 18% |
| Stated | 65% | 27% |
| Revealed | 67% | 28% |
| Mutual interest | 62% | 34% |

- **Revealed vs stated, yeses:** +2.3 pp (CI 0.6 to 4.1). At top 1, not clear (+2.2 pp, CI -1.5 to 6.4).
- **Revealed vs stated, matches:** +0.7 pp (CI -1.0 to 2.3), so not clear.
- **Mutual interest, matches:** +4.1 to +7.1 pp, at a cost of 3.5 to 5.0 pp of yeses.
- **Who benefits:** in the least-chosen third, people with no match in their top 3 fall from 73% to 57%.

**Three checks:**
1. **Five random group splits** give +4.1 to +7.1. That's why you quote the range: +7.1 is the luckiest split.
2. **A stricter history design** (`split_history`): the first half of the evening is history, and only the second half is ranked, so no feature can hold a later outcome. This gives +2.5 pp at top 3 (CI 1.2 to 3.9) and +5.1 pp at top 1.
3. **Removing a piece:** without the partner's general yes rate the lift is -0.7 pp. The whole gain comes from steering people towards partners who say yes often, hence the cap on mass-likers.

**Exposure** (`ext_a5_exposure.csv`): the top 20% of users get 30% of slots under random order, against 56% (stated), 61% (revealed) and 54% (mutual interest). Ranking concentrates attention on popular users.

**Files:** `ext_a5_lifts.csv`, `ext_a5_seed_sensitivity.csv`, `ext_a5_history_design.csv`, `ext_a5_partner_model.csv`, `ext_a5_who_benefits.csv`, `ext_a5_exposure.csv`.

### 5b. Where interest is wasted (A5j, new)

**Question:** how much interest goes unreturned, and does ranking for both sides fix it?

**Method** (`endings_table` in `src/ext_market.py`; the idea comes from a Kaggle notebook's "four endings" chart):
- Every date ends one of four ways: match, only I said yes, only they said yes, or neither.
- First counted over all 8,378 dates. Then counted in each person's top 3 under each ranking rule, from the same pre-date replay as 8c.
- The change is tested with the usual paired person-level bootstrap.

**Result:**

| | Match | Only I said yes | Only they said yes | Neither |
|---|---:|---:|---:|---:|
| All dates | 16% | 26% | 25% | 33% |
| Top 3, stated weights | 27% | 38% | 14% | 21% |
| Top 3, revealed weights | 28% | 40% | 13% | 20% |
| Top 3, both likely to say yes | 34% | 28% | 20% | 19% |

- 51% of dates are one-sided, and **61% of yeses are never returned** (men 65%, women 55%).
- Ranking for both sides cuts unreturned-yes slots by 10.6 pp (CI 8.3 to 12.7).

**The catch:** the top-3 rows are a replay of people already in the room, as in 8c.

**Files:** `ext_a5_endings.csv`, figure `ext_a5_endings.png`.

### 5c. Why not just predict matches directly? (A5k, new)

**Question:** our 8c rule combines two models (I'd say yes × they'd say yes). Would one model trained on matches do just as well?

**Method** (`direct_vs_two_sided`, `rank_quality`):
- In `ranking_frame`, a logistic regression is trained on `match` with exactly the facts the two-sided rule uses: the partner as rated by others in the opposite fold, the partner's yes rate, their stated weights and the pair facts.
- It's trained on the other events and tested on the held-out one, as before.
- Top-1 and top-3 match rates are compared with the paired bootstrap, over the same five random splits.
- How well each rule *orders* a list is also scored within each person, with ROC-AUC and average precision (PR-AUC) against the base rate. PR-AUC is the Kaggle notebook's suggestion for a rare outcome. Within-person scoring matters because a ranking only reorders one person's list.

**Result:**
- **The two-sided rule wins:** top-3 matches 34.0% against 30.0%, +4.0 pp (CI 2.4 to 5.5). Across five splits the range is +1.2 to +4.0, and four of five are clearly above zero.
- **Ranking quality for matches:** ROC-AUC .71 (two-sided), .67 (direct model), .63 (stated or revealed). Average precision .55 against a 21% base rate.
- **A match is only a little harder to rank than a yes:** .71 vs .76.

**The catch:** the reason it wins is plausible but untested. A match model only learns from the rarer combined outcome, while two models learn from every yes and no.

**Files:** `ext_a5_direct_vs_two_sided.csv`, `ext_a5_rank_quality.csv`.

### 5d. Felt vs actual shared interests (S4, new, supports decision 4)

**Question:** decision 4 says to lead with shared interests. Should an app match on hobby overlap, or help people feel common ground?

**Method** (`s4_common_ground`):
- Match rate by band, on our core sample, for:
  - felt shared interests (`shar`, rated after the date);
  - actual hobby overlap (`int_corr`, the correlation of the two people's hobby ratings from sign-up);
  - age gap and same race.
- Intervals resample people, not rows.
- Slopes come from a linear probability model per SD, with person-clustered errors, both raw and within each rater (subtracting a rater's mean removes how readily they say yes).

**Result:**
- **Felt shared interests:** 5% → 35% match from lowest to highest band (+12.9 pp per SD).
- **Actual hobby overlap:** 16% → 20% (+1.1 pp per SD, p=.035, **adjusted .052**, so borderline).
- **Age gap:** 20% → 9% from 0-2 to over 10 years.
- **Same race:** no difference (18% vs 18%).

**The catch:**
- Felt shared interests is rated after the date and partly reflects overall liking, so it isn't something an app sees in advance.
- The point is the contrast: the feeling matters, the measured overlap barely does.
- **Use:** prompts and icebreakers that surface common ground, not hobby filters.

**Files:** `ext_s4_common_ground.csv`, `ext_s4_common_ground_slopes.csv`, figure `ext_s4_common_ground.png`.

**About the Kaggle notebook:** it uses a *synthetic* dataset (14,504 dates, 2,548 people, no missing values), so none of its numbers are cited. Only three ideas were reused: four endings, the direct-vs-two-sided question with PR-AUC, and banded charts.

## 6. Slide 8d: behaviour beats surveys (A3)

**Question:** what predicts how much a person will be wanted later, their sign-up survey or their first three speed dates?

**Method** (`a3_survey_vs_behaviour` in `src/ext_market.py`):
- **Target:** the share of partners in dates 4 onwards who said yes to the person.
- **Survey predictors (about 40):** demographics, hobbies, stated weights, self-ratings and lifestyle.
- **Behaviour predictors (4):** ratings and yeses received in dates 1 to 3.
- **Model:** ridge regression, with 7-fold cross-validation grouped by event, scored by out-of-sample R². The gap between the two R² values is bootstrapped.

**Why ridge:** ridge is ordinary regression plus a penalty on large coefficients.
- With about 275 people per gender and about 40 correlated survey columns, ordinary regression would fit noise.
- `RidgeCV` picks the penalty strength from 20 values between 1 and 10,000.
- Predictors are standardised and missing values filled with the median.
- Ridge gives the survey its best chance, so "the survey still loses" is a strong claim.

**Why cross-validation:**
- In-sample R² always rises with more columns, and the survey has ten times more. Testing on unseen events removes that head start.
- Grouping folds by event matters because people at one event share partners. Mixing them between training and testing would flatter the model.
- Some survey rows have negative R², worse than guessing the average. In-sample R² would hide that.

**Result:**

| | Ridge R² | Boosting R² |
|---|---:|---:|
| Women: full survey | .01 | -.03 |
| Women: first 3 dates | .29 | .25 |
| Men: full survey | .10 | .03 |
| Men: first 3 dates | .29 | .25 |

- Gap: +.29 (CI .19 to .38) for women, +.19 (CI .10 to .28) for men. Gradient boosting agrees.
- **Person 5 (section 7):** predicting a person's own yes from ratings gives AUC .827 with or without stated weights.

**The catch:**
- "Dates" are four-minute speed dates. In an app, the equivalent would be early swipes or messages.
- Race, field and career are number codes treated as amounts, which slightly understates the survey. Boosting handles this better and agrees.
- A3 predicts how much a person is wanted, while Person 5 predicts what a person chooses. They're different targets that point the same way.

**Files:** `ext_a3_prediction.csv`, `ext_a3_r2_gap.csv`.

## 7. Slide 9: market simulator (S3)

**Question:** if every person gets just one exclusive recommendation, like a Daily Pick, which pairing rule produces the most matches?

**Method** (`history_candidates`, `simulate_market`, `s3_market_simulator`):
- Run each event separately with the history design: the first half of the evening predicts the second half.
- Each person gets exactly one partner, with no double-booking.

| Rule | How it pairs |
|---|---|
| Random | Random pairing, averaged over 30 tries |
| Stated stable matching | Gale-Shapley on stated scores. Pairs swap until no two people would both rather be together |
| Revealed stable matching | The same, on revealed scores |
| Optimised | The Hungarian algorithm (`linear_sum_assignment`) maximises the total P(I say yes) × P(they say yes) |

- **Outcome:** did the assigned pair actually match?
- **Interval:** resampling the 21 events.

**Result:**
- Match rates: random 17.7%, revealed 20.0%, stated 22.1%, optimised 24.1%.
- **Optimised vs random:** +6.3 pp (CI 2.5 to 10.3).
- **Optimised vs stated:** +1.9 pp (CI -4.4 to 8.8), not clear.
- With two slots, all three structured rules beat random by 4.4 to 5.0 pp.

**The catch:** only 21 events, so the intervals are wide. Coordination clearly helps, but no structured rule clearly beats another.

**Files:** `ext_s3_lifts.csv`.

## 8. Slide 9: the revenue chain

The data has no revenue. Each link uses a measured stand-in.

| Link | Number | Source | Strength |
|---|---|---|---|
| Many users get nothing | 18% get no match | `ext_a5_funnel.csv` | Solid |
| No match, less engaged | Next-day response 80% vs 90% (p=.007) | `ext_a5_funnel.csv` | **Weakest:** survey response is only a stand-in for retention |
| Matches become dates | +6.2 pp per extra match (CI 3.5 to 8.9, n=226) | `ext_a5_funnel.csv` | Moderate: only 48% answered the follow-up |
| Ranking can reduce "nothing" | 73% → 57% no match, least-chosen third | `ext_a5_who_benefits.csv` | Replay, not a live test |
| Attention is concentrated | Top 20% get 56-61% of slots | `ext_a5_exposure.csv` | Descriptive |

The step from matches to retention to subscription revenue is an industry assumption, not something measured here. That's why every decision ships as an A/B test with match rate, 30-day retention and upgrade rate.

## 9. Slide 10: robustness checks

| Check | Method | Result |
|---|---|---|
| Other people's ratings (R1) | Replace your rating with the partner's average from others and refit. "Everyone except you" is fine for a regression, just not for ranking | Attractiveness still top: 54.2 (women), 68.9 (men) |
| Survey format (R4) | Waves 6-9 used a 1-10 scale. Rerun on 100-point waves only | Gap 18.4 and 17.3, against 20.2 and 20.7 overall. Disclose it |
| Person 5: consensus split | Split your rating into peer consensus plus your deviation, and use a likelihood-ratio test | p < .001: both matter |
| Person 5: order | Rescale position to 0-1 and add it to the fixed-effects logit | p = .78 and .86: no drop in yeses. This doesn't prove anything about fatigue as such |

## 10. Person 5's other methods

**Recall shift:**
- **Method:** sign-up weights (`*1_1`) against the next-day "what mattered in your decisions" (`*7_2`). Both are normalised to 100, then a paired bootstrap.
- **Result:** attractiveness +8.3 points (CI 6.0 to 10.9, n=123).

**Say-do gap groups:**
- **Method:**
  1. Per person, compare ratings of accepted and rejected dates trait by trait.
  2. Turn the positive differences into behavioural weights that sum to 100.
  3. The gap is half the sum of absolute differences from stated weights.
  4. Split people into thirds by gap.
- **Result:** the high-gap group states 24.8 on attractiveness but behaves like 31.6. Match rates are 20%, 16% and 19%, with no trend.

**Event condition:**
- **Method:** a logistic regression with clustered errors.
- **Result:** extensive-choice events show lower yes odds (OR .82, p=.05), but this is confounded with event.

## 11. Previous-partner effect (Q&A backup, S2)

- **Method:** a fixed-effects regression of this yes on the previous partner's consensus attractiveness.
- **Placebo:** shuffle each person's date order many times and rerun. Shuffling gives the "order means nothing" distribution, and the p-value is the share of shuffles at least as large as the real effect.
- **Result:** -1.4 pp per standard deviation (SD) (p=.008, adjusted .018), mostly among men (-1.7 pp). This replicates Bhargava and Fisman (2014).

## 12. Ready Q&A answers

- **Why not 20 points of lift?** Seventy percent of the time both rules pick the same person, and a yes has a ceiling. The 20 is an importance score, not a yes rate.
- **Isn't the 8a result unrealistic?** Yes, it uses post-date ratings. Slide 8c is the realistic version: +2.3 pp at top 3.
- **Why cross-validation for a regression?** We're comparing prediction, and in-sample R² always favours the side with more columns. Testing on unseen events is the fair test.
- **Where's the revenue?** It isn't measured. Matches stand in for it, linked by the chain in section 8, and each decision is an A/B test on retention and upgrades.
- **Doesn't mutual ranking just push mass-likers?** Yes, all of the lift comes from partner yes rate, so we recommend a cap.
- **Is "Daily Pick" better because of revealed weights?** No. Coordination beats random, but the weights can't be separated.
- **Halo effect?** Using other people's ratings, attractiveness stays top.
- **Why not just predict matches directly?** We tried it. One model trained on matches, using the same facts, gets 30% top-3 matches. Ranking on both sides' yes gets 34% (+4.0 pp, CI 2.4 to 5.5; +1.2 to +4.0 across five splits).
- **How much interest is wasted?** Half of all dates are one-sided, and 61% of yeses are never returned. Ranking for both sides cuts unreturned yeses in the top 3 from 38% to 28%.
- **Does model B check whether you meet the partner's preferences?** We tested it as a mirror of model A: the partner's weights × how others rated you. It adds about one point at most (+1.4 pp with their stated weights, +0.7 pp with weights from their own choices; side test, A5l). Their pickiness already explains most of their yes, and about 15 dates are too few to learn one person's taste.
- **Isn't revealed better only with lots of data?** Partly. Everyone's pooled revealed weights predict *each person's own* held-out choices better than that person's survey: AUC .76 vs .73 before the date, .90 vs .88 after (A5m). But weights learned from one person's own dates don't catch up within 12 dates. So start from everyone's behaviour and blend in personal choices slowly.
- **Would the partner's behaviour beat their survey with more data?** We tested it: learning each partner's taste from 2 up to 12 of their dates never beat their survey on top-3 matches (−0.2 to −1.5 pts, no steady trend; A5n). One evening isn't enough to learn one person's taste, so use their survey first and switch to behaviour as it builds up.
- **Should apps match on shared hobbies?** No. The feeling of shared interests goes with a 5% → 35% match rate, but actual hobby overlap only 16% → 20%. Build prompts, not filters.

## 13. Lines cut from the script

The full previous script is below, word for word. When to use the cut lines:

| Cut line | Use it when |
|---|---|
| 8a "We replay each participant's evening… Same four hundred fifty-two people…" | Someone asks how the replay works |
| 8a "Jiayi's twenty is an importance score…" | "Why isn't the lift 20 points?" |
| 8b "We're not comparing the same person twice…" and the women/men line | Someone misreads 78% vs 88% |
| 8b "If someone asks what would show an even bigger gap…" | "Would a real app show more?" |
| 9 "In the data, the decision is already first contact…" | "Do you model second dates?" |
| 9 the feed paragraph and the researcher/product/platform line | Someone asks about feeds vs scarce slots |
| 10 "Person 5's checks show no drop in yeses…" | Order or fatigue questions |

### Previous script (before the revenue rewrite)

> #### Person 6 speaking script (~5.5 minutes at 150 words a minute, slides 8a–8d, 9–10)
> 
> ##### Slide 8a (~90 sec) — Pooled result + FAQ
> 
> *Question this slide answers:* if a platform ranked your dates by revealed weights instead of your survey weights, would you say yes to the top pick more often?
> 
> "Person 5 showed that after the event, people's own reports of attractiveness rise about eight points. So they partly see the gap themselves. What should a company do with it?
> 
> Jiayi showed stated and revealed weights diverge—especially for attractiveness. Our business question: if a platform ranked your dates using revealed instead of stated weights, would you say yes more often to whoever is shown first?
> 
> We replay each participant's evening: score every date with the six ratings, rank with either their survey weights or our gender-specific revealed weights, and measure yes rate on the top one and top three. Same four hundred fifty-two people as the core model. One caution: this replay uses ratings made after each date, so treat it as a best case. I'll test a realistic version shortly.
> 
> Here's the pooled result. Top-one yes rate goes from about eighty-seven to ninety percent—roughly three percentage points overall. Top-three moves about two points. So the headline is modest but positive.
> 
> Why not twenty points? About seventy percent of the time both rules pick the same person, so there's no lift for them, and a yes or no has a ceiling. Jiayi's twenty is an importance score, and this is a yes rate on people already in the room.
> 
> So where does it actually matter? Next slide: only the people whose number one changes."
> 
> ##### Slide 8b (~70 sec) — Where #1 changes + payoff
> 
> *Question this slide answers:* when the two rules disagree on who should be first, whose pick gets the yes?
> 
> "The question here: when the two rules disagree on who should be first, whose pick gets the yes? This chart is one hundred thirty-eight participants—about thirty-one percent—where stated number one and revealed number one are different partners.
> 
> We're not comparing the same person twice. We're comparing: if the platform surfaced stated's top pick, yes about seventy-eight percent of the time; if it surfaced revealed's top pick, about eighty-eight percent—a ten-point lift. Women in this slice see an even larger bump; men about four points—see the footnote on the chart.
> 
> That ten points is where the pooled three points comes from: roughly thirty percent of people times about ten points when ranking actually changes who is first.
> 
> This is the customer-value punchline for our project title. People say they want brains and sincerity; choices put more weight on looks. When survey weights and choice weights disagree on who should be first, in this replay, ranking by revealed preferences moves real yes decisions—not just what people write on the form.
> 
> If someone asks what would show an even bigger gap: a full profile catalog instead of fifteen met dates, or pickier users."
> 
> *Speaker note (Person 5's segments, if asked):* the high-gap group states attractiveness at 24.8 but behaves closer to 31.6, and match rates don't step up or down across the gap groups (20%, 16%, 19%).
> 
> ##### Slide 8c (~50 sec) — Pre-date test and mutual matches
> 
> *Question this slide answers:* using only what an app knows before the date, does revealed ranking still help, and does it raise matches, not just yeses?
> 
> "Now the realistic question: if a platform only knows what it knows before two people meet, does revealed ranking still help? We test it on events the models never saw. Any ranking beats the forty-five percent average yes rate. Revealed adds about two points of yeses in the top three, and at top one the difference isn't clear.
> 
> But yes isn't the goal. A match needs both people. Revealed ranking doesn't clearly raise mutual matches. Ranking for mutual interest does: from about twenty-seven percent to thirty-one to thirty-four, and about two and a half points on our stricter test. It costs three to five points of yeses, and the lift comes from favouring people who say yes more often, so a real product would need a cap on mass-likers."
> 
> ##### Slide 8d (~20 sec) — Behaviour beats surveys
> 
> *Question this slide answers:* what predicts how much a person will be wanted later: their full sign-up survey, or their first three speed dates?
> 
> "So should a company ask more questions, or watch what people do? Three speed dates predict who'll be wanted later far better than the whole sign-up survey: twenty-nine percent of the variation, against one to ten. Person 5 found the same from another angle: adding stated weights to a prediction model leaves accuracy unchanged. Behaviour is the better signal."
> 
> ##### Slide 9 (~70 sec)
> 
> *Questions this slide answers:* what should marketers do? And if every person gets just one exclusive recommendation, like a Daily Pick, which pairing rule produces the most matches?
> 
> "So what should marketers do—and how does this translate to a real app?
> 
> In the data, the decision is already first contact: my yes versus mutual match. We are not modelling second dates.
> 
> Our slides eight a and b are the infinite-feed problem: who is number one in my queue? Revealed-style ranking can lift yeses there, but pre-date the gain is small, about two points in the top three.
> 
> Limited recommendations raise a different question: if everyone gets just one exclusive pick, like a Daily Pick, which pairing rule produces the most matches? In our simulation, optimising for both people beats random by about six points, but the structured rules are too close to separate. So the value is coordinating scarce slots, not which weights you use. This is mutual interest at a first introduction, not second dates.
> 
> For researchers: surveys are hypotheses, and use them alongside early behaviour. For product: don't build copy only from stated weights. For platforms: pick the KPI—engagement on my feed versus mutual matches on scarce slots—then pilot the matching policy that fits. Test yes, match, and retention in production; our numbers are historical replays."
> 
> ##### Slide 10 (~35 sec)
> 
> *Question this slide answers:* does our headline survive the obvious objections, and what can't this data tell us?
> 
> "We're limited to Columbia grad speed dating twenty years ago, observational data, and a historical simulation—not a live test. Does our headline survive the obvious objections? It holds up: using other people's ratings, attractiveness stays the top trait, and the survey format difference only trims the gap to about eighteen points. Person 5's checks show no drop in yeses across the evening.
> 
> Bottom line: customers tell you what they value; choices tell you the truth. The gap is large enough to change how we research, design, and rank. Thank you."
