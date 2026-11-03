# Jiayi: Part 3 speaking notes

About 2–2.5 minutes. Use the main chart and, if space permits, the gap-interval chart.

## Slide 1 — How we compare statements and choices

“We compare what people said they valued before the event with the attributes associated with saying yes after a date.

We use a conditional logit, separately for women and men. This compares a person's choices across their own dates, accounting for their overall tendency to say yes. The predictors are the six ratings they gave each partner, standardized so coefficients describe a one-standard-deviation change.

We use 6,222 decisions from 452 people. We exclude incomplete records and people whose remaining decisions are all yes or all no. To keep the comparison fair, we calculate stated preferences from those same people, counting each person once.”

## Slide 2 — Attractiveness receives more weight in choices

“The blue bars show stated importance; the orange bars show model-derived importance. Following our group's method, we set negative coefficients to zero and normalize the positive standardized coefficients to sum to 100.

For women, attractiveness rises from 17.7 stated points to 37.9 revealed points. For men, it rises from 26.7 to 47.4. The gaps are approximately 20 points in both groups.

We resample whole people 1,000 times per gender and rerun the analysis. The 95% gap intervals are 15.3 to 24.8 for women and 15.6 to 25.3 for men. Both exclude zero under this model and bootstrap procedure.

Intelligence and sincerity receive less relative weight in choices, while shared interests receive more. Fun's gap is inconclusive for women. These are individual comparison intervals; we have not adjusted for testing multiple attributes.

The takeaway for market researchers is to combine survey preferences with behavioral validation. These findings do not show that people lied or that attractiveness causes a yes. Ratings can reflect a halo effect or chemistry, and this measures an initial meeting rather than long-term compatibility.

Next, we check whether a more flexible machine-learning model tells a similar story.”

## If asked

- **Why is men's ambition zero?** Its coefficient is slightly negative and the agreed importance transformation clips it to zero. That is not proof that ambition is irrelevant. Signed coefficients and intervals are retained in the results table.
- **Why drop always-yes/always-no people?** They provide no within-person outcome variation for the conditional likelihood. We report the exclusions.
- **Can we say men care more than women?** Not as a statistical conclusion from these separate fits. That requires a formal gender-difference test.
- **Are these literal true preference percentages?** No. They are relative, standardized model-importance shares, compared descriptively with survey allocations.
- **What is the main uncertainty limitation?** People within the same event share partners; person-level resampling does not capture all cross-person dependence.

Chart footer: “Conditional logit; standardized ratings; negative coefficients clipped to zero; importance sums to 100; 1,000 participant bootstrap draws; seed 42. Source: Columbia Speed Dating data / Fisman et al. (2006).”
