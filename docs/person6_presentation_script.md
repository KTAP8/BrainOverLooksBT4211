# Person 6 speaking script (~5.2 minutes at 150 words a minute, slides 8a–8d, 9–10)

Every slide ends on a decision a marketer can take. Methods, cut lines and Q&A answers are in [person6_methods_explainer.md](person6_methods_explainer.md).

## Slide 8a (~45 sec) — Pooled result + FAQ

*Question this slide answers:* if a platform ranked your dates by revealed weights instead of your survey weights, would you say yes to the top pick more often?

"Person 5 showed that after the event, people's reports of attractiveness rise about eight points—they partly see the gap themselves. So what should a company do with it?

Our question: if a platform ranked your dates by revealed instead of stated weights, would you say yes to the first one more often? We replay each person's evening both ways. It uses ratings made after each date, so it's a best case.

Top-one yes goes from eighty-seven to ninety percent—about three points. Why not twenty? Seventy percent of the time both rules pick the same person, and a yes has a ceiling.

So where does it matter? Only where number one changes."

## Slide 8b (~40 sec) — Where #1 changes + payoff

*Question this slide answers:* when the two rules disagree on who should be first, whose pick gets the yes?

"The question here: when the two rules disagree on who should be first, whose pick gets the yes? That's thirty-one percent of people. Stated's pick gets a yes seventy-eight percent of the time; revealed's gets eighty-eight—ten points. Thirty percent times ten is the three points we just saw.

People say they want brains and sincerity; their choices put looks first. So the first decision is cheap: rank on what people do, not what they write on the form. But a yes is engagement, not revenue. Does it hold before the date, and does it create matches?"

*Speaker note (if asked):* women in this group gain about seventeen points and men about four. Person 5's high-gap group states attractiveness at 24.8 but behaves closer to 31.6, and match rates don't step up or down across gap groups (20%, 16%, 19%).

## Slide 8c (~50 sec) — Pre-date test and mutual matches

*Question this slide answers:* using only what an app knows before the date, does revealed ranking still help, and does it raise matches, not just yeses?

"Now the realistic test: only what a platform knows before two people meet, on events the models never saw. Revealed adds about two points of yeses in the top three; at top one it isn't clear.

But a company doesn't earn from yeses. It earns when people match, because matches keep users on the app. Revealed ranking doesn't clearly raise matches. Ranking for mutual interest does: from twenty-seven percent to thirty-one to thirty-four, and two and a half points on our stricter test. The gain goes to the users most likely to leave: in the least-chosen third, people with no match in their top three fall from seventy-three to fifty-seven percent. It costs a few yeses and needs a cap on mass-likers.

Decision: rank for mutual matches, not swipes."

## Slide 8d (~50 sec) — Behaviour beats surveys

*Question this slide answers:* what predicts how much a person will be wanted later: their full sign-up survey, or their first three speed dates?

"Should a company ask more questions, or watch what people do? We ran it as a prediction contest. The target: what share of partners say yes to a person from date four onwards. One model gets their whole sign-up survey, about forty answers. The other gets only their first three dates: how those partners rated them, and whether they said yes. Both use ridge regression, so forty answers can't just fit noise, and we always score on events the model never saw.

Three speed dates explain twenty-nine percent of the differences between people; the whole survey explains one to ten. A second model, gradient boosting, agrees. Person 5 found adding stated weights doesn't improve prediction.

Decision: shorten sign-up and learn from early activity. The long survey adds friction and buys almost nothing."

## Slide 9 (~95 sec) — Four decisions that drive revenue

*Questions this slide answers:* how do these results become revenue, and what should the company do first? And if every person gets just one exclusive recommendation, like a Daily Pick, which pairing rule produces the most matches?

"So how does this become revenue? A dating app makes money when users stay and pay. And users stay when the app works for them, which means they get matches. Our data has no revenue, but it lets us check each step of that chain.

First, the problem: eighteen percent of people went home with zero matches. Second, those people drift away: only eighty percent of them even answered the next-day follow-up survey, against ninety percent of people who matched. Third, matches turn into real results: weeks later, every extra match raised the chance of going on an actual date by six points.

So more matches mean users who stay engaged and get real dates, and that's what keeps them paying. Here are four ways to get more matches.

One: rank for mutual matches, and cap mass-likers and the most popular profiles—ranking today gives the top fifth of users nearly sixty percent of the slots.

Two: shorten onboarding and treat survey answers as a starting guess, then update from behaviour.

Three: sell scarce introductions, like a Daily Pick. With one exclusive pick each, optimising for both people beats random by six points; coordinating matters more than which weights you use.

Four: in ads and profile prompts, lead with what drives choices—photos, fun and shared interests—not career and ambition.

These are replays, so each one ships as an A/B test: match rate, thirty-day retention and upgrade rate."

## Slide 10 (~30 sec) — Limits and close

*Question this slide answers:* does our headline survive the obvious objections, and what can't this data tell us?

"Limits: Columbia grad students twenty years ago, observational data, replays rather than live tests, and no revenue data—matches stand in for it. But the headline holds: using other people's ratings, attractiveness stays top, and the survey format only trims the gap to eighteen points.

Bottom line: customers tell you what they value; choices tell you the truth. Rank, onboard and market on choices, and optimise for matches. Thank you."
