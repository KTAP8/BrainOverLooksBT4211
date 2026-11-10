# Person 6 speaking script (~5.5 minutes at 150 words a minute, Canva pages 20–26)

Every slide ends on a decision a marketer can take. This file matches the Canva speaker notes word for word. Lines in brackets are Q&A backup and aren't spoken. Methods, cut lines and Q&A answers are in [person6_methods_explainer.md](person6_methods_explainer.md).

## Page 20, slide 8a (~45 sec): Rank by what people do?

*Question this slide answers:* if a platform ranked your dates by revealed weights instead of your survey weights, would you say yes to the top pick more often?

"Person 5 showed that after the event, people's reports of attractiveness rise about eight points—they partly see the gap themselves. So what should a company do with it?

Our question: if a platform ranked your dates by revealed instead of stated weights, would you say yes to the first one more often? We replay each person's evening both ways. It uses ratings made after each date, so it's a best case.

Top-one yes goes from eighty-seven to ninety percent—about three points. Why not twenty? Seventy percent of the time both rules pick the same person, and a yes has a ceiling.

So where does it matter? Only where number one changes."

## Page 21, slide 8b (~40 sec): When the top pick changes

*Question this slide answers:* when the two rules disagree on who should be first, whose pick gets the yes?

"The question here: when the two rules disagree on who should be first, whose pick gets the yes? That's thirty-one percent of people, the ones whose number one changes. Stated's pick gets a yes seventy-eight percent of the time; revealed's gets eighty-eight. That's ten points. Thirty percent times ten is the three points we just saw.

People say they want brains and sincerity; their choices put looks first. So the first decision is cheap: rank on what people do, not what they write on the form. But a yes is engagement, not revenue. And here's the leak."

(If asked: women in this group gain about seventeen points and men about four. Person 5's high-gap group states attractiveness at 24.8 but behaves closer to 31.6, and match rates don't step up or down across gap groups: 20%, 16%, 19%.)

## Page 22, new slide (~30 sec): Most yeses go nowhere

*Question this slide answers:* how much interest is wasted, and what question should a platform ask instead?

"A yes only pays off if it comes back. Across all eight thousand dates, half end one-sided. Out of every hundred yeses, sixty-one are never returned—these faded hearts. For the user, that's effort with no payoff, and a reason to leave.

So the question for a platform isn't 'who will you like?' but 'who will like you back?' Ranking for both sides cuts those wasted slots in the top three from thirty-eight to twenty-eight percent. Let's test that properly."

## Page 23, slide 8c (~45 sec): Rank for both sides

*Question this slide answers:* using only what an app knows before the date, does revealed ranking still help, and does it raise matches, not just yeses?

"Now the realistic test: we give the model only what a platform knows before two people meet, on events it never saw. Revealed weights add about two points of yeses in the top three, but they don't clearly raise matches.

Ranking for both sides does: from twenty-seven to thirty-four percent. It also beats the obvious alternative, one model trained directly on matches, by about four points. And the gain goes to the users most likely to leave: among the least-chosen third, people with no match in their top three fall from seventy-three to fifty-seven percent. It costs a few yeses and needs a cap on mass-likers.

Decision: rank for mutual matches, not swipes."

(If asked: the lift is +2.5 to +7.1 points across our tests; the stricter early-evening test gives +2.5.)

## Page 24, slide 8d (~45 sec): Watch, don't ask

*Question this slide answers:* what predicts how much a person will be wanted later: their full sign-up survey, or their first three speed dates?

"Should a company ask more questions, or watch what people do? We ran it as a prediction contest: what share of partners say yes to a person from date four onwards? One model gets their whole sign-up survey, about forty answers. The other gets only their first three dates: how those partners rated them, and whether they said yes. Both use ridge regression, scored on events the model never saw.

Three speed dates explain twenty-nine percent of the differences between people; the whole survey explains only one to ten percent. Person 5 found adding stated weights doesn't improve prediction either.

Decision: shorten sign-up and learn from early activity. The long survey adds friction and buys almost nothing."

(If asked: ridge stops forty correlated answers from fitting noise. A second model, gradient boosting, gives the same ranking: dates .25, survey -.03 to .03.)

## Page 25, slide 9 (~95 sec): Matches pay the bills

*Questions this slide answers:* how do these results become revenue, and what should the company do first? And if every person gets just one exclusive recommendation, like a Daily Pick, which pairing rule produces the most matches?

"So how does this become revenue? A dating app makes money when users stay and pay, and users stay when the app works for them, which means they get matches. Our data has no revenue, but it lets us check each step of that chain.

First, the problem: eighteen percent of people went home with zero matches. Second, those people drift away: only eighty percent of them even answered the next-day follow-up survey, against ninety percent of people who matched. Third, matches turn into real results: weeks later, every extra match raised the chance of an actual date by six points.

Picture two users. Julia gets zero matches, decides the app doesn't work, and deletes it. Sam gets four, chats, goes on a date and stays, so he's the one paying twenty dollars a month. More matches turn Julias into Sams. Four ways to get there.

One: rank for mutual matches, and cap mass-likers and the most popular profiles; ranking today gives the top fifth of users nearly sixty percent of the slots.

Two: shorten onboarding. Treat survey answers as a starting guess, then update from behaviour.

Three: sell scarce introductions, like a Daily Pick. With one exclusive pick each, optimising for both people beats random by six points.

Four: in ads and profile prompts, lead with what drives choices: photos, fun and shared interests, not career and ambition.

These are replays, so each one ships as an A/B test: match rate, thirty-day retention and upgrade rate."

(If asked about the Daily Pick: coordinating the picks matters more than which weights you use; the structured rules can't be separated.)

## Page 26, slide 10 (~30 sec): Limits and close

*Question this slide answers:* does our headline survive the obvious objections, and what can't this data tell us?

"Limits: Columbia grad students twenty years ago, observational data, replays rather than live tests, and no revenue data—matches stand in for it. But the headline holds: using other people's ratings, attractiveness stays top, and the survey format only trims the gap to eighteen points.

Bottom line: customers tell you what they value; choices tell you the truth. Rank, onboard and market on choices, and optimise for matches. Thank you."
