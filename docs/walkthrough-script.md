# Walkthrough video: narration script

For the 3–5 minute early-submission demo (the tuned model succeeding where the base model fails). Read it over the screen recording, `docs/walkthrough.mp4` (4:43, silent). Each `## NN.` section is one clip you record in the studio (`demo/video/studio.py`). It starts on the recording's title card with the same `caption()` id. About 670 words; each scene is sized to its section at about 2.5 words a second, so a calm pace fits.

Every number comes from [results/slm_results.md](../results/slm_results.md), [results/ablation1.md](../results/ablation1.md) or the run files in `runs/`. The page shows answers from the models live, except the Claude column, which is read from Claude's eval run. The page's answers matched the eval runs on all 40 pressure lines and all 8 benchmark lines it shows.

## 00. Intro (title card: "League chat toxicity: a 1.7B model vs prompting")

This is my Week 7 project: a small model that reads one League of Legends chat line and answers with one word, toxic or nontoxic. It's Qwen3-1.7B with a LoRA adapter. The data is L2DTnH: about fifteen thousand real chat lines labelled by experienced players, and I trained on about eleven thousand of them. The spec is on screen. What matters most is precision, because a false flag punishes an innocent player. So the bar, set before any run, is precision of at least 0.80 and F1 of at least 0.75.

## 01. Prompting has a ceiling (title card: "1. Prompting has a ceiling")

First: does prompting already solve this? I ran Claude Opus 5.5 and GPT-5.6 Sol, three prompts each, on the same thousand benchmark lines. The best cell, Claude with a plain zero-shot prompt, reached precision 0.624. GPT's best F1 was 0.568, about what a keyword list scores. Longer prompts with the annotators' guideline made Claude worse, not better. Neither family came close to 0.80, so a tuned small model had something to win.

## 02. Base vs tuned on real chat (title card: "2. The untrained model misjudges real chat")

Here's the untrained base model next to two tuned versions on real benchmark lines. I picked these from the lines the untrained model gets wrong, leaving out slurs, so this is the flattering view. It lets plain insults through, like "gg easy noobs", and it flags harmless lines, like a report call or someone apologising. On one it doesn't answer at all: it just says "troll". Across all thousand lines, v2 fixes 77 of the untrained model's mistakes and makes 20 new ones.

## 03. Typed live (title card: "3. Lines it has never seen, typed live")

These lines are typed live and don't appear in any training or eval file, so they have no label; right and wrong here is my reading of the rules. "Go back to iron noob" is an insult: the untrained model lets it through, both tuned models flag it. "Report mid for afk" is a report call: the untrained model flags it, the tuned ones don't. And an honest one, "trash jungler": v1 flags it, v2 lets it through.

## 04. The data fix (title card: "4. A data fix for what v1 got wrong")

v1 learned only from the experts' labels. On my forty-line pressure set it caught 3 of 11 toxic lines and flagged 5 harmless ones, mostly lines with hostile-looking words. So for v2 I changed only the data. Claude wrote 720 lines aimed at those patterns, a GPT critic checked them against my written rules, and 676 went into training. On the pressure set, v2 stops flagging "fucking lag again" and "cry more top", and now catches "jg diff". On the benchmark, the change is within noise.

## 05. Still missed (title card: "5. What it still misses")

But the main failure isn't fixed. Hostility without insult words gets "nontoxic" from all three small models: sarcasm like "wow great gank jungle, really helpful", "gg ez", and polite put-downs. Claude's best prompt catches all four. The new lines are six percent of the training data, seen once, and the expert lines outnumber them sixteen to one. Next I'd weight them more, and test on lines from a different writer.

## 06. Data-efficiency curve (title card: "6. More expert data stops helping")

This is the data-efficiency curve: v1 trained on nested subsets, from 1,364 to 10,919 expert-labelled rows. F1 climbs steeply up to about 5,500 rows, then flattens. Doubling to all the data adds 0.01, which is within noise. No size reaches the 0.75 F1 bar, and the pressure-set score doesn't improve with size. With four sizes and one seed, about 5,500 rows is an estimate, not a measured minimum. My guess is that different data would help more than more of the same.

## 07. The numbers (title card: "7. The numbers, and how to check them")

The shipped model is v2. On the benchmark, its precision is 0.857, which clears the bar, and its F1 is 0.698, which misses it. That's level with Claude's best prompt on F1, with much higher precision, at about an eighth of a second per line in batches on a laptop GPU. Every number comes from per-line run files in the repo. The adapter is public on Hugging Face, pinned to a commit, and one command reruns the eval. Running it from the Hub matched my local run on every line.

## 08. Close (title card: "Thanks for watching")

Thanks for watching. The code, the data and every run are on GitHub, and the model is on Hugging Face.
