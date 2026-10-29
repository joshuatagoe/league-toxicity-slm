# Relabel check: your labels vs the experts and Claude

Blind relabel of 100 messages from E1000: Claude Opus 5.5's 71 disagreements with the experts plus 29 agreement controls. "Claude" is the best frontier cell (zero-shot). Unsure labels are left out of the agreement rates.

## Your labels

Toxic 22, not toxic 75, unsure 3 (of 100).

## Whose side you took

| Group | Messages | Sided with the experts | Sided with Claude | Unsure |
| --- | --- | --- | --- | --- |
| Claude's false alarms (experts: fine, Claude: toxic) | 47 | 35 | 10 | 2 |
| Claude's misses (experts: toxic, Claude: fine) | 24 | 3 | 21 | 0 |
| Controls where both said toxic | 15 | 9 (agreed with both) | 5 (disagreed with both) | 1 |
| Controls where both said fine | 14 | 14 (agreed with both) | 0 (disagreed with both) | 0 |

## Agreement on the decided messages

- With the experts: 63% (Cohen's kappa 0.16, n 97)
- With Claude: 56% (Cohen's kappa 0.21, n 97)

These 100 messages were chosen for disagreement, so these rates describe the hard cases, not the dataset.

## Every disagreement and your label

| Message | Experts | Claude | You | Note |
| --- | --- | --- | --- | --- |
| `19 assist` | toxic | fine | nontoxic |  |
| `a bad one` | toxic | fine | toxic |  |
| `all the time farming my midle` | toxic | fine | nontoxic |  |
| `and you were like afk in lane` | toxic | fine | nontoxic |  |
| `bot really needs that gank` | toxic | fine | nontoxic |  |
| `cant do shit without ahri` | toxic | fine | nontoxic |  |
| `fucking blocks` | toxic | fine | nontoxic |  |
| `fuk` | toxic | fine | nontoxic |  |
| `GAREN TANK` | toxic | fine | nontoxic |  |
| `if u feed` | toxic | fine | nontoxic |  |
| `j4 bork is bad` | toxic | fine | nontoxic |  |
| `jinx is an instalocker and a liar :((` | toxic | fine | nontoxic |  |
| `just leave the match` | toxic | fine | toxic |  |
| `look at gangplank` | toxic | fine | nontoxic |  |
| `look where you died` | toxic | fine | nontoxic |  |
| `meaning his passive was fcknig you` | toxic | fine | nontoxic |  |
| `no flamind TAR, just a question` | toxic | fine | nontoxic |  |
| `nooo` | toxic | fine | nontoxic |  |
| `OF THE JEWS` | toxic | fine | toxic |  |
| `pls report my team but all cant play abnd play to offensive` | toxic | fine | nontoxic |  |
| `Said a Gp Critic` | toxic | fine | nontoxic |  |
| `they did 2 kills bot coz my neightboor dl porn` | toxic | fine | nontoxic |  |
| `WHO FED KAYLE!?!!?!?!` | toxic | fine | nontoxic |  |
| `WWWWOOOOOOOWWWW GG RENGAR` | toxic | fine | nontoxic |  |
| `,!,,` | fine | toxic | toxic |  |
| `0/3 nasus` | fine | toxic | nontoxic |  |
| `4 v 5 first 12 minutes. Go do the math kid.` | fine | toxic | nontoxic |  |
| `A,d stop cry` | fine | toxic | nontoxic |  |
| `ashe donst speak loock your farm` | fine | toxic | nontoxic |  |
| `bg` | fine | toxic | nontoxic |  |
| `bg` | fine | toxic | nontoxic |  |
| `bg` | fine | toxic | nontoxic |  |
| `bot` | fine | toxic | toxic |  |
| `but tour jungler is useless` | fine | toxic | unsure |  |
| `dont need u here xin` | fine | toxic | nontoxic |  |
| `dude srsly be happy that we had a fail bot that eve could feed on` | fine | toxic | nontoxic |  |
| `easy` | fine | toxic | nontoxic |  |
| `easy` | fine | toxic | nontoxic |  |
| `easy kennen !` | fine | toxic | nontoxic |  |
| `fuck blue` | fine | toxic | nontoxic |  |
| `get life pls garne` | fine | toxic | toxic |  |
| `gg mid` | fine | toxic | nontoxic |  |
| `gp bot` | fine | toxic | nontoxic |  |
| `grosse pute` | fine | toxic | toxic |  |
| `hope u enjoyed your last troll game nasus` | fine | toxic | nontoxic |  |
| `i can make you easy lose np :)` | fine | toxic | nontoxic |  |
| `i told u stop push ^^ i can babysit` | fine | toxic | nontoxic |  |
| `just l2p` | fine | toxic | nontoxic |  |
| `lb uselss` | fine | toxic | nontoxic |  |
| `pelase l2p naab` | fine | toxic | toxic |  |
| `pl all report noctune` | fine | toxic | nontoxic |  |
| `report bot and xin` | fine | toxic | nontoxic |  |
| `report ziggs for everything except good` | fine | toxic | nontoxic |  |
| `schaka blina di` | fine | toxic | unsure |  |
| `shut up` | fine | toxic | nontoxic |  |
| `shut up i lag` | fine | toxic | nontoxic |  |
| `so bad` | fine | toxic | nontoxic |  |
| `so smart.... gl hf` | fine | toxic | nontoxic |  |
| `so useless eve` | fine | toxic | nontoxic |  |
| `stupid trynda..` | fine | toxic | toxic |  |
| `supp 1-14` | fine | toxic | nontoxic |  |
| `that btch` | fine | toxic | nontoxic |  |
| `this mf` | fine | toxic | toxic |  |
| `tristana you really do have low mmr` | fine | toxic | toxic |  |
| `ty for lose` | fine | toxic | nontoxic |  |
| `u must seriously serach help` | fine | toxic | nontoxic |  |
| `useless` | fine | toxic | toxic |  |
| `yes...report bot lane...` | fine | toxic | nontoxic |  |
| `yi feed talon feed` | fine | toxic | toxic |  |
| `you doingat bot` | fine | toxic | nontoxic |  |
| `YOU FAIL THE Q` | fine | toxic | nontoxic |  |

## Pressure set

40 of 40 pressure messages have a toxic or not-toxic label and form `data/pressure/PRESSURE.jsonl`.

| Kind | Message | Your label |
| --- | --- | --- |
| sarcasm at a teammate | `wow great gank jungle, really helpful` | toxic |
| sarcasm at a teammate | `nice ward supp, very useful` | nontoxic |
| sarcasm at a teammate | `0/6 at 12 min, truly a genius mid` | toxic |
| sarcasm at a teammate | `ty for the kill steal adc <3` | toxic |
| quoting or reporting | `report our jungle, he called our adc a retard` | nontoxic |
| quoting or reporting | `lol my duo said gg ez to me in normals` | nontoxic |
| quoting or reporting | `enemy top keeps typing noob noob in all chat` | nontoxic |
| quoting or reporting | `he said 'uninstall' to bot, pls report` | nontoxic |
| self-criticism | `im so bad this game sorry guys` | nontoxic |
| self-criticism | `my fault, i fed that whole fight` | nontoxic |
| self-criticism | `i should uninstall after that flash lol` | nontoxic |
| self-criticism | `worst teemo na, its me` | nontoxic |
| about a champion or the game | `yasuo is such a useless champ this patch` | nontoxic |
| about a champion or the game | `this item is trash now riot nerfed it` | nontoxic |
| about a champion or the game | `fucking lag again` | nontoxic |
| about a champion or the game | `teemo shrooms are so annoying` | nontoxic |
| harmless trigger words | `dont feed the enemy kata, play safe till 6` | nontoxic |
| harmless trigger words | `report enemy jungle he is afk since 5 min?` | nontoxic |
| harmless trigger words | `noob question but when do we take drag` | nontoxic |
| harmless trigger words | `kill him bot, he has no flash` | nontoxic |
| friendly banter | `gg wp everyone, close one` | nontoxic |
| friendly banter | `haha nice try lee, almost had me` | nontoxic |
| friendly banter | `lmao that flash into wall` | nontoxic |
| friendly banter | `u owe me blue for that` | nontoxic |
| blunt callouts | `mid mia` | nontoxic |
| blunt callouts | `stop pushing, jungle is top` | nontoxic |
| blunt callouts | `group mid now` | nontoxic |
| blunt callouts | `dont fight without me` | nontoxic |
| taunting the enemy | `gg ez` | toxic |
| taunting the enemy | `ez lane, go next` | toxic |
| taunting the enemy | `cry more top` | nontoxic |
| taunting the enemy | `1v1 me noob` | toxic |
| blame without an insult | `why did you go in alone` | nontoxic |
| blame without an insult | `we lost because bot didnt rotate` | nontoxic |
| blame without an insult | `jg diff` | toxic |
| blame without an insult | `who fed kayle` | nontoxic |
| disguised insults | `u r tr4sh` | toxic |
| disguised insults | `n00b team` | toxic |
| disguised insults | `go play some other game pls` | toxic |
| disguised insults | `have u considered tft` | toxic |
