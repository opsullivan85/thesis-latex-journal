# Gait timing on the G125 benchmark: GaitNet vs OCS2's own trot

- **Source:** `rss/timming.md`, a CSV despite the extension, with 134,412 swings.
- **Scripts:** `rss/figures/gait_stats.py` holds the statistics; run `python gait_stats.py` to print the full report. `rss/figures/fig_gait.py` draws the figure (`rss/figures/fig_gait.pdf` / `.png`).
- **Checking:** two independent recomputations, one with an exact event sweep and one with a 0.5 ms grid, written without seeing these scripts, reproduced every headline number exactly. Two skeptical reviews followed. Their corrections are applied below, including a run-up step-count bug in the first version of `gait_stats.py`.

## 1. What the file contains

- **Arms:** `c0` = GaitNet (the iteration-350 checkpoint with the OCS2 IDM), `trot` = OCS2's own trot.
- **Cells:** 0.10 and 0.20 m/s × 26, 38 and 49 % voids, world seeds 111–210, 20 s trials.
- **Only successful trials (confirmed by the author).** The number of trials in each cell equals the benchmark's success count for that cell: GaitNet 100 93 96 | 96 87 77 = 549, trot 97 89 80 | 94 81 63 = 504. Every trial also runs to about 20 s. So every number below is conditional on success, and the failed trials, where gait would matter most, are missing. Checks on the world seeds that both arms passed change little: the step rate at 49 %/0.20 m/s is 7.28 vs 7.29 steps/s, and the non-diagonal overlaps are 2.17 vs 2.27 per trial.
- **Measured swings, not commanded ones.** The events come from simulation ground truth (author). The documented rule (gaitnet-docs, `experiment-setup.md`) counts a foot as up while it is 1.5 cm above its rolling minimum; the author has not confirmed that this rule made this file. The trot's same-leg liftoff interval is 0.598 s, which matches its 0.6 s period. However, its measured swing is 0.235 s against the scheduled 0.3 s, and 21 % of its time measures as all four feet down, which its schedule never has. Absolute swing times, duty factors and feet-up shares therefore depend on the detector. Only comparisons between the arms under the same detector are meaningful.
- **Time resolution is about 21 ms.** Every timestamp is a sample of the ground-truth log. The log is documented as 50 Hz, but samples are 20–34 ms apart in simulated time (median 21 ms), and the spacing varies within a trial. A guess, not checked: a 50 Hz wall-clock logger on a simulation that runs slightly slower than real time. Both arms sit on this grid; for example, the trot's swings cluster at 213, 235 and 256 ms. A lag or overlap of one sample is at the resolution limit.
- **The `counted` column** means the touchdown lies between field entry and 20 s. 441 swings that land after 20 s are logged but not counted. Field entry is when the rear feet reach the void field (author). It is at about 5.3 s at 0.10 m/s and about 2.8 s at 0.20 m/s. The front feet reach the field earlier, so the end of the run-up is already on the field.

## 2. Figure (`fig_gait.pdf`)

- **(a) Footfall diagram.** World seed 134 at 49 % voids and 0.20 m/s, the first 3 s after field entry, with the same world for both arms. The seed is chosen by a rule, not by eye: of the 53 seeds that succeeded in this cell for both arms, it is the one whose GaitNet feet-up shares are closest to the cell's pooled shares. A bar is a measured swing. Filled bars are the FL+RR pair and open bars the FR+RL pair.
- **(b) Steps per second against void fraction.** GaitNet per speed, and the trot.
- A feet-in-swing panel (share of time with 0, 1 or 2 feet up) was removed on 2026-11-04 at the author's request. Its numbers stay in F3 and in `gait_stats.py`'s report.

Draft caption:

> Gait timing on OCS2, successful benchmark trials only (549 GaitNet, 504 trot), measured on the void
> field. (a) Measured swings of each leg in one world (49 % voids, 0.20 m/s), 3 s after field
> entry. Filled: FL and RR; open: FR and RL. (b) Steps per second, GaitNet at each speed and OCS2's
> own trot. Of the time GaitNet had two feet up, 99 % was on a diagonal pair, against 100 % for the
> trot.

## 3. Findings

Each finding states what was measured; an interpretation, where one is given, is labelled.

**F1. GaitNet pairs diagonal legs, like the trot.**
- 99.0 % of its two-feet-up time is on a diagonal pair.
- Of its 24,793 overlapping swing pairs, 24,370 (98.3 %) are diagonal, 227 rear, 158 same-side and 38 front.
- The trot is 100 % diagonal (26,713 pairs).
- This repeats the docs' finding for SC-idm@1590 on exact terrain (2874 diagonal, 15 rear, 8 same-side) on the benchmark checkpoint with depth terrain.
- Measured swings are shorter than commanded ones, so the non-diagonal counts are a lower bound. At 0.20 m/s and 49 %, 175 non-diagonal pairs overlap, but 761 come within 50 ms of each other.

**F2. A diagonal pair does not lift together.**
- The partners' liftoffs are a median 42 ms apart (10–90 %: 21–66 ms, i.e. 1–3 log samples). Only 3 % fall in the same sample, against 97 % for the trot.
- The touchdowns are not systematically staggered. The signed touchdown lag (the later lifter minus the earlier) has a median of 0 ms (IQR −41 to +24 ms), and in 64 % of pairs the later lifter lands no later than the first.
- This stagger is built into the action space: GaitNet starts at most one step per 40 ms tick (methodology.tex:15), so it is **not a learned behaviour**. What the network chooses is to start the partner on a later tick (measured lags of 1–3 samples), and to give it the shorter swing.

**F3. GaitNet spends more time with all four feet down and steps singly more often.**
- Measured all-feet-down time:

  | | all four feet down | of which pauses > 0.5 s |
  | --- | --- | --- |
  | GaitNet, 0.10 m/s | 51 % | 14 % of field time |
  | GaitNet, 0.20 m/s | 39 % | 0.7 % |
  | Trot (both speeds) | 21 % | 0 % |

  So at 0.10 m/s, much of GaitNet's four-feet-down time is standing still, not double support.
- One foot up: GaitNet 29–33 %, trot 1 %.
- 10.9 % of GaitNet's field steps overlap no other swing (7.1 % if a one-sample gap still counts as touching). The trot has none, apart from 5 edge cases.
- Consecutive diagonal-pair events alternate between the two pairs in 71 % of cases (65 % at 0.10 m/s, 75 % at 0.20 m/s), against 100 % for the trot. Of the repeats, 82 % are the same pair stepping twice in a row with no other step in between.

**F4. GaitNet changes its step rate with speed; the trot does not.**

| | 26 % voids | 38 % voids | 49 % voids |
| --- | --- | --- | --- |
| GaitNet, 0.10 m/s | 5.35 | 5.47 | 5.63 |
| GaitNet, 0.20 m/s | 7.04 | 7.17 | 7.29 |
| Trot, every cell | 6.66–6.67 | 6.66–6.67 | 6.66–6.67 |

(steps/s)

- The trot's rate is exactly 4 legs / 0.6 s.
- The speed effect holds world by world: all 255 worlds that GaitNet passed at both speeds step faster at 0.20 m/s.
- Part of the lower rate at 0.10 m/s comes from pauses (F3), not slower stepping.
- The rise with void fraction (about +0.25 steps/s from 26 to 49 %) is small and compares different survivor sets.
- Measured swing length does not change with speed: the median is 128 ms at both, against 235 ms for the trot. So GaitNet takes about as many steps as the trot, each with about half the measured swing.

**F5. GaitNet's stride is irregular and bimodal.**
- Same-leg liftoff interval: median 0.556 s, 10–90 % 0.23–1.09 s, CV 0.61 (0.67 at 0.10 m/s, 0.47 at 0.20 m/s). The trot: 0.598 s, CV 0.02.
- About 26 % of GaitNet's strides are under 0.3 s, quick re-steps of the same leg. The main mode is 0.5–0.8 s, and the tail is pauses (7.6 % of strides at 0.10 m/s are over 1.5 s).
- The median is therefore a poor summary. GaitNet has no fixed period, which agrees with "no gait schedule".

**F6. Non-diagonal pairs concentrate in the hardest cell.**

| | 26 % voids | 38 % voids | 49 % voids |
| --- | --- | --- | --- |
| 0.10 m/s, share of pairs | 0.33 % | 0.56 % | 1.51 % |
| 0.10 m/s, per trial | 0.11 | 0.19 | 0.53 |
| 0.20 m/s, share of pairs | 1.33 % | 1.92 % | 3.95 % |
| 0.20 m/s, per trial | 0.75 | 1.10 | 2.27 |

- At 0.20 m/s and 49 %, 68 of the 77 successful trials have at least one non-diagonal pair.
- Within a trial, non-diagonal overlaps are about 3× as frequent on the field as in the run-up at 0.20 m/s (0.077 vs 0.026 per s). The run-up is not all flat ground, because the front feet enter the field before field entry.
- **Interpretation (correlation only):** this is the cell where GaitNet-Diagonal lost the most (59 vs 77 %), so it fits the Leg-pairs paragraph. Three things weaken it:
  - The link across cells is not monotone. At 0.20 m/s and 26 % voids, 53 of 96 trials use a non-diagonal pair, yet GaitNet-Diagonal lost 0 points there.
  - The 18-point loss is partly within run-to-run variation, since the rerun scored 74 in that cell.
  - Even in that cell, 96 % of pairs are diagonal.
- "Relies on other pairs" is therefore stronger than the data show. "Uses other pairs more often there" is what was measured.
- The causal reading cannot be tested with the existing logs: there are no swing logs for the failed trials or for GaitNet-Diagonal (author).

**F7. GaitNet is left-right asymmetric; the trot is not.**
- Right legs spend more time in swing (measured duty):

  | | FL | FR | RL | RR |
  | --- | --- | --- | --- | --- |
  | 0.10 m/s | .865 | .770 | .864 | .812 |
  | 0.20 m/s | .810 | .765 | .778 | .746 |

- Right mean duty is below left in 99 % of trials, and in all six cells.
- Right legs lead staggered diagonal pairs more often: FR 37 % and RR 31 % against FL 20 % and RL 13 % at 0.10 m/s.
- The trot measures 0.604–0.610 on every leg with the same detector, so this is not a detector effect.
- **Untested hypothesis (the author does not know whether G125 used this config):** gaitnet-docs (experiment-setup.md:241) records a sim config with `dead_contact_feet [RF_FOOT, RH_FOOT]`, the two right feet. If the G125 runs used it, the right feet's contact came from the MPC schedule, and GaitNet's history and contact inputs would differ between the sides.

**F8. Run-up vs field.**
- Within each speed, step rate and feet-up shares barely change: at 0.10 m/s, 5.46 vs 5.48 steps/s; at 0.20 m/s, 6.82 vs 7.16, which is +5 %.
- Non-diagonal overlaps rise 1.6× (0.10 m/s) and 3× (0.20 m/s) on the field (F6).
- The run-up includes the start-up transient, and its end has the front feet on the field (field entry is the rear feet's). So this is not a flat-vs-voids comparison, and the run-up/field contrast understates any effect of the voids.

**F9. The trot runs on its stated schedule.**
- The partners lift in the same sample in 97 % of pairs, and the pairs alternate in 100 % of events.
- Period 0.598 s, step rate 6.66 steps/s, and the same on every leg and in every cell.
- The 0.3 s scheduled swing cannot be checked with this detector. The 78 % two-feet-up share is about 0.235/0.3, so it is consistent with it.

**Data quality.**
- No leg has overlapping swings with itself.
- 421 GaitNet field swings are under 50 ms and 70 are over 300 ms, both rare. The short ones are not bounces: they follow a median 0.41 s of stance.
- Three feet were up for 23 ms (one sample) in one trial (GaitNet, 0.20 m/s, 49 %, seed 206).
- 709 trot swings start at t = 0, so the trot is mid-swing when recording starts. This affects only the run-up.
- Field entry is the rear feet reaching the field (author). It coincides with a foot event in 21 % of trials. About 20 % of log samples carry a foot event, so 21 % is what chance gives for any time on the log grid. The coincidence says nothing about how field entry was computed.

## 4. What this means for the paper

These are proposals only; no paper file was edited.

1. **conclusions.tex:17.** "We have not measured which legs GaitNet swings together on OCS2. Since GaitNet-Diagonal lost success mainly in the hardest cell, GaitNet may have learned a mostly trot-like pairing with free timing and footholds."
   - The first sentence is now outdated, and F1, F4 and F5 confirm the hypothesis in the second.
   - Proposed: "In GaitNet's successful benchmark trials, 98 % of its overlapping measured swings paired diagonal legs, as in a trot, but its timing was not a trot's. Its stride varied widely, it often paused or stepped one leg at a time, and its step rate rose with speed."
2. **results.tex:233–234 (Leg pairs).** "the drop suggests that it relies on other pairs on dense voids at the faster speed".
   - Proposed: keep the hedge, add the measurement, and soften "relies on". For example: "Unrestricted, GaitNet paired diagonal legs in 98 % of its overlapping measured swings, but it used other pairs most often in that cell, about twice per successful trial against 0.1 at 0.10 m/s and 26 % voids. The drop therefore suggests that it uses other pairs on dense voids at the faster speed, …"
   - Mention that measured counts understate how often the diagonal limit binds.
3. **results.tex:132–135 ("Our untested interpretation …").** F3 and F5 show, in aggregate, that GaitNet waits and moves single legs. The link to which footholds were in reach remains untested, so "untested" could become "tested only in part". Optional.
4. **results.tex:88 (`\verify{0.3 s swings, and a 0.6 s period}`).** The 0.6 s period is now confirmed (0.598 s). The 0.3 s swing is not measurable here, so keep `\verify` on the swing only.
5. **methodology.tex:15.** If F2 is cited, say that the one-step-per-tick rule is why diagonal partners never lift together, so that readers do not take the stagger as learned.
6. **constants.tex:209–212.** `\corlGzDiagonalPairs/RearPairs/SidePairs/NoopPct` come from SC-idm@1590 on exact terrain and are used nowhere in the .tex files. Mark them as superseded, or replace them with macros from this file.
7. The figure is not yet referenced in any .tex file. Whether it goes in the main text, the appendix (not reviewed at L4DC) or nowhere is the author's call, given the page limit.

## 5. Author's answers (2026-11-04) and what stays open

1. **Only successful trials?** Yes. Every number here is conditional on success.
2. **How were liftoff and touchdown detected?** From the simulation. Still open: whether it is the documented 1.5 cm rolling-minimum rule, and why the samples are about 21 ms apart rather than 20.
3. **How is `field_entry_s` defined?** The answer was "read entry". It is read here as "rear entry", the rear feet reaching the void field; please correct this if it meant something else. F6, F8 and the data notes now treat the run-up as partly on the field.
4. **Did G125 load `dead_contact_feet [RF_FOOT, RH_FOOT]`?** Unknown. F7's cause stays open, so the paper should not name one.
5. **Can failed trials and GaitNet-Diagonal be logged the same way?** No. F6 stays a correlation. The Leg-pairs paragraph can report the measured counts but cannot claim GaitNet needs the other pairs.
