# ICRA rescue draft — open items

Build: `cd ICRA-rescue && latexmk -pdf main.tex` (uses `../refs.bib`, `../images/`).
Orange `\pending{}` = experiment not finished. Magenta `\verify{}` = taken from
run READMEs / code comments, confirm against raw data. Values live in `constants.tex`.

## Implementation facts that constrain the claims (checked in gait-net @ 5c13681)

1. **Hardware decision rate is not 25 Hz.** `gaitnet_min_request_period` = 0.25 s gates every
   request and `gaitnet_noop_cooldown` = 0.2 s (`load_controller.launch:288,306`). The launch
   comment measures a 0.516 s per-leg slot. Paper states "at most 4 Hz" on hardware.
2. **Duration head is overridden on hardware.** Floor `max(0.25, travel/0.8)`, cap 0.6 s
   (`gaitnetservice.py:_finalize_action`). Sim default floor is 0.
3. **Hardware multi-leg is diagonal-only** (`gaitnet_diagonal_only` true on hw). Hardware
   overlap was 0.3 % of time (`runs/4_*/README.md`). Do not claim unrestricted overlap on hardware.
4. **Hardware logit shaping:** `noop_logit_penalty` 7.0, `repeat_penalty` 1.5, starve-force
   step (`starve_force_trail` 0.15). The one terrain failure was a starve-forced step.
   Consider ablating or at least reporting how often starve-force fired.
5. **Gazebo default selection mode is `pga`, not `dense`** (`load_controller.launch:117`).
   The Gazebo benchmark must pass `gaitnet_selection_mode:=dense` or the paper must say otherwise.
6. **Categories 3 vs 4 in `runs/` may be the same config.** `gaitnet_single_swing` already
   defaulted to false at 4d8b6f8 (2026-09-08) and only acts when a leg is disabled; the
   obsavoid_13 bags are from 2026-09-10. Check the `[config]` line in those bags.
7. **Foothold is projected** onto a planar region and the swing target is constrained to a
   ±0.10 m square (`ConvexRegionSelector.cpp`). The executed foothold ≠ GaitNet's argmax in
   general; log the projection distance.
8. **Leg orderings differ:** feet in Isaac Lab order FL FR RL RR; contacts and terrain slots
   FR FL RR RL (`gaitnetservice.py:62-81`). Confirm `FootstepAction.msg` leg-id documentation.
9. **Training controller source is not inspectable** (empty `gaitnet/gaitnet/control`
   submodule → opsullivan85/rl-mpc-locomotion). Confirm the "Mini-Cheetah convex MPC" description.
10. **Gazebo WBC rate** not confirmed (Table II `XX`).
11. **Newer checkpoint on origin/master** (37×37 grid at 1.0 cm). All numbers here assume the
    25×25 / 1.5 cm checkpoint `gaitnet_20251101_012343-c01ed84-footstep-cost_700.pt`.
12. **Architecture figure** (`images/diagrams/gaitnet-architecture.png`) is reused from the
    journal; check its labels still match Table II.

## Experiments still needed
- Gazebo benchmark (Table III, Fig. 2): 5 conditions × difficulty × speed, matched seeds.
- Concurrency stats in Gazebo (overlap %, pair distribution, blocked requests).
- Enumeration study: dense vs 65-sampled vs intermediate K.
- End-to-end latency and realized decision interval in Gazebo.
- Scheduled NMPC baseline definition (native legged_perceptive gait + footholds).
