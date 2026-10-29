# Full Growth RL Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train and verify stats, level growth, every evolution branch and mastery using main game interfaces.

**Architecture:** Preserve the trained 54-action combat/recovery policy. Add independent masked stat and evolution categorical heads; adapt the three choices to main's existing 1,620-action decoder. Main mechanics and reward/config defaults remain unchanged. Real evaluations start at level 1; fixture XP is used only by mechanics tests.

**Tech Stack:** Gymnasium, Stable-Baselines3 PPO, PyTorch, unittest.

- [ ] Add failing `tests/test_full_growth.py`: action encoding, unavailable choices, stats, levels 10/20/30/40, policy masks, transfer/save/resume.
- [ ] Extend training settings/environment/runtime for optional growth and factorized actions; keep combat checkpoints compatible.
- [ ] Add growth-aware observation features and policy, preserving trained combat weights and legal growth masks.
- [ ] Add CLI migration, growth metrics and monitor displays.
- [ ] Add reproducible verification command reporting mechanics tests separately from natural-play progression and branch coverage.
- [ ] Run new tests, existing job/dashboard checks and actual short PPO job; record unrelated pre-existing failures separately.
- [ ] Migrate the best combat checkpoint without deleting it; launch full-growth training and verify progress/window.

Training episodes allow 600 virtual seconds for later growth. Standard score evaluation remains capped at 100 seconds or death. An optional longer natural-play verification reports the levels actually reached, without treating fixture coverage as learned competence.
