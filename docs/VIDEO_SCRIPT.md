# Voiceover Script — Bobathon Video 4 min

> **Recommended format:** light slides + code / Skore screen in the background.
> Target duration: exactly 4 min (≈ 490 words at a comfortable reading pace).
> The text below is to be read as-is; `[SHOW]` blocks indicate what should be on screen.

---

## INTRO — 0:00 → 0:15

**[SHOW: title slide "Bobathon ESILV — Parkinson True-OFF Score" + Kaggle logo + IBM Bob logo]**

> Hi. Our Bobathon project: predicting the true motor OFF score of Parkinson's patients — and showing how IBM Bob shaped every step of that work.

---

## DIM 1 — CHALLENGE & LEARNING — 0:15 → 1:10

**[SHOW: Figure 4 (patient trajectory), then JOURNAL.md RMSE column from 16.50 down to 3.35]**

> The goal is to predict the **true OFF** score: a debiased version of the MDS-UPDRS motor rating, stripped of drug-timing effects, rater subjectivity, and missing values.
>
> The data is 44,590 clinical visits from 5,576 patients. Up to 42% of visits have no OFF measurement — that missingness is itself a signal, not just noise.
>
> The biggest learning was **honest evaluation**. A naive random split gave us RMSE 2.5 on LightGBM — impressive, but wrong. Once we grouped the split by patient, the real score was 4.5. The Kaggle holdout is by patient too, so that gap would have shown up immediately on the leaderboard.
>
> Beyond that, we learned how to build longitudinal features — summarising each patient's history of visits into slopes, means, and progression curves — without ever touching the target labels.

---

## DIM 2 — IBM BOB USAGE — 1:10 → 2:10

**[SHOW: Bob interface with a journal conversation, then Skore Hub report exp 33]**

> We used IBM Bob from day one. For **data exploration**, Bob ran the EDA, generated an interactive table report, and wrote a summary file — two hours of manual work done in minutes.
>
> For **each experiment**, Bob drafted a design note in `journal/` before writing a single line of code. That step forces you to articulate *why* you are trying something, not just run it.
>
> The clearest example: experiment 33, our best model. We described a three-part architecture — calibrate the OFF/ON measurements, fit a progression curve per patient, then blend two LightGBMs. Bob generated the full script, ran it, checked the Skore report fold by fold, and updated the journal automatically.
>
> Without Bob, we would not have reached experiment 33 in the time available. We would have stopped around experiment 12.

---

## DIM 3 — TECHNICAL CONTINUITY — 2:10 → 3:05

**[SHOW: project file tree in Bob (`experiments/`, `journal/`), then RMSE progression chart]**

> Three things from this setup we will keep for all future ML projects.
>
> First, **plain Python scripts** instead of notebooks — one file per experiment, versioned in git, with its design note alongside. No broken notebooks, no output drift.
>
> Second, **Skore Hub** — every cross-validation report gets a permanent URL. We pushed 33 experiments. Without that, results disappear into terminal output.
>
> Third, **patient-grouped cross-validation by default** on any longitudinal data. That is the single most important technical lesson: a correct split is worth more than a better model.
>
> What we would do differently: run hyperparameter search with Optuna from experiment 8, GroupKFold-aware. That would have saved several manual iterations.

---

## DIM 4 — OVERALL EXPERIENCE — 3:05 → 3:50

**[SHOW: full JOURNAL.md History table, experiments 01 → 33]**

> The hackathon was intense and well-structured. We split the work into two tracks — features and model architecture — and the Bob journal kept a shared log of every decision and result. No duplication, no "did we already try that?".
>
> The main difficulty: the transformer experiments took far too long to train and never beat LightGBM. We should have checked the compute cost earlier.
>
> What to improve: GPU access from the start, and more time on individual patient trajectories — that is where the gap to the top of the leaderboard sits.

---

## OUTRO — 3:50 → 4:00

**[SHOW: Skore Hub report exp 33 — RMSE 3.35]**

> Our best model reaches **RMSE 3.35**, down from 16.5 for the dummy. IBM Bob was the tool that made 33 iterations possible in two days. Thank you.

---

## Summary — what to show on screen

| Moment | Recommended screen |
|---|---|
| 0:00 – 0:15 | Title slide + IBM logo + Kaggle logo |
| 0:15 – 1:10 | Figure 4 (patient trajectory) + JOURNAL.md RMSE column |
| 1:10 – 2:10 | Bob interface (journal conversation) + Skore Hub exp 33 |
| 2:10 – 3:05 | Project file tree in Bob (`experiments/`, `journal/`) + RMSE chart |
| 3:05 – 3:50 | Full JOURNAL.md History table, 01 → 33 |
| 3:50 – 4:00 | Skore Hub report exp 33 — RMSE 3.35 |
