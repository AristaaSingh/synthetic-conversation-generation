# The Evaluator (Objective C)

The evaluator is the classifier that scores generated conversations for misogyny —
whether, how severe, and which kind. It exists because the CHI paper (Lagos Rojas
et al., 2026) shows a binary LLM judge ceiling-rates everything with near-zero
variance; a purpose-trained, multi-head classifier is the instrument that can
actually discriminate. It is also the load-bearing component for the whole
fine-tuning plan: it establishes the baseline, and it is what proves (or disproves)
that the injector improved anything — so it must be built *before* the injector, and
kept independent of it (project_record.md §21).

This document has two parts:

- **Part I — Dataset preprocessing** (§1–§5): how the raw datasets become the
  training/evaluation files under `data/evaluator/`. Every dropped row, label
  mapping, and integrity check is recorded so the pipeline is reproducible and the
  choices are defensible.
- **Part II — The model and training** (§6–§10): the architecture, the multi-task
  design, how partial labels and class imbalance are handled, evaluation, and how to
  run it.

See `taxonomy_mapping.md` for the canonical 6-category taxonomy the category labels
map onto, and `project_record.md` §19–§21 for the surrounding decisions.

---

# Part I — Dataset preprocessing

## 1. What is produced

| File | Role | Trained on? |
|---|---|---|
| `biasly_{train,val,test}.csv` | Evaluator — the only source with **severity** and rich categories | train/val yes; **test held out** |
| `cmsb_all.csv` | Evaluator — bulk negatives, tweet register | yes |
| `guest_all.csv` | Evaluator — **hard negatives** (rude-not-sexist), Reddit register | yes |
| `sws_all.csv` | Evaluator — **only workplace register** | yes |
| `selfma_eval.csv`, `selfma_eval_workplace.csv` | **OOD test set** — real workplace dialogue | **NEVER** |
| `biasly_rewrite_pairs.csv`, `cmsb_pairs.csv` | Injector (Objective B) — parallel neutral↔sexist pairs | injector only |

**Final evaluator pool (post-dedup): 28,587 texts, 5,451 positive (19.1%).**

---

## 2. The shared column contract

Every evaluator training file emits the columns in
`evaluator/dataset_common.py::COMMON_COLUMNS`:

`source, text, is_misogynistic, severity, canonical_categories, in_scope, rationale`

**Partial labels are expected and encoded as blanks/NA, not zeros:**

| Label | Present in | Trainer behaviour where absent |
|---|---|---|
| `is_misogynistic` (binary) | all four | — always present |
| `severity` (0–1000) | **Biasly only** | skip the severity loss (NA ≠ 0) |
| `canonical_categories` | Biasly, Guest (partial) | skip the category loss (blank ≠ negative) |

> **`in_scope` warning.** This is a Biasly-specific, Objective-B (generation) flag —
> it marks datapoints whose category is in scope for the *generator* (the violent
> Biasly classes are excluded there). It is meaningless for the evaluator, where it
> just mirrors `is_misogynistic`. **The evaluator trainer must not filter on
> `in_scope`** — the evaluator wants all data, including the violent classes that
> anchor the top of the severity scale.

---

## 3. Per-dataset preparation

### 3.1 Biasly — `prepare_biasly.py`

Movie-subtitle misogyny; the workhorse and the only source with severity.

- 30,000 annotations → **10,000 datapoints** by aggregating 3 annotators: a
  datapoint is misogynistic if **≥1 annotator** says so (preserves minority
  judgements of subtle misogyny; reproduces the paper's 31.6%).
- Category labels split on **`;` only** (two category names contain commas); the
  `"Add optional explanation"` UI artefact is stripped; the two sexualization
  variants merged. (Full detail in `taxonomy_mapping.md` §4.)
- 80/10/10 stratified split **at datapoint level**, so annotator siblings cannot
  leak across splits.
- `biasly_rewrite_pairs.csv`: 2,977 expert rewrites (misogynistic↔neutral) for the
  injector, over 1,985 unique datapoints (multiple annotators rewrote the same
  text — legitimate augmentation).

### 3.2 SWS — `prepare_sws.py`

Sexist Workplace Statements. The only workplace-register data.

- Ships as a one-sheet `.xlsx` (a TSV exported to Excel); columns `Sentences`,
  `Label` (1/0).
- Binary only — no severity, no category.
- 1,137 → **1,126** after removing 11 same-label exact duplicates. 0 rows dropped
  for empty text or bad labels.

### 3.3 CMSB — `prepare_cmsb.py`

"Call me sexist, but…" tweets. Bulk negatives + a second injector corpus.

- `sexism_data.csv` (aggregated), **not** `sexism_annotations.csv` (raw per-worker).
  The `sexist` boolean is the dataset's provided binary label; used as-is. (CMSB
  also distinguishes content- vs phrasing-sexism internally; we use the aggregate,
  consistent with the dataset's headline label.)
- 13,631 → **13,321** after dedup: **294 same-label duplicates removed**, and **16
  rows in 8 conflicting-label groups dropped entirely** — the same tweet labelled
  both sexist and non-sexist across CMSB's sub-datasets (`callme`, `hostile`,
  `benevolent`, `scales`, `other`). The label is unreliable in those 8 cases and
  there is no principled way to pick a side, so all copies are dropped.
- Binary only for the evaluator.
- **Injector pairs (`cmsb_pairs.csv`, 2,222):** reconstructed from CMSB's
  adversarial examples. An adversarial row (`of_id ≠ -1`) is a minimally-edited
  *neutral* version of the sexist source it points to; reversed, that is a
  (neutral → sexist) demonstration. Of 2,292 adversarial rows, **70 were dropped**
  because they were not clean neutral→sexist pairs: 45 where both sides remained
  sexist (the edit did not neutralise), 25 where the source was not sexist. Only
  the 2,222 clean-direction pairs are kept.

### 3.4 Guest et al. — `prepare_guest.py`

Expert-annotated Reddit misogyny. The source of **hard negatives**.

- `final_labels.csv` (adjudicated, one label per post), **not**
  `original_labels.csv` (52k raw per-annotator rows).
- `is_misogynistic = (level_1 == "Misogynistic")`. Text taken from `body`.
- **12 image-only posts dropped** (empty `body` — no text to classify).
- 6,383 unique posts → 6,371 (after image drop) → **6,141** after removing 230
  same-label duplicate texts (Reddit reposts/copypasta across distinct posts).
- Multi-label posts (699 misogynistic *rows* → 516 misogynistic *posts*)
  aggregated with **"any misogynistic wins"** — matches Biasly's ≥1 philosophy.
  Exactly **1 post** carries both a Misogynistic and a Nonmisogynistic
  (Counter_speech) tag — a counter-speech post quoting a pejorative; this rule
  labels it misogynistic. n=1, documented.
- **Hard negatives:** 43 posts tagged `Nonmisogynistic_personal_attack` — hostile
  but not sexist. Flagged via `is_hard_negative`. These teach the classifier the
  *rude ≠ misogynistic* boundary that the generation pipeline's perpetrator (terse,
  dismissive, not overtly sexist) sits on.
- **Category mapping is deliberately partial.** Guest's `level_2` classifies
  misogyny by **linguistic form** (Derogation, Pejorative, Personal attack,
  Treatment), which is orthogonal to Capodilupo's **social mechanisms**. Only the
  two unambiguously-lexical classes are mapped — `Derogation` and
  `Misogynistic_pejorative` → `use_of_sexist_language`. The rest contribute to the
  binary head only; forcing them would inject label noise.

### 3.5 SELFMA — `prepare_selfma.py` (OOD test set, never trained on)

Real self-reported microaggressions from microaggressions.com.

- The evaluator would be circular if trained *and* tested on Biasly-family data.
  SELFMA is the independent check: the only data that is simultaneously real,
  gendered, workplace, and dialogue.
- Multi-turn dialogue exists **only** in the `transcript` field of `type=="chat"`
  records; ~74% of transcript lines carry a `SPEAKER::` marker, the rest is
  scene-setting narration (kept separately, not treated as turns).
- Gender filter applied via the raw JSONL `tags` field (the annotation sheet has
  no gender tag — the two are joined on Post ID).
- 153 gender chat dialogues → **138** with ≥2 parseable turns → **25 workplace**.
- **Every row is a positive** (`is_misogynistic = True`) — SELFMA has no negatives.
  This is why it can validate the **binary** head ("does a movie-trained classifier
  recognise real workplace microaggressions?") and serve as the injector's target
  distribution, but **cannot** give per-category metrics (1–9 examples/category).
  Fitting anything on positives-only would teach it to answer "yes" — hence
  test-only.

---

## 4. Integrity checks (all passing)

Run during the audit; must be re-run if any prep script changes.

| Check | Result |
|---|---|
| **Train ∩ SELFMA (OOD test) leakage** | **0** — evaluation is uncontaminated |
| **Training pool ∩ Biasly held-out test** | **0** — in-distribution test held out |
| Internal exact-duplicate texts, per source | **0** after dedup (was CMSB 302 / Guest 230 / SWS 11) |
| Conflicting-label duplicates | dropped (CMSB: 8 groups / 16 rows) |
| CMSB label mapping vs raw `sexist` | exact (1,798 = 1,798) |

**One residual, handled at pool time, not in the prep scripts:** CMSB and SWS share
**12 identical tweets** (SWS's "filtered Twitter" subset overlaps CMSB), all with
matching labels. The trainer must `deduplicate()` the concatenated pool to remove
these before training.

**Injector-pair caveat:** `biasly_rewrite_pairs.csv` and `cmsb_pairs.csv` contain
repeated *source* texts (multiple neutral rewrites of one sexist source — 992 and
928 respectively). This is intended augmentation for the injector, **but any
train/val split of the injector data must be by source text**, or the same source
leaks across the split.

---

## 4a. Limitations — state these in the dissertation

Honest limitations of the prepared data, to be reported explicitly rather than
discovered by an examiner.

1. **The category head is essentially Biasly-only.** Guest's taxonomy classifies
   misogyny by *linguistic form* (Derogation, Pejorative, Personal attack,
   Treatment), which is **orthogonal** to Capodilupo's *social mechanisms*
   (inferiority, roles, objectification). Only its two lexical classes could be
   honestly mapped (→ `use_of_sexist_language`); the rest contribute to the binary
   head only. CMSB and SWS have no category labels at all. **So category
   supervision comes almost entirely from Biasly** — the multi-source pooling
   strengthens the *binary* and *severity* heads far more than the *category* head.
   Per-class category metrics must be read in that light.

2. **Two canonical categories are barely supported, one register-mismatched.**
   `denial_of_reality_of_sexism` has ~34 Biasly training examples; the pooled data
   does not meaningfully improve it. `use_of_sexist_language` gains breadth from
   Guest, but from Reddit rather than workplace register. Report **per-class**
   metrics, never macro-only.

3. **Severity is single-source.** Only Biasly has the 0–1000 scale, so the severity
   head is trained and validated on movie-subtitle text alone. Its transfer to
   workplace register is unverified except qualitatively via SELFMA.

4. **The evaluation of the *category* head is weak.** SELFMA, the only real-data
   check, has 1–9 labelled dialogues per category (and 0 for
   `use_of_sexist_language`). It validates the **binary** head convincingly but can
   only give **qualitative** evidence for categories.

5. **Register skew.** Three of four training sources are social media / film; only
   SWS (1,126 statements) matches the target workplace-texting register. The
   classifier is asked to generalise across a register gap, which the SELFMA OOD
   test is designed to measure.

6. **Class imbalance is inherent** (19.1% positive) and handled by weighting, not
   resampling — a modelling choice to record.

---

## 5. Reproduce

```bash
IP="Individual Project"
python -m synthetic_conversation_generation.evaluator.prepare_biasly  --input "$IP/Biasly Data/biasly_dataset.csv"
python -m synthetic_conversation_generation.evaluator.prepare_cmsb    --input "$IP/CSMB Dataset/sexism_data.csv"
python -m synthetic_conversation_generation.evaluator.prepare_guest   --input "$IP/Guest Dataset/final_labels.csv"
python -m synthetic_conversation_generation.evaluator.prepare_sws     --input "$IP/SWS Dataset/ISEP Sexist Data labeling.xlsx"
python -m synthetic_conversation_generation.evaluator.prepare_selfma  --data-dir "$IP/SelfMA Dataset"
```

---

# Part II — The model and training

## 6. Architecture

A **shared transformer encoder with three task heads** (`evaluator/model.py`,
class `MultiHeadEvaluator`):

```
                          ┌─> binary head    (1 logit)   -> misogynistic?
text -> encoder -> mean-pool ─> severity head  (1 scalar)  -> 0-1000 intensity
                          └─> category head  (6 logits)  -> multi-label kind
```

| Spec | Choice | Justification |
|---|---|---|
| **Encoder** | `microsoft/deberta-v3-base` (default; configurable) | The Biasly paper's best detector (F1 0.807). `--model-name roberta-base` is a no-sentencepiece fallback. |
| **Pooling** | Masked **mean** of the last hidden state | DeBERTa has no NSP objective, so `[CLS]` is not a calibrated sequence summary; masked mean pooling is the robust cross-family default. |
| **Heads** | Three linear layers on the pooled vector | Shared trunk = the auxiliary heads regularise the representation; multi-task learning across related signals. |
| **Severity output** | `sigmoid(...) * 1000` | Bounds the regression to Biasly's [0, 1000] scale rather than letting it predict out of range. |

**Why three heads and not just binary.** The CHI paper's central finding is that a
binary judge cannot discriminate — it ceiling-rates. Severity gives a continuous,
comparable signal; category gives *which* mechanism. Together they are what let the
evaluator rank two generations rather than call both "misogynistic, yes".

## 7. The defining constraint — partial labels

Most rows carry only a binary label (see Part I §2). Severity is Biasly-only;
category is Biasly + partial Guest. The loss for each head is therefore **masked to
the rows that actually carry that label**:

```
loss = binary_loss                                    # every row
     + λ_sev * severity_loss[ ~isnan(severity) ]      # Biasly rows only
     + λ_cat * category_loss[ category_mask ]         # Biasly + Guest rows only
```

An absent label contributes **nothing** — it is never treated as a zero (which would
teach the model that a tweet with no severity annotation has severity 0, or belongs
to no category). This is the single most important correctness property of the
trainer, and is unit-testable via `--smoke`.

## 8. Handling class imbalance

The pool is **19.1% positive**. Left unweighted, a classifier reaches ~81% accuracy
by always answering "no" — the exact ceiling-rating failure inverted. Handled by
**class weighting, not resampling**: the binary loss uses
`pos_weight = n_neg / n_pos` (computed from the actual pool at train time, ~4.3),
so a missed positive is penalised proportionally more. Resampling was avoided because
it would duplicate the scarce positives and distort the severity/category signal
attached to them.

## 9. Evaluation — two held-out sets

| Set | What | What it measures |
|---|---|---|
| **`biasly_test.csv`** | In-distribution (same family as most training data) | Standard binary P/R/F1, per-class category F1 (**with support counts**), severity MAE + Pearson r |
| **`selfma_eval.csv`** | **OOD** — real, human-reported workplace dialogue, never trained on | **Catch-rate (recall)** — does a model trained on movie subtitles / tweets recognise real workplace microaggressions? |

Two evaluation rules, both consequences of the data (Part I §4a):

1. **Per-class category metrics, never macro-only.** Category support is
   Biasly-dominated and wildly uneven, so a macro average would hide that (e.g.)
   `denial_of_reality_of_sexism` is unlearned. The trainer prints per-class F1 *and*
   support.
2. **On SELFMA, only recall is meaningful.** SELFMA is 100% positive, so precision
   and F1 are undefined; the reported number is the catch-rate. This is the headline
   generalisation claim, and the reason SELFMA exists (Part I §3.5).

Both sets are provably disjoint from the training pool (Part I §4: zero leakage).

## 10. Running it

### Locally

```bash
# Correctness check — tiny model, 200 rows, 1 epoch, no GPU, ~15s.
# Verifies pooling, dedup, the masked multi-task loss, and both eval paths.
python -m synthetic_conversation_generation.evaluator.train_evaluator --smoke

# Real training.
python -m synthetic_conversation_generation.evaluator.train_evaluator \
    --data-dir data/evaluator --output-dir models/evaluator \
    --epochs 3 --batch-size 16 --lr 2e-5
```

### On AIRE (`train_evaluator.slurm`)

Unlike the generation pipeline, this needs **no Ollama and no Apptainer** — it loads
DeBERTa directly, so the job is just conda Python on the GPU. Two one-time steps on a
**login node** (compute nodes are offline):

```bash
pip install -r requirements.txt                       # incl. the CUDA torch build
export HF_HOME=$SCRATCH/hf_cache                       # or add to ~/.bashrc
python scripts/predownload_model.py microsoft/deberta-v3-base   # committed, reproducible
sbatch train_evaluator.slurm
```

`scripts/predownload_model.py` fetches the model **and verifies it loads offline**,
reproducing the compute node's conditions — so a success there means the job will not
fail on a missing model. The SLURM script sets `HF_HUB_OFFLINE=1` and runs a
model-cache pre-flight before touching the GPU.

- **Device** is auto-selected (CUDA > MPS > CPU); override with `--device`.
- **Outputs:** `evaluator.pt` (weights), the tokenizer, and `metrics.json`
  (in-distribution + OOD results, pool size, pos_weight).
- **Data** (`data/evaluator/*.csv`) is tracked in git, so it arrives with a pull;
  only re-running preprocessing needs the raw datasets.
- **Dependencies:** see `requirements.txt` (evaluator subset: `transformers`,
  `torch`, `scikit-learn`, `sentencepiece`, `protobuf`, …).

**Expected shape of results** (recorded so a weak category head is not mistaken for a
bug): the **binary** and **severity** heads are well-supported across four registers;
the **category** head is Biasly-dominated and will be strong only on
`assumptions_of_inferiority`, weak-to-zero elsewhere (Part I §4a). Report accordingly.

## 11. Limitations of the trained evaluator

In addition to the data limitations (Part I §4a):

- **Single-run, no cross-validation reported by default.** For a headline number,
  average over seeds.
- **Threshold is fixed at 0.5.** The binary threshold is not tuned; a
  precision/recall trade-off could be chosen on the Biasly validation split if a
  specific operating point is wanted.
- **Dialogue vs statement mismatch at inference.** The evaluator is trained on
  single statements but scores multi-turn generated conversations. SELFMA (also
  dialogue) is the check that this transfers; if it does not, per-message scoring
  is the fallback.

---

## References

- Sheppard et al. (2024). Biasly. *Findings of ACL 2024*.
- Samory et al. (2021). "Call me sexist, but…". *ICWSM 2021*.
- Grosz & Conde-Céspedes (2020). Sexist Statements at the Workplace. *PAKDD 2020 LDRC*.
- Guest et al. (2021). An Expert Annotated Dataset for the Detection of Online Misogyny. *EACL 2021*.
- Breitfeller et al. (2019). Finding Microaggressions in the Wild. *EMNLP-IJCNLP 2019*.
- He et al. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training. *arXiv:2111.09543*.
- Lagos Rojas et al. (2026). "Are Compliments Bad Now?". *CHI 2026*. — motivates the multi-head design.
