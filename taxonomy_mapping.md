# Canonical Taxonomy and Cross-Dataset Mapping

**Purpose.** Four overlapping taxonomies are in play across this project's theory papers and
its two training datasets. They do not align 1:1. This document defines a single **canonical
taxonomy** for the project and maps every source scheme onto it, so that generation
(Objective B) and evaluation (Objective C) speak the same vocabulary.

**Counts are from the actual downloaded data**, not the papers — see `project_record.md` §19
for the audit.

---

## 1. The four source taxonomies

| Source | Type | Scheme | Role here |
|---|---|---|---|
| **Capodilupo et al. (2010)** | Foundational theory | **7 gender-MA themes** | **Anchor for the canonical scheme** |
| **WoMenS** (Miyake et al., 2025) | Psychometric scale | 8 factors validating Capodilupo's themes (EFA + CFA) | **Validation of the anchor** |
| **CHI 2026** (Lagos Rojas et al.) | HCI study | 8 workplace gender-MA categories (Sue + Gartner + Kim & Meister) | Workplace lens; evaluation warning |
| **Biasly** (Sheppard et al., ACL 2024) | NLP dataset (movie subtitles) | 12 misogyny categories + severity 0–1000 | **Primary training data** |
| **SELFMA** (Breitfeller et al., EMNLP 2019) | NLP dataset (microaggressions.com) | 4 themes / 12 sub-themes | Real-dialogue anchor |

All of CHI, WoMenS and SELFMA descend from **Sue et al. (2007)** and **Capodilupo et al. (2010)**,
so they converge closely. **Biasly is the outlier** — it was built inductively from annotator
observation of movie subtitles, so its categories are lexical/behavioural rather than theoretical.
Reconciling Biasly is the main work of this document.

> ### ⚠️ Provenance of this mapping — read before citing
>
> **No published crosswalk exists between these taxonomies. The mapping in §3 is our own
> construction.** The individual taxonomies are each quoted from their source papers; the
> *correspondences between them* are our judgement, arrived at by comparing published category
> definitions. Where a source category spans two canonical themes, we record the primary mapping and
> flag the ambiguity explicitly. This is a methodological contribution of the project, not an
> inherited standard, and must be presented as such — with its low-confidence mappings named
> (see §3) and per-class metrics reported throughout, since class support is highly uneven.

---

## 2. The canonical taxonomy (6 categories)

**Anchored on Capodilupo et al. (2010)**, as psychometrically validated by **WoMenS**
(Miyake et al., 2025). *Environmental* (Capodilupo's 7th theme) is dropped: it is macro-level
(media portrayal, the pay gap) and cannot be expressed in a two-person dialogue.

| # | Canonical key | Definition |
|---|---|---|
| 1 | `assumptions_of_inferiority` | Assuming she is less capable; infantilising or paternalistic language; over-explaining; not taking her seriously |
| 2 | `traditional_gender_roles` | Essentialist assumptions about what women are/do; steering her toward stereotypically "female" tasks |
| 3 | `second_class_citizenship` | Treating her as lower-status; denying autonomy; assuming she defers; fewer opportunities |
| 4 | `sexual_objectification` | Reducing her to appearance or body; comments on looks in a professional context |
| 5 | `use_of_sexist_language` | Degrading gendered language; slurs; dehumanising comparisons |
| 6 | `denial_of_reality_of_sexism` | Dismissing or invalidating her account of bias ("it was just a joke", "sexism isn't a thing anymore") |
| — | ~~`environmental`~~ | *Dropped — macro/systemic, not expressible in a dyadic conversation* |

### Why Capodilupo rather than the CHI 8 — decision record

The CHI 8 was the initial anchor. It was **rejected on empirical grounds** after running the
preprocessing: it left `pathologizing_character` and `exclusion` with **zero** Biasly support and
`denial_of_experience` with **42**. A taxonomy in which 3 of 8 classes cannot be learned does not
fit the evidence. Capodilupo's themes were adopted instead because:

1. **Common ancestor.** CHI, WoMenS and SELFMA all descend from Sue (2007) → Capodilupo (2010).
   Anchoring at the root gives the shortest, least lossy crosswalk to every dataset, rather than
   forcing all sources through a derivative scheme.
2. **The only psychometrically validated scheme.** WoMenS ran EFA + CFA on it. This supports the
   claim that the taxonomy is *measured*, not asserted.
3. **It fits the data.** Five well-populated classes instead of five-plus-three-empty (see §3).
4. **The workplace framing is retained.** The CHI 8 *is* Capodilupo + Kim & Meister's workplace lens.
   We cite Kim & Meister (2023) for the workplace interpretation layered on top of Capodilupo's
   validated themes.

---

## 3. Mapping table

`→` = primary mapping. `~` = partial / secondary. Counts are from the actual data.

| Canonical (Capodilupo) | **Biasly** (count) | **SELFMA** gender sub-theme (count) | **CHI 8 equivalent** |
|---|---|---|---|
| `assumptions_of_inferiority` | → Trivialization (**1,652**) | ~ Stereotype; → Myth of Meritocracy (11) | undermining competence |
| `traditional_gender_roles` | → Gender essentialism *(ambiguous — see below)* (**1,287**) | → Stereotype (**185**) | restrictive gender roles |
| `second_class_citizenship` | → Lacking autonomy or agency (**634**) | → Second-Class Citizen (83); ~ Ownership (23) | gender as liability |
| `sexual_objectification` | → Sexualization (**850**) *(two variants merged — see §4)* | → Objectification (**125**) | sexual objectification |
| `use_of_sexist_language` | → Gendered slurs + Dehumanization (**500**) | → Overt Aggression (11) | gender hostility |
| `denial_of_reality_of_sexism` | ~ Anti-feminism (**42**) *(low confidence)* | → Denial of Lived Exp. (**38**) | denial of experience |

*(Biasly counts here are at datapoint level post-aggregation, hence lower than the raw annotation counts in §5.)*

### Low-confidence and ambiguous mappings — stated openly

- **`anti_feminism → denial_of_reality_of_sexism` is weak.** Biasly defines anti-feminism as
  *"Feminism is a bad idea… women shouldn't have equal rights"*, which is **not** the same as denying
  that sexism is real. It is merely the nearest Capodilupo theme. With n=42, per-class metrics for
  this category will be unreliable and should be reported as such.
- **`gender_essentialism → traditional_gender_roles` is ambiguous.** Biasly's definition spans both
  role assumptions (*"women are good at childrearing"*) **and** pathologising content
  (*"women are untrustworthy and overly emotional"*). The primary mapping is recorded; the
  pathologising sense is lost. This is a known information loss.
- **No CHI category for `pathologizing_character` / `exclusion` survives** — neither has Biasly support.
  If those dynamics matter for the write-up, they must come from SELFMA (Abnormality n=81,
  Erasure n=27) or be acknowledged as out of the classifier's reach.

### Out of scope (excluded from the canonical scheme)

| Source category | Count | Why excluded |
|---|---|---|
| Biasly: Domestic violence / VAW | 313 | Overt violence — off-domain for subtle workplace MAs |
| Biasly: Rape and other sexual violence | 251 | As above |
| Biasly: Phallocentrism | 199 | Not expressible in a workplace dyad |
| Biasly: Intersectional, identity-based | 148 | Out of scope (project is gender-only, per CHI's own scoping) |
| Biasly: Transmisogyny/Homophobia | 43 | Out of scope (see limitations) |
| Biasly: "Other" | 232 | Unlabelled residue |
| Biasly: "Add optional explanation" | 635 | **UI artefact, not a category — must be stripped** |
| SELFMA: Criminal Status, Alien in Own Land, Monolith | 1 / 4 / 21 | Race-specific sub-themes |

**Important distinction between B and C:**
- **Objective B (generation)** trains only on the **in-scope** categories — the violent classes
  would push the generator off-domain.
- **Objective C (evaluator)** should **retain the full Biasly range**, including the violent
  classes, because they anchor the top of the 0–1000 severity scale. Removing them would
  compress the severity distribution and degrade the regressor.

---

## 4. Data-hygiene rules (apply before any training)

Derived from the audit; each is a real defect in the raw files:

1. **Split Biasly's `misogynistic_inferences` on `;` only** — *not* on commas. Two categories
   contain commas in their own names (e.g. *"Sexualization (focus on appearance, degrading
   language)"*) and a comma-split silently shatters them.
2. **Drop `"Add optional explanation"`** (n=635) — an annotation-UI artefact.
3. **Merge Biasly's two sexualization labels** — *"Sexualization (focus on appearance, degrading
   language)"* (966) and *"Objectification/sexualization (focus on appearance)"* (417). These are
   a mid-annotation taxonomy revision, not distinct classes. Combined: **1,383**.
4. **Aggregate Biasly to datapoint level** following the paper: a datapoint is misogynistic if
   **≥1 of its 3 annotators** says so (→ 3,159 positives, 31.59%). Do not majority-vote — the
   paper deliberately preserves minority annotator judgements.
5. **SELFMA has no gender tag in the annotation sheet.** Join `microaggressions_v1.json` on
   `Post ID` and filter on the `tags` field to isolate the 1,411 gender posts.
6. **SELFMA dialogues live in `transcript`, not `quote`**, and only on `type == "chat"` records
   (a list of `SPEAKER:: line` strings). This is the only place real multi-turn dialogue exists.

---

## 5. What each dataset actually yields, per objective

| | Biasly | SELFMA |
|---|---|---|
| **Objective B** (content grounding) | 3,159 misogynistic datapoints + **2,977 parallel rewrite pairs** (reverse them: mitigated → misogynistic) | **153 real gender dialogues** (mean 4.7 turns), of which **30 workplace** — few-shot / reference anchor |
| **Objective C** (evaluator) | **Severity regression (0–1000)** + multi-label category over 5,600 annotations — the training signal | Gold reference set; the 30 workplace dialogues as a human-authored sanity check |
| **Taxonomy** | 12 categories (inductive) | 4 themes / 12 sub-themes (theory-driven) |

---

## 6. Honest limitations (for the write-up)

- **Register mismatch.** Biasly is *movie subtitles* — dramatised speech, not workplace texting.
  It grounds *what misogyny sounds like*, not *how colleagues text*.
- **Coverage is uneven.** `denial_of_reality_of_sexism` has only 42 Biasly datapoints (and via a
  low-confidence mapping), so the classifier will be unreliable on it. Report per-class metrics, not
  just macro. CMSB, SWS and Guest et al. are candidates for filling this gap.
- **Two dynamics are unreachable from Biasly.** Pathologising character and exclusion have no Biasly
  support at all. If they matter to the argument, they must be sourced from SELFMA or acknowledged
  as outside the classifier's range.
- **The taxonomy is one interpretive frame, not ground truth.** Per Lagos Rojas et al., microaggression
  interpretation is *situated*; these labels reflect the annotators' positions. Severity and category
  should be reported as annotator-derived, never as objective fact.
- **Gender-only scope.** Intersectional cases are excluded for tractability, following the CHI paper's
  own stated scoping decision — and, like them, this is acknowledged as a substantive limitation.

---

## References

- Lagos Rojas, C., Genç, H. U., Bozzon, A., & Colombo, S. (2026). "Are Compliments Bad Now?": Comparing LLMs and Human Interpretations of Gender Microaggressions in the Workplace. *CHI 2026*.
- Miyake, E., Ahn, L. H., Tran, A. G. T. T., & Atkin, A. L. (2025). Women's Microaggressions Scale (WoMenS). *The Counseling Psychologist*, 53(2), 174–209.
- Sheppard, B., Richter, A., Cohen, A., et al. (2024). Biasly: An Expert-Annotated Dataset for Subtle Misogyny Detection and Mitigation. *Findings of ACL 2024*, 427–452.
- Breitfeller, L., Ahn, E., Jurgens, D., & Tsvetkov, Y. (2019). Finding Microaggressions in the Wild. *EMNLP-IJCNLP 2019*, 1664–1674.
- Sue, D. W., et al. (2007). Racial microaggressions in everyday life. *American Psychologist*.
- Capodilupo, C. M., et al. (2010). The manifestation of gender microaggressions. In *Microaggressions and Marginality*.
