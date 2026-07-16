"""
Canonical gender-microaggression taxonomy — the single source of truth.

This module is the *only* place the project's category vocabulary is defined. Three
consumers depend on it and must not redefine it:

  1. The evaluator     -- as the classifier's output space (`evaluator/prepare_biasly.py`)
  2. The generator     -- as a control input on dialogue-flow beats
  3. The injector      -- as a conditioning signal alongside severity

Anchor
------
Capodilupo et al. (2010) gender microaggression themes, as psychometrically
validated by WoMenS (Miyake et al., 2025).

Capodilupo is chosen over the CHI 8 because (a) it is the common theoretical
ancestor of the CHI, WoMenS and SELFMA schemes, giving the shortest and least
lossy crosswalk to every dataset; (b) it is the only scheme in play with
psychometric validation (EFA + CFA), supporting the claim that the taxonomy is
*measured* rather than asserted; and (c) it fits the data -- the CHI 8 left two
categories with zero Biasly support and one with 42. See `taxonomy_mapping.md`
and project_record.md section 21.2 for the full decision record.

Capodilupo's 7th theme, "environmental", is deliberately omitted: it is
macro-level (media portrayal, the pay gap) and cannot be expressed in a
two-person dialogue.

References
----------
[1] Capodilupo, C. M., Nadal, K. L., Corman, L., Hamit, S., Lyons, O. B., &
    Weinberg, A. (2010). The manifestation of gender microaggressions. In
    D. W. Sue (Ed.), Microaggressions and Marginality: Manifestation, Dynamics,
    and Impact (pp. 193-216). Wiley.
[2] Miyake, E., Ahn, L. H., Tran, A. G. T. T., & Atkin, A. L. (2025). Women's
    Microaggressions Scale (WoMenS): A Comprehensive Sexism Scale. The
    Counseling Psychologist, 53(2), 174-209.
[3] Kim, J. Y., & Meister, A. (2023). Microaggressions, Interrupted: The
    experience and effects of gender microaggressions for women in STEM.
    Journal of Business Ethics, 185(3), 513-531.  [workplace lens]
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroaggressionCategory:
    """One canonical category.

    definition:      the general (Capodilupo) sense of the theme
    workplace_form:  how it concretely manifests between colleagues, per Kim &
                     Meister (2023). This is the form the generator should realise
                     and the form a beat's category refers to.
    """
    key: str
    label: str
    definition: str
    workplace_form: str


TAXONOMY: dict[str, MicroaggressionCategory] = {
    "assumptions_of_inferiority": MicroaggressionCategory(
        key="assumptions_of_inferiority",
        label="Assumptions of inferiority",
        definition=(
            "Assuming a woman is less capable, less intelligent, or less competent "
            "than a man; infantilising or paternalistic treatment."
        ),
        workplace_form=(
            "Over-explaining things she already knows; checking her work when he would "
            "not check a male peer's; treating her conclusions as needing verification; "
            "patronising reassurance."
        ),
    ),
    "traditional_gender_roles": MicroaggressionCategory(
        key="traditional_gender_roles",
        label="Assumptions of traditional gender roles",
        definition=(
            "Essentialist assumptions about what women are and what women do; "
            "expectations that she will occupy a stereotypically feminine role."
        ),
        workplace_form=(
            "Assuming she will take the notes, organise the social event, or handle "
            "the client-soothing; framing her as naturally 'good with people' rather "
            "than technically strong."
        ),
    ),
    "second_class_citizenship": MicroaggressionCategory(
        key="second_class_citizenship",
        label="Second-class citizenship",
        definition=(
            "Treating a woman as lower-status; denying her autonomy or authority; "
            "affording her fewer opportunities than a man in the same position."
        ),
        workplace_form=(
            "Deciding or speaking for her; assuming she will defer; routing decisions "
            "around her; re-attributing her ideas to himself or to others."
        ),
    ),
    "sexual_objectification": MicroaggressionCategory(
        key="sexual_objectification",
        label="Sexual objectification",
        definition=(
            "Reducing a woman to her body or appearance; treating her as an object "
            "to be looked at rather than a person."
        ),
        workplace_form=(
            "Comments on her looks or clothing in a professional context; remarks "
            "about her appearance in place of her contribution."
        ),
    ),
    "use_of_sexist_language": MicroaggressionCategory(
        key="use_of_sexist_language",
        label="Use of sexist language",
        definition=(
            "Degrading gendered language; slurs; dehumanising comparisons; hostility "
            "expressed through gendered terms."
        ),
        workplace_form=(
            "Gendered put-downs framed as banter or humour; diminutives; 'jokes' that "
            "make objecting feel like overreacting."
        ),
    ),
    "denial_of_reality_of_sexism": MicroaggressionCategory(
        key="denial_of_reality_of_sexism",
        label="Denial of the reality of sexism",
        definition=(
            "Dismissing or invalidating a woman's account of bias; contending that "
            "sexism is no longer a real problem."
        ),
        workplace_form=(
            "'It was just a joke'; 'you're overthinking it'; 'that's not what happened'; "
            "reframing her discomfort as oversensitivity."
        ),
    ),
}

# Ordered, stable list of keys — use this for classifier label indices so that
# label order never depends on dict iteration or insertion accidents.
CATEGORY_KEYS: list[str] = sorted(TAXONOMY.keys())

# Capodilupo's 7th theme, retained here only to document the omission.
OMITTED_THEMES: dict[str, str] = {
    "environmental": (
        "Macro/systemic manifestations (media portrayal, the gender pay gap). Omitted: "
        "not expressible in a two-person dialogue."
    ),
}


def definition_block(include_workplace: bool = True) -> str:
    """Render the taxonomy as a prompt-injectable block.

    Used where an LLM must classify or plan against the taxonomy (e.g. the
    dialogue-flow planner, the state assessor). Per the CHI paper's finding, and
    Kumar et al. as cited therein, supplying explicit definitions materially
    affects detection sensitivity — so the definitions travel with the labels
    rather than the label names being used bare.
    """
    lines = []
    for key in CATEGORY_KEYS:
        c = TAXONOMY[key]
        lines.append(f"- {c.key} ({c.label}): {c.definition}")
        if include_workplace:
            lines.append(f"    In a workplace: {c.workplace_form}")
    return "\n".join(lines)


def is_valid(key: str) -> bool:
    """True if `key` is a canonical category key."""
    return key in TAXONOMY
