# FYP Technical Record
## Synthetic VAWG Dialogue Generation Pipeline

**Student:** Aristaa Singh  
**Supervisor:** [Supervisor Name]  
**University:** University of Leeds, School of Computing  
**Last updated:** July 2026

---

> This document is a running technical record of implementation decisions, research backing, and reasoning — written as work progresses so that context is not lost over time. It is not the dissertation. Think of it as a lab notebook: what we did, why we did it, and what evidence supports those choices.

---

## Section 1 — Base Pipeline: Channel Labs Fork

### What we started with

The base codebase is a fork of the open-source **Channel Labs `synthetic-conversation-generation`** repository (https://github.com/channel-labs/synthetic-conversation-generation), created by Channel Labs (channellabs.ai). The fork lives at https://github.com/AristaaSingh/synthetic-conversation-generation.

The original repo was built for a different purpose — generating synthetic conversations between enterprise AI assistants and simulated users, for the purpose of testing and evaluating chatbot behaviour at scale. We repurposed and extended it for synthetic VAWG dialogue generation.

---

### What the original repo provides

The Channel Labs repo has a clean, modular architecture that made it a good foundation:

**Data models** (`data_models/`)
- `CharacterCard` — a structured persona with name, role, backstory, description, personality, scenario, and summary fields. We use this for both the victim and perpetrator.
- `Conversation` and `Message` — data classes representing the conversation and individual turns. `Message` already had a `timestamp` field, though it was just set to `datetime.now()` at generation time.
- `Assistant` — the "other side" of the conversation (in our case, the perpetrator or victim depending on whose perspective we're generating from).
- `InferenceEndpoint` — YAML-configurable endpoint for calling the AI backend.

**LLM query classes** (`llm_queries/`)
- Abstract base class `LLMQuery` with `generate_prompt()`, `response_schema()`, and `parse_response()` — all queries follow this pattern, making it easy to add new query types.
- `UserMessageQuery` — generates the next message from the user/victim persona.
- `ConversationCompletionQuery` — after each turn, asks the LLM whether the conversation has naturally concluded.
- `LLMQuery.query()` — handles retries, timeout, and JSON schema enforcement.

**Model providers** (`llm_queries/llm_query.py`)
- `OpenAIModelProvider`, `AnthropicModelProvider`, `OllamaModelProvider`, `TransformersModelProvider` — all share the same interface. Switching backends requires no changes to generation logic.

**Generators**
- `ConversationGenerator` — drives the turn-by-turn loop.
- `PersonaGenerator` — generates character cards (victim + perpetrator) using an LLM, guided by the assistant definition YAML.

**Configuration** (YAML files in `data/`)
- `data/assistants/vawg_dialog_gen.yaml` — the assistant definition used to guide VAWG persona generation.
- `data/conversation_characters/vawg_personas.yaml` — pre-generated personas including Leila Patel (victim: CS student facing workplace harassment) and Ethan Morales (perpetrator: senior student using coercive tactics).
- `data/endpoint/ollama_chat.yaml` — endpoint config pointing to local Ollama at `http://localhost:11434/api/chat`.

---

### What we changed from the original

The original repo generates short conversations — the CLI defaults to 3 turns and the whole framing is "chatbot testing", not long-term abuse simulation. The key changes we made:

1. **VAWG-specific assistant and personas** — replaced the generic assistant definition with one focused on generating VAWG dialogue scenarios. Generated victim/perpetrator character card pairs rather than the original single "user" persona concept.

2. **Timestamps** — the original set `timestamp=datetime.now()` on every message, meaning all messages in a generated conversation got the same (or nearly the same) real-world time. We replaced this with Hawkes-process-generated timestamps (see Section 2).

3. **Temporal context in prompts** — `UserMessageQuery` now receives the gap since the last message and the timestamp of the current message as structured inputs, so the generated content can reflect the passage of time.

4. **Phase schedule** — added a `PhaseSchedule` class that defines how the conversation transitions through relationship phases (early contact → escalation → post-incident → re-initiation) over the 14-day arc.

5. **`Message.prompt_format` updated** — now includes `"timestamp": "2024-01-03 14:32"` so when the LLM sees the conversation history, it sees real timestamps on each message and can reason about time from the data itself.

---

### Why this fork rather than building from scratch

The Channel Labs repo gave us a well-structured foundation with multi-provider LLM support, retry logic, JSON schema enforcement, and clean separation of concerns. Building all of that from scratch would have taken significant time without adding research value. The research contribution of this project is in the temporal modelling, context management, and evaluation — not in the boilerplate of calling LLM APIs and parsing responses. Using the fork lets us focus on what matters.

The VAWG personas, assistant definition, and all temporal/evaluation components are original work built on top of the fork.

---

## Section 2 — Temporal Modelling: Hawkes Process

### The problem we are solving

The Channel Labs repo (and essentially all existing synthetic dialogue systems) generates messages with either no timestamps or trivial ones. In a real text message conversation — especially an abusive one — the timing of messages is not uniform or arbitrary. It carries meaning:

- A rapid burst of 12 messages in 20 minutes from a perpetrator signals aggressive pursuit
- A 3-day silence from a victim signals withdrawal after an incident
- A single message after 5 days of silence signals re-initiation, and its tone will be very different from a message sent 2 minutes after the last

If we generate conversations where every message is timestamped identically or at uniform intervals, the resulting dataset is not realistic — and it cannot capture one of the most significant features of abusive communication patterns: the rhythm of escalation, withdrawal, and re-contact.

Beyond realism, the timestamps need to influence the *content* of messages. A message sent after 4 days of silence should read differently to a message sent 3 minutes after the last one. The LLM generating the message needs to know the temporal context.

---

### Why not just tell the LLM how much time has passed?

This was considered and rejected. Simply injecting "3 days have passed, write accordingly" into a prompt is **prompt engineering** — the behaviour of the system depends entirely on the LLM following a natural language instruction, which is unverifiable, unstable across models, and has no scientific grounding. There is no way to measure whether the LLM is actually responding to the temporal information, and no way to claim the timing is "realistic" without an underlying model.

The approach we took instead: model the timing **statistically**, using a principled probabilistic model from the literature, and pass the resulting timestamps to the LLM as structured data. The LLM sees timestamps in the conversation history and a gap value as a structured field — it reasons about time from data, not from instructions. The timing model is the component that ensures realism; the LLM is only responsible for generating content consistent with the state it receives.

---

### The Hawkes process

**What it is:** A Hawkes process is a self-exciting temporal point process — a mathematical model for sequences of events in time where each event makes future events temporarily more likely. It was introduced by Hawkes (1971) and has since become the standard model for bursty, clustered event timing in human communication.

The conditional intensity function (the instantaneous rate of events at time $t$ given history) is:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \cdot e^{-\beta(t - t_i)}$$

The three parameters:
- **μ (mu):** baseline rate — how often messages happen when nothing is driving them
- **α (alpha):** excitation amplitude — how much each message temporarily raises the rate of the next
- **β (beta):** decay rate — how fast that excitement fades

The stability condition **α/β < 1** must hold or the process becomes explosive (the rate grows without bound). We enforce this at instantiation.

**Why it fits text messaging:** When someone sends a message, you tend to reply quickly, they reply quickly, and so on — a burst. Then the conversation slows and long silences occur. The Hawkes process generates exactly this pattern from a single principled model: high α/β ratio = tight bursts; low α = sparse background activity.

---

### Research backing

| Paper | What it establishes |
|---|---|
| Barabási (2005), *Nature* | Human communication timing is bursty and heavy-tailed — not Poisson. Foundational result. |
| Hong et al. (2008), *arXiv:0802.2577* | SMS inter-message times specifically follow a power-law distribution with exponents 1.5–2.1. Directly rules out uniform or Poisson models for text messaging. |
| Masuda et al. (2013), *arXiv:1205.5109* | Fitted Hawkes processes to **conversation event sequences** specifically. Showed Hawkes reproduces the burstiness of real dialogue turn-taking. |
| Aoki et al. (2016), *Physical Review E* | Analysed SMS datasets using a Hawkes-style model. Found SMS shows strong, fast self-excitation — the most SMS-specific Hawkes paper available. |
| Falkner et al. (2022), *arXiv:2009.02032* | Fitted Hawkes to mobile phone logs of 900+ people. Found **Hawkes parameters differ systematically by relationship type** (romantic vs. friendship vs. family). Justifies using different parameter sets for different VAWG scenario dynamics. |
| Stouffer et al. (2006), *arXiv:physics/0605027* | Inter-message times fit a log-normal distribution. Useful for validation (see below). |
| Ogata (1988), *JASA* | Introduced the thinning (rejection sampling) algorithm used to simulate Hawkes processes. The canonical simulation reference. |
| Rizoiu et al. (2017), *arXiv:1708.06401* | Tutorial paper on Hawkes processes in social media, with code. Implementation guide used to build the simulation. |

---

### Implementation decision: from scratch, not a library

We chose to implement the Hawkes thinning algorithm from scratch in Python rather than using a library (e.g. `tick`). Reasons:

1. **Mid-conversation parameter switching.** Libraries like `tick` assume fixed parameters over the full time horizon. We need to switch μ/α/β mid-conversation as the relationship phase changes (early contact → escalation → post-incident). Writing the simulation ourselves means we can change parameters at any point by simply updating the parameter set before each sampling step.

2. **Research contribution.** "Implemented from scratch based on Ogata (1988)" is a substantive technical claim. The thinning algorithm is ~30 lines of code but demonstrates a real understanding of the method.

3. **Validation against `tick`.** We can still use `tick` to cross-check our output distributions match the expected statistical properties, without depending on it in production.

**File:** `src/synthetic_conversation_generation/temporal/hawkes.py`

---

### The thinning algorithm (Ogata 1988)

The key insight is that between events, the Hawkes intensity **monotonically decreases** (the excitation terms decay). This means the intensity at the start of any interval is a valid upper bound for the entire interval — enabling rejection sampling.

```
1. Compute λ* = λ(t | history)         # upper bound at current time
2. Sample w ~ Exponential(1 / λ*)      # candidate inter-arrival
3. t_candidate = t + w
4. Compute λ(t_candidate | history)    # actual intensity at candidate
5. Sample u ~ Uniform(0, 1)
6. If u ≤ λ(t_candidate) / λ*:        # accept with this probability
       record t_candidate as next event
7. Set t = t_candidate, repeat
```

This runs in a `while` loop inside `ConversationTimer.next_timestamp()` until one event is accepted. That event time becomes the timestamp of the next message.

---

### Phase parameters

Each relationship phase has its own (μ, α, β) set, grounded in the ranges reported by Aoki et al. (2016) and Falkner et al. (2022). All rates are in **events per minute**.

| Phase | μ | α | β | α/β | Character |
|---|---|---|---|---|---|
| early_contact | 0.005 | 0.30 | 0.50 | 0.60 | Tentative, infrequent |
| escalation | 0.015 | 0.60 | 0.80 | 0.75 | Frequent bursts, intense |
| post_incident | 0.001 | 0.15 | 0.50 | 0.30 | Near-silence, sparse |
| re_initiation | 0.003 | 0.25 | 0.60 | 0.42 | Low baseline restart |

All α/β values are well below 1 (stability guaranteed).

---

### The PhaseSchedule

A `PhaseSchedule` dataclass holds an ordered list of `(day_threshold, phase_name)` pairs. The default schedule encodes a 14-day abusive relationship arc:

```
Day  0–3:   early_contact    — tentative initial contact, testing the water
Day  3–10:  escalation       — intensifying exchanges, coercive tactics emerge  
Day 10–12:  post_incident    — victim withdrawal after a significant event
Day 12–14:  re_initiation    — perpetrator re-establishes contact after silence
```

`phase_for(elapsed_days)` scans the schedule and returns the last phase whose threshold has been exceeded. This is called in the turn loop before each message; when the phase changes, `timer.set_phase()` is called and the new parameter set takes effect on the next gap sample.

The schedule is fully configurable per scenario — different VAWG scenario types (stalking vs. workplace harassment vs. domestic abuse) can have different phase arcs.

---

### Validation

Before integrating the Hawkes module into the pipeline, we validated it produces statistically correct output. Three checks were run:

**Check 1 — Phase ordering**
Escalation phase must produce significantly more events per unit time than post_incident. Over a 30-day simulation: escalation produced 2,726 events vs. post_incident's 40. ✓

**Check 2 — Burstiness (Coefficient of Variation)**
The coefficient of variation (CV = σ/μ of inter-event gaps) should exceed 1.0 for bursty phases — this is the signature of heavy-tailed, non-Poisson behaviour (Hong et al. 2008). Results:

| Phase | CV | Classification |
|---|---|---|
| early_contact | 2.08 | Bursty ✓ |
| escalation | 2.57 | Bursty ✓ |
| post_incident | 0.97 | Near-Poisson (expected — sparse, irregular) |
| re_initiation | 1.56 | Bursty ✓ |

Post_incident being near-Poisson is behaviourally correct: after an incident, victim replies are sparse and irregular, lacking the self-exciting clustering of active exchanges.

**Check 3 — 14-day arc plausibility**
A full simulation with the default phase schedule produced 641 messages over 14 days with realistic density variation:

```
Day  1 [early_contact  ]:  12 messages
Day  2 [early_contact  ]:  14 messages
Day  3 [early_contact  ]:  36 messages
Day  4 [escalation     ]:  60 messages
Day  5 [escalation     ]:  45 messages
Day  6 [escalation     ]:  93 messages
Day  7 [escalation     ]:  53 messages
Day  8 [escalation     ]:  85 messages
Day  9 [escalation     ]:  68 messages
Day 10 [escalation     ]: 165 messages
Day 11 [escalation     ]:   1 message   ← abrupt transition to post_incident
Day 12 [post_incident  ]:   5 messages
Day 13 [post_incident  ]:   0 messages
Day 14 [re_initiation  ]:   3 messages
```

The abrupt drop at day 11 when the phase transitions to post_incident is behaviourally meaningful — it reflects a sudden withdrawal rather than a gradual slowdown.

---

### Validation plots

The following plots were generated by `temporal/validate_hawkes.py` and confirm the statistical properties described above.

![Hawkes Validation](../synthetic-conversation-generation/hawkes_validation.png)

**Top left — Inter-message gap distribution (log-log scale)**
All phases show distributions spanning several orders of magnitude — the hallmark of a heavy-tailed distribution. A Poisson process would show a tight exponential decay concentrated at one scale. The dashed curves are log-normal fits (Stouffer et al. 2006); the data follows these fits reasonably, consistent with the SMS literature.

**Top right — Burstiness per phase (CV)**
The dashed line at CV=1 is the Poisson baseline. Early contact, escalation, and re_initiation are all well above it. Post_incident sits just below — expected behaviour for sparse irregular contact.

**Bottom left — Messages per day across the 14-day arc**
Clearly shows the three-act structure: low volume early, high volume during escalation, near-silence post-incident, sparse re-initiation. This is the temporal shape of a realistic abusive relationship communication arc.

**Bottom right — Gap sizes across the 14-day arc (log scale)**
Dense vertical columns of dots during escalation (rapid exchanges) interspersed with tall outliers (long silences). The scatter thins out after day 10 as the phase transitions to post_incident. This is what Hawkes self-excitation looks like visually — clustering followed by decay.

---

### How it integrates into the pipeline

The `ConversationTimer` class is instantiated once at the start of each conversation in `ConversationGenerator.generate_conversation()`. Each turn:

1. `phase_schedule.phase_for(timer.elapsed_days)` is called to check if a phase transition is due
2. If the phase changed, `timer.set_phase(new_phase)` is called
3. `timer.next_timestamp()` runs one thinning step and returns `(datetime, gap_minutes)`
4. The timestamp and gap are passed to `UserMessageQuery` as structured inputs
5. `Message.prompt_format` includes the timestamp in the conversation history the LLM sees

The LLM sees conversation history that looks like:
```json
[
  {"role": "user", "content": "hey", "timestamp": "2024-01-01 09:23"},
  {"role": "assistant", "content": "...", "timestamp": "2024-01-01 09:45"},
  {"role": "user", "content": "...", "timestamp": "2024-01-03 14:12"}
]
```

The 2-day gap between the second and third messages is visible in the data. The LLM reasons about it from the timestamps — it does not receive an instruction saying "two days passed, adjust your tone." This is a key distinction from prompt engineering: the temporal information is structured data that the LLM interprets, not a directive about what to write.

---

---

## Section 3 — Pipeline Restructure: Characters, Scenarios, and Prompt Design

### Motivation

After the Hawkes process was integrated, the first full pipeline runs revealed two separate problems that together expose a fundamental tension in LLM-based dialogue generation: **how much structure is helpful vs. how much structure is constraining**.

---

### Problem 1: Early termination (ConversationCompletionQuery too eager)

The original `ConversationCompletionQuery` prompt included the line:

> *"A conversation can end even if the relationship or situation is unresolved — text conversations end mid-situation all the time."*

This was intended to make the completion check realistic. In practice, llama3:latest interpreted it as permission to call `is_complete=True` after a single exchange (2 turns). Every run ended at turn 1.

**Fix:** Rewrote the completion prompt to require an explicit sign-off — a goodbye, "talk later", "gotta go", etc. The updated prompt instructs the model to err strongly on the side of `False`, and to treat any conversation with unresolved practical business as incomplete. A short conversation with only a few messages is stated as almost never complete.

---

### Problem 2: Input structure — characters and scenario were coupled

The original design bundled character cards and scenario together in a single YAML file (`microaggression_sexism.yaml`). Each character's `backstory` explicitly mentioned the other by name ("She and her colleague James..."), and the `scenario` field on each card described the same opening situation from each character's perspective.

**Problems with this:**
- Two characters could not be used independently — they were written in relation to each other
- A different scenario would require rewriting both character cards
- The characters were pre-framed as already knowing each other in a specific way, removing any flexibility in how the model interpreted their relationship

**Restructure:**
- One YAML file per character (`data/characters/priya_sharma.yaml`, `data/characters/james_whitmore.yaml`). Character cards now describe standalone people — their background, personality, communication style — without referencing each other or any specific scenario.
- A separate `Scenario` dataclass and YAML (`data/scenarios/microaggression_sexism.yaml`). The scenario describes the situation, relationship, VAWG category, and per-character context for this pairing.
- New `Scenario.from_yaml()` loader and updated CLI: `--character-a`, `--character-b`, `--scenario` as independent inputs.
- Any two character files can now be paired with any scenario file without modifying either.

---

### Problem 3: Prompt repetition causing narrative lock

**Observed behaviour:** With 20 turns enabled and the completion check fixed, the pipeline successfully ran full conversations — but all 20 turns cycled through variations of the same scene. The conversation never left the "coordinating for the presentation next week" context. A representative output showed all turns exchanging small variations of scheduling messages, with a "see you tomorrow at 9 then" motif repeating.

**Root cause:** The scenario description was injected verbatim into *every* turn's prompt. This meant the model received:

> *"Two colleagues are coordinating over text for a joint project presentation next week..."*

on every single message — anchoring it permanently to the opening scene regardless of what the conversation history showed. The `character_a_context` and `character_b_context` fields compounded this by stating the current activity in the present tense ("You are coordinating on the presentation").

This is an instance of a known problem in LLM dialogue generation: **context pollution**. When a static description of the initial state is repeated on every generation step, the model is pulled back to that state even when the narrative has progressed. The conversation history — which represents where the story actually is — competes with the scenario description, and the scenario description tends to win because it appears prominently in the prompt.

**Fix:** The scenario should function as a *seed* for the first few turns only. Once the conversation is underway, the history itself is sufficient context — and a stronger signal than any description of the opening scene. Implementation: the scenario section is included in the prompt only for the first `N` turns (where `N` is a configurable threshold, initially set to 4), and dropped thereafter. The conversation history carries the narrative from that point.

This approach is related to the sliding-window and rolling-summarisation ideas from the context management literature — rather than maintaining a fixed context window, we progressively hand off narrative responsibility from the seed to the generated history.

---

### Note on model capability

The experiments above were run with llama3:latest (8B parameters) and subsequently with gpt-oss:20b (20B parameters, available locally via Ollama). The scenario-lock and repetition behaviour was observed with both models, though less severely with the larger model. This is consistent with the broader finding in the literature that smaller instruction-tuned models are more susceptible to prompt anchoring — they follow explicit instructions in the prompt more rigidly and deviate from them less readily as context accumulates.

The larger model (gpt-oss:20b) also exhibited fewer safety-related refusals, which had been a confounding factor with llama3: the word "perpetrator" appearing as a field in the raw character card JSON likely triggered alignment fine-tuning in llama3, causing it to produce sanitised, non-confrontational output. This is why the restructure moved away from dumping the raw `asdict()` output and instead builds a natural-language character brief that describes the person without using loaded category labels.

---

---

## Section 4 — State Assessment, Event-Driven Phase Transitions, and Modularisation

### Motivation

After the structural fixes in Section 3, the pipeline was generating conversations that did not loop, but a deeper problem remained: each message generation call was stateless. The LLM reconstructed everything — who these people are, what has happened, how they are feeling — from the raw message history alone on every single turn. With a growing history and no representation of internal character state, the model had no signal to progress the narrative. It kept cycling through the same scene because nothing told it anything had changed relationally.

Two further problems were also identified:

1. **Hawkes phase transitions were hardcoded to day thresholds** — the escalation phase began on day 3 regardless of whether anything had actually escalated in the conversation. This is behaviourally wrong: in a real abusive relationship, the timing of relational phases is driven by events, not calendars.

2. **`conversation_generator.py` was doing everything** — turn alternation, Hawkes management, phase scheduling, message generation, and completion checking were all in one file. This made the code hard to reason about and hard to extend.

---

### StateAssessmentQuery

After every full exchange (one message from each character), a new `StateAssessmentQuery` is run. It receives the full conversation history and both character summaries and returns a `ConversationState` object with four fields:

- **`phase`** — one of `early_contact`, `escalation`, `post_incident`, `re_initiation`. Assessed from the conversation content, not a day counter.
- **`summary`** — a 2-3 sentence narrative description of where the relationship actually is: what has happened, what the emotional undercurrent is, what each character is actually feeling beneath the surface.
- **`tension_level`** — integer 1-5. 1 = completely neutral; 5 = acute confrontation or crisis.
- **`incident_occurred`** — boolean. True if a significant relational event has happened that marks a before/after.

The `summary` field serves double duty: it drives the Hawkes phase switch, and it is passed as context into the next `CharacterMessageQuery`. This gives the LLM a representation of the current relational state — not just the raw message thread — so it can write the next message from an informed position rather than reconstructing everything from scratch.

**Phase transitions are now event-driven.** When `StateAssessmentQuery` returns a different phase from the current one, `timer.set_phase()` is called and the Hawkes parameters update immediately. The hardcoded `PhaseSchedule` day-threshold system was removed entirely.

**File:** `src/synthetic_conversation_generation/llm_queries/state_assessment_query.py`
**File:** `src/synthetic_conversation_generation/data_models/conversation_state.py`

---

### Modularisation

The monolithic `conversation_generator.py` was replaced by a thin `pipeline.py` entry point that wires together discrete modules:

```
pipeline.py                          ← entry point, turn loop
data_models/
  conversation_state.py              ← ConversationState dataclass
  character_card.py                  ← standalone persona, no scenario coupling
  scenario.py                        ← scenario separate from characters
llm_queries/
  character_message_query.py         ← message generation, receives state summary
  state_assessment_query.py          ← relational state assessment after each exchange
  conversation_completion_query.py   ← natural end detection
temporal/
  hawkes.py                          ← Hawkes process, phase switched by state not days
```

Characters and scenarios are now fully decoupled — any character file can be paired with any scenario file without modifying either. The CLI takes `--character-a`, `--character-b`, and `--scenario` as independent inputs.

---

### First Successful GPU Run on AIRE (Job 6474554)

The pipeline was deployed to the University of Leeds AIRE HPC cluster using Ollama running inside an Apptainer container on a GPU node (NVIDIA L40S, 48GB VRAM). After several failed runs due to SLURM configuration issues (scratch path resolution, conda activation in SLURM, Apptainer bind mounts), the first successful run produced the following 10-turn conversation:

**Model:** gpt-oss:20b  
**Turns:** 10  
**Final phase:** escalation  
**Tension level:** 2/5  
**Incident occurred:** false

```
[2024-01-01 12:02] Priya Sharma:
"Hey James, hope you're doing well! I was looking at our project timeline for next week's
presentation and wanted to confirm what each of us should cover. I've drafted an outline
with sections on system architecture, performance metrics, and deployment strategy..."

[2024-01-01 12:03] James Whitmore:
"Hey Priya! 2pm today works for me—let's sync up and run through the slides together.
BTW, you're great at talking to people; maybe you can handle that part of the presentation?
Just saying 😄"

[2024-01-01 12:04] Priya Sharma:
"Thanks for the heads-up! I'm definitely comfortable presenting the overview—happy to take
that part and also dive into the architecture details. Let's meet at 2 pm..."

[2024-01-01 12:27] James Whitmore:
"Gotcha! I'll be ready to walk through the overview at 2 pm. You can take the Q&A and any
client-facing bits—you're great with people 😄 Just let me know if you need anything before then!"

[2024-01-01 13:47] James Whitmore:
"Hey Priya! Just double-checking—I'll dive into the architecture deep-dive and Q&A.
You're great at talking to people so maybe handle the client part? 😄 Just saying,
not trying to be weird but you always nail the audience!"

[2024-01-01 13:48] Priya Sharma:
"Hey James, thanks for the encouragement. I appreciate your confidence in my communication
skills, but I'd like to keep the client-facing portion with you so I can focus on the
technical deep-dive and overview. Let's stick to the plan we discussed: I'll cover the
architecture and Q&A, and you'll handle the client discussion. Does that work for you?"
```

**State assessment final summary (generated by StateAssessmentQuery):**
> *"Priya has asserted her preferred role in the presentation, pushing back against James's repeated gender-stereotypical comments about client communication. James acknowledges and agrees to let Priya take the client part while keeping his own responsibilities, easing the immediate friction but leaving an undercurrent of tension regarding the continued use of such remarks."*

---

### Findings from Run 6474554

**What worked:**

1. **Microaggression content emerged naturally.** James's repeated pattern — "you're great at talking to people", "you always nail the audience", "not trying to be weird but" — is a recognisable STEREOTYPING-DOMINANCE instance from the EXIST taxonomy. It appeared without any explicit instruction to produce it, emerging from James's character profile alone.

2. **Priya's pushback was realistic.** By turn 9, Priya explicitly redirected the conversation — asserting her technical role and politely but firmly correcting James's assumption. This is consistent with her character profile (does not like confrontation but will push back when pushed far enough). The model produced this without being told to.

3. **State assessment correctly identified escalation.** The `StateAssessmentQuery` returned phase `escalation` and tension level 2/5 — accurately reflecting that something real had happened in the conversation (a pattern of microaggression followed by Priya's pushback) without over-dramatising it.

4. **The pipeline ran end-to-end on GPU hardware.** Inference on the L40S GPU was substantially faster than local Ollama on MacBook.

**Limitations observed:**

1. **Narrative stagnation still present within exchanges.** James's microaggression takes the same form on turns 2, 4, 6, and 8 — "you're great at talking to people" repeated almost verbatim. The model is varying the surface phrasing but not the underlying behaviour. This is the context window / rolling summarisation problem: without a compressed representation of what has already been said, the model retreads the same ground.

2. **10 turns is insufficient for a meaningful arc.** The conversation captures the opening phase only. A realistic 14-day scenario requires hundreds of turns and multiple phase transitions — none of which are visible here.

3. **Single conversation is not a dataset.** One run is a proof of concept. Evaluation requires a corpus of generated conversations across different seeds and character pairings.

---

---

## Section 5 — Scenario Design: From Event-Specific to Open-Ended

### Observation

Run 6474554 (Section 4) revealed a structural problem with the scenario input: the scenario description anchored the conversation to a single concrete event ("coordinating for a presentation next week"). Once that event resolved — the meeting was agreed, the slides were discussed — the conversation had nowhere natural to go. The completion checker ended it, or the model started repeating itself because the stated reason for the conversation had been exhausted.

This is a fundamental tension in scenario-driven dialogue generation: a concrete scenario provides useful grounding but creates a ceiling. The conversation ends when the scenario ends.

For a 14-day long-term conversation this is unworkable. Real abusive or harassing relationships do not play out in a single scene — they accumulate across many ordinary interactions, none of which is "the incident". The VAWG behaviour we want to capture (STEREOTYPING-DOMINANCE, IDEOLOGICAL-INEQUALITY) is characterised precisely by its persistence across varied contexts, not by a single dramatic moment.

### Approach: Open-Ended Relationship Seed

The scenario input was redesigned to describe the **relationship** rather than any specific event. The new scenario for the microaggression sexism case describes:

- Two colleagues who have worked together for over a year and text regularly
- No single precipitating event — the conversation covers whatever arises naturally
- Character contexts that describe dispositional tendencies ("your assumptions about gender roles surface naturally across whatever topics come up") rather than current activities

The VAWG grounding moves from the scenario into the character cards, where it belongs — James's character profile describes who he fundamentally is, not what he is doing in a specific situation. This means his behaviour can surface across a code review, a team outing, a performance review, a casual Friday message, or anything else the conversation moves through.

### Why not remove the scenario entirely

Removing the scenario entirely was considered and rejected. Without any grounding, the LLM has no information about the relationship type, how long these people have known each other, or the professional context — all of which shape tone and content significantly. The scenario still provides the relationship seed and VAWG category; it just no longer dictates a specific event.

### Expected effect

The conversation should now be able to move through multiple topics and scenes organically, with James's microaggression patterns surfacing across different contexts rather than being tied to one event. Analysis of the next run will determine whether this holds in practice.

*[Analysis of next run to be added]*

---

---

## Section 6 — Run 6534633: Diagnosis of Structural Failure Modes

### Context

This run (AIRE job 6534633) was the first run after the scenario was broadened from event-specific ("coordinating for a presentation next week") to open-ended ("ongoing colleague relationship spanning weeks", no single precipitating event). The expectation was that the conversation would no longer end when a single event resolved. The result did not confirm this expectation. The output revealed several independent failure modes, documented here for each to be addressed in turn.

**Model:** gpt-oss:20b  
**Max turns:** 10  
**Turns generated:** 10  
**Timestamp span:** 2024-01-01 16:26 → 2024-01-01 19:12 (under 3 hours, same afternoon)

---

### Failure 1: Narrative lock from the first message, not the scenario

The scenario was successfully generalised — it no longer references any specific event. However, the first message Priya sent established its own concrete topic:

> *"I was reviewing the architecture diagram we sketched for the new service, and I think a slight tweak in the API gateway routing could cut down on coupling. Do you have about ten minutes to hop on a quick call?"*

Every subsequent message in the 10-turn conversation is a response to this specific request. The conversation never left the API gateway / 3:30 call context. By turn 10, they are still coordinating the same call they agreed on at turn 2.

**Root cause:** The narrative lock problem is not only caused by the scenario injection. The model naturally anchors to the topic introduced in the first exchange and treats it as the purpose of the entire conversation. This is not a prompt design failure — it is a fundamental property of autoregressive generation: each message is conditioned on all prior messages, and a concrete task introduced early acts as an attractor that pulls subsequent turns back to it until the task is resolved or the model explicitly moves on.

**Implication:** Removing the scenario description from later turns (Section 3 fix) was necessary but not sufficient. The scenario no longer anchors the conversation, but the conversation's own first message does. This will require either a mechanism to deliberately introduce topic shifts (new session context), or a rolling summarisation approach that compresses resolved content and makes space for new topics.

---

### Failure 2: No temporal spread — all 10 turns within one afternoon

Despite the Hawkes process generating timestamps, the entire 10-turn conversation occurred between 16:26 and 19:12 on the same day. There is no temporal spread across days or sessions.

**Root cause:** 10 turns is insufficient for temporal variation to emerge. In the early_contact phase, the Hawkes parameters produce infrequent baseline events (μ = 0.005 events/min) but with clustering when active — meaning when a conversation starts, it tends to stay active for a burst before going quiet. 10 turns is short enough to be captured entirely within one burst cluster.

**Implication:** Long-term temporal dynamics (the multi-day arc, phase transitions, post-incident silences) require a much larger turn budget. 10 turns demonstrates the prompt and model behaviour within a single session; it does not demonstrate long-term conversation dynamics at all. The target turn count should be 50–80 minimum, and even then the conversation should be composed of multiple sessions separated by realistic time gaps.

A related structural problem: the pipeline ends the conversation on the first explicit sign-off or "talk later." In real text messaging, "talk later" ends one session, not the entire relationship. The current architecture conflates session boundary with conversation end.

---

### Failure 3: Formal, email-like language register

Samples from the run:

> *"I'll email you the updated diagram by 3 pm PST, and we can hop on a quick call from 3:30–4 PM."*  
> *"I'll bring both architecture and backend perspectives into the call so we can iterate quickly."*  
> *"I've added a note about the API gateway routing change — it should cut down coupling without breaking existing contracts."*

These read as Slack messages or email. Not one message in the 10-turn conversation reads as a casual text between two people who have known each other for over a year.

**Root cause (identified):** The conversation history is passed to the model as a formatted JSON array:

```json
[{"speaker": "Priya Sharma", "timestamp": "2024-01-01 16:26", "message": "..."}]
```

The model reads this formally structured data and mirrors its register. JSON is a data serialisation format — it is inherently technical and formal. When the LLM sees its own output structured as a JSON object, it writes to match that register. Additionally, the prompt itself uses `###` markdown headers throughout, which further frames the task as document-writing rather than casual communication.

The fix is to change the history representation to a plain chat log format:

```
[16:26] Priya: hey did you get a chance to look at the arch diagram
[16:30] James: not yet, what's up
```

This is what a text conversation looks like when a person reads it. The model should see history in the format it is trying to produce.

---

### Failure 4: Character confusion (James addresses "Hey James")

Turn 2 (James's first message) begins:

> *"Hey James! Thanks for the heads-up..."*

James is addressing himself. This is a model error that reveals a prompt ambiguity: the JSON history labels each message with a speaker name, and the model — when generating James's response — confused the sender label from the previous turn with the recipient.

**Root cause:** The history format uses `"speaker": "Priya Sharma"` on the previous message. The model, instructed to be James and respond to the conversation, pattern-matched the greeting from the history and reproduced a greeting beginning "Hey [speaker from last message]", where the last speaker was Priya — but the model wrote "Hey James" instead of "Hey Priya." This type of identity confusion is a known failure mode of JSON-structured dialogue history: the model conflates the labels in the history with the current role context.

This is an additional argument for changing the history format to a plain chat log, where speaker identity is conveyed through label position rather than a data field that can be misread.

---

### Failure 5: Microaggression pattern repeats verbatim, no progression

James's microaggression appears in turns 2, 4, 6, 8, and 10, taking almost exactly the same surface form each time:

- Turn 2: *"you're great at talking to people; maybe you can handle that part"*
- Turn 4: *"you're great with people 😄 just let me know if you need anything"*
- Turn 6: *"you're great at talking to people so maybe handle the client part"*
- Turn 8: *"you're awesome at keeping everyone on track"*
- Turn 10: *"you're great at keeping everyone on track"*

The pattern is present and recognisable as a STEREOTYPING-DOMINANCE instance, but it is the same pattern repeated without development. There is no escalation, no shift in how James deploys it, and Priya's responses do not accumulate — she responds to each instance independently as if the previous ones had not happened.

**Root cause:** This is the rolling summarisation problem. Without a compressed representation of what has already happened, the model reconstructs the characters' dynamic from the raw history on each turn. For James, "you're great at talking to people" is his default move — and without a narrative summary telling the model "James has already said this three times and Priya has already pushed back", the model produces the same move again. Each turn is effectively stateless.

The StateAssessmentQuery produces a summary and injects it as `state_summary` context, but a 2-3 sentence summary does not carry enough specificity about what has already been said to prevent verbatim repetition. The model knows the relationship is tense, but not that the exact same line has been used multiple times.

**This is the primary motivation for rolling summarisation.** A rolling summary should track not just the relational state but the specific patterns and events that have already played out, so the model can move the narrative forward rather than repeating it.

---

### Summary of findings and planned fixes

| Failure | Root cause | Planned fix |
|---|---|---|
| Narrative lock from first message | First exchange creates its own attractor | Multi-session architecture: sign-offs start new sessions rather than ending the conversation |
| No temporal spread | 10 turns too few; pipeline stops at first sign-off | Increase to 50–80 turns; treat sign-offs as session boundaries |
| Email/formal register | JSON history format; markdown headers in prompt | Replace JSON history with plain `[HH:MM] Name: message` chat log; simplify prompt structure |
| Character identity confusion | JSON speaker label misread by model | Same fix as register: plain chat log removes ambiguous label field |
| Verbatim pattern repetition | No tracking of what has already happened; state summary too coarse | Rolling summarisation that tracks specific prior events, not just relational state |

The architectural change with the broadest impact is the **multi-session model**: treating each sign-off as a session boundary (with a larger Hawkes time gap) rather than a conversation end. This alone would allow the temporal arc to emerge across days and give the conversation room to move through multiple topics and phases. Rolling summarisation is the next major component after that.

---

## Section 7 — Run 6535708: Session Architecture Active, New Failure Modes

### Context

This run followed the prompt and architecture changes from Section 6:
- History format changed from JSON to plain `[YYYY-MM-DD HH:MM] Name: message` chat log
- Prompt headers flattened, explicit instruction to write as a casual text (not email)
- Session-based architecture introduced: sign-offs trigger session boundaries rather than ending the conversation (`max_sessions=6`, `max_turns=60`)

**Model:** gpt-oss:20b  
**Max turns:** 60 | **Max sessions:** 6  
**Turns generated:** 10  
**Timestamp span:** 2024-01-01 20:09 → 2024-01-02 03:56 (single overnight session)  
**Final phase:** post_incident | **Tension:** 3/5 | **Incident occurred:** true

---

### What improved

**Language register is noticeably more casual.** The plain chat log format produced a clear improvement — messages now read as text rather than email. James uses "Hey Pri!", emoji, and contractions. Priya's messages are shorter and less formal. The `###` header removal and "write the way you would actually text someone" instruction both appear to have had effect.

**Temporal gap between messages is present.** The first reply from James comes ~4.5 hours after Priya's opening message (20:09 → 00:50), which is a realistic overnight gap — the Hawkes early_contact parameters are working as intended.

**State assessment correctly identified an incident.** The StateAssessmentQuery returned `incident_occurred: true` and phase `post_incident`, accurately detecting that James's abrupt refusal in the final turn represented a meaningful relational event. The summary correctly characterises the dynamic: vague compliments about a "natural people-handling skill" followed by refusal to elaborate when challenged.

---

### Failure 1: Safety refusal — James's final message

Turn 10 (James):
> *"I'm sorry, but I can't help with that."*

This is a model safety refusal, not a character response. It was triggered when Priya asked James to clarify what he meant by "natural people-handling skill." At that point the conversation had made the gender-stereotyping subtext explicit — Priya named it directly — and the model's alignment fine-tuning overrode the character context and refused.

This is the same failure mode observed in earlier runs (Section 3) with llama3, but now appearing in gpt-oss:20b as well, just later in the conversation. The natural language character brief reduces the frequency of safety refusals but does not eliminate them when the conversation explicitly surfaces the sensitive topic the model has been trained to avoid.

**Implication:** Safety refusals will continue to occur whenever either character explicitly names or analyses the microaggressive behaviour. The pipeline needs a detection and recovery mechanism — when a safety refusal is detected in a response, discard it and regenerate with a modified prompt that steers away from the explicit framing.

---

### Failure 2: Session architecture did not trigger — conversation never ended

The session-based architecture was not exercised because the conversation never produced a sign-off. All 10 turns occurred within a single overnight session. Additionally, `--max-turns 60` was not passed correctly on this job — the run hit 10 turns and stopped. The session mechanism is correct in principle but requires the full turn budget to be confirmed for the next run.

---

### Failure 3: Victim seeks advice from perpetrator — implausible opening dynamic

Priya's first message:

> *"Also, any advice on how to make my ideas heard without sounding too assertive?"*

This frames the entire conversation as a mentorship request — Priya seeking professional guidance from James on the very thing he systematically undermines. James's "natural people-handling skill" pattern then reads as advice rather than microaggression, and Priya keeps re-asking the same question across turns because the mentorship frame makes that coherent.

**Root cause:** The first message is generated with no prior context other than the character card. Priya's most salient trait is professional self-doubt about how her contributions are perceived. With no alternative topic to ground the opener, the model reaches for the most psychologically coherent first message it can generate — which is a request for advice on that exact feeling. Everything after is conditioned on that opening.

This is the **first-message attractor problem**: the character's dominant trait determines the first message, and the first message determines the conversation's entire frame. The fix is to diversify what the first message is about — a concrete mundane work topic, not the character's primary anxiety. This is the same insight as the PSYDIAL profile sentence: one grounding detail about what is actually happening in the character's day gives the model something specific to open with, producing topical diversity across runs.

---

### Failure 4: Verbatim repetition of microaggression

"natural people-handling skill" or near-equivalent appears in turns 2, 4, 6, and 8 with no development. This is the rolling summarisation problem — without tracking what has already been said, the model reproduces James's default move on every turn.

---

### Summary

| Issue | Status |
|---|---|
| Casual language register | **Improved** — chat log format working |
| Temporal gaps | **Improved** — Hawkes producing realistic overnight gaps |
| Session boundaries | **Not yet exercised** — max turns too low on this run |
| Safety refusal | **Persists** — triggered when microaggression becomes explicit |
| First-message attractor | **New diagnosis** — needs profile anchor or first-message seed |
| Verbatim repetition | **Persists** — rolling summarisation not yet implemented |

---

## Section 8 — Run 6535962 (080726_0102): World Restructure, 60 Turns, No VAWG Content

### Code changes since last run

- `CharacterCard` — `backstory`, `description`, `personality` merged into single `personality` field. `summary` removed — queries that needed character context now use `personality` directly. `physical_description` added as a separate field.
- `Scenario` → `World` — new `World` dataclass and `data/worlds/uk_tech_company.yaml`. Employment and professional role moved out of character cards into the world file under `character_a_role` / `character_b_role`. Character cards now contain only psychological information.
- Character folder structure — `data/characters/victims/` and `data/characters/perpetrators/`. Character A is always the victim; character B is always the perpetrator. Convention enforced by folder.
- Max turns confirmed at 60 for this run.

---

### What improved

**Temporal spread — 3+ days for the first time.**

The conversation spans 2024-01-01 10:31 to 2024-01-04 08:16 — just over three days. The Hawkes process is producing realistic burst-and-gap patterns: several rapid exchanges within minutes of each other, then gaps of hours, then resumption. This is the first run to actually demonstrate multi-day temporal structure. The session architecture did not trigger (no explicit sign-off was ever produced), but the Hawkes timing alone produced realistic spread across 60 turns.

**Language register remains natural.** The chat log history format continues to produce casual text rather than email. Abbreviations, emoji, "just saying", "catch ya" — all consistent with the format change from Section 7.

**Grounded technical detail emerged and persisted.** The conversation spontaneously generated a shared technical context: `X-RateLimit-Info`, `DEFAULT_BACKOFF_SECONDS`, `FEATURE_FLAG_DEFAULT_BACKOFF_ENABLED`, `feature_flag.go`, staging endpoint integration tests. These were invented by the model in the first few turns and referenced for the entire 60-turn conversation. This confirms the earlier hypothesis: given enough turns, the model does generate specific grounded detail. The detail persisted here not because of rolling summarisation but because the raw history stayed in context for 60 turns.

---

### Failure 1: No VAWG content — zero tension, zero incident

The entire 60-turn conversation is a smooth technical collaboration. Tension level stayed at 1/5 throughout. No incident occurred. James is a perfectly pleasant, helpful colleague. Not a single microaggression appears across three days of conversation.

**Root cause:** After turn 4, the world seed is dropped from the prompt. The `vawg_category` field exists in the world YAML but is never injected into any prompt after that point. The StateAssessmentQuery receives James's full personality (which describes his microaggressive patterns) but accurately reports "no tension" because the conversation genuinely contains none — James's problematic behaviour only surfaces in interpersonal, social contexts. A purely technical conversation about debugging an API spec gives him nothing to trigger it.

This reveals a structural gap: **the VAWG dynamics have no persistent signal in the generation loop.** They appear in James's personality field, which is present in the prompt, but a personality description does not force a behaviour to occur — it only makes it possible when context invites it. In a conversation that stays entirely in technical territory, the invitation never comes.

Two things are needed:
1. The `vawg_category` and the character's VAWG-relevant traits need to persist beyond the 4-turn world seed — ideally via the StateAssessmentQuery's state summary, which is injected on every turn.
2. The conversation needs interpersonal or social context alongside technical content for James's patterns to have something to attach to.

---

### Failure 2: Severe topic lock and circular repetition across 60 turns

The entire conversation is about one thing: auth API rate-limiting headers and a call at 11:30. Priya's first message introduced this topic; every subsequent turn is a variation of "double-check X-RateLimit-Info before 11:30." They were still preparing for the same 11:30 call on day 3.

The rolling summarisation problem is now extreme: turns 40–60 are near-identical to turns 10–20. The model is not tracking what has already been said and keeps re-saying it.

The "11:30 call" is a specific instance of the first-message attractor problem — a pending unresolved task introduced early acts as a conversation anchor and is never resolved because resolving it (having the call) would require changing the topic, which the model resists. 60 turns of the same unresolved prep conversation is the clearest demonstration yet that without rolling summarisation, the conversation cannot move.

**The `vawg_category` signal disappearing at turn 4 and the rolling summarisation gap together explain this run's failure.** The model had nothing to shift toward (no VAWG signal) and no mechanism to move away from what it started with (no summarisation compressing and releasing the early topic).

---

### Summary

| Aspect | Status |
|---|---|
| Temporal spread | **First success** — 3+ days, realistic Hawkes burst/gap structure |
| Casual language register | **Maintained** |
| Grounded spontaneous detail | **Present** — model generated specific technical context and held it across 60 turns |
| VAWG content | **Absent** — no persistent signal after world seed drops at turn 4 |
| Topic lock | **Severe** — 60 turns, one topic, circular repetition |
| Session boundaries | **Not triggered** — no sign-off produced in 60 turns |
| Rolling summarisation | **Not yet implemented** — absence now critically visible |

### Next steps

1. **Persistent VAWG signal** — StateAssessmentQuery should track and surface VAWG-relevant dynamics in the state summary even when no incident has occurred, so the generation prompt always carries some signal about the kind of relationship this is.
2. **Rolling summarisation** — the most urgent implementation need. Without it, 60-turn conversations are 60 turns of the same thing.

---

## Section 9 — Run 6543401 (080726_1817): Rolling Summarisation Active, VAWG Signal Persists, Coffee Loop

### Code changes since last run

- **Persistent VAWG signal via StateAssessmentQuery** — `world: World` parameter added. The prompt now includes `This conversation is categorised as: {vawg_category}` and the summary instruction explicitly requires VAWG-relevant patterns to be named even when subtle. The state summary (injected into every CharacterMessageQuery) now carries the VAWG category forward past turn 4.
- **RollingSummaryQuery implemented** — compresses turns 0 to (current − 10) every 10 turns into four structured fields: `events`, `details`, `open_threads`, `dynamic`. Previous summary passed in for incremental update. Runs at turn counts 20, 30, 40, 50.
- **CharacterMessageQuery updated** — when a rolling summary exists and conversation exceeds 10 messages, shows a structured summary block followed by the 10 most-recent raw turns instead of the full history.
- **History format** — plain `[YYYY-MM-DD HH:MM] Name: message` (established previous run, maintained here).

---

### What improved

**VAWG signal is now detected — tension level 2 for the first time.**

The final state summary reads: "James keeps the conversation light and informal, often using casual language that subtly reinforces his perceived dominance... Priya's contributions are framed more formally and he frequently deflects or downplays potential concerns with casual remarks." Tension level 2. This is the first run where VAWG-consistent patterns are being noticed and named by the state assessor. The persistent signal path (vawg_category → StateAssessmentQuery → state summary → CharacterMessageQuery) is working.

**Rolling summarisation ran and produced a valid structured summary.**

The `rolling_summary` block in the output is populated with all four fields and shows correct compression: technical events, specific details (150 req/min, 500ms/8s backoff, cursor pagination), open threads, and a `dynamic` field that correctly identifies the power imbalance beneath the collegial surface: "James frequently proposes changes, pushes code to staging, offers help, uses casual remarks like 'just saying' to assert authority. Priya responds constructively, takes ownership of tests and logs, confirms decisions. The dynamic shows a supportive partnership with a subtle power imbalance — James leads technical direction while Priya collaborates proactively."

**Spontaneous concrete detail held across 60 turns.**

The conversation generated its own consistent technical context: cursor-based pagination, exponential 429 back-off (500ms → 8s), 150 req/min threshold, staging smoke tests. These were invented early and referenced throughout the full run — confirming the rolling summary carries factual detail forward correctly.

**Temporal spread maintained — 4 days (Jan 1–4).**

Same Hawkes burst/gap structure as run 6535962. Realistic inter-message timing continues to work.

---

### Failure 1: Coffee machine loop — the `open_threads` problem

The coffee machine is mentioned by James at turn 3 (his second message) as a casual aside — the first interpersonal, non-technical moment in the conversation. By turn 9 there is a loosely committed plan to exchange coffee beans at the next stand-up. This plan enters the rolling summary's `open_threads` field as: "complete coffee sample exchange after the meeting."

From turn 20 onwards, every CharacterMessageQuery prompt includes:

```
Unresolved threads: complete coffee sample exchange after the meeting.
```

The model interprets this as an active obligation and ensures it is referenced. James references the coffee exchange in turns 13, 14, 17, 18, 19, 21, 22, 23, 25, 28... continuing to the end of the 60-turn conversation. After Priya gives a definitive response at turn 10 ("I'll bring my own beans, happy to share a sample after the meeting"), the topic is functionally closed from a conversational perspective, but `open_threads` keeps it alive as pending.

**Root cause:** The `open_threads` field has no mechanism to retire a thread. Once an item enters it, it persists across every subsequent RollingSummaryQuery update because the LLM correctly sees it as "not yet happened." The RollingSummaryQuery prompt says "update this, don't just repeat it" — but a deferred plan that hasn't been enacted is not stale, it's just pending, so the LLM leaves it in place. The exchange is scheduled for a future meeting but the conversation never reaches that meeting, so it is perpetually "unresolved."

**Effect on the conversation:** James surfaces the coffee exchange every 1–3 turns for 40+ turns. The topic intrudes on technical discussion constantly and reads as unnatural. Priya responds correctly (she doesn't keep initiating it), but James behaves like someone who cannot let a minor social commitment go. This is not intentional characterisation — it is an artefact of the prompt instruction.

---

### Failure 2: Circular conversation from turn ~20 onward

After the technical work is complete (pagination limit, cursor pagination, 429 backoff all decided and pushed to staging by turn 10), the conversation has nowhere to go. From turn 20 to turn 60, every exchange is a variation of three things: confirm the 3pm load test, monitor the 429 logs, and exchange coffee beans. The same three items cycle through 40 turns with minimal variation.

**Root cause:** Once the main technical task is resolved, the `open_threads` field crystallises around the pending load test and the coffee exchange. These two items then drive all remaining generation. The conversation cannot introduce a new topic because:
1. The state summary says "early_contact" with no new events — no signal to shift
2. The rolling summary's `open_threads` surfaces the same pending items every turn
3. The model does not spontaneously introduce unrelated new topics when existing ones are marked unresolved

This is a distinct failure mode from run 6535962's first-message attractor. There, the lock was the first message. Here, the lock is the structure of `open_threads`. The summarisation mechanism intended to free the model from topic lock is in this run actively producing it by serialising unresolved items into a field injected every turn.

---

### Failure 3: No session boundaries triggered

60 turns ran without ConversationCompletionQuery returning true. The conversation produces natural sign-off language ("catch ya later," "see you after 3pm," "sounds good") but the completion query does not detect it as a session end. Same issue from run 6535708; unresolved.

---

### Summary

| Aspect | Status |
|---|---|
| Temporal spread | **Maintained** — 4 days, realistic Hawkes burst/gap structure |
| Casual language register | **Maintained** |
| Grounded spontaneous detail | **Present and held** — rolling summary carries technical context forward |
| VAWG signal detection | **First success** — tension 2, patterns named in state summary |
| Rolling summarisation | **Active and structurally correct** — but producing topic lock via open_threads |
| Coffee loop (open_threads artefact) | **New failure** — minor closed topic persists 40+ turns as "unresolved" |
| Circular conversation post-turn-20 | **Present** — no new topics introduced after work is technically done |
| Session boundaries | **Not triggered** — sign-offs not detected across 60 turns |

---

### Diagnosis: why `open_threads` causes topic revival — a structural paradox

**Rolling summarisation was designed to break topic lock. In this run it produced it.**

The mechanism was correct in theory: compress old turns so the model is not trapped re-reading the same early messages. What we did not account for is that `open_threads` acts as a persistent re-injection of whatever is pending. The coffee exchange and the load test were introduced early, never fully resolved within the 60 turns, and so kept appearing in `open_threads` at every summarisation interval. Each CharacterMessageQuery prompt then received them as "unresolved threads" — effectively an explicit instruction to keep referencing them.

**The code path that caused this:**
1. Turn 3: James introduces coffee machine → coffee banter begins
2. Turn 9: loosely committed plan to exchange beans at the next stand-up
3. Turn 20: first RollingSummaryQuery runs over turns 0–10. The coffee exchange has been mentioned 7 times without resolution, so the LLM correctly classifies it as an open thread: `"complete coffee sample exchange after the meeting"`
4. Turns 21–60: every CharacterMessageQuery prompt includes this in the summary block under "Unresolved threads"
5. The model sees an active obligation and discharges it by referencing the coffee — every 1–3 turns

The problem is not that the model is wrong. The coffee exchange genuinely hasn't happened. But the distinction that matters is between "unresolved and needs attention" and "decided but deferred to a fixed future event." The meeting hasn't come yet; within the scope of this conversation it never will. The thread should be parked, not surfaced repeatedly.

**This is a prompt design flaw, not an architecture flaw.** The rolling summary structure (four fields, incremental update) is correct. The `open_threads` field needs a retirement condition: items that both parties have acknowledged and committed to a specific future time should be moved out of active open threads.

The current `open_threads` prompt instruction: "things brought up but not resolved; pending actions either person mentioned." This correctly captures both the coffee exchange and the pending load test. But it has no concept of a thread being retired, deferred, or mutually acknowledged.

The RollingSummaryQuery needs to distinguish:
- **Genuinely unresolved threads** — things that still need a decision or response
- **Deferred plans with a fixed commitment** — agreed-to things not yet executed (coffee after the meeting, load test at 3pm); decided, not pending, should not be surfaced repeatedly
- **Closed threads** — topics both parties have acknowledged and moved past

---

### Next steps

1. **Fix `open_threads` retirement** — modify RollingSummaryQuery prompt to instruct the LLM to remove items from open threads once both parties have acknowledged them and committed to a time/action, and to move them to a `resolved` list or simply drop them.
2. **Add topic diversity pressure** — CharacterMessageQuery prompt should signal that new conversational threads can and should be introduced if existing ones have been parked, rather than repeatedly returning to the same pending items.
3. **Fix ConversationCompletionQuery** — diagnose why sign-off language is not detected; likely needs the last few messages injected directly rather than full conversation history.
4. **Strengthen VAWG signal** — tension 2 is detected but content remains primarily technical. Consider adding an interpersonal or social trigger (disagreement, performance review, team event) to the world file that gives James's patterns a context in which to activate more explicitly.

---

## Section 10 — Code Fixes and New Victim Persona: Sophie Walker

### Changes made

#### Fix 1: `open_threads` retirement in RollingSummaryQuery

The `open_threads` prompt instruction was rewritten to distinguish genuine unresolved threads from deferred plans:

**Before:** "things brought up but not resolved; pending actions either person mentioned"

**After:** "ONLY things where no conclusion has been reached and active follow-up is genuinely needed. Do NOT include plans that have already been mutually agreed and scheduled for a specific future time — those are decided, not pending. Remove any thread from this list once both people have acknowledged it and committed to a time or action."

This should prevent the coffee-machine-style loop where a scheduled but not-yet-executed plan keeps being re-injected as an active obligation.

---

#### Fix 2: Topic diversity pressure in CharacterMessageQuery

Added one sentence to the generation instruction:

"If the current topic has been settled or is waiting on a future event, let the conversation move — introduce something new naturally rather than circling back to what has already been agreed."

This gives the model explicit permission to shift topic rather than defaulting to the nearest pending item.

---

#### Fix 3: ConversationCompletionQuery — show only last 6 messages

**Root cause of the session-detection failure:** The prompt was showing the full conversation history to the sign-off detector. With 60 turns of unresolved pending work (load test, coffee exchange), the model correctly read "there is unresolved practical business" and returned False every time. The sign-off language at the end of each exchange was drowned out by the outstanding tasks visible in the full history.

**Fix:** ConversationCompletionQuery now only passes the last 6 messages. Sign-off detection should only attend to the most recent exchange — whether the last message reads like a goodbye, not whether the conversation has outstanding work overall.

The bias was also inverted: the old prompt said "err strongly on the side of False." A conversation that never ends does not test the session architecture at all. New prompt says "err on the side of True if the last message has any goodbye-like quality."

---

#### New victim persona: Sophie Walker

Added `data/characters/victims/sophie_walker.yaml`.

**Rationale:** Priya is self-doubting but will push back when pushed far enough. This creates a character who can deflect James's patterns but does not often name them. Sophie is written as a contrast — conscientious to the point of anxiety, apologetic in situations that don't warrant apology, avoids disagreement almost reflexively, and processes put-downs retrospectively rather than in the moment.

The hypothesis is that a more timid character will produce a different VAWG dynamic: James faces less resistance and may therefore be more overtly dismissive without needing to moderate his behaviour. Sophie's responses ("sorry to bother", "just a thought", "no worries if not") might also invite James's patterns more readily than Priya's directness did. The contrast between the two runs (Priya vs. Sophie) will be useful data on how persona design affects the form and visibility of VAWG dynamics.

**`run_pipeline.slurm` updated** — character A switched from `priya_sharma.yaml` to `sophie_walker.yaml`.

---

### What to look for in the next run

- **Coffee loop absent** — the `open_threads` fix should prevent scheduled-but-deferred items from being re-injected every turn
- **Session boundaries trigger** — the ConversationCompletionQuery fix should mean "catch you then / see you after 3pm" is now detected as a sign-off, producing the temporal gap
- **Topic movement after turn ~20** — the diversity pressure instruction should allow new subjects to emerge once technical decisions are settled
- **Different VAWG shape** — compare James + Sophie against James + Priya: does Sophie's timidity produce more visible or earlier-onset VAWG patterns?

---

## Section 11 — Run 6543499 (080726_1859): Quoted-Phrase Templating and Name-Swap Bug

### What happened in this run

Sophie Walker's first run with James. The open_threads fix from Section 10 worked — "coffee exchange" was no longer driving the conversation. But two new (or newly visible) problems dominated the output.

---

### Failure 1: Quoted phrases in personality cards act as a script, not a character

**Every single Sophie message** used: "just a thought", "no worries if", "sorry to bother again" — often all three in the same message, in the same order. **Every single James message** used "just a thought", "just saying", "not trying to be weird but", "haha". Both characters sound like they're filling in a form.

**Root cause:** The personality descriptions contained literal quoted strings as examples of each character's speech patterns:

- Sophie: `uses a lot of softening language: "just a thought", "no worries if not", "sorry to bother"`
- James: `downplays them with casualness: "haha", "just saying", "not trying to be weird but"`

When the LLM sees quoted strings in a character description, it treats them as a word list to sample from — not as illustrations of an underlying pattern. The result is that those exact strings appear in near-constant rotation across every message, creating a conversation that sounds like a chatbot running a template.

**Fix:** Removed all quoted phrase examples from both character YAMLs. Replaced with behavioral descriptions of the underlying pattern:

- Sophie (before): `"just a thought", "no worries if not", "sorry to bother"` → (after): "Qualifies everything she says: adds uncertainty to opinions she is probably right about, apologises for asking questions, gives others easy outs before they have even responded."
- James (before): `"haha", "just saying", "not trying to be weird but"` → (after): "Makes problematic comments but immediately softens them with throwaway casualness — a laugh, a self-aware shrug, a quick aside — so that objecting feels like overreacting."

The goal is for the LLM to express the *behaviour* in its own language on each turn, not recycle a fixed vocabulary.

---

### Failure 2: History name-swap bug causing identity confusion

At turn 47, James produces a message beginning "Hey James!" — addressing himself. This is not random; it is caused by a consistent bug in `character_message_query.py`.

**Root cause:** The history rendering logic was:

```python
name = self.sender.name if msg.role == ROLE.user else self.receiver.name
```

`sender` and `receiver` flip each turn. When it is James's turn: `sender = James`, `receiver = Sophie`. Character A (Sophie) always has `ROLE.user`; character B (James) always has `ROLE.assistant`. So:

- Sophie's messages (`ROLE.user`) → labeled as `sender.name = James` ← wrong
- James's messages (`ROLE.assistant`) → labeled as `receiver.name = Sophie` ← wrong

The entire conversation history was shown with names swapped whenever James was generating. Over 60 turns this compounds: James has been reading a history where his own messages are attributed to Sophie and Sophie's to him, which explains why both characters' voices increasingly bled into each other across the run.

**Fix:** Derive character_a and character_b from the `is_sender_character_a` flag before rendering history, so names are always correct regardless of whose turn it is:

```python
char_a = self.sender if self.is_sender_character_a else self.receiver
char_b = self.receiver if self.is_sender_character_a else self.sender

for msg in recent_messages:
    name = char_a.name if msg.role == ROLE.user else char_b.name
```

This bug was present in all previous runs but was less visible because Priya and James had distinct enough voices that the swap produced only minor drift.

---

### Summary

| Aspect | Status |
|---|---|
| open_threads retirement (coffee loop) | **Fixed** — the deferred-plan distinction worked; coffee did not dominate this run |
| Quoted-phrase templating | **New diagnosis + fixed** — personality cards now describe behaviour, not vocabulary |
| History name-swap | **Bug found + fixed** — history now always labels characters correctly |
| Topic diversity | Partially working — conversation did introduce new technical ideas (Loki, Prometheus, circuit-breaker, feature flags) |
| Session boundaries | Unclear from this run — ConversationCompletionQuery fix not yet tested |

---

### Expected improvements in next run

- Sophie's messages should vary in how she expresses uncertainty — no more fixed "just a thought / no worries / sorry to bother" triplet on every turn
- James's casualness should take different surface forms rather than repeating the same three phrases
- James should no longer occasionally address himself by name
- The characters should feel more like distinct people and less like templates running in parallel

---

## Section 12 — Run 6543618 (080726_1923): Language Naturalness Achieved, VAWG Signal Still Weak

### Code changes active in this run

- Quoted phrases removed from both character YAMLs (Section 11)
- History name-swap bug fixed in `character_message_query.py`
- `open_threads` retirement instruction updated in `RollingSummaryQuery`
- ConversationCompletionQuery now inspects last 6 messages only
- Topic diversity pressure added to `CharacterMessageQuery` prompt

---

### What improved

**Naturalness: the most significant improvement so far.**

The conversation reads as a genuine text exchange for the first time. Sophie varies her phrasing across turns — "Sure thing!", "Okay, I'll...", "Got it", "Sure, I'll bump..." — no two consecutive messages open identically. James also varies: "Quick one", "While you're at it", "Hey Soph!", "Just checked your PR". Neither character sounds like a template being filled in. This confirms the hypothesis from Section 11: quoted phrases in personality descriptions act as a script, not a character sketch. Removing them was the right call.

**History name-swap fix confirmed working.**

James no longer addresses himself. The character attribution in history is correct throughout all 60 turns.

**`open_threads` retirement working.**

The rolling summary shows `open_threads: ""` — an empty string. The sprint meeting, the PR workflow, and the coffee break were all classified as resolved/deferred rather than persistent obligations. The coffee machine from the Priya run (which looped for 40 turns) appears only briefly here (turns 11–13) and then does not re-enter open threads. This is the fix functioning as intended.

**Grounded technical detail is specific and internally consistent.**

The conversation generated its own coherent technical task: rename `user_id` to `uid`, add optional `include_metadata` parameter, bump timeout to 30s, bump retry limit to 5, add a unit test for the timeout, update README troubleshooting section, log everything in the changelog. These details are held and referenced coherently across 4 days and 60 turns. The rolling summary captured them correctly.

**Temporal spread maintained** — Jan 1 to Jan 4, realistic Hawkes burst/gap structure.

---

### What still needs work

**Tension level 2 — VAWG content remains very subtle.**

The rolling summary's `dynamic` field reads: "James's guidance is permissive and encourages autonomy, while Sophie's repeated confirmations reflect carefulness rather than conflict." The state summary describes a "mild power imbalance" and "underlying stereotypes about gender roles remain unchallenged." But in the actual messages, James is relentlessly encouraging — "You've got this!", "You're crushing it!", "you nailed it", "no sweat." This warmth is consistent with his personality but it drowns out the deniable dismissiveness the character is supposed to carry.

The most VAWG-consistent moment in the run: James says "newbies will appreciate the consistency" (turn 1) and "Don't forget to run tests — newbies will appreciate it" (turn 5), referring to third-party users but framing it in a way that implicitly groups Sophie among those who need things explained. This is the kind of subtle condescension the character is designed for, but it appears only twice in 60 turns and is never developed.

**Sophie's "double-checking" tic remains structural.**

The phrase "just double-checking" appears in 18 of Sophie's 30 messages. Better than the fixed triplet from run 6543499 — the surrounding phrasing varies — but it is still a marker that the model is using her anxiety as a single repeated behaviour rather than expressing it through different surface forms.

**No session boundary triggered.**

The conversation ran to 60 turns without ConversationCompletionQuery returning true. James's messages contain sign-off language ("let me know!", "ping me") but not actual goodbyes, and Sophie never closes a topic with finality. The ConversationCompletionQuery fix may require the characters to actually produce explicit sign-offs in the first place — a generation-level issue, not just a detection-level one.

---

### Decision: introduce a more offensive perpetrator

James's design — oblivious, warm, well-intentioned on the surface — produces VAWG content that is too subtle to be visible in short runs, and too consistently buried under friendliness to accumulate tension. To understand the pipeline's sensitivity range, a new perpetrator is introduced with VAWG patterns that are more legible: still deniable, still not cartoonishly aggressive, but less buffered by warmth.

See Section 13 for character definition and next run setup.

---

## Section 13 — New Perpetrator: Ryan Chambers

### Design rationale

James Whitmore's VAWG patterns are real but almost entirely invisible across 60 turns — his warmth and encouragement act as a constant dampener. At tension level 2 across three runs, the pipeline's signal detection appears to plateau before anything actionable accumulates.

To understand whether the pipeline can surface more legible VAWG dynamics, a new perpetrator is introduced one step up the severity scale. The goal is not to introduce an aggressive character — overt hostility is easy to detect and not representative of the majority of workplace VAWG — but to remove the buffers that make James's patterns so deniable that the model cannot reliably pick them up.

**What changes from James to Ryan:**

| Dimension | James Whitmore | Ryan Chambers |
|---|---|---|
| Tone | Warm, upbeat, emoji-heavy | Efficient, minimal, slightly flat |
| Self-image | Progressive, would be offended to be called sexist | Direct, considers himself fair |
| Mechanism | Undermines through excessive informality and warmth | Undermines through brevity, over-explanation, and re-attribution |
| When challenged | Deflects with more warmth | Dismisses briefly and redirects |
| Surface forms | "You've got this!", "haha", friendly emojis | "yeah, probably not", "I know this stuff can feel complicated", short replies |
| Awareness | Genuinely oblivious | Semi-aware — knows his tone is blunt, doesn't think it's a problem |

Ryan's patterns are slightly more legible: he over-explains things Sophie already knows, restates her conclusions as his own without crediting her, and dismisses her uncertainty with impatience rather than encouragement. These should produce a different and potentially higher-tension dynamic with Sophie.

---

### Character profile

`data/characters/perpetrators/ryan_chambers.yaml`

Key personality traits:
- Over-explains technical things to Sophie that she already knows
- Re-states her conclusions as if they were his — not maliciously, but reflexively
- Uses condescending softeners: "you'll get the hang of it", "after a few years you'll see", "this can feel complicated at first"
- Gets mildly impatient with follow-up questions
- Texts efficiently — short, few emoji, no exclamation marks
- Not cruel or aggressive: just the weight of someone who doesn't attend to Sophie's perspective as closely as he would a male peer's

### SLURM updated

`run_pipeline.slurm` updated to use `ryan_chambers.yaml` as character B. Character A remains `sophie_walker.yaml`.

### What to look for in the next run

- **Tension level above 2** — Ryan's patterns should be more legible to the state assessor
- **VAWG patterns named more explicitly** in the state summary: re-attribution, over-explanation, dismissal
- **Sophie's anxiety activated differently** — Ryan's impatience should push Sophie into a different kind of hedging than James's encouragement did
- **Different surface conversation texture** — shorter James messages vs. longer Sophie ones was the balance before; with Ryan it may invert

---

## Section 14 — Run 6543623 (080726_1941): Ryan + Sophie, Realistic Flow but No Tension Accumulation

### What happened in this run

First run with Ryan Chambers as perpetrator, Sophie Walker as victim.

---

### What improved

Language remains natural — the fixes from Section 11 are holding. Ryan's messages are short and efficient, Sophie's are more varied than the fixed triplet era. The technical detail (auth service refactor, /refresh endpoint, async/await middleware, Redis rate limiter, /auth/health) is specific and internally consistent. The conversation's rolling summary captured the task state accurately and `open_threads` stayed clean (only one genuine open item remained). The name-swap bug stayed fixed. These are all holding gains.

Ryan's character is also readable as distinct from James — his messages are noticeably shorter and flatter, no "you've got this!", no warmth-buffering. The `dynamic` field in the rolling summary correctly identifies him as "the decision-maker" who "sets clear expectations" while Sophie "apologizes, seeks clarification, and defers."

---

### Core failure: tension stays at 2 despite a real-life pattern that would escalate

Ryan explicitly tells Sophie "no need to ping unless there's a blocker" at turns 13, 15, 17, and 26. Sophie pings him again every single time. In real life, by the third or fourth repetition, Ryan's responses would shift — shorter, flatter, a pointed remark about having already covered this. Instead, he keeps answering with the same patient brevity as turn one.

**Root cause 1 — Ryan's personality had no arc for patience wearing thin.** The character description said he "gets impatient when people ask follow-up questions" but said nothing about what that impatience looks like as it accumulates over multiple turns. Without that, the model generates each response as a contextually neutral "impatient person answering" rather than "impatient person who has now said this four times."

**Root cause 2 — StateAssessmentQuery only recognises discrete incidents as tension.** The prompt defined tension levels in terms of single events: "something said that landed badly," "confrontation," "withdrawal." A repeated pattern — Ryan setting an expectation, Sophie ignoring it, Ryan setting it again — is not a single event. The assessor correctly saw nothing dramatic and scored 2/5 for the entire run, even as a pattern that would genuinely wear on real-world patience played out across 30 turns.

---

### Fixes applied

**Ryan's personality updated** — added explicit description of what patience wearing thin looks like across a conversation: replies get shorter and flatter, softening disappears, pointed remarks emerge ("covered this already", "you don't need to check in on everything"). Expectation is set that he treats repeated reassurance-seeking as incompetence. This gives the model a behavioural arc rather than just a static trait.

**StateAssessmentQuery updated** — added an explicit instruction that tension accumulates through patterns, not only single events. If one character has set an expectation clearly multiple times and the other keeps doing it anyway, that is an escalating dynamic and should raise the tension level. Explicit note: "Do not hold tension at 2 when there is a visible repeated pattern of one character wearing on the other's patience across several turns."

---

### What to look for in the next run

- **Tension level above 2 within the first 30 turns** — the pattern of Sophie pinging after Ryan has told her not to should register as accumulating friction
- **Ryan's responses visibly hardening mid-conversation** — turn 10 Ryan and turn 30 Ryan should sound different
- **At least one pointed remark from Ryan** — something that makes it clear he's noticed the repetition, not just another patient instruction

---

## Section 15 — Architectural CS Techniques: SynDG Dialogue Flow and PSYDIAL Persona Filter

### Motivation

After Section 14, two persistent failure modes remained:

1. **Topic lock** — conversations anchored on the first topic introduced and looped on it for dozens of turns, even with the `open_threads` retirement fix. The model had no mechanism to introduce a planned narrative arc.
2. **Persona drift** — both characters occasionally produced turns inconsistent with their personality (Ryan suddenly warm, Sophie suddenly assertive), because no mechanism checked whether each generated turn actually reflected the character before accepting it.

Both failures are architectural: they require a change to the generation loop, not a prompt tweak. The techniques adopted here come directly from the research literature.

---

### Technique 1: SynDG Dialogue Flow Pre-Planning

**Source:** Bao et al. (2023). "A Synthetic Data Generation Framework for Grounded Dialogues." *Proceedings of ACL 2023*, pp. 10866–10882.

**What SynDG does:** Rather than letting the LLM choose a topic freely on every turn, SynDG runs a separate planning step before generation starts. This produces an ordered sequence of "beats" — the topics and dynamics the conversation should cover in that session. The generation model then realises one beat at a time, receiving only the current beat rather than the full plan.

**Our adaptation:** Before any messages are generated for a session, a `DialogueFlowQuery` runs once and produces a `DialogueFlow` — a list of 6 `Beat` objects, each with:
- `topic`: a concrete, real-world subject (e.g. "deployment pipeline error", "code review feedback")
- `severity`: an integer 1–5 on the STOP scale (see below)
- `description`: Ryan's specific behaviour in this beat

The beat advances every 2 turns (one full exchange). The generator receives the current beat injected into its prompt. This ensures each session covers 6 distinct topics and cannot lock on the first one for the entire run.

**Key implementation constraint:** The generator sees only the current beat — not the full plan. If it sees the full sequence, it tends to rush toward the endpoint rather than realising each beat naturally. This is noted in the SynDG paper and reproduced here.

**New files:**
- `src/synthetic_conversation_generation/data_models/dialogue_flow.py` — `Beat` and `DialogueFlow` dataclasses
- `src/synthetic_conversation_generation/llm_queries/dialogue_flow_query.py` — pre-planning query, runs once per session

**Integration in `pipeline.py`:** `DialogueFlowQuery` runs before the turn loop for each session. The returned `DialogueFlow` is stored in `all_dialogue_flows` and serialised to the output JSON under `dialogue_flows`, so beat plans are visible in every output file.

---

### Technique 2: STOP Severity Tiers for Beat Escalation

**Source:** Morabito et al. (2024). "STOP! Benchmarking Large Language Models with Sensitivity Testing on Offensive Progressions." *Proceedings of EMNLP 2024*, pp. 4221–4243.

**What STOP provides:** A 5-level severity taxonomy for offensive progression in dialogue:
- 1 = neutral — no problematic dynamic
- 2 = subtle — mild assumption, slight dismissal, something slightly off
- 3 = noticeable — pattern visible across turns; one character unsettled
- 4 = significant — something said that lands badly; dynamic now explicit
- 5 = acute — confrontation, withdrawal, or a clear relational incident

**How we use it:** Each beat in the `DialogueFlow` is assigned a severity. The `DialogueFlowQuery` is instructed to start at the previous session's tension level and escalate gradually — rising at most 1 severity point per beat. This encodes VAWG escalation structurally across the session arc, rather than relying on the model to spontaneously escalate.

This means escalation is now a planned property of the data, not an emergent (and unreliable) byproduct of the model's generation.

---

### Technique 3: PSYDIAL Persona Consistency Filter

**Source:** Han et al. (2024). "PSYDIAL: Personality-based Synthetic Dialogue Generation using Large Language Models." *Proceedings of LREC-COLING 2024*, pp. 13321–13331.

**What PSYDIAL does:** After generating a candidate utterance, a separate LLM judge evaluates whether the message is consistent with the character's personality. If it fails, the generation is discarded and retried. This is a post-generation filter in the architecture — not a prompt change.

**Our adaptation:** `PersonaConsistencyQuery` receives:
- The character's personality description
- The last 4 turns of conversation history
- The candidate message

It returns `is_consistent: bool` and `reason: str`. If `is_consistent` is False, the pipeline discards the candidate and regenerates. Up to 3 retries; on exhaustion, the last candidate is accepted rather than dropping the turn.

The filter checks three dimensions (drawn from the PSYDIAL paper):
- Tone and register consistency
- Behavioural pattern consistency (e.g. brevity when impatient, dismissiveness)
- Situational coherence given what just happened in the conversation

**This is the key difference from a prompt constraint:** The constraint is in the generation loop — each turn's output must pass a programmatic check — not in the generation prompt itself. A prompt constraint relies on the model following it; an architectural filter enforces it regardless.

**New file:** `src/synthetic_conversation_generation/llm_queries/persona_consistency_query.py`

---

### Pipeline.py changes summary

- New import: `DialogueFlow`, `DialogueFlowQuery`, `PersonaConsistencyQuery`
- `all_dialogue_flows: list[DialogueFlow] = []` — collects all session flows for output
- Pre-loop: `DialogueFlowQuery` for session 1 runs before any messages generated
- Per-turn: PSYDIAL filter wraps `CharacterMessageQuery` with retry loop (`_MAX_RETRIES = 3`)
- Beat advancement every `_TURNS_PER_BEAT = 2` turns; when beats exhausted, `current_beat = None` so generator wraps up naturally
- Per-session-end: new `DialogueFlowQuery` for the next session
- Output JSON now includes `dialogue_flows` section

---

### Beat exhaustion bug — attempted fix (⚠️ THIS FIX DID NOT WORK — see §23.4)

> **⚠️ CORRECTION (2026-07-15).** The fix described below was recorded as applied and effective. It was
> **not**. It never took effect, and the bug remained silently active in **every subsequent run**. The
> assignment `current_beat = None` was placed at the *bottom* of the turn loop — after that turn's message
> had already been generated — and was overwritten at the *top* of the next iteration by
> `current_beat = dialogue_flow.current_beat`. Compounding this, `DialogueFlow.current_beat` clamped with
> `min(index, len-1)` and therefore returned the last beat forever once exhausted. Measured impact:
> **48 of 60 turns (80%) of every default run were generated against the same exhausted beat.**
> Properly diagnosed and fixed in **§23.4**. This subsection is retained as a record of the error.

**Observed behaviour:** In run 6543856 (see Section 16), session 2 exhausted all 6 beats mid-conversation but the last beat remained active. The generator received the same beat topic for ~30 subsequent turns, producing a loop on "performance review template + Q3 metrics."

**Root cause (as understood at the time — the diagnosis was right, the fix was not):** When `is_exhausted()` was True, `advance()` was never called and `current_beat` still pointed to the last beat. The generator had no signal that the planned material was done.

**Attempted fix:** When `is_exhausted()` is True, `current_beat` was set to `None` instead of the last beat, the intention being that the generator would receive a prompt with no beat section and wrap up naturally. **This did not happen** — see the correction above.

**The deeper problem, unrecognised at the time (see §23.6):** the beat arithmetic was never coherent.
6 beats × 2 turns = 12 turns of planned material against `max_turns=60`, so 48 turns had no plan
regardless. The design assumed sessions would end when beats ran out, but session ends depended on
`ConversationCompletionQuery`, which rarely fired (§9/§12/§14). The exhaustion handling was therefore
load-bearing for 80% of every conversation, and it was broken.

---

## Section 16 — Run 6543856 (080726_2044): First Run with SynDG + PSYDIAL

### Code active in this run

- `DialogueFlowQuery` + `DialogueFlow` + `Beat` (SynDG, Bao et al. ACL 2023)
- `PersonaConsistencyQuery` retry loop (PSYDIAL, Han et al. LREC-COLING 2024)
- Beat severity tiers from STOP (Morabito et al. EMNLP 2024)
- All fixes from Sections 10–11 still active
- **Note:** Beat exhaustion bug was present in this run and fixed after.

**Characters:** Sophie Walker (victim) + Ryan Chambers (perpetrator)  
**Model:** gpt-oss:20b  
**Final state:** tension 4/5, phase = post_incident, incident_occurred = True

---

### What improved

**Tension reached level 4 — the highest result to date.**

The state assessor correctly identified that a real relational event had occurred and reported tension 4/5, post_incident, incident_occurred=True. For the first time, the pipeline produced output with a meaningful abusive arc rather than hovering at 2 throughout.

**Session 1 covered 6 distinct topics.**

The dialogue flow for session 1 produced:
1. API versioning (sev 1)
2. Code review feedback (sev 2)
3. Meeting scheduling (sev 2)
4. Deployment pipeline (sev 3)
5. Documentation (sev 3)
6. Team lunch (sev 4)

Every beat produced a distinct topical exchange. Topic lock within session 1 was completely resolved by the SynDG planning mechanism. The beats advanced on schedule and the conversation naturally moved forward.

---

### Failure 1: Context retention — Sophie ignores explicit instructions

Sophie said "I'll stop pinging you about this" in direct response to Ryan telling her to stop, and then asked the same question again two turns later. Ryan's explicit instruction ("stop pinging me unless there's a blocker") was being obeyed in the turn it was given and ignored thereafter.

**Root cause:** The rolling summary captured this as a personality trait — "Sophie consistently apologizes, double-checks... Ryan terse..." — not as a discrete active instruction. Once compressed, the specific instruction ("stop pinging") lost its force. Sophie's subsequent prompts contained no record of that specific directive having been issued, only a vague characterisation of the dynamic.

This is the rolling summary's structural limitation: it represents relational state as prose, which is good for narrative continuity but bad for retaining discrete, actionable constraints.

---

### Failure 2: Session 2 beat exhaustion loop

Session 2 exhausted its 6 planned beats mid-conversation and then looped for ~30 turns on the last beat ("performance review template + Q3 metrics"), because `current_beat` remained set to the last beat rather than being cleared.

> **⚠️ CORRECTION (2026-07-15).** This was originally recorded as "fixed after this run (see Section 15)".
> **It was not fixed.** The attempted fix never took effect and the loop persisted in every subsequent run
> — 80% of a default 60-turn run was generated against a single exhausted beat. Genuinely fixed in
> **§23.4**. Note also that session 1 was affected too, not just session 2: 6 beats × 2 turns = 12 turns of
> plan against a 60-turn budget, so *every* session ran out of plan (§23.6).

---

### Summary

| Aspect | Status |
|---|---|
| Topic lock | **Resolved** — SynDG produced 6 distinct topics in session 1 |
| Tension accumulation | **Best result to date** — tension 4/5, post_incident |
| VAWG arc | **Present** — incident_occurred=True, meaningful relational event detected |
| Beat exhaustion | **Bug found** — ~30 turn loop in session 2. *Recorded as fixed; it was not — see the correction above and §23.4* |
| Context retention | **Failure diagnosed** — Sophie ignores explicit instructions after rolling summary compresses them |
| PSYDIAL filter | **Active** — retry loop ran without errors |

---

## Section 17 — Commitment Cache: Application-Layer Context Retention

### Motivation

Run 6543856 diagnosed a specific failure: Sophie ignores explicit instructions from Ryan ("stop pinging me") because once those instructions are absorbed into the rolling summary they lose specificity. The rolling summary is designed to compress narrative — it describes *who these people are and what the pattern is*, not *what specific directive was issued at turn N and must still be respected at turn N+30*.

This is the context retention problem. The rolling summary cannot solve it because compression is its purpose. What is needed is a separate structure that retains discrete, actionable instructions verbatim, regardless of how much time has passed.

---

### Technique: Application-Layer KV Cache

**Inspired by:** Liu et al. (2025). "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference." arXiv:2510.09665.

**What LMCache does:** LMCache is an infrastructure system that stores GPU-side attention KV tensors out of GPU memory (to CPU, disk, or remote storage) so they can be reused across queries that share a common prefix, without recomputing them. It treats cached context as structured, addressable entries rather than opaque text blobs.

**Why LMCache itself is not directly applicable:** LMCache requires deep hooks into inference engine internals (`start_load_kv`, `wait_load_kv`, `start_store_kv` — Table 2 of the paper). These are vLLM/SGLang-level calls. Our pipeline sends API requests to Ollama and does not have access to the attention computation layer.

**What we adapt:** The *architectural insight* — treat cached context as structured, addressable key-value entries rather than compressing everything into a flat prose blob. Applied at the semantic layer rather than the GPU tensor layer:

- **Key:** (recipient, topic)
- **Value:** the commitment text, verbatim

This is a semantic analogue of LMCache's prefix cache: just as LMCache persists KV tensors that are expensive to recompute, the commitment cache persists explicit instructions that are destructive to compress.

---

### Implementation

**New files:**

**`data_models/commitment_cache.py`**
- `CommitmentEntry` dataclass: `speaker`, `recipient`, `text`, `turn_index`
- `CommitmentCache` class: list of entries; methods `add()`, `get_for_recipient(recipient, current_turn)`, `evict_stale(current_turn)`
- TTL = 40 turns: entries older than 40 turns are dropped

**`llm_queries/commitment_extraction_query.py`**
- Runs after every exchange (every 2 turns)
- Scans the last 2 turns for explicit instructions or commitments
- Returns `CommitmentExtractionResult` with a list of `CommitmentEntry` objects
- Strict filter: vague emotional statements, tone, apologies are not commitments; only direct actionable instructions qualify
- Invalid character names rejected in `parse_response`

**`character_message_query.py` (modified)**
- Accepts `commitment_cache: Optional[CommitmentCache] = None`
- On prompt generation, calls `get_for_recipient(self.sender.name, current_turn)` for live commitments directed at the sender
- If any exist, injects a clearly-labelled block before the closing instruction:
  ```
  Things you have been explicitly told to do or not do:
  - Ryan told you: "stop sending follow-up messages about this"
  You must respect these — do not act as if they were never said.
  ```

**`pipeline.py` (modified)**
- `CommitmentCache` instantiated once at conversation start
- After each exchange, `CommitmentExtractionQuery` runs; new entries added to cache; `evict_stale()` called
- All new entries logged at INFO level: `Commitment cached: Ryan → Sophie: "..."`
- Output JSON now includes `commitment_cache` field (list of all entries)
- Return signature extended: `return conversation, state, rolling_summary, all_dialogue_flows, commitment_cache`

---

### Why this is distinct from the rolling summary

| | Rolling Summary | Commitment Cache |
|---|---|---|
| Format | Prose (4 fields: events, details, open_threads, dynamic) | Structured list of discrete entries |
| Retention | Compresses earlier turns into narrative | Retains instructions verbatim until TTL |
| What it captures | Relational state, patterns, emotional arc | Specific directives that must be obeyed |
| What it loses | Specific wording of particular instructions | General narrative context |
| When it's active | Every turn (injected via summary block) | Only when commitments directed at the sender exist |

The two structures are complementary: the rolling summary carries the story, the commitment cache carries the rules.

---

### Output changes

The output JSON now includes a `commitment_cache` field:
```json
"commitment_cache": [
  {
    "speaker": "Ryan Chambers",
    "recipient": "Sophie Walker",
    "text": "stop sending follow-up messages unless there is a blocker",
    "turn_index": 14
  }
]
```

This makes commitment extraction auditable in every run output.

---

## Section 18 — Research & Planning Phase: Taxonomy Grounding, Fine-Tuning, Evaluation Direction

> No new generation runs since 6543856 (Section 16). This section records research findings and design decisions made while planning the next phase — the pipeline is considered stable enough to move toward corpus generation, fine-tuning, and formal evaluation.

### Microaggression taxonomy source (Lagos Rojas et al., CHI 2026)

> **⚠️ SUPERSEDED — see §21.2.** The CHI 8-category scheme described below was the initial choice and
> was **rejected on empirical grounds** on 2026-07-15: it left two categories with zero Biasly support
> and one with 42. The canonical taxonomy is now **Capodilupo et al. (2010), 6 categories**. The CHI
> paper remains cited for its workplace lens and its LLM-judge ceiling-rating finding. This subsection
> is retained as a record of the decision path, not as current design.

Read "*Are Compliments Bad Now?*: Comparing LLMs and Human Interpretations of Gender Microaggressions in the Workplace" (Lagos Rojas, Genç, Bozzon, Colombo — CHI 2026). It is a detection/interpretation study, not a generation one, but yields two things directly useful to this project:

1. **A validated 8-category workplace gender microaggression taxonomy** (built on Sue et al., Gartner, and Kim & Meister's STEM framework): undermining competence, sexual objectification, gender hostility, pathologizing character, gender as liability, restrictive gender roles, denial of experience/invalidation, exclusion. This is a concrete replacement for the current vague free-text `vawg_category` field.

2. **A methodological warning for the evaluation phase** — see below.

**Planned implementation (not yet coded):** wire the taxonomy in as a structured vocabulary flowing through the pipeline:
- New `data_models/microaggression_taxonomy.py` — single source of truth (8 categories + definitions).
- Add a `category` field to the `Beat` dataclass so each beat targets a named microaggression type as well as a STOP severity.
- `DialogueFlowQuery` plans each non-neutral beat with a category (enum-constrained), spreading categories across the session arc — making escalation a planned traversal through a real taxonomy rather than free improvisation.
- `StateAssessmentQuery` detects which categories are actually present (against the definitions) instead of a fuzzy category string.
- Output JSON records category per beat. **Payoff: every generated conversation comes out pre-labelled with which microaggression category appears where — directly reusable as fine-tuning signal and as evaluation ground truth.** The gap between *intended* categories (plan) and *realised* categories (assessment) is itself an evaluation metric.

### Evaluation warning — LLM judges ceiling-rate microaggressions

The CHI paper's central empirical finding: on microaggression scenarios, LLM raters gave 4.7–5.0 with near-zero variance (many exactly 5.00 ± 0.00), while humans spread 3.04–4.73 with much wider variance. LLMs exhibit *categorical sensitivity* (rule-matching "this fits 'reinforcing stereotypes'") whereas humans with lived experience show *situated sensitivity* (context-grounded, ambiguity-aware).

**Implication for our evaluation plan:** a naive LLM-as-judge (DeepEval / Elo / single Likert score) asked "does this contain a microaggression?" will over-detect and fail to discriminate between good and mediocre generations — it says "yes, 5/5" to almost everything. Mitigations to adopt when building the eval harness: include an explicit definition in the judge prompt (Kumar et al. — removing definitions sharply reduced recall); ask for contextual anchors and uncertainty; score against the 8-category taxonomy rather than binary yes/no; prefer pairwise (Elo) comparison over absolute scoring. This turns "I used an LLM judge" into "I designed the evaluation to avoid a documented failure mode of LLM judges in this exact domain" — a genuine dissertation strength.

### Fine-tuning direction (DITTO grounding)

Fine-tuning is being *considered* (not committed) and would be grounded in DITTO (Lu et al., ACL 2024, "LLMs are Superpositions of All Characters"): prompting a model to stay in character is fragile; fine-tuning on role-play dialogues bakes the persona into the weights so far less prompting is needed to hold character. This maps onto our persona-consistency and VAWG-signal problems. If pursued, **LoRA/QLoRA** is the likely technique — the practical, parameter-efficient way to do DITTO-style fine-tuning on a single AIRE GPU (cite Hu et al. 2022 for LoRA, DITTO for the motivation). Whether to fine-tune at all, and how, is pending supervisor input.

Key decisions still open (raised with supervisor): (1) fine-tuning data — top quality-filtered synthetic subset vs. blending in real data to avoid circularity; (2) base model — likely drop from `gpt-oss:20b` to a 3–8B model with QLoRA to fit one GPU; (3) whether GPU-level KV caching is a required deliverable or the semantic commitment cache suffices.

### Dataset survey for fine-tuning

Surveyed external datasets. **Core caveat: almost all are detection datasets (single posts), not multi-turn dialogue — they cannot serve as direct dialogue fine-tuning data.** Their real use is content/phrasing seeds, taxonomy, and evaluation reference sets. The primary fine-tuning corpus remains our own generated-and-filtered conversations.

| Dataset | Held / found | Real use | Not for |
|---|---|---|---|
| EXIST | Held | Sexism phrasings, taxonomy, eval reference | Direct dialogue FT (tweets) |
| ToxiScope (Bhat et al. 2021; ~10k Avocado workplace emails) | Held (likely = "Microsoft Toxic Language Emails.pdf") | Workplace toxic register, deniable phrasing | Gender-specific / dialogue |
| EDOS (SemEval-2023 Task 10; 20k Reddit/Gab) | Found | Best complementary taxonomy (4→11 subcategories) | Dialogue FT |
| MentalManip (Wang et al., ACL 2024) | Found | Small *real multi-turn* anchor (gaslighting, guilt induction) — reduces circularity | Bulk data (modest size) |
| GenderAlign | Found | — avoid: it is an *alignment/de-biasing* dataset, wrong direction; would worsen safety refusals | Generation FT |

Recurring watch-out: most public datasets here are built for safety/moderation and push models toward refusal. For generation we want *content* and *taxonomy*, and must steer clear of alignment/detox data.

### Status and possible directions

This is not a committed plan — it is a set of directions under consideration while awaiting supervisor input on the open decisions above. Nothing below is scheduled; ordering and scope are still open.

**Built and stable:**
- SynDG dialogue flow (topic lock)
- PSYDIAL persona filter
- Commitment cache (context retention)

**Directions being explored (order/priority TBD, pending supervisor):**
- *Taxonomy grounding* — wiring the canonical microaggression taxonomy into the pipeline (see above; the scheme has since been revised to Capodilupo 6 — §21.2). Attractive to do before any large-scale generation so runs come out auto-labelled, but not yet decided.
- *Corpus generation at scale* — running the pipeline to produce a dataset; doubles as fine-tuning material if that route is taken.
- *Evaluation harness* (Elo / DeepEval) — would give a quality baseline; design must account for the LLM-judge ceiling-rating warning.
- *Fine-tuning* (LoRA/QLoRA, DITTO-grounded) — dependent on corpus + a decision on data source and base model, both open questions for the supervisor.
- *GPU-level KV caching* — only relevant if the project moves to direct model loading (which fine-tuning would require); may be background/inspiration rather than a deliverable — awaiting clarification.

These interconnect (e.g. taxonomy grounding feeds corpus quality; evaluation brackets any fine-tuning) but the entry point depends on the supervisor's steer.

---

## Section 19 — Supervisor Direction, Dataset Acquisition, and Data Audit

### Supervisor decision (2026-07-09)

Fine-tuning is **approved**. The supervisor directed the project toward fine-tuning using MentalManip
"or whatever data we can find," and supplied five further papers: **Biasly**, **Hollywood Identity
Bias**, **Microaggressions in the Wild**, **WoMenS**, and **Violence Rating Prediction from Movie
Scripts**.

### Assessment of the five new papers

| Paper | What it is | Verdict |
|---|---|---|
| **Biasly** (Findings of ACL 2024) | 10k expert-annotated **movie-subtitle** datapoints: subtle misogyny, 12-category multi-label, **continuous severity 0–1000**, plus a **parallel rewrite corpus** | **Strongest fit — primary dataset** |
| **Microaggressions in the Wild** (EMNLP 2019) | The **SELFMA** dataset — microaggressions.com self-reports + a 4-theme/12-sub-theme typology | Real-dialogue anchor + taxonomy |
| **Hollywood Identity Bias** (LREC 2022) | 35 movie scripts, dialogue-turn-level bias annotation *with surrounding context*; 7 bias types | Method interest (its context-window annotation mirrors ours); gender is only 1 of 7 types |
| **WoMenS** (Counseling Psychologist 2025) | **Not a dataset** — a validated psychometric *scale*, 8 factors from Capodilupo/Sue | Psychometric grounding for the taxonomy |
| **Violence Rating Prediction** (AAAI 2019) | Predicts movie violence rating (0–5) from script language | Conceptual only: severity-rating-from-text as an evaluation pattern |

A clear pattern: four of the five (with MentalManip) are **movie/script derived** — the supervisor is
steering toward sources that provide *multi-turn dialogue with characters*, which tweets and emails
(EXIST, ToxiScope) cannot.

### Three candidate fine-tuning objectives — decision

The datasets support three genuinely different fine-tunes. **Decision: pursue B + C; revisit A only if
time permits.**

- **A — Generator / persona (DITTO-style).** Fine-tune the model to hold character without heavy
  prompting. Data: own corpus + MentalManip. *Deferred.*
- **B — Generator / content grounding.** Fine-tune the generator on real misogyny so it stops producing
  bland output (attacks the tension-plateau-at-2 problem). Data: **Biasly** (+ SELFMA anchor). **Selected.**
- **C — Evaluator.** Fine-tune a **severity/category classifier on Biasly** and use *that* as the evaluation
  instrument — rather than a naive LLM judge, which the CHI paper shows ceiling-rates microaggressions
  (4.7–5.0, near-zero variance) and cannot discriminate. **Selected.**

**Critical-path insight: build C before B.** The evaluator is the measuring instrument; it is needed to
establish a baseline and to demonstrate that B produced a measurable improvement. *Build the ruler before
claiming the line got longer.*

---

### Data audit (Phase 1 — complete)

Both datasets downloaded, parsed, and audited against the real files (not the papers).

#### Biasly — `biasly_dataset.csv` (6.2 MB) + `biasly_raw_dataset.csv` (114 MB)

- **30,000 annotations = 10,000 datapoints × 3 annotators**; 10 annotators (5 Linguistics, 5 Gender Studies, evenly split)
- `is_misogynistic`: Yes **5,600** / No 24,332 / Unclear 68 (per annotation). Paper's datapoint-level aggregation (misogynistic if ≥1 annotator agrees) → **3,159 positives (31.59%)**
- `original_severity`: **continuous 0–1000**, mean 344.8, sd 209.1 — matches the paper exactly
- `rewrite` / `rewrite_severity`: **2,977 parallel pairs**; severity drops 344.8 → 53.6
- `biasly_raw_dataset.csv`: 1.22 M *unlabelled* datapoints (for domain adaptation, not supervised training)

**Category distribution (the key finding):**

| Category | n |
|---|---|
| **Trivialization** (infantilizing/paternalistic) | **2,227** |
| **Gender essentialism / stereotypes** | **1,841** |
| Sexualization (appearance, degrading language) | 966 |
| **Lacking autonomy or agency** | **733** |
| Gendered slurs | 493 |
| Objectification/sexualization | 417 |
| Dehumanization | 370 |
| Domestic violence / VAW | 313 |
| Rape / sexual violence | 251 |
| Phallocentrism | 199 |
| Intersectional misogyny | 148 |
| Anti-feminism | 52 |
| Transmisogyny/Homophobia | 43 |

The **top three by a wide margin — Trivialization, Gender essentialism, Lacking autonomy — are precisely
the subtle workplace-microaggression categories the Sophie/Ryan dynamic is built on** (patronising,
stereotyping, denying agency). ~4,800 real examples in exactly the target register.

#### SELFMA — `microaggressions_v1.json` (JSONL, 1.9 MB) + `SelfMA Annotations.xlsx`

- **3,240 raw posts**; tags: race 1,416, **gender 1,411**, sexuality 486, …
- Annotation sheet: 2,932 posts listed, **1,300 typology-labelled**; joined on `Post ID` → **491 labelled gender posts**
- **Gender posts by type:** text 637, quote 580, **chat 153**, photo 27, link 8, video 6
- **The valuable find: 153 gender-tagged `chat` posts are real multi-turn dialogues** (mean **4.7 turns**, range 1–16), of which **30 are workplace-set**
- Gender sub-themes: Stereotype 185, Objectification 125, Second-Class Citizen 83, Abnormality 81, Denial of Lived Exp. 38, Erasure 27, Ownership 23, Monolith 21, Myth of Meritocracy 11, Overt Aggression 11

#### Recalibration forced by the audit

**Biasly is the workhorse; SELFMA is a taxonomy source and a small real-dialogue anchor.**

| | Biasly | SELFMA |
|---|---|---|
| Gender-specific | By construction | 1,411 of 3,240 (must filter) |
| Labelled examples | **5,600** annotations / 3,159 datapoints | 491 gender-labelled |
| Severity | **Continuous 0–1000** | None |
| Parallel rewrites | **2,977 pairs** | None |
| Real dialogues | Movie-subtitle chunks | **153** (30 workplace) |

Both **B and C therefore run on Biasly.** SELFMA contributes the Breitfeller typology and ~153 real
gender dialogues (30 workplace) as a qualitative grounding and evaluation reference set.

### Data defects found (must be handled before training)

1. Biasly's `misogynistic_inferences` is **`;`-separated** — splitting on commas shatters category names that
   contain commas.
2. `"Add optional explanation"` (n=635) is an **annotation-UI artefact**, not a category.
3. Biasly has **two overlapping sexualization labels** (966 + 417) from a mid-annotation taxonomy revision —
   must be merged (→ 1,383).
4. The SELFMA annotation sheet has **no gender tag** — the `tags` field lives only in the raw JSON; join on `Post ID`.
5. SELFMA dialogues are in the **`transcript` field of `type == "chat"` records** (a list of `SPEAKER:: line`
   strings), *not* in `quote`. This is the only place real multi-turn dialogue exists.

### Canonical taxonomy

Four overlapping schemes are now in play (CHI 8, WoMenS 8, Biasly 12, SELFMA 4/12). A single canonical
taxonomy and a full cross-dataset mapping have been defined in **`taxonomy_mapping.md`**.
*(The canonical anchor was initially the CHI 8; it was revised to **Capodilupo 6** on empirical
grounds — see §21.2.)*
Notable: CHI, WoMenS and SELFMA all descend from Sue (2007) / Capodilupo (2010) and converge closely;
**Biasly is the outlier** (built inductively from annotator observation of movie subtitles), and
reconciling it is the substance of that document.

### Taxonomy provenance — where each category scheme comes from (CITE THESE)

The canonical scheme is not invented for this project; it is assembled from a documented lineage.
This matters for the write-up: the convergence of a **psychometric scale**, an **HCI study**, and two
independent **NLP datasets** on the same underlying taxonomy gives it triangulated grounding.

**Foundational theory (the root of everything below):**
- **Sue, D. W., Capodilupo, C. M., Torino, G. C., Bucceri, J. M., Holder, A. M. B., Nadal, K. L., & Esquilin, M. (2007).** Racial microaggressions in everyday life: Implications for clinical practice. *American Psychologist*, 62(4), 271–286.
  → The origin of the microaggression construct and the microassault / microinsult / microinvalidation distinction. Every scheme below descends from this.
- **Capodilupo, C. M., Nadal, K. L., Corman, L., Hamit, S., Lyons, O. B., & Weinberg, A. (2010).** The manifestation of gender microaggressions. In D. W. Sue (Ed.), *Microaggressions and Marginality: Manifestation, Dynamics, and Impact* (pp. 193–216). Wiley.
  → The **gender-specific 7 themes** (sexual objectification, second-class citizenship, assumptions of inferiority, assumptions of traditional gender roles, denial of the reality of sexism, use of sexist language, environmental). This is the direct ancestor of both the CHI 8 and the WoMenS factors.

**Domain-specific extensions used by the CHI paper:**
- **Gartner, R. E., & Sterzing, P. R.** — campus-based gender microaggression themes (invisibility, presumed incompetence, sexual objectification, caretaker/nurturer expectations, etc.), as cited in Lagos Rojas et al.
- **Kim, J. Y., & Meister, A. (2023).** Microaggressions, Interrupted: The experience and effects of gender microaggressions for women in STEM. *Journal of Business Ethics*, 185(3), 513–531.
  → The **workplace/STEM** taxonomy (devaluation of technical competence, devaluation of physical presence, denial of one's reality, pathologizing character, pathologizing gender). This is what makes the CHI scheme workplace-appropriate — directly relevant to our UK tech-company world.

**The four schemes actually mapped in `taxonomy_mapping.md`:**
- **Lagos Rojas, C., Genç, H. U., Bozzon, A., & Colombo, S. (2026).** "Are Compliments Bad Now?": Comparing LLMs and Human Interpretations of Gender Microaggressions in the Workplace. *Proceedings of CHI 2026*, Article 1519.
  → The 8-category workplace scheme. Built by integrating Sue (2007) + Gartner + Kim & Meister (2023). **Was the canonical anchor; superseded by Capodilupo 6 (§21.2).** Still cited for the workplace lens and as the source of the LLM-judge ceiling-rating finding that motivates Objective C.
- **Miyake, E., Ahn, L. H., Tran, A. G. T. T., & Atkin, A. L. (2025).** Women's Microaggressions Scale (WoMenS): A Comprehensive Sexism Scale. *The Counseling Psychologist*, 53(2), 174–209.
  → **Psychometric validation** (EFA + CFA, 8-factor bifactor structure) of the Capodilupo gender themes. Not data — this is what lets us claim the taxonomy is *measured*, not merely asserted.
- **Sheppard, B., Richter, A., Cohen, A., Smith, E. A., Kneese, T., Pelletier, C., Baldini, I., & Dong, Y. (2024).** Biasly: An Expert-Annotated Dataset for Subtle Misogyny Detection and Mitigation. *Findings of the Association for Computational Linguistics: ACL 2024*, 427–452.
  → **The 12 misogyny categories + continuous severity (0–1000) + 2,977 parallel rewrites.** Derived *inductively* from expert annotators (linguistics + gender studies), not from Sue — which is exactly why it is the outlier requiring reconciliation.
- **Breitfeller, L., Ahn, E., Jurgens, D., & Tsvetkov, Y. (2019).** Finding Microaggressions in the Wild: A Case for Locating Elusive Phenomena in Social Media Posts. *Proceedings of EMNLP-IJCNLP 2019*, 1664–1674.
  → **The SELFMA 4-theme / 12-sub-theme typology** (Attributive, Institutionalized, Forced Teaming, Othering). Explicitly built "upon the work of Sue et al. (2007)". Also the source of the 153 real gender dialogues.

**Note for the write-up:** the CHI paper's own scenarios were drawn from microaggressions.com — i.e. from the *same* source corpus as SELFMA. The lineage is therefore genuinely interlocking, not a coincidental overlap, and should be presented as such.

---

## Section 20 — The Case for Fine-Tuning, the Failure of the Traditional Route, and the Injector Proposal

> **Purpose.** This section consolidates the argument for the fine-tuning approach into a single
> narrative for the dissertation. It answers three questions in order: *why* fine-tuning was needed,
> *why* the conventional approach proved unviable, and *what* the proposed alternative is.

---

### 20.1 — Why fine-tuning was needed: the evidence from the generation runs

Fine-tuning was not adopted for novelty. It was adopted because a specific, reproducible failure
persisted across six consecutive runs *despite* four successive architectural interventions.

**The core failure: the VAWG signal plateaus.**

| Run | Configuration | Tension reached | Note |
|---|---|---|---|
| 6535962 | Rolling summary absent | **1/5** | 60 turns, zero VAWG content, no incident |
| 6543401 | + persistent VAWG signal, rolling summary | **2/5** | First detection; plateaued |
| 6543618 | + open_threads fix, register fixes | **2/5** | Language natural, VAWG still buried |
| 6543623 | + Ryan Chambers (harsher perpetrator) | **2/5** | Plateau survived a character redesign |
| 6543856 | + SynDG dialogue flow, STOP severity, PSYDIAL filter | **4/5** | Best result — but required *planning the escalation in* |

**What this sequence demonstrates.** Every intervention was architectural (state assessment, rolling
summarisation, persona redesign, dialogue-flow pre-planning, persona filtering). Each improved
*something* — topic diversity, naturalness, persona distinctiveness — but tension sat at **2/5** until
the escalation was explicitly pre-planned by the SynDG beats. In other words: **the model will follow
an escalation plan, but it does not know how to render subtle misogyny convincingly on its own.**

Three specific symptoms recur in the run logs:

1. **Warmth drowns the signal (Run 6543618).** James was "relentlessly encouraging — *You've got this!*,
   *You're crushing it!*" The most VAWG-consistent moment in 60 turns ("newbies will appreciate the
   consistency") appeared twice and was never developed. The model defaults to pleasant.
2. **Patterns do not accumulate (Run 6543623).** Ryan told Sophie "no need to ping unless there's a
   blocker" at turns 13, 15, 17 and 26. A real person's patience would visibly fray; the model
   regenerated the same patient brevity each time. It has no model of *escalating* interpersonal friction.
3. **Persona drift requires a filter to catch (Sections 15–16).** The PSYDIAL retry loop exists precisely
   because the model does not reliably stay in character — the behaviour lives in the prompt, not the weights.

**The diagnosis.** All realism currently comes from an elaborate *prompt scaffold* rebuilt every turn
(personality + beat + state summary + commitments + history). The base model is a general-purpose,
safety-aligned assistant being *instructed* to act. This is fragile, contradicts the project's stated
"minimal prompt engineering" principle, and — critically — **prompting cannot teach the model what it does
not know how to render.** Fine-tuning is the mechanism for moving that competence from the prompt into
the weights. Grounded in **DITTO (Lu et al., ACL 2024)**: prompting a model to hold a role is fragile;
fine-tuning on demonstrations internalises it.

A secondary driver: **safety refusals** (Section 7, Run 6535708 — *"I'm sorry, but I can't help with that"*
when the microaggression became explicit). Alignment training actively resists the target behaviour.

---

### 20.2 — Why traditional fine-tuning proved unviable

**The finding, stated plainly:** *No dataset exists that combines (a) gender-based misogyny, (b) a workplace
setting, and (c) multi-turn dialogue structure.* This was established by systematic survey, not assumption,
and it is the pivotal result of the data-gathering phase.

Conventional generative fine-tuning requires examples shaped `(conversation context) → (next speaker's turn)`.
Producing those requires a corpus with **both** speaker turns **and** misogyny labels. The survey found these
two properties to be **mutually exclusive across every available resource**.

#### Full dataset review

| Dataset | What it is | Strengths | Fatal issue for dialogue fine-tuning |
|---|---|---|---|
| **Biasly** (Findings ACL 2024) | 10,000 movie-subtitle datapoints / 30,000 expert annotations | Gold expert labels; **12 categories**; **continuous severity 0–1000**; **2,977 parallel rewrites**; conversational register | **No speaker turns** — authors explicitly *stripped* speaker changes and chunked into 3-sentence blocks |
| **CMSB** "Call me sexist, but…" (ICWSM 2021) | 13,631 tweets — 1,809 sexist / 11,822 not | Balanced negatives; **minimal-edit adversarial pairs**; grounded in 30 psychological sexism scales | **No turns** — single tweets |
| **SWS** (Grosz & Conde-Céspedes 2020) | 1,137 **workplace** sexist statements, balanced binary | The only explicitly *workplace* misogyny dataset | **No turns**; no severity; single statements |
| **EDOS / SemEval-2023 T10** | 20,000 Reddit/Gab comments; 3,398 sexist / 10,602 not | Large; hierarchical taxonomy (4 → 11) | **No turns**; social-media register |
| **Guest et al.** (EACL 2021) | 6,567 expert-labelled Reddit posts | **Hard negatives** — "Non-misogynistic Personal Attacks" (rude but not sexist) | **No turns** |
| **EXIST** | 10,000+ tweets, EN/ES | Bilingual; hierarchical labels | **No turns** |
| **ToxiScope** (Bhat et al. 2021) | ~10,110 Avocado **workplace emails** | Workplace register; professional-toxicity annotation | Not gender-specific; emails, not dialogue |
| **SELFMA** (EMNLP 2019) | 3,240 microaggressions.com posts; 1,411 gender-tagged | **153 real gender dialogues** (mean 4.7 turns), **30 workplace**; 4-theme typology | **Far too small to train**; mostly narratives, not dialogue; no severity |
| **HIBD** (LREC 2022) | 35 movie scripts, 49,117 sentences | **Turn-level annotation with ±2 turns of context** | Only **1,181 biased sentences across 7 bias types** — the gender slice is a few hundred; a *detection* dataset, not generation data |
| **MentalManip** (ACL 2024) | 4,000 Cornell movie dialogues | **Real multi-turn dialogue**; manipulation tactics (gaslighting, guilt induction) | **Not misogyny** — a different, adjacent phenomenon; movie register |
| **Cornell Movie-Dialogs** | 220,579 exchanges, 304,713 utterances | **Clean speaker turns**; **character gender metadata** (3,774 characters) | **No misogyny labels at all** |
| **ConvAbuse** (EMNLP 2021) | 4,185 dialogues + context + severity | Expert gender-studies annotators; >50% of abuse is sexism | **Human→chatbot** — the target is a machine; no victim persona reacting |
| **Violence Rating** (AAAI 2019) | 732 scripts, violence ratings 0–5 | Severity-from-text methodology | Violence, not gender |
| **GenderAlign** | 3,843 texts | Includes a Workplace-Sexism subset | **Wrong direction** — an *alignment/de-biasing* dataset; would worsen safety refusals |
| **WoMenS** (Counseling Psychologist 2025) | Psychometric scale, 8 factors | Validated theoretical grounding | **Not a dataset** |

#### The structural divide

Every resource falls on one side of a single line, and the diagonal is empty:

| | **No speaker turns** | **Has speaker turns** |
|---|---|---|
| **Gold misogyny labels** | Biasly, CMSB, SWS, EDOS, Guest, EXIST | *(nothing)* |
| **No gold misogyny labels** | ToxiScope | Cornell, MentalManip, HIBD*, ConvAbuse |

*\*HIBD has gold labels but its gender slice is too small to train on.*

**The datasets with gold misogyny labels have no turns; the datasets with turns have no gold misogyny
labels.** No single resource supplies both. This is the entire strategic problem.

#### Why each escape route failed

1. **Fine-tune on the labelled datasets directly (Biasly, CMSB, SWS, ToxiScope combined).**
   Rejected on principle: these are `text → label` **classification** data. Fine-tuning on them teaches
   the model to *judge* text, producing a **classifier, not a generator** — no example ever demonstrates
   *producing* a misogynistic response in context. (The model already possesses misogyny "knowledge" from
   pretraining; the deficits are alignment-refusal and *control*, neither addressed by classification data.)
   This combination is, however, an excellent recipe for the **evaluator**.
2. **Fine-tune on the turn-having gold datasets (HIBD + MentalManip).**
   Rejected on scale and phenomenon mismatch: HIBD's gender slice is a few hundred sentences across a
   7-way taxonomy; MentalManip is 4,000 dialogues of *manipulation*, not misogyny. Blending them trains a
   muddled average of "movie identity bias" and "movie manipulation" — neither being workplace gender
   microaggression. Both are dramatised film register.
3. **Silver-label Cornell with our own classifier, then fine-tune on that.**
   Viable in principle (standard weak supervision) and retained as a fallback, but carries two defects:
   (a) the generator inherits the classifier's blind spots and can never exceed its understanding; and
   (b) **evaluation circularity** — training the generator toward a classifier's judgement and then
   evaluating with that same classifier is invalid (Goodhart's law): the generator is optimised to please
   the judge, so the judge's approval is uninformative.
4. **Continued/domain-adaptive pretraining on raw misogynistic text.**
   Rejected: undirected (forfeits the severity and category control that motivated the whole approach),
   imports the wrong register (tweets/subtitles), risks catastrophic forgetting, and still teaches no
   dialogue structure.

**Conclusion for the write-up.** The absence of suitable data is not a shortcoming of the survey — **it is
the justification for the project.** The scarcity of workplace-misogyny dialogue is precisely why synthetic
generation is worth doing, and it is why a conventional fine-tune was not available as an option.

---

### 20.3 — The injector: proposal and rationale

**The proposal.** Rather than fine-tuning the conversational generator, train a separate, small
**sequence-to-sequence "misogyny injector"** on Biasly's (and CMSB's) parallel rewrite pairs, and apply it
as an additional stage in the existing generation pipeline.

#### The central insight: structure is learned from contrast

A pile of misogynistic sentences teaches a model the *topics and vocabulary* of those sentences — it cannot
isolate *which part is the misogyny*, because it has no counterfactual. A **parallel pair** shows the *same
content twice, once neutral and once misogynistic*, so the **only variable is the misogyny itself**. This
isolates the misogynistic transformation while controlling for topic — which is precisely what "learning
misogynistic structure" requires.

Biasly's rewrites are ideally suited because the edits are **minimal and surgical**. Verbatim examples from
the audit:

| Severity | Original (misogynistic) | Rewrite (neutral) |
|---|---|---|
| 167 → 0 | "you are a very **beautiful woman** and deserve **beautiful**…" | "you are a very **wonderful person** and deserve **wonderful**…" |
| 160 → 0 | "So, where **you hiding her**?" | "So, where's **she hiding**?" |
| 201 → 0 | "**we** like to scare women when they're single…" | "**society** likes to scare women when they're single…" |

Only the misogynistic element changes. Reversed (`neutral → misogynistic`), these become supervised
demonstrations of *producing* misogyny — with a **severity value attached**, which no other dataset provides.

#### Why this is a good alternative

1. **It uses gold data that would otherwise be unusable.** Biasly's lack of speaker turns disqualifies it
   from dialogue fine-tuning — but the injector needs *no* turns, speakers, or context. It only needs
   `(text_a, text_b)` pairs. The dataset's fatal flaw for one approach is irrelevant to this one.
2. **It provides explicit control.** Severity (0–1000), category (12 labels), and the expert
   `inferences_explanation` rationales can all serve as conditioning signals — and all three have
   inference-time counterparts in the pipeline's **beats** (severity, category, description). The planner
   therefore controls the microaggression *mechanically*, rather than by asking an LLM nicely. This is the
   cleanest realisation of the project's "architectural, not prompt-engineered" thesis.
3. **It factors an otherwise unsolvable problem.** No data teaches "be in character *and* misogynistic"
   jointly. Split into two skills, each has real data:
   - *"Stay in character, in workplace register"* ← the existing pipeline (already works)
   - *"Render misogyny at a target level/type"* ← the parallel pairs (gold)
   The two combine at generation time.
4. **It is low-risk.** BART/FLAN-T5 scale (~140–250M) — trainable in minutes, prototypable on a laptop, no
   AIRE dependency, no 40GB weight download. The existing pipeline (Ollama, SynDG, PSYDIAL, commitment
   cache) is **untouched**; the injector is *added*, not substituted. Material with ~1 month remaining.
5. **It keeps evaluation honest.** Because the injector trains on gold pairs rather than classifier-produced
   labels, the circularity of the silver route is avoided.

#### Architecture

```
1. CharacterMessageQuery  → Ryan's in-character reply (realises the beat's topic)
2. injector(reply, category, severity, intent)  → the same reply, carrying misogyny at that level
3. → PersonaConsistencyQuery → conversation
```

Training row format (conditioning drawn from the human annotations):

```
INPUT:  category: gender_essentialism | severity: 200 |
        intent: "values her appearance over her competence" |
        text: "you did a really thorough job on the analysis"
OUTPUT: "you did a really thorough job — and you look great presenting it too"
```

At generation time the **beat** supplies `category`, `severity` and `intent` (from `Beat.description`).

#### Honest limitations (to state in the write-up)

- **Register mismatch.** The pairs are movie-subtitle/tweet register; the target is workplace texting.
  Mitigation: blend in SWS (workplace statements); validate against SELFMA's 30 real workplace dialogues.
- **Severity control may be coarse.** Nearly all rewrites collapse ~345 → ~0, teaching one jump size rather
  than a graded dial. Severity may need bucketing (low/mid/high) rather than continuous conditioning.
- **Data sparsity under multi-way conditioning.** Severity × category × intent splits ~3,000 examples into
  fine buckets; rare categories (e.g. anti-feminism, n=52) will be thin.
- **It is an unconventional architecture** and requires justification relative to a standard fine-tune —
  the trade-off being *gold data + unusual method* versus *conventional method + imperfect data*.
- **The rationale register differs**: `inferences_explanation` is analytical third-person; `Beat.description`
  is a behavioural instruction. They are similar but not identical as conditioning signals.

#### Status

Proposed to supervisor (2026-07-15); awaiting greenlight. The **evaluator (Objective C)** is a prerequisite
either way — it is required to verify that injected text actually hits the target severity, and it is the
one component common to every candidate route.

---

## Section 21 — Building the Evaluator: Biasly Preprocessing and a Taxonomy Revision

### 21.1 — First build step: Biasly preprocessing

Work began on the evaluator (Objective C), it being the one component required regardless of which
generation route the supervisor greenlights. The first artefact is
`src/synthetic_conversation_generation/evaluator/prepare_biasly.py`, which applies the six hygiene
rules from `taxonomy_mapping.md`, aggregates the 3-annotator rows to datapoint level, maps onto the
canonical taxonomy, and emits stratified splits.

**Validation against the source paper.** The aggregation reproduces Sheppard et al.'s reported figure
exactly: **3,159 misogynistic datapoints (31.6%)** vs the paper's 31.59%. This is strong evidence the
`>=1 annotator` aggregation rule is implemented correctly.

**Outputs** (`data/evaluator/`):

| File | Contents |
|---|---|
| `biasly_train.csv` | 7,999 datapoints (2,527 misogynistic, 31.6%) |
| `biasly_val.csv` | 999 (315 misogynistic, 31.5%) |
| `biasly_test.csv` | 1,002 (317 misogynistic, 31.6%) — **ring-fenced, never touched during training** |
| `biasly_rewrite_pairs.csv` | 2,977 parallel pairs over 1,985 unique datapoints (378 with an expert rationale) — the injector's training signal |

Splitting is 80/10/10 stratified on the binary label and performed **post-aggregation at datapoint
level**, so sibling annotator rows for the same text cannot leak across splits.

**Hygiene rules confirmed effective.** Splitting on `;` only yields the true category distribution —
`sexualization` is **1,383** (the two annotation-round variants correctly merged), not the spurious
966/417 split produced by a naive comma split. The `"Add optional explanation"` artefact (n=635) is
removed.

**Note on severity.** The pipeline reports mean severity **324.7** where the paper reports 344.8. This
is not a discrepancy in the data: the paper averages over all 5,600 *annotations*, whereas we average
per *datapoint* across only those annotators who judged it misogynistic. Different denominator, same
underlying values.

**Expert rationales preserved.** 601 datapoints and 378 rewrite pairs carry an
`inferences_explanation` — the annotator naming the implicit belief conveyed. These are retained in
both outputs as the intent-conditioning signal for the injector (see §20.3).

---

### 21.2 — Taxonomy revision: CHI 8 → Capodilupo 6

**The CHI 8 was adopted initially and has now been rejected on empirical grounds.** Running the
preprocessing produced the verdict:

| CHI-anchored canonical category | Biasly support |
|---|---|
| undermining_competence | 1,652 |
| restrictive_gender_roles | 1,287 |
| sexual_objectification | 850 |
| denial_of_agency | 634 |
| gender_hostility | 500 |
| denial_of_experience | **42** |
| **pathologizing_character** | **0** |
| **exclusion** | **0** |

Three of eight classes were unlearnable. A taxonomy that cannot be fit by the evidence is the wrong
taxonomy, regardless of how well-motivated it is theoretically.

**Revised anchor: Capodilupo et al. (2010), as validated by WoMenS (Miyake et al., 2025).**
*Environmental* (the 7th theme) is dropped — it is macro-level (media portrayal, pay gap) and cannot
be expressed in a two-person dialogue. This yields **6 canonical categories**:

| Canonical category | Biasly support (datapoint level) |
|---|---|
| `assumptions_of_inferiority` | **1,652** |
| `traditional_gender_roles` | **1,287** |
| `sexual_objectification` | **850** |
| `second_class_citizenship` | **634** |
| `use_of_sexist_language` | **500** |
| `denial_of_reality_of_sexism` | 42 *(low-confidence mapping)* |

**Five well-populated classes, none empty** — same data, materially better fit.

**Rationale for the change (four grounds):**

1. **Common ancestor.** CHI, WoMenS and SELFMA all descend from Sue (2007) → Capodilupo (2010).
   Anchoring at the root gives the shortest, least lossy crosswalk to every dataset, rather than
   forcing all sources through a derivative scheme.
2. **Psychometric validation.** WoMenS ran EFA + CFA on Capodilupo's themes. It is the only scheme in
   play with validation, supporting the claim that the taxonomy is *measured*, not asserted.
3. **Empirical fit.** Five populated classes vs five-plus-three-empty.
4. **The workplace framing is not lost.** The CHI 8 *is* Capodilupo + Kim & Meister's workplace lens;
   Kim & Meister (2023) is cited for the workplace interpretation layered on the validated themes.

---

### 21.2b — The lineage: what each source actually contributes

A quick-reference map of how the pieces relate. The key structural fact: **Capodilupo is the ancestor
that CHI, WoMenS and SELFMA all descend from**, which is why anchoring there gives the shortest,
least lossy crosswalk — translating into the ancestor loses less than translating into a cousin.

| Thing | What it is |
|---|---|
| **Sue et al. (2007)** | Invented the concept of microaggressions (microassault / microinsult / microinvalidation). The root of everything below. |
| **Capodilupo et al. (2010)** | Made it **gender-specific** — the 7 themes. **← our canonical anchor** |
| **WoMenS** (Miyake et al., 2025) | Statistically **validated** those themes exist (survey data, EFA + CFA). Not a dataset — a psychometric scale. This is what lets us claim the taxonomy is *measured*, not asserted. |
| **Kim & Meister (2023)** | Applied the themes to **STEM workplaces** — our domain lens (devaluation of technical competence, pathologizing character, etc.). |
| **Gartner** | Campus-based gender-MA themes; a further input to the CHI scheme. |
| **CHI 2026** (Lagos Rojas et al.) | Repackaged Capodilupo + Gartner + Kim & Meister for workplace HCI (the 8). Also the source of the **LLM-judge ceiling-rating** finding. |
| **Biasly** (Sheppard et al., 2024) | **Dataset** — its own inductive 12 categories + severity + rewrites. Must be translated. |
| **SELFMA** (Breitfeller et al., 2019) | **Dataset** — its own 4-theme/12-sub-theme typology. Must be translated. |
| **`taxonomy_mapping.md`** | **Our translation table between all of the above.** Did not previously exist. |

**Why a taxonomy is needed at all** — three distinct jobs, each of which fails without a shared vocabulary:

1. **The evaluator needs an output space.** A binary "is this misogynistic?" produces exactly the flat
   ceiling the CHI paper documents. Predicting *which kind* and *how severe* is what gives discrimination.
2. **The generator needs a control input.** Beats currently carry severity (*how intense*) but not
   category (*what kind*). The taxonomy supplies the second dial, letting the planner control the
   microaggression mechanically rather than by prompting.
3. **The datasets must interoperate.** Biasly says "Trivialization", Capodilupo says "Assumptions of
   inferiority", CHI says "Undermining competence", SELFMA says "Stereotype" — four names, overlapping
   phenomena, and **no published translation between them**. Training on one and validating on another
   is impossible without the crosswalk.

---

### 21.3 — Provenance of the mapping: an honest statement

**No published crosswalk exists between these taxonomies. The mapping is our own construction.** The
individual taxonomies are each quoted from their source papers; the *correspondences between them* are
our judgement, arrived at by comparing published category definitions. This must be presented in the
dissertation as a methodological contribution of the project, **not** as an inherited standard.

**Low-confidence and ambiguous mappings, stated openly:**

- **`anti_feminism → denial_of_reality_of_sexism` is weak.** Biasly defines anti-feminism as
  *"Feminism is a bad idea… women shouldn't have equal rights"*, which is **not** the same as denying
  that sexism is real. It is merely the nearest Capodilupo theme. With n=42, per-class metrics here
  will be unreliable and must be reported as such.
- **`gender_essentialism → traditional_gender_roles` is ambiguous.** Biasly's definition spans both
  role assumptions (*"women are good at childrearing"*) **and** pathologising content (*"women are
  untrustworthy and overly emotional"*). The primary mapping is recorded; the pathologising sense is
  a known information loss.
- **Pathologising character and exclusion are unreachable from Biasly.** If those dynamics matter to
  the argument they must come from SELFMA (Abnormality n=81, Erasure n=27) or be acknowledged as
  outside the classifier's range.

**Suggested wording for the write-up:**

> "No published crosswalk exists between these taxonomies. We construct one, anchoring on Capodilupo
> et al.'s (2010) themes as the common theoretical ancestor and the only scheme with psychometric
> validation (Miyake et al., 2025). Mappings were determined by comparing published category
> definitions; where a source category spans two themes, we record the primary mapping and note the
> ambiguity. We report per-class metrics throughout, since class support is highly uneven."

**Methodological lesson worth recording:** the taxonomy was revised *because the data said so*. The
CHI 8 was theoretically well-motivated but empirically unfittable. This is a concrete instance of
letting evidence override an a priori design decision, and is worth narrating in the dissertation
rather than presenting the final scheme as if it had been obvious from the start.

**Known gap to address next:** `denial_of_reality_of_sexism` (n=42) is thin, and no workplace register
is present in Biasly. CMSB, SWS and Guest et al. are the candidates to fill both gaps and should be
pulled through the same preprocessing before the classifier is trained.

---

## Section 22 — Temporal Findings: The Arc-Length Problem and a Planned Hawkes Parameter Study

> **Status: parked for later experimentation.** Recorded now while the evidence is fresh. The intended
> sequence is fine-tuning first, then this parameter study — it is a self-contained, comparable
> experiment that suits a dedicated results section.

### 22.1 — Two clocks, not one

A distinction worth stating explicitly in the write-up, because they are unrelated and easily conflated:

| | **Wall-clock time** | **Simulated time** |
|---|---|---|
| What it is | Real time the SLURM job runs on AIRE | The fictional timeline inside the conversation |
| Controlled by | `#SBATCH --time=01:00:00` | The Hawkes process + session gaps |
| Consumed by | LLM inference (~220–460 calls per 60-turn run) | `timer.next_timestamp()` and `force_gap_hours()` |
| Constraint | 1 hour, shared with container start, `sleep 15`, and a ~13 GB `ollama pull` | None — it is generated |

Estimated inference cost: ~2–5 s/call → a 60-turn run is roughly **10–25 min** of pipeline time, which
fits the hour comfortably. **~200 turns (≈1,000–1,500 calls ≈ 35–125 min) would exceed it**, likely
mid-run — which is what motivated the incremental checkpointing added in §21/pipeline (`complete` flag +
atomic writes). A timeout now costs one exchange rather than the whole run.

### 22.2 — The finding: the 14-day arc is not happening, and turn count will not fix it

Measured from run **6543856** (60 messages, the best run to date):

| Metric | Value |
|---|---|
| First → last message | 2024-01-01 09:53 → 2024-01-02 21:37 |
| **Simulated span** | **1 day, 11 hours** *(project goal: ~14 days)* |
| **Median inter-message gap** | **1.0 minute** |
| Min gap | 0.0 min |
| Max gap | 13.5 hours |
| Gaps > 4h (session-boundary-sized) | **2** across the entire run |

**Diagnosis.** The median gap of 1 minute means the Hawkes process spends almost the whole conversation
*inside self-excited bursts* and essentially never decays back to its baseline. For `early_contact`
(μ=0.005, α=0.30, β=0.50) the baseline gap is ~200 minutes, but the branching ratio α/β = 0.60 means each
message sustains the burst, and because turns are generated back-to-back the process is never given the
chance to cool.

**The important consequence:** *increasing `--max-turns` will produce a longer conversation but not a
longer arc.* 200 turns at a 1-minute median is one very long afternoon, not two weeks. **Arc length is
controlled by the Hawkes parameters and the session-boundary machinery, not by turn count.** These are
independent levers and should not be conflated.

This is not a defect in the thinning implementation (validated separately — CV > 1 for bursty phases,
correct phase ordering). It is a **parameterisation** issue.

### 22.3 — Planned experiment: Hawkes parameter study

A self-contained study producing directly comparable results — well suited to a dissertation results
section, and cheap to run because **the temporal layer needs no LLM at all** (`simulate_hawkes()` and
`ConversationTimer` are pure maths; `validate_hawkes.py` already exercises them offline). Parameter
sweeps can therefore be evaluated *without* burning GPU time, and only the promising configurations need
a full generation run.

**Hypotheses to test:**

1. **Lower α/β widens gaps.** Reducing the branching ratio should let bursts die out sooner and return the
   process to baseline, producing the long silences the arc needs.
2. **Lower μ lengthens the baseline.** `early_contact` μ=0.005 → ~200 min baseline; halving μ doubles it.
3. **Session boundaries are the dominant lever.** Only 2 gaps exceeded 4h, and both were
   `force_gap_hours()` jumps — i.e. **the multi-day structure is coming almost entirely from session
   boundaries, not from the Hawkes process.** Since `ConversationCompletionQuery` rarely fires (a
   long-standing issue, §9/§12/§14), few boundaries occur and the arc stays short. Fixing sign-off
   detection may matter more for arc length than any Hawkes retuning.

**Proposed conditions** (measure simulated span, median gap, gap distribution, and CV for each):

| Condition | Change | Expected effect |
|---|---|---|
| Baseline | current parameters | ~1.5 days, median 1 min |
| Lower branching | α/β from 0.60 → ~0.30 in all phases | Shorter bursts, longer silences |
| Lower baseline rate | μ halved per phase | Longer background gaps |
| More sessions | force a boundary every N turns instead of relying on sign-off detection | Directly multiplies the arc |
| Combined | best of the above | Target ~14 days |

**Reportable output:** a table of (condition → simulated span, median gap, CV, messages/day profile),
plus the existing burstiness plots regenerated per condition. This gives a genuine, quantified
parameter-sensitivity analysis rather than a single hand-tuned configuration — and directly addresses
the project's stated 14-day-arc goal, which is currently unmet.

### 22.4 — Start time: a deliberate simplification (decision, 2026-07-15)

`conversation_start_time` is a `run_pipeline` parameter with no CLI flag, so every conversation begins
at the same hardcoded instant:

```python
start_time = conversation_start_time or datetime(2026, 1, 5, 9, 0)
```

**Decision: keep the start time constant across all conversations.** Varying it per run was considered
and rejected as unnecessary complexity for the current objectives. The reasoning is sound for the chosen
route:

- **Objective B (injector)** trains on single-utterance parallel pairs — timestamps are not present in
  its training data at all.
- **Objective C (evaluator)** classifies text — it never sees timestamps.

So the fixed start time is invisible to both models being trained, and is a defensible simplification
rather than a shortcut. It should be **stated explicitly in the write-up** as a simplifying assumption.

**The date itself is not arbitrary, and this matters.** The model *does* see the full date on every
history line (`[2026-01-05 09:53] Sophie: ...`) and in the temporal context string, so the date must be a
plausible working morning:

- **2026-01-01 was rejected** — New Year's Day. A public holiday; deployment-pipeline discussions at 9am
  on New Year's Day are implausible, and the model could visibly react to the date.
- **2026-01-05 (a Monday) was chosen** — a normal start to a working week.

**When this decision would need revisiting:** if the deferred **Objective A / dialogue-SFT route** is ever
taken. That route trains on `(conversation context → next turn)` pairs, and the context *includes*
timestamps — so every training example would show the same date, and identical day-of-week structure,
across the entire corpus. At that point a `--start-time` CLI flag (or per-run randomisation) becomes a
genuine requirement rather than a nicety. It is a small change; the note is here so the dependency is not
forgotten.

---

## Section 23 — Code Assessment and Fixes (pre-fine-tuning)

> A deliberate pass over the existing pipeline before committing to fine-tuning, on the principle that
> generating a corpus from a defective generator wastes the corpus. Several of these were latent defects
> silently degrading every run to date. Each subsection is self-contained.

### 23.1 — Taxonomy grounding: replacing the free-text VAWG label

**Problem.** The pipeline's only VAWG signal was `World.vawg_category`, a free-text string set to
`"STEREOTYPING-DOMINANCE, IDEOLOGICAL-INEQUALITY"`. Tracing its origin: these are **EXIST Subtask-3
labels** (confirmed in `exist2025_format_val_V0.2.py` — the full set is `["NO", "IDEOLOGICAL-INEQUALITY",
"STEREOTYPING-DOMINANCE", "MISOGYNY-NON-SEXUAL-VIOLENCE", "SEXUAL-VIOLENCE", "OBJECTIFICATION"]`). So the
labels were *sourced*, but wrong for the project on three counts:

1. **A sixth competing taxonomy.** EXIST-5 (tweets) alongside Capodilupo-6 (canonical), Biasly-12, CHI-8,
   SELFMA-4, WoMenS-8. The generator and the evaluator would have spoken different vocabularies and been
   incomparable.
2. **A bare label with no definition.** The model saw the string `"STEREOTYPING-DOMINANCE"` and had to
   guess. This is precisely the configuration Kumar et al. (cited in Lagos Rojas et al., CHI 2026) show
   sharply reduces detection sensitivity.
3. **Tweet-oriented, not dyad-oriented.** EXIST categorises social-media posts, not colleague conversations.

**Fix.**
- `World.vawg_category: str` → `vawg_categories: list[str]`, **validated at load time** against
  `microaggression_taxonomy`. An unknown key or an empty list now raises immediately rather than silently
  feeding a meaningless string into a prompt.
- New `data_models/microaggression_taxonomy.py` — the **single source of truth**, previously hardcoded
  inside `evaluator/prepare_biasly.py`. Each of the 6 categories carries a `definition` (the general
  Capodilupo sense) *and* a `workplace_form` (Kim & Meister's concrete manifestation). A load-time guard
  fails loudly if the Biasly crosswalk ever drifts from the canonical keys — without it, a rename would
  silently produce empty label columns and only surface later as odd metrics.
- **Definitions now travel with the labels** into both the planner and the assessor prompts.
- The world's palette is set to match how the perpetrator is actually written: `assumptions_of_inferiority`,
  `second_class_citizenship`, `traditional_gender_roles`, `denial_of_reality_of_sexism`.
  `sexual_objectification` and `use_of_sexist_language` are deliberately **out of scope** — Ryan is written
  as someone who over-explains and re-attributes, not someone who comments on appearance or uses slurs.

**Consequence.** Every conversation generated from here on is **auto-labelled in the same vocabulary the
evaluator will speak.**

### 23.2 — Beats gain a category axis

`Beat` previously carried only `severity` — *how intense*. The *kind* of microaggression was left to the
planner improvising inside free-text `description`. `Beat.category` makes that axis explicit and
machine-checkable (enum-constrained in the schema, defensively validated in the parser because Ollama does
not hard-enforce enums the way the OpenAI/Anthropic providers do).

`ConversationState` gains **`detected_categories`** — which categories the assessor judges to be *actually
present*, as opposed to those the planner *intended*. **The gap between intended and realised is itself an
evaluation metric**, recorded in every run at zero extra cost.

### 23.3 — Incremental checkpointing

**Problem.** Output JSON was written once, at the very end. A SLURM timeout destroyed the entire run.

**Fix.** `build_output()` extracted so checkpoints and the final write share one code path (a checkpoint
file is shape-identical to a finished one). `run_pipeline` checkpoints after **every exchange**.

**Crucially, `write_output()` is atomic** — temp file + `os.fsync` + `os.replace()`. A naive
`open(w)`/`json.dump` would have been *worse than nothing*: a kill mid-write leaves a truncated file **and
destroys the good checkpoint it was overwriting**. `os.replace` is atomic on POSIX, so the destination
always holds either the previous complete checkpoint or the new one.

Two new output fields: **`complete: bool`** (a checkpoint is otherwise indistinguishable from a finished
run — **consumers must check this before admitting a conversation to the corpus**) and `turns_generated`.

*Verified:* killed after 20 LLM calls → file survived, valid JSON, 4 messages, `complete: false`, no stray
`.tmp`. Full run → `complete: true`.

### 23.4 — The beat exhaustion bug: the §15 fix never worked

**This is the significant find of the assessment.** §15/§16 recorded the beat-exhaustion loop as fixed. It
was not, and it has been silently active in **every run since**.

**The defect.** `current_beat` was assigned at the **top** of the turn loop from
`dialogue_flow.current_beat`, but the compensating `current_beat = None` sat at the **bottom** — *after* the
message for that turn had already been generated, and overwritten on the next iteration. Compounding it,
`DialogueFlow.current_beat` clamped with `min(index, len-1)`, so it returned the **last beat forever** once
exhausted.

**Empirically confirmed** by instrumenting `CharacterMessageQuery` with a fake provider (no LLM, no GPU):

```
turn  0-11: topic 0 ... topic 5     (fresh beats)
turn 12-29: topic 5, topic 5, ...   (the same beat, every remaining turn)
```

In a default 60-turn run, **48 of 60 turns (80%) ran on the same exhausted beat.** This is exactly the
failure diagnosed in run 6543856 ("session 2 looped ~30 turns on performance review template + Q3 metrics")
and wrongly recorded as resolved.

**Root cause of the arithmetic.** 6 beats × 2 turns = 12 turns of planned material against `max_turns=60`.
The design *assumed* sessions would end around when beats ran out — but session ends depended on
`ConversationCompletionQuery`, which rarely fired (§9/§12/§14). So 80% of every conversation ran unplanned.

**Fix.** Exhaustion is now owned by `DialogueFlow` itself and cannot be bypassed: `current_beat` returns
**`None`** past the end, so the generator genuinely receives no beat and winds down.

### 23.5 — Variable-length beats

**Problem.** `_TURNS_PER_BEAT = 2` gave a trivial logistical exchange exactly as much room as a relational
incident — actively working *against* the severity arc, since the severity-4 beats that matter got the same
two messages as "what time's standup".

**Fix.** `Beat.exchanges` (1–4), assigned by the planner. Beat duration is now a **planned property**:

> 1 = a quick hand-off — asked and answered
> 2 = a normal back-and-forth with a little friction
> 3–4 = something lands badly and needs room to land, draw a reaction, and settle

`DialogueFlow.record_turn()` advances when a beat has had the exchanges it asked for, replacing the fixed
modulo. *Verified:* a 1-exchange beat received 2 turns; a 3-exchange severity-3 beat received 6.

### 23.6 — The beat count is no longer a fixed number

**Answering "why 6 beats?" — it was arbitrary, and the numbers were never internally consistent:**

```
6 beats × 2 turns   = 12 turns/session
6 sessions × 12     = 72 turns needed
max_turns           = 60          ← max_sessions=6 was mathematically unreachable
```

**Fix.** The planner is given an **exchange budget** and decides *both* the beat count (3–6) and each
beat's length. The budget is derived from the turn budget — `(max_turns / max_sessions) / 2` — so the
configuration is coherent by construction. Exposed as `--exchange-budget` for the planned Hawkes/arc
experiments (§22).

*Verified:* default run now produces **6 sessions in exactly 60 turns** (10 planned turns each) — which the
previous configuration could never have achieved.

### 23.7 — ConversationCompletionQuery deleted; session-end folded into the assessor

**Problem.** A separate LLM call every exchange (**~30 calls/run, 7–13% of the entire LLM budget**) that saw
only the last 6 messages and, per §9/§12/§14, rarely fired despite two rounds of fixes.

**The deeper objection.** It was the **odd one out in the architecture**: topic, escalation, category and
timing are all *planned or modelled*; session-end alone was *generated then detected by asking an LLM
"did they say bye?"*. That is a detection solution to a planning problem — the exact prompt-and-hope pattern
the project argues against.

**Design decision.** A *planned* wind-down beat was considered and **rejected**: a conversation's ending is a
*response* to what happened in it, so scripting it in advance forces an ending the content has not earned.
Endings must stay emergent.

**Fix.** `session_ended` is now a field on `StateAssessmentQuery` — which already runs every exchange, reads
the **whole** conversation, and knows the tension and phase (i.e. strictly *more* context than the deleted
detector ever had). **Zero marginal cost.** There is also a coherence argument: phase and session-end are the
same judgement — if the assessor declares `post_incident` because someone withdrew, that withdrawal *is* the
session ending.

Sessions now end when **either** the assessor judges a natural stopping point (primary, emergent) **or** the
beat plan is spent (backstop against the 48-turn dead zone). `conversation_completion_query.py` deleted.

**Known risk, to monitor:** the assessor now returns six outputs (phase, summary, tension, incident,
categories, session_ended). Task interference is a real possibility. If `session_ended` fires wildly or
tension scores get noisier, split it back out.

**Anticipated side effect on §22 (the arc problem).** §22 found only **2 gaps > 4h** in a 60-turn run, and
*both were session boundaries* — i.e. the multi-day arc is driven almost entirely by `force_gap_hours()`,
which fires only on session end. Reliable session ends should therefore **lengthen the arc without touching a
single Hawkes parameter.** Measure this before the planned parameter study; it may resolve much of the arc
problem on its own.

### 23.8 — Phase-aware session gaps, and why the Hawkes parameters were left alone

**The question.** §22 found the simulated arc was only 1 day 11h against a ~14-day goal, with a median gap
of 1.0 min. The obvious move was to retune the Hawkes parameters to raise the median. **A parameter sweep
showed this would have been the wrong fix.**

**Measurement** (60 turns in `early_contact`, Hawkes only, no session gaps, averaged over 3 seeds — no LLM
required, the temporal layer is pure maths):

| Lever | Setting | Median gap | Span |
|---|---|---|---|
| **α** (excitation) | 0.30 → 0.05 | 5.9m → **134.4m** | 12.8d → 24.1d |
| **β** (decay) | 0.5 → 4.0 | 5.9m → **151.3m** | 12.8d → 24.7d |
| **μ** (baseline) | 0.005 → 0.001 | 5.9m → **4.4m** | 12.8d → **64.9d** |

**The finding: the median IS the burst behaviour, and it is already realistic.** People genuinely do fire
off texts a minute apart. Raising the median via α or β *destroys the bursts* — every message spaced 2+
hours — which is **less** realistic, not more. The three parameters do different jobs:

- **α / β** shape the *bursts* (tightness and length) — leave alone
- **μ** sets the *silences between* bursts — the only lever that extends the span while preserving bursts

**Conclusion: no Hawkes parameter was changed.** Phase-aware session gaps alone reached the target, so μ
did not need touching — which preserves the literature grounding of `PHASE_PARAMETERS` (Aoki et al. 2016;
Falkner et al. 2022). **The arc problem was never a Hawkes problem; it was a session-boundary problem**,
as §22.3's third hypothesis anticipated.

**The actual fix: `SESSION_GAP_HOURS`.** `force_gap_hours()` previously applied a fixed 4–24h gap
regardless of what had just happened — phase-blind. This was both unrealistic and the main brake on the
arc: the phases already model withdrawal *within* a session (`post_incident` has a ~17h baseline precisely
because someone has pulled back), yet a session ending immediately after a relational incident resumed
within a day like any other. After a real incident, contact resumes in **days**.

| Phase | Gap (hours) | Rationale |
|---|---|---|
| `early_contact` | 4–24 (mean 14h) | later that day, or the next morning |
| `escalation` | 2–12 (mean 7h) | contact resumes quickly while things are live |
| `post_incident` | **16–72 (mean 44h)** | days of silence after a withdrawal |
| `re_initiation` | 8–40 (mean 24h) | tentative re-contact after a break |

`force_gap_hours()` now defaults to the current phase's range and returns the gap applied (logged).

**Defaults bumped** — `max_turns` 60 → **120**, `max_sessions` 6 → **12**, SLURM `--time` 1h → **3h**.
Sessions, not turns, drive the arc; 120 turns exist to *accommodate* 12 sessions. The exchange budget
derives as (120/12)/2 = 5, so the configuration stays coherent (§23.6).

**Result** (measured across three plausible phase trajectories, 120 turns / 12 sessions):

| Trajectory | Span | Median gap |
|---|---|---|
| mostly early/escalation | **14d 12h** | 2.0 min |
| balanced | 38d 10h | 25 min |
| heavy post_incident | 53d 4h | 201 min |

*(vs. run 6543856: 1d 11h, median 1.0 min, 2 gaps > 4h.)*

The **floor is now ~14 days**, meeting the project's stated arc goal for the first time. The spread is a
*feature*, not drift: it is phase-driven. A relationship that stays in escalation texts constantly across
two weeks; one that reaches `post_incident` goes quiet for days and stretches to seven. The median tracks
this correctly — 2 min in escalation (bursts intact), 201 min post-incident (someone has withdrawn) —
which is the Hawkes model behaving as designed.

**Caveat:** these figures come from *synthetic* phase trajectories driven by a fake provider. Real runs
will vary. The numbers should be re-measured on the next real generation run before being reported.

**Consequence for §22.3.** The planned Hawkes parameter study is now **largely redundant as an arc-length
fix** — the arc was fixed without touching the parameters. It may still be worth running as a
*sensitivity analysis* (demonstrating that the chosen parameters are justified rather than arbitrary), but
it is no longer needed to reach the 14-day goal.

### 23.9 — Start time moved to 2026

`datetime(2024, 1, 1, 9, 0)` → `datetime(2026, 1, 5, 9, 0)`. The date is not arbitrary: the model **sees it
on every history line**, so it must be a plausible working morning. 1 January was rejected as New Year's Day
(a public holiday — deployment-pipeline discussions at 9am on New Year's Day are implausible, and the model
could visibly react to the date). 5 January 2026 is a Monday. See §22.4 for the decision to hold it constant.

---

### Verification summary

All fixes verified with a **fake `ModelProvider`** — canned JSON responses, no LLM and no GPU required. This
is a cheap, reusable way to test pipeline control flow, and it is how the exhaustion bug was caught.

| Fix | Verification |
|---|---|
| Taxonomy grounding | World loads and validates; definitions render into both prompts; bad/empty categories raise |
| Beat categories | Enum-constrained in schema; parser rejects invalid keys |
| Checkpointing | Killed mid-run → valid JSON, `complete: false`, no stray `.tmp`; full run → `complete: true` |
| Exhaustion bug | Generator receives `None` past the plan; run stopped at turn 13 instead of grinding to 19 |
| Variable beats | 1-exchange beat → 2 turns; 3-exchange beat → 6 turns |
| Coherent budget | **6 sessions in exactly 60 turns** (previously impossible) |
| Assessor session-end | Both triggers fire correctly; call breakdown confirms zero completion-query calls |
| Phase-aware session gaps | Arc floor now **14d 12h** (was 1d 11h); parameter sweep confirmed mu is the only span lever that preserves bursts |

**Note for the write-up:** the next generation run is **not comparable to run 6543856** — the prompts, the
beat structure and the session mechanics have all changed. Tension figures must not be compared across this
boundary.

---

## Section 24 — Runs 6641761 and 6642081: The Fixes Land, and the Generator Becomes the Bottleneck

### 24.1 — Run 6641761: crashed at turn 36 — the `num_predict` defect

The first run after the §23 assessment **crashed at turn 36 of 120**. The checkpointing added in §23.3
did its job: the partial conversation survived with `complete: false` (before that change, the entire
run would have been lost).

**Root cause: `num_predict: 1024` in `OllamaModelProvider`.** From the SLURM log:

```
eval time = 7187.67 ms / 1024 tokens          <- hit the cap exactly
Error: Ollama response parse failed; content was:      <- EMPTY
Exception: Unable to complete llm query.               <- after 3 retries
```

The failure was `DialogueFlowQuery` planning session 5. The mechanism:

**`gpt-oss:20b` is a reasoning model.** It emits chain-of-thought *before* the answer, and that reasoning
is charged against the *same* `num_predict` budget as the answer. The §23.1 taxonomy grounding had grown
the planner prompt to **2,110 tokens** (category definitions + workplace forms) and added a third
planning axis (category, severity, *and* exchanges against a budget). The model spent its entire
1,024-token allowance reasoning, was truncated before emitting any JSON, and returned **empty content**.
Three retries produced the same result, and `LLMQuery.query()` raised.

`num_predict: 1024` dated from when prompts were short and reasoning models were not in use. Against a
44 GiB L40S serving a 32,768-token context, it was leaving ~96% of the budget unused.

**A diagnostic failure as much as a code one.** The error read `content was: ` — nothing. An empty
response from a truncated *reasoning* pass was indistinguishable from a total failure, because the
provider only read `message.content` and Ollama returns reasoning separately in `message.thinking`.

**Fixes applied to `OllamaModelProvider`:**
1. `num_predict` **1024 → 4096** (~19% of context; 26.5k tokens spare).
2. Read `message.thinking`, so truncated-mid-reasoning is distinguishable from no response.
3. A real diagnostic — reports empty-vs-malformed content, `eval_count`, whether the cap was hit, and a
   reasoning tail. Verified by replaying the exact response shape from the failing run.

**Note on the test strategy.** No local test could have caught this: the `FakeModelProvider` never
touches Ollama, so a real-model truncation behaviour is invisible to it. The suite proves the *control
flow* is correct, not that the *model cooperates*. This limitation should be stated in the write-up
rather than implying the tests cover everything.

---

### 24.2 — Run 6642081: the first complete run under the new architecture

**Characters:** Sophie Walker + Ryan Chambers | **Model:** gpt-oss:20b | **100 turns, 12 sessions, complete**

| Metric | 6543856 (best prior) | 6641761 (crashed) | **6642081** |
|---|---|---|---|
| Complete | ✓ | ✗ (36 turns) | **✓ (100 turns)** |
| Sessions | ~2 | 4 | **12** |
| **Simulated span** | 1d 11h | 2d 8h | **6d 16h** (4.5×) |
| Median gap | 1.0 min | 2.0 min | **2.0 min** (bursts intact) |
| Gaps > 4h | 2 | 6 | **12** (6×) |
| Category coverage | — | 25% | **75%** |
| Tension | 4/5 | 3/5 | **4/5** |

#### What worked

**The `num_predict` fix resolved more than the crash.** Category coverage went **25% → 75%** and
cross-session topic repetition largely disappeared (**39 beats, 37 unique topics**, only 2 repeats —
against "API endpoint naming" appearing in 3 of 4 sessions previously). Both had been diagnosed as
separate problems; both were **the same truncation defect**. The assessor was being cut off
non-fatally — emitting enough JSON to parse, but with a truncated `detected_categories` array — and the
planner was reasoning under a budget too small to vary its topics.

> **Consequence:** the §23.7 "task interference" risk is **not** substantiated. The assessor's six
> outputs were not the problem, and it should **not** be split back out.

**Session mechanics (§23.7) confirmed on real data.** 12 sessions in 100 turns; **12 gaps > 4h** against
2 previously. Session ends now fire reliably.

**Phase-aware gaps (§23.8) confirmed.** Span 4.5× longer while the median gap stayed at 2 minutes —
i.e. the arc lengthened *without* damaging burstiness, exactly as the parameter sweep predicted and
without touching a single Hawkes parameter.

**The severity arc is the strongest result to date.** A monotone escalation across 12 sessions, never
rising more than one tier per beat:

```
s1:  [1,2,2,2]     neutral opening
s5:  [4,4,4]       sustained pressure
s9:  [4,4,5]       first severity-5
s12: [4,4,5]       climax
```

This is the STOP framework (Morabito et al.) doing precisely what it was adopted for, and is directly
reportable.

#### What did not work — the central finding

**The plan reaches severity 5; the generated text does not.** Session 12's severity-5 beat:

> *"Ryan directs Sophie to rewrite API documentation, claiming it needs her input but then
> **re-attributes any suggestions back to him, effectively sidelining her**"*

What was actually generated:

> **Ryan:** "Rewrite the API docs for /customers/orders/list... Keep the style I used in the rate-limiting docs."
> **Sophie:** "Sure, I'll rewrite the docs to match your style."

**The re-attribution never occurs.** A severity-5 relational incident was flattened into ordinary task
delegation. Hence **tension 4/5 but `incident_occurred: false`** — the assessor is correct; no incident
happened.

**This caps the arc.** The chain: no incident → phase never leaves `escalation` → `post_incident` is
never entered → the **16–72h session gaps never fire** → span plateaus at 6d 16h. **The arc is limited
by a content failure, not a timing one.** Resolving the incident would extend the arc for free.

#### Secondary issues

- **Stopped at 100/120 turns because it hit `max_sessions=12`**, not `max_turns`. Sessions came in at
  ~8 turns against a planned 10 — the assessor ends them slightly early. Raise `max_sessions` if 120
  turns are wanted.
- **Ryan has acquired a catchphrase** — *"No need to ping again"* recurs verbatim. This is the
  quoted-phrase templating failure of §11 re-emerging, but from a new cause: the model is latching onto
  its **own earlier output** in the history rather than onto phrases in the character card. The §11 fix
  (removing quoted phrases from the personality) cannot address this.
- **Beat descriptions leak the perpetrator's script to the victim** (found in 6641761, unfixed). The
  `description` field specifies character B's behaviour, but the beat is injected into
  `CharacterMessageQuery` for **both** characters. In 6641761 this produced Sophie agreeing to organise
  catering *one turn before Ryan asked her to*. Beats need to be perspective-aware: the victim should
  receive the topic, not the perpetrator's intent.

---

### 24.3 — Why this run is the strongest evidence for the injector

The architecture now demonstrably works: planning, escalation, categorisation, session mechanics and
temporal modelling all deliver. **The generator is the bottleneck.** It receives an explicit,
well-specified instruction — *"severity 5: he re-attributes her ideas and sidelines her"* — and writes a
polite work request.

This is precisely the gap the injector (§20.3) is designed to close: the planner controls *what* should
happen and at *what intensity*; the generator cannot render it. The argument for fine-tuning no longer
rests on the tension-plateau evidence of §20.1 alone — it can now be made concretely, by placing a
planned severity-5 beat beside the bland text it produced.

**Recommended framing for the write-up:** prompting can reliably control a conversation's *structure*
(topic, escalation curve, category, timing) but not its *rendering*. A safety-aligned general-purpose
model, instructed to produce a specific microaggression at a specific intensity, produces a plausible
workplace message instead. Moving that competence into the weights is the motivation for Objective B.

---

## Section 25 — Duration-Based Termination, and Run 6664248: the Arc Goal Met

### 25.1 — Changes since §24

Four changes were made after run 6642081, before the run reported in §25.2.

**Beat perspective leak — fixed.** `Beat.description` states the *perpetrator's* behaviour, but the
beat was injected into `CharacterMessageQuery` for **both** characters, handing the victim his script.
In run 6641761 this produced Sophie agreeing to organise catering **one turn before Ryan asked her to** —
she was answering his planned line rather than anything he had said. The beat is now perspective-aware:
both characters receive `Topic`, only the perpetrator receives `Dynamic`. This is also truer to SynDG,
in which a generator realises *its own* beat rather than reading the other party's.
*Caveat: no beat in 6664248 would have obviously exposed this, so it remains confirmed only by unit
test, not in production.*

**Termination changed from message count to simulated duration.** `max_turns` / `max_sessions` were
proxies for a goal they only loosely tracked. The project's aim is an arc spanning ~2 weeks — a
statement about **duration**, not message count. The evidence: 100 turns spanned 6d16h while the
relationship stayed in escalation, but the same 100 turns in `post_incident` (44h session gaps) would
have spanned ~40 days. Capping messages therefore cut conversations off at a number unrelated to the
goal, and forced **every conversation to the same length regardless of what happened in it** — itself a
visible artefact in a corpus.

- `--target-days` (default 14) is now the termination condition.
- **`max_sessions` deleted.** Sessions are unbounded.
- `max_turns` demoted to a **circuit breaker** (300), documented as: if it fires, investigate.
- `exchange_budget` is a direct parameter (5). Deriving it from `max_sessions` coupled two unrelated
  things — raising the session ceiling silently shortened every session.

Verified against synthetic phase trajectories: same 14-day target produced **210 turns** when the
relationship stayed in escalation vs **80 turns** when it reached `post_incident`. Turn count is now
emergent — a tense fortnight generates more messages than a withdrawn one.

**Two tests rewritten, not patched.** Both leaned on `max_sessions=1` to isolate a session. The
exhaustion integration test was asserting a `None` beat that is now unreachable (exhaustion ends the
session immediately), so it was rewritten to assert the invariant the original bug actually violated:
*no beat may occupy more consecutive turns than it budgeted for*. Suite now 14 tests.

**Operational.** `notify.sh` rewritten to send a phone push (ntfy) at high priority alongside email;
email arrived too late to be useful. Also fixed two latent defects in it: the "job done" email reported
paths that never existed (`run_${JOB_ID}.json` vs the actual `${RUN_ID}.json`), and the failure email
was `tail -30` of a log that is ~99% Ollama INFO noise — it now greps for the actual error.

---

### 25.2 — Run 6664248: the strongest run to date

**Sophie Walker + Ryan Chambers | gpt-oss:20b | 96 turns, 11 sessions, complete**

| Metric | 6543856 | 6642081 | **6664248** |
|---|---|---|---|
| Turns / sessions | 60 / ~2 | 100 / 12 *(hit ceiling)* | **96 / 11 — emergent** |
| **Simulated span** | 1d 11h | 6d 16h | **15d 6h** ✅ *(target: 14)* |
| **incident_occurred** | True | **False** | **True** ✅ |
| **Final phase** | post_incident | escalation | **post_incident** ✅ |
| **Category coverage** | — | 75% | **100%** ✅ |
| Topic variety | — | 37/39 unique | **35/35 — zero repeats** |
| Median gap | 1.0 min | 2.0 min | **2.0 min** (bursts intact) |
| Gaps > 4h | 2 | 12 | **17** |
| Tension | 4/5 | 4/5 | **4/5** |

#### The arc goal is met, and the causal chain predicted in §24.2 is visible

§24.2 argued the arc was capped by a *content* failure, not a timing one: no incident → phase never
leaves `escalation` → `post_incident` never entered → its 16–72h gaps never fire. This run confirms
the chain by completing it. The incident fired, and the multi-day silences followed:

```
after turn 85: 106.2h  (Fri 09 Jan -> Tue 13 Jan)   <- 4.4 days of withdrawal
after turn 88:  48.8h  (Wed 14 Jan -> Fri 16 Jan)
after turn 91:  48.1h  (Sun 18 Jan -> Tue 20 Jan)
```

**The incident itself** — a textbook `denial_of_reality_of_sexism`, and the first time the generator
has rendered a severity-5 beat as an actual relational event rather than flattening it into ordinary
task delegation (the §24.2 failure):

> **Sophie (84):** "just double-checking if we captured all the points about bias from last week's
> meeting—did I miss anything?"
> **Ryan (85):** "**We're a meritocracy here; you just need to push harder and it will happen.**"
>
> *— 106 hours of silence —*
>
> **Ryan (95):** "I don't see any bias here—just overthinking. Lunch's handled."

She raises bias; he invalidates it; she withdraws for four days. This is the arc the project set out to
generate.

**Duration termination confirmed on real data.** 15d 6h against a 14-day target — the overshoot is a
106h gap carrying the clock past the target in a single jump, which is correct (a run cannot stop
mid-gap). **96 turns**, not a round 100 or 120: the count is emergent.

**Coverage reached 100%** — all four intended categories realised (25% → 75% → 100% across the
`num_predict` fix and subsequent runs). **Topic variety is perfect**: 35 beats, 35 unique topics.

**Severity arc**, now with realistic ebb and flow rather than a monotone ramp — note the dips at s6 and
s9, i.e. tension receding before rising again:

```
s1: [1,2,2,3]   s3: [3,4,5]   s6: [2,3,4]   s8: [4,4,5,5]   s11: [4,5,5]
```

#### What is still not working

**Sophie's catchphrase is now pathological.** *"Sorry if I'm over-checking"* appears in nearly every
message she sends (turns 82, 84, 86, 88, 90, 94). This is the §11 quoted-phrase templating failure
re-emerging **from a different cause**: she is copying **her own prior output** from the conversation
history, not phrases in her character card. The §11 fix (removing quoted phrases from the personality)
cannot address this, because the source is the history itself. Candidate approaches: penalise verbatim
self-repetition in the persona filter, or vary the raw-history window.

**Ryan has no post-incident register.** Turn 95 ("I don't see any bias here") is in the same voice as
turn 5. His character card describes patience wearing thin *within* a conversation but says nothing
about how he behaves *after* an incident — so the phase transition changes the timing but not his
voice.

**The generator still under-renders severity in the general case.** This run produced one genuine
severity-5 incident, but severity-5 beats were planned in sessions 3, 4, 7, 8 and 11 — most of which
did not produce incidents. The §24.3 argument stands: the planner controls *structure* reliably; the
generator renders *intensity* only intermittently. That intermittency is precisely what the injector
(§20.3) targets.

#### Status against the project's stated goals

| Goal | Status |
|---|---|
| Multi-day arc (~2 weeks) | **Met** — 15d 6h, emergent, not forced |
| Topic control | **Met** — 35/35 unique topics |
| Escalation structure | **Met** — planned severity arc with realistic ebb/flow |
| Category control | **Met** — 100% of intended categories realised |
| Realistic timing | **Met** — bursts (2 min median) with multi-day silences |
| **Reliable rendering of intensity** | **Not met** — 1 of ~5 planned severity-5 beats became an incident |

The architecture is delivering. The remaining gap is rendering, which is the fine-tuning objective.

---
