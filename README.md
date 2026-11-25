# Task 6: Multilingual and Multimodal Culturally-Grounded Spoken and Visual Question Answering

## Summary of findings so far

| Variant | Eval set | BERTScore-F1 | P | R | ROUGE-L | BLEU |
|---|---|---|---|---|---|---|
| v0: zero-shot baseline (Qwen2.5-Omni-3B) | dev, 1000 | 0.7392 | 0.6908 | 0.7974 | 0.2510 | 9.68 |
| v1: v0 + answer post-processing | dev, 1000 | 0.7825 | 0.7736 | 0.7945 | 0.3356 | 15.09 |
| v2: new prompt + post-processing | dev, first 300 | 0.7841 | 0.7648 | 0.8075 | 0.3308 | – |
| **v3: QLoRA fine-tuned (2000 train items, 1 epoch)** | dev, first 300 | **0.8489** | **0.8596** | **0.8402** | **0.4840** | **28.25** |

Key observations:

- **Post-processing alone gives about +0.04 F1** (v0 to v1). The base model often keeps talking after the answer:
  it appends `Human:` turns, asks follow-up questions, or stops mid-sentence. Cutting these off mainly raises precision.
- **The new prompt alone adds almost nothing** (v1 to v2, +0.002, on a different subset, so not directly comparable).
- **Fine-tuning gives a clear gain of +0.065 F1 over v2** on the same 300 dev questions. 
- **Much of the gain comes from matching the answer style.** Precision jumps from 0.76 to 0.86 and ROUGE-L from
  0.33 to 0.48. The model learned to answer in one or two factual sentences phrased like the references
  ("The image depicts…"). BERTScore rewards this strongly.
- **Content looks largely correct in spot checks.** Landmarks such as the Burj Khalifa, Carthage and the
  Anti-Lebanon range are identified correctly. Some answers are plausible but generic, for example on the cultural
  meaning of lanterns. BERTScore barely penalises that, so a manual error analysis is still needed.
- The fine-tuned answers tend to be **shorter than the references**. 

### However

- v0 and v1 were scored on all 1000 dev questions, v2 and v3 only on the **first 300** (generation runs at about
  0.10 items/s on a T4). Only the v2 vs v3 comparison is strictly like-for-like.
- The final number should be confirmed on the full 1000 dev questions (`EVAL_LIMIT = None`, `TRAIN = False`).


## Analysis of the fine-tuned model


### Answer length and overlap

| Variant | Mean words per answer | ROUGE-L | Word-overlap F1 |
|---|---|---|---|
| v0 baseline | 50.6 | 0.247 | 0.320 |
| v1 baseline + post-processing | 31.6 | 0.331 | 0.411 |
| **v3 fine-tuned** | **23.1** | **0.484** | **0.556** |
| Reference answers | 25.2 | – | – |

- The baseline answers are twice as long as the references. The fine-tuned model almost matches the reference
  length (1.14 vs 1.22 sentences per answer).

### Generation behaviour

- No empty answers.
- The model stops cleanly after the answer. In 299 of 300 cases generation ends at the `Human:` stop string,
  which `clean_answer` removes. 


### Error analysis

We read the 15 answers with the lowest word-overlap F1 by hand and found three error types:

| Error type | Examples |
|---|---|
| **Wrong recognition / hallucination** (most severe) | Souq Waqif: the model names the "Al-Bidda Tower" instead of the spiral minaret of the Fanar Islamic Cultural Center. A horse-riding stunt show is called "Buzkashi". A cosplay panel is described as a "press conference". |
| **Too generic** | Mosaic: "part of a historical or religious building" instead of the paradise symbolism of Islamic art. Protest signs: "justice, freedom, change" instead of opposition to foreign interference in Libya. |
| **Correct but phrased differently** | Syrian opposition flags, port explosion, flute, currency exchange: correct in content, but shorter or worded differently from the reference. |

**Conclusion:** Fine-tuning mainly improved the **form and style** of the answers (length, structure,
phrasing). It added little **cultural knowledge**, which is expected with 2000 examples and a 3B model.
Part of the low-scoring answers are actually correct, which shows a limitation of the overlap-based metrics
rather than of the model.

## Where the improvement comes from

Overall, BERTScore-F1 rises by about **+0.11**, from 0.739 (baseline) to 0.849 (fine-tuned). The gain comes from
two sources: cleaning up the answers (+0.04) and fine-tuning (+0.065).

| Step | Eval set (dev questions) | BERTScore-F1 | Gain | Source of the gain |
|---|---|---|---|---|
| v0: baseline | 1000 | 0.7392 | – | starting point (zero-shot Qwen2.5-Omni-3B) |
| v1: + post-processing | 1000 | 0.7825 | **+0.043** | `clean_answer` removes invented `Human:` turns, follow-up questions and cut-off sentences |
| v2: + new prompt | 300 | 0.7841 | +0.002 | almost no effect |
| v3: + QLoRA fine-tuning | 300 | **0.8489** | **+0.065** | training on 2000 train items |
| **Total** | | | **≈ +0.11** | |


- About 40 % of the total gain needs **no training at all**. It comes from removing the text the base model
  keeps generating after the actual answer.
- About 60 % of the gain comes from **fine-tuning**, which teaches the model the short, factual answer style of
  the references.
- Only the step **v2 → v3 (+0.065) is an exact like-for-like comparison**, because both are scored on the same
  300 dev questions. v0 and v1 were scored on all 1000 questions, so the total of +0.11 compares slightly
  different question sets. Evaluating v2 and v3 on all 1000 questions (see [Next steps](#next-steps)) will give
  the exact total.


## Next steps

Ordered by expected value for effort:

1. **Evaluate on the full dev set.** Score v2 and v3 on all 1000 dev questions (`TRAIN = False`,
   `EVAL_LIMIT = None`, `EVAL_PROMPT_ONLY = True`). This takes about 3 h on a Kaggle T4 without training and
   gives the final, like-for-like numbers.
2. **Manual evaluation.** Draw a random sample of about 50 answers and label each one as *correct*, *generic* or
   *wrong*. This measures factual accuracy, which BERTScore and ROUGE-L do not capture, and quantifies the
   error types above.
3. **Train on more data.** Only 2000 of the 10,000 train items are used, and validation loss was still
   decreasing at the end. Training on 5000–10,000 items (about 10–21 h on a T4, resumed across several Kaggle
   runs) should mainly help the knowledge-heavy "significance" questions.
4. **Cheap post-processing fixes.** Remove repeated sentences and test slightly longer answers. The fine-tuned
   answers are a bit shorter than the references and often omit the extra context sentence, which limits recall.
5. **Ablations.** Compare LoRA rank (8 / 16 / 32) and training-set size (500 / 1000 / 2000 / more) to show how
   much data and capacity the gain actually needs.
