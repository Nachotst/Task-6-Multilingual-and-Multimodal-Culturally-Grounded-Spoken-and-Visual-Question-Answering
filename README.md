# Task 6: Multilingual and Multimodal Culturally-Grounded Spoken and Visual Question Answering

## Summary of findings so far

| Variant | Eval set | BERTScore-F1 | P | R | ROUGE-L | BLEU |
|---|---|---|---|---|---|---|
| v0: zero-shot baseline (Qwen2.5-Omni-3B) | dev, 1000 | 0.7392 | 0.6908 | 0.7974 | 0.2510 | 9.68 |
| v1: v0 + answer post-processing | dev, 1000 | 0.7825 | 0.7736 | 0.7945 | 0.3356 | 15.09 |
| v2: new prompt + post-processing | dev, first 300 | 0.7841 | 0.7648 | 0.8075 | 0.3308 | – |
| **v3: QLoRA fine-tuned (2000 train items, 1 epoch)** | dev, first 300 | **0.8489** | **0.8596** | **0.8402** | **0.4840** | **28.25** |
| v4: QLoRA fine-tuned (all 9800 train items, 1 epoch) | dev, first 300 | 0.8511 | 0.8575 | 0.8466 | 0.4904 | 29.87 |
| **v4: QLoRA fine-tuned (all 9800 train items, 1 epoch)** | **dev, 1000** | **0.8533** | **0.8616** | **0.8470** | **0.4961** | **30.47** |

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
- **Five times more training data (v4) barely helps:** +0.002 F1 over v3 on the same 300 questions, which is not
  statistically significant.

### However

- v0, v1 and v4 were scored on all 1000 dev questions, v2 and v3 only on the **first 300** (generation runs at
  about 0.10 items/s on a T4). Like-for-like comparisons are v2 vs v3 and v3 vs v4 on the first 300.
- The first 300 dev questions are slightly harder than the rest: v4 scores 0.8511 on them and 0.8542 on the other
  700. A 300-item score therefore slightly underestimates the 1000-item score.


## Analysis of the fine-tuned model


### Answer length 

| Variant | Mean words per answer |
|---|---|
| v0 baseline | 50.6 |
| v1 baseline + post-processing | 31.6 | 
| **v3 fine-tuned** | **23.1** |
| Reference answers | 25.2 |

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
phrasing). It added little **cultural knowledge**.
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
- v4 (all train data) adds only +0.002 on top of v3, so it is not a separate source of gain. On all 1000 dev
  questions the fine-tuned model scores **0.8533**, i.e. **+0.114 over the baseline**.


## Training on the full train set (v4)

**Question:** does more data help? v3 used only 2000 of the 9800 train items, and its validation loss was still
decreasing.

**Setup:** start from the v3 adapter and train only on the **7800 train items it has not seen** (same shuffle seed,
so no item is repeated). Same hyperparameters, fresh optimizer with warmup, 975 steps. Because of Kaggle's 12 h
limit the run was split in two: run 1 stopped itself after 11 h at step 550 and saved a checkpoint, run 2 resumed,
finished training and evaluated on all 1000 dev questions. Results: `results/qlora_r16_n9800/`.

**Validation loss** (same 200 items as for v3):

| Step | 0 | 200 | 400 | 600 | 800 | 975 |
|---|---|---|---|---|---|---|
| Validation loss | 0.930 | 0.906 | 0.877 | 0.863 | 0.850 | **0.845** |

Step 0 equals the final v3 value, which confirms that the v3 adapter was loaded. The loss keeps falling, with no
sign of overfitting.

**Result on the same 300 dev questions:**

| | v3 (2000 items) | v4 (9800 items) | Difference |
|---|---|---|---|
| BERTScore-F1 | 0.8489 | 0.8511 | +0.0023 |
| Precision | 0.8596 | 0.8575 | −0.0021 |
| Recall | 0.8402 | 0.8466 | +0.0064 |
| Mean words per answer (reference: 24.9) | 22.9 | 24.1 | +1.2 |

- **The gain is not statistically significant.** 

- Answers became slightly longer and closer to the reference length: recall up, precision slightly down.
- Content changes in both directions. Better: v4 names the *Fanar Islamic Cultural Center* where v3 hallucinated
  the *Al-Bidda Tower*, and calls the winged bulls *lamassu*. Worse: v4 calls the *Erbil Citadel*, which v3 named
  correctly, only "a large fortress with a tall minaret".

  

**Conclusion:** more of the same data is not the bottleneck. Further gains are more likely
from adding background knowledge, or from better post-processing.

## Next steps

Ordered by expected value for effort:

1. **Manual evaluation.** Draw a random sample of about 50 answers and label each one as *correct*, *generic* or
   *wrong*. This measures factual accuracy, which BERTScore and ROUGE-L do not capture, and could show content
   differences between v3 and v4 that BERTScore hides.
2. **Cheap post-processing fixes.**
3. **Evaluate v2 and v3 on the full dev set** for exact like-for-like totals (`TRAIN = False`, `EVAL_LIMIT = None`,
   `EVAL_PROMPT_ONLY = True`, about 3 h each on a Kaggle T4).
4. Compare LoRA rank (8 / 16 / 32) and training-set size (500 / 1000 / 2000 / 9800) to show how much
   data and capacity the gain actually needs. The v3/v4 result suggests that a few hundred items may already be
   enough for the style.
5. **Other Backbone Model.** A larger base model (e.g. Qwen2.5-Omni-7B,Qwen2.5-VL). More
   knowledge for the "significance" questions.
