# Fine-tuned small model vs frontier API: clinical note extraction

## Accuracy

| model | exact match (ID) | exact match (OOD) | field acc (ID) | field acc (OOD) | valid JSON (OOD) |
|---|---|---|---|---|---|
| haiku-5.5 | 94.3% | 96.0% | 99.6% | 99.1% | 99.3% |
| qwen3-8b-base | 17.7% | 19.7% | 92.7% | 92.5% | 100.0% |
| qwen3-8b-ft | 99.7% | 15.7% | 100.0% | 90.7% | 100.0% |
| sonnet-5.5 | 93.0% | 97.3% | 99.5% | 99.5% | 99.7% |

ID = same note styles as training, new patients. OOD = new note styles, Latin abbreviations, Celsius, follow-up in days, and conditions/drugs never seen in training.

## Generalisation: recall on seen vs never-seen vocabulary (OOD split)

| model | ICD-10 seen | ICD-10 held-out | meds seen | meds held-out | false positives / note |
|---|---|---|---|---|---|
| haiku-5.5 | 96.8% | 98.5% | 98.5% | 100.0% | 0.03 |
| qwen3-8b-base | 48.3% | 43.4% | 97.4% | 97.9% | 1.12 |
| qwen3-8b-ft | 99.1% | 35.8% | 99.7% | 100.0% | 0.58 |
| sonnet-5.5 | 97.1% | 100.0% | 99.0% | 100.0% | 0.02 |

## Cost and speed

| model | pricing | cost / 1,000 notes | cost / 1,000 *correct* notes (OOD) | mean tokens in / out | latency p50 / p95 | notes / hour |
|---|---|---|---|---|---|---|
| haiku-5.5 | api:claude-haiku-5-5 | $0.38 | $0.40 | 728 / 623 | 2.83s / 3.93s | 9,835 |
| qwen3-8b-base | gpu | $0.084 | $0.43 | 517 / 250 | 4.93s / 6.52s | 42,090 |
| qwen3-8b-ft | gpu | $0.061 | $0.39 | 517 / 175 | 3.63s / 4.85s | 57,251 |
| sonnet-5.5 | api:claude-sonnet-5-5 | $4.99 | $5.13 | 728 / 354 | 2.48s / 4.13s | 10,644 |

## When does self-hosting beat the API?

- **qwen3-8b-base** vs **haiku-5.5**: self-hosting is cheaper above **220,432 notes/day**
- **qwen3-8b-base** vs **sonnet-5.5**: self-hosting is cheaper above **16,963 notes/day**
- **qwen3-8b-ft** vs **haiku-5.5**: self-hosting is cheaper above **220,432 notes/day**
- **qwen3-8b-ft** vs **sonnet-5.5**: self-hosting is cheaper above **16,963 notes/day**

Assumptions: one always-on GPU at $3.52/hour (set with --gpu-hourly), training 1.5 GPU-hours amortised over 30 days, API list prices. Self-hosted throughput is what vLLM measured at the concurrency used in the run.
