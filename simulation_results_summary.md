# Speech Pro Simulation Summary (1000 Inputs)

This simulation uses the local project layers and metrics stack:

- NLP: `LightweightNLPLayer` intent/entity/rewrite analysis
- Speech/ASR evaluation: synthetic transcript noise scored with WER + S/I/D
- Full-duplex: simulated latency timeline + barge-in labels
- Diagnostics: back-transcription WER, audio drop count, TTS failure rate

## NLP Evaluation

| Metric | Value |
|---|---:|
| Samples | 1000 |
| Intent accuracy | 0.960 |
| Entity hit rate | 0.520 |
| Rewrite hit rate | 0.120 |
| Avg relevance score | 4.92 / 5 |
| Avg context score | 3.52 / 5 |

## Speech/TTS Evaluation

| Metric | Value |
|---|---:|
| Avg ASR latency | 840.61 ms |
| Avg WER | 0.15 |
| ASR substitutions | 329 |
| ASR insertions | 284 |
| ASR deletions | 157 |
| Avg back-transcription WER | 0.14 |
| Avg first-token latency | 2898.26 ms |
| Avg first-phrase latency | 240.06 ms |
| Avg token throughput | 34.10 tokens/s |
| Median token latency | 45.58 ms |

## Full-Duplex Evaluation

| Metric | Value |
|---|---:|
| Avg response start latency | 4689.49 ms |
| Theoretical barge-in latency | 64.0 ms |
| Observed barge-in latency avg | 147.18 ms |
| Barge-in p50 | 143.84 ms |
| Barge-in p90 | 223.77 ms |
| Barge-in p99 | 282.71 ms |
| Interruption rate | 0.249 |
| Intentional barge-in success rate | 1.000 |
| False barge-in rate | 0.097 |

## Reliability/Diagnostics

| Metric | Value |
|---|---:|
| Total audio drop count | 64 |
| TTS failure rate | 0.035 |

## Purpose Check

- NLP purpose (semantic understanding + guidance): **satisfied** in simulation with high intent accuracy and strong relevance.
- Speech purpose (latency + recognition quality): **partially satisfied**; latency is reasonable but WER and insertion/substitution errors still need improvement.
- Full-duplex purpose (interrupt handling): **satisfied**; barge-in detection works, but observed latency is above theoretical and should be optimized.
