# AgentRoute: completed quality-panel results

1540 held-out method/question/seed evaluations; MATH and SVAMP excluded. Physical transport is reported separately.

## MMLU

| Method | Correct / N | Accuracy (95% Wilson CI) | Mean online tokens |
|---|---:|---:|---:|
| AgentRoute | 63/100 | 63.0% (53.2–71.8) | 2,726.8 |
| Dense | 63/100 | 63.0% (53.2–71.8) | 2,976.7 |
| No communication | 64/100 | 64.0% (54.2–72.7) | 2,344.6 |
| Single agent | 58/100 | 58.0% (48.2–67.2) | 354.2 |
| SafeSieve | 62/100 | 62.0% (52.2–70.9) | 5,302.3 |
| CARD | 61/100 | 61.0% (51.2–70.0) | 5,199.9 |
| SelfOrg | 60/100 | 60.0% (50.2–69.1) | 7,933.8 |
| AgentNet | 76/100 | 76.0% (66.8–83.3) | 5,215.4 |

Positive quality change favors AgentRoute; positive token saving means AgentRoute used fewer tokens.

| Reference | Accuracy difference (pp) | Conservative 95% CI (pp) | Holm-adjusted p | Token saving (95% paired bootstrap CI) |
|---|---:|---:|---:|---:|
| Dense | +0.0 | -7.7 to 7.7 | 1 | 8.4% (4.9–11.5) |
| No communication | -1.0 | -10.4 to 8.5 | 1 | -16.3% (-21.4–-10.5) |
| Single agent | +5.0 | -8.6 to 18.1 | 1 | -669.8% (-743.5–-590.5) |
| SafeSieve | +1.0 | -11.9 to 13.8 | 1 | 48.6% (43.4–54.1) |
| CARD | +2.0 | -9.8 to 13.6 | 1 | 47.6% (41.9–53.6) |
| SelfOrg | +3.0 | -9.9 to 15.6 | 1 | 65.6% (59.8–71.0) |
| AgentNet | -13.0 | -24.6 to -0.1 | 0.01645 | 47.7% (41.0–54.5) |

Initial-to-final diagnostic: 11/74 questions with a correct initial candidate ended incorrectly. This does not identify aggregation as the sole cause.

## BBH

| Method | Correct / N | Accuracy (95% Wilson CI) | Mean online tokens |
|---|---:|---:|---:|
| AgentRoute | 11/50 | 22.0% (12.8–35.2) | 3,112.1 |
| Dense | 12/50 | 24.0% (14.3–37.4) | 3,567.4 |
| No communication | 10/50 | 20.0% (11.2–33.0) | 2,334.4 |
| Single agent | 9/50 | 18.0% (9.8–30.8) | 346.4 |
| SafeSieve | 11/50 | 22.0% (12.8–35.2) | 4,325.9 |
| CARD | 10/50 | 20.0% (11.2–33.0) | 4,242.8 |
| SelfOrg | 8/50 | 16.0% (8.3–28.5) | 6,000.1 |
| AgentNet | 9/50 | 18.0% (9.8–30.8) | 4,991.5 |

Positive quality change favors AgentRoute; positive token saving means AgentRoute used fewer tokens.

| Reference | Accuracy difference (pp) | Conservative 95% CI (pp) | Holm-adjusted p | Token saving (95% paired bootstrap CI) |
|---|---:|---:|---:|---:|
| Dense | -2.0 | -15.3 to 11.8 | 1 | 12.8% (9.2–16.4) |
| No communication | +2.0 | -8.4 to 12.1 | 1 | -33.3% (-40.4–-25.4) |
| Single agent | +4.0 | -8.1 to 15.3 | 1 | -798.4% (-893.4–-715.0) |
| SafeSieve | +0.0 | -14.9 to 14.9 | 1 | 28.1% (19.0–37.3) |
| CARD | +2.0 | -8.4 to 12.1 | 1 | 26.7% (17.5–35.8) |
| SelfOrg | +6.0 | -7.4 to 18.2 | 1 | 48.1% (42.2–54.2) |
| AgentNet | +4.0 | -15.4 to 22.6 | 1 | 37.7% (29.2–45.9) |

Initial-to-final diagnostic: 1/12 questions with a correct initial candidate ended incorrectly. This does not identify aggregation as the sole cause.

## Interpretation limits

- Fixed selected panels: MMLU 100 questions and BBH 50 questions, not full benchmark populations.
- Baseline implementations are documented adaptations; native-paper reproduction is not established.
- Accuracy and token comparisons are paired by question. The primary family has seven Holm-corrected comparisons per dataset.
- The 2 percentage-point noninferiority margin is a screening threshold; inconclusive does not establish preservation.
- Quality-run latency cohorts differ in telemetry overhead. No maximum capacity or universal bottleneck-free claim.
- Physical transport replay is a separate subsystem experiment; MATH-based live anchors were excluded by the user.
- Inference call tokens include recorded online model calls. Startup, fitting, warmup and failed attempts belong in the separate cost ledger, not these held-out token totals.

Rescoring: all recorded quality/format outcomes matched an offline recomputation with the frozen scorer. Raw evidence hashes were checked before analysis. Full ablation, seed-cluster, timing-cohort and answer-survival results are in QUALITY_STATISTICS.json.
