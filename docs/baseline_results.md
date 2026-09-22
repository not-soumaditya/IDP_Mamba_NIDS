# Baseline Model Results — UNSW-NB15

**Model:** RandomForest  
**Dataset:** UNSW-NB15 (flat features, non-sequential)  
**Training time:** 2.79s

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7539 |
| Macro F1 | 0.4721 |
| Weighted F1 | 0.7227 |
| Inference time | 343.76 µs/sample |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Analysis | 0.0000 | 0.0000 | 0.0000 | 2000 |
| Backdoor | 0.9836 | 0.0344 | 0.0664 | 1746 |
| DoS | 0.3423 | 0.5513 | 0.4223 | 12264 |
| Exploits | 0.7453 | 0.6186 | 0.6761 | 33393 |
| Fuzzers | 0.6618 | 0.1129 | 0.1929 | 18184 |
| Generic | 0.9519 | 0.9832 | 0.9673 | 40000 |
| Normal | 0.7458 | 0.9856 | 0.8491 | 56000 |
| Reconnaissance | 0.9297 | 0.7174 | 0.8099 | 10491 |
| Shellcode | 0.4717 | 0.5287 | 0.4985 | 1133 |
| Worms | 0.8571 | 0.1385 | 0.2384 | 130 |
