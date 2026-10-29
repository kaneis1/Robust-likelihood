# Rewrite acceptance

Run started 2026-10-29T10:32:56Z: 235 clean items, 3262 rewrite units, 3933 calls to `claude-fable-5-1` (temperature sent: 1.0), 314,493 input and 127,399 output tokens.

A rewrite is kept if its regex slot tuple is unchanged and NLI entailment is at least 0.8 both ways (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`); pad is checked by regex only. Up to 3 attempts per unit.

| Kind | Accepted | Units | Rate |
|---|---|---|---|
| hpara1 | 459 | 463 | 99.1% |
| hpara2 | 460 | 463 | 99.4% |
| hpara3 | 459 | 463 | 99.1% |
| hzh | 443 | 463 | 95.7% |
| pad | 229 | 235 | 97.4% |
| para1 | 229 | 235 | 97.4% |
| para2 | 230 | 235 | 97.9% |
| typo | 229 | 235 | 97.4% |
| verbose | 202 | 235 | 86.0% |
| zh | 202 | 235 | 86.0% |

Accepted rewrites identical to their original (not counting Chinese, whose back-translation may match): 1 (u:SNG02148:para2).
