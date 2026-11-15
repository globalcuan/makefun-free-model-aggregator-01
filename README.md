# makefun-free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/globalcuan/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/globalcuan/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator server in pure Python 3.13 (standard library only). It reads the trial brief, probes ten simulated free-tier providers, registers the exact free-model catalog, and serves the OpenAI Chat Completions and Anthropic Messages APIs — JSON and streaming — routing every request to the highest-rated free model that meets its requirements (tools, vision, JSON mode, minimum context) and failing over to the next candidate whenever a provider refuses.

## Run it

```sh
python3 main.py --list   # print the catalog document
python3 main.py          # serve (inside a trial: MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL set)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=phpx09k3&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
