# default

A few basic tests to check that the policy is running properly.

This is cyclotron's default locomotion benchmark: what `./cyclotron.sh --benchmark <policy>` runs, and what
`./cyclotron.sh --view <policy>` opens, when no other benchmark is given. It is laid out in the benchmark format
of [humanoid-policy-viewer](https://github.com/menloresearch/humanoid-policy-viewer). Each row of
`data/<config>/test.jsonl` is one test scenario: velocity commands over time, timed pushes and the floor
friction. `benchmark.yaml` says how often each test runs, how runs are randomised and scored, and which numbers
become leaderboard tasks.

It started as a copy of the viewer's `benchmarks/default` at commit `a90090b` and is maintained in cyclotron
from then on: later changes to the viewer's copy do not reach it. With the viewer at `4e62f70` it moved to
scoring protocol 2 (version 2): its tasks use `survival_rate@2` and `survival_time@2`, which judge a run by how
it ends. The tests themselves are unchanged.

## Run it

```
./cyclotron.sh --benchmark <org>/<policy>
```

## Change it

```
./cyclotron.sh --view <org>/<policy>    # edit in the viewer; saves go straight into this folder
git diff benchmarks/default
```

Bump `version` in `benchmark.yaml` whenever a change can move a score.
