// Offline compatibility bridge. HPV owns format validation and scoring semantics;
// cyclotron owns Hub access and publication. Never call an HPV HF operation here.
import { readFileSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [hpv, command, directory, commit] = process.argv.slice(2);
const load = (path) => import(pathToFileURL(resolve(hpv, path)).href);
try {
  if (command === 'policy') {
    const { findLocalPolicy } = await load('scripts/localPolicy.mjs');
    const { policyFileHashes } = await load('scripts/benchmarkProvenance.mjs');
    const policy = findLocalPolicy(directory);
    console.log(JSON.stringify({ ...policy, hashes: policyFileHashes(policy.policyDir, policy.primary) }));
  } else if (command === 'inspect') {
    const { loadBenchmarkDir, cardFrontMatter, licenseFields } = await load('scripts/benchmarkDir.mjs');
    const { checkTests } = await load('src/benchmark/check.js');
    const { compatibilityProblems, buildEvalYaml } = await load('src/benchmark/settings.js');
    const { planCells } = await load('src/benchmark/run.js');
    // These exported pure functions have no top-level I/O. Importing them does not
    // invoke publish/pull or any HF client; keep HPV's canonicalization verbatim.
    const { scoredContentHash, withTestedCommit } = await load('scripts/hfBenchmark.mjs');
    const { BENCHMARK_FILES } = await load('scripts/hfDataset.mjs');
    const { collectProvenance, onPublicRepo } = await load('scripts/benchmarkProvenance.mjs');
    const loaded = loadBenchmarkDir(directory);
    const { benchmark, rawBenchmark, rows } = loaded;
    const checks = checkTests(loaded);
    const problems = compatibilityProblems(benchmark);
    const name = rawBenchmark.title ?? benchmark.name;
    const tasks = buildEvalYaml(benchmark, { name, description: rawBenchmark.description ?? '' }).tasks;
    const plan = checks.errors.length ? { cells: [], tests: [] } : planCells(benchmark, rows);
    const walk = (dir, prefix = '') => readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const name = prefix + entry.name;
      return entry.isDirectory() && !entry.name.startsWith('.') ? walk(join(dir, entry.name), `${name}/`)
        : entry.isFile() && BENCHMARK_FILES.some((re) => re.test(name)) ? [name] : [];
    });
    const provenance = collectProvenance(hpv);
    console.log(JSON.stringify({
      schema: 'cyclotron-hpv-inspection/1', raw: rawBenchmark, rows, tasks,
      errors: checks.errors, warnings: checks.warnings, problems,
      cells: plan.cells, tests: plan.tests, scored_hash: scoredContentHash(rawBenchmark, rows),
      files: walk(directory), provenance, public: onPublicRepo(hpv, provenance.hpv.commit),
      card: cardFrontMatter(rows, {
        license: licenseFields(directory, rawBenchmark.license, benchmark.name),
        tags: rawBenchmark.tags, prettyName: name,
      }),
      benchmark_yaml: commit ? withTestedCommit(readFileSync(join(directory, 'benchmark.yaml'), 'utf8'), commit) : null,
    }));
  } else {
    throw new Error(`Unknown offline inspection command: ${command}`);
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
