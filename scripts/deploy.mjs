#!/usr/bin/env node
import { mkdtemp, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(scriptDir, '..');
const dist = join(root, 'dist');
const args = new Set(process.argv.slice(2));
const noBuild = args.has('--no-build');
const dryRun = args.has('--dry-run');
const branch = 'master';
const remote = 'origin';

function run(command, commandArgs, options = {}) {
  const result = spawnSync(command, commandArgs, {
    cwd: options.cwd || root,
    encoding: 'utf8',
    stdio: options.capture ? ['ignore', 'pipe', 'pipe'] : 'inherit'
  });

  if (result.status !== 0) {
    const rendered = [command, ...commandArgs].join(' ');
    if (options.capture) {
      if (result.stdout) process.stdout.write(result.stdout);
      if (result.stderr) process.stderr.write(result.stderr);
    }
    throw new Error(`Command failed: ${rendered}`);
  }

  return options.capture ? result.stdout.trim() : '';
}

function ensureRoot() {
  const actualRoot = run('git', ['rev-parse', '--show-toplevel'], { capture: true });
  if (resolve(actualRoot) !== root) {
    throw new Error(`Run this script from the blog repository root: ${root}`);
  }
}

async function main() {
  ensureRoot();

  if (!noBuild) {
    run(process.execPath, [join(root, 'scripts', 'build.mjs')]);
  }

  if (!existsSync(dist)) {
    throw new Error(`Missing build output: ${dist}. Run npm run build first or omit --no-build.`);
  }

  run('git', ['fetch', remote, branch]);

  const worktree = await mkdtemp(join(tmpdir(), 'msxf-pages-deploy-'));
  let addedWorktree = false;

  try {
    run('git', ['worktree', 'add', '--detach', worktree, `${remote}/${branch}`]);
    addedWorktree = true;

    const rsyncArgs = ['-a', '--delete', '--exclude', '.git', `${dist}/`, `${worktree}/`];
    if (dryRun) rsyncArgs.unshift('--dry-run');
    run('rsync', rsyncArgs);

    if (dryRun) {
      console.log(`[dry-run] Would deploy dist/ to ${remote}/${branch}.`);
      return;
    }

    run('git', ['add', '-A'], { cwd: worktree });
    const changed = run('git', ['status', '--porcelain'], { cwd: worktree, capture: true });

    if (!changed) {
      console.log(`No deployment changes for ${remote}/${branch}.`);
      return;
    }

    const sourceCommit = run('git', ['rev-parse', '--short', 'HEAD'], { capture: true });
    run('git', ['commit', '-m', `Deploy site from ${sourceCommit}`], { cwd: worktree });
    run('git', ['push', remote, `HEAD:${branch}`], { cwd: worktree });
    console.log(`Deployed dist/ to ${remote}/${branch}.`);
  } finally {
    if (addedWorktree) {
      run('git', ['worktree', 'remove', worktree]);
    } else {
      await rm(worktree, { recursive: true, force: true });
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
