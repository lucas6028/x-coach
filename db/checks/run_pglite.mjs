// LOCAL DEV ONLY: the Docker-free twin of run_local.sh. Same steps -- Supabase shim, every
// migration in filename order, re-apply the newest, then each *_check.sql -- on PGlite (Postgres
// compiled to WASM, running inside Node). Exists because `docker pull` fails from non-interactive
// sessions on Windows (the credential helper needs a logon session), while Node always works.
//
//   npm install --prefix db/checks/.pglite @electric-sql/pglite   # once; .pglite/ is gitignored
//   node db/checks/run_pglite.mjs
//
// The check files are written for psql. The only psql features they use are emulated here:
// `\set NAME 'value'` (with '' for a quote), a line that is exactly `:NAME`, and `\echo 'text'`.
// RLS is enforced for real: the checks `set role authenticated`, which PGlite honours.
import { createRequire } from "node:module";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const requireLocal = createRequire(join(here, ".pglite", "package.json"));

let PGlite;
try {
  ({ PGlite } = requireLocal("@electric-sql/pglite"));
} catch {
  console.error("PGlite is not installed. Run once:");
  console.error("  npm install --prefix db/checks/.pglite @electric-sql/pglite");
  process.exit(2);
}

const extensions = {};
try {
  extensions.pgcrypto = requireLocal("@electric-sql/pglite/contrib/pgcrypto").pgcrypto;
} catch {
  // gen_random_uuid() is core since PG13; without the contrib build the one
  // `create extension pgcrypto` line is stripped instead (see plain()).
}
const db = new PGlite({ extensions });

async function run(label, sql) {
  try {
    await db.exec(sql);
  } catch (err) {
    console.error(`FAILED in ${label}: ${err.message}`);
    if (err.where) console.error(`  where: ${err.where}`);
    process.exit(1);
  }
}

function plain(sql) {
  return extensions.pgcrypto ? sql : sql.replace(/create extension if not exists pgcrypto;/gi, "");
}

async function runCheck(label, text) {
  const macros = {};
  let chunk = [];
  const flush = async () => {
    if (chunk.join("").trim()) await run(label, chunk.join("\n"));
    chunk = [];
  };
  for (const line of text.split(/\r?\n/)) {
    const set = line.match(/^\\set (\w+) '(.*)'$/);
    if (set) {
      macros[set[1]] = set[2].replace(/''/g, "'");
      continue;
    }
    const echo = line.match(/^\\echo '(.*)'$/);
    if (echo) {
      await flush();
      console.log(echo[1]);
      continue;
    }
    const use = line.match(/^:(\w+)$/);
    if (use) {
      if (!(use[1] in macros)) {
        console.error(`FAILED in ${label}: unknown macro :${use[1]}`);
        process.exit(1);
      }
      chunk.push(macros[use[1]]);
      continue;
    }
    chunk.push(line);
  }
  await flush();
}

await run("shim", readFileSync(join(here, "00_local_supabase_shim.sql"), "utf8"));

const migDir = join(root, "db", "migrations");
const migrations = readdirSync(migDir).filter((f) => f.endsWith(".sql")).sort();
for (const f of migrations) {
  console.log(`apply ${f}`);
  await run(f, plain(readFileSync(join(migDir, f), "utf8")));
}
const latest = migrations[migrations.length - 1];
console.log(`re-apply ${latest}`);
await run(latest, plain(readFileSync(join(migDir, latest), "utf8")));

for (const f of readdirSync(here).filter((f) => f.endsWith("_check.sql")).sort()) {
  console.log(`check ${f}`);
  await runCheck(f, readFileSync(join(here, f), "utf8"));
}
console.log("OK");
