#!/usr/bin/env bash
# LOCAL DEV ONLY: apply every migration in db/migrations/ to a throwaway postgres:16 container
# behind the Supabase shim, re-apply the newest one to prove it is re-runnable, then run the RLS
# checks. Exits non-zero on the first SQL error or failed assertion. Needs Docker.
#
#   bash db/checks/run_local.sh
#
# Nothing here touches the Supabase project; the container is removed on exit.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
name="xcoach-migration-check"

docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" -e POSTGRES_PASSWORD=postgres postgres:16-alpine >/dev/null
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

# TCP, not the socket: the image's init phase runs a socket-only server, so 127.0.0.1 answering
# means the real server is up.
until docker exec "$name" pg_isready -h 127.0.0.1 -U postgres >/dev/null 2>&1; do sleep 1; done

run() { docker exec -i "$name" psql -h 127.0.0.1 -U postgres -d postgres -v ON_ERROR_STOP=1 -q "$@"; }

run < "$here/00_local_supabase_shim.sql"

latest=""
for f in "$root"/db/migrations/*.sql; do
    echo "apply $(basename "$f")"
    run < "$f"
    latest="$f"
done

echo "re-apply $(basename "$latest")"
run < "$latest"

for check in "$here"/*_check.sql; do
    echo "check $(basename "$check")"
    run < "$check"
done

echo "OK"
