#!/usr/bin/env bash
#
# run_campaign.sh — campaign runner for the differential cargo-fuzz targets.
#
# Runs a given target with libFuzzer -fork=N (N = CPUs by default), writes a
# log, crash artifacts, and a final rc file, and stops when coverage has not
# grown for PLATEAU_SECS (default 1800) or after MAX_SECS (default 7200),
# whichever comes first. Prints a one-line summary (execs, cov, ft, crashes,
# stop reason).
#
# Usage:
#   fuzz/run_campaign.sh <target> [corpus_dir] [max_secs] [plateau_secs] [forks]
#
# Outputs (under fuzz/campaign/<target>/):
#   log.txt      full libFuzzer output
#   crashes/    crash artifacts (artifact_prefix)
#   corpus/     the working corpus (the seed corpus is copied in, or the given
#               corpus directory is used directly when it exists)
#   rc          0 = clean stop, 1 = crash found, 2 = infrastructure error
#   summary.txt  the one-line summary
#
# Prerequisites: nightly toolchain per rust-toolchain.toml. cargo-fuzz is NOT
# required: the build step reproduces cargo-fuzz's sancov build directly (see
# the build section for why -Clink-dead-code and ASan are dropped). On macOS
# hosts with the shared-build courtesy lock, the build step is wrapped in
# `lockf` when available (CARGO_LOCKF_PATH, default /tmp/cass-cargo.lock);
# the fuzz run itself does not compile anything and does not hold the lock.
#
# Coverage plateau detection: polls the log for libFuzzer's `cov:` / `ft:`
# counters; when the best (cov, ft) pair has not improved for PLATEAU_SECS,
# the fuzzer is stopped with SIGINT (libFuzzer's graceful stop, which merges
# fork-mode corpora). MAX_SECS is passed to libFuzzer as -max_total_time as a
# hard backstop.

set -uo pipefail

TARGET=${1:?usage: run_campaign.sh <target> [corpus_dir] [max_secs] [plateau_secs] [forks]}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORPUS_DIR=${2:-}
MAX_SECS=${3:-7200}
PLATEAU_SECS=${4:-1800}
FORKS=${5:-}

OUT_DIR="$REPO_ROOT/fuzz/campaign/$TARGET"
LOG="$OUT_DIR/log.txt"
CRASH_DIR="$OUT_DIR/crashes"
RC_FILE="$OUT_DIR/rc"
SUMMARY_FILE="$OUT_DIR/summary.txt"

if [ -z "$FORKS" ]; then
    if command -v nproc >/dev/null 2>&1; then
        FORKS=$(nproc)
    else
        FORKS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
    fi
fi
if [ -z "$CORPUS_DIR" ] || [ ! -d "$CORPUS_DIR" ]; then
    CORPUS_DIR="$REPO_ROOT/fuzz/corpus/$TARGET"
fi
# The working corpus must be writable (libFuzzer adds units to the first
# corpus directory). Seed from the requested dir into the campaign corpus.
WORK_CORPUS="$OUT_DIR/corpus"
mkdir -p "$WORK_CORPUS" "$CRASH_DIR"
if [ -n "$(ls -A "$WORK_CORPUS" 2>/dev/null)" ]; then
    : # resume an existing campaign corpus
elif [ -d "$CORPUS_DIR" ] && [ -n "$(ls -A "$CORPUS_DIR" 2>/dev/null)" ]; then
    cp -R "$CORPUS_DIR"/. "$WORK_CORPUS"/
fi

cd "$REPO_ROOT" || exit 2

# --- build (lockf-wrapped where the courtesy lock exists) ----------------
# This reproduces exactly what `cargo fuzz build` runs internally: the sancov
# instrumentation flags libFuzzer needs plus `--cfg fuzzing`. Two of
# cargo-fuzz's own flags are dropped on purpose:
#   * -Clink-dead-code: it forces EVERY extern in every rlib to resolve, and
#     the dependency chain asupersync -> sysinfo -> objc2-io-kit declares
#     IOURLWriteDataAndPropertiesToResource, which no macOS SDK tbd provides
#     (verified against the SDK on macOS 27). Plain builds dead-strip that
#     unreferenced extern and link cleanly, so the workspace's normal
#     binaries are unaffected; only link-dead-code builds break.
#   * -Zsanitizer=address: the differential signal is panic-based (engine
#     divergence vs stock SQLite), not memory-error-based, and ASan roughly
#     halves exec/s for no signal gain here.
# The resulting binary is an ordinary libFuzzer binary: libfuzzer-sys links
# the runtime, sancov provides the coverage hooks, and every libFuzzer CLI
# flag (-fork, -max_total_time, -minimize_crash, ...) behaves as usual.
#
# `--target <host triple>` is REQUIRED, not cosmetic: it is what makes cargo
# scope RUSTFLAGS to target artifacts only. Without it the sancov flags also
# instrument host build scripts, whose executables then fail to link
# (__sanitizer_cov_* is only provided by the libFuzzer runtime in the final
# binary). This mirrors cargo-fuzz's own invocation.
SANCOV_FLAGS="-Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=4 -Cllvm-args=-sanitizer-coverage-inline-8bit-counters -Cllvm-args=-sanitizer-coverage-pc-table -Cllvm-args=-sanitizer-coverage-trace-compares"
# Assemble without a leading space when RUSTFLAGS starts unset: the resulting
# string must be byte-identical to a proven manual build's RUSTFLAGS, or
# cargo's fingerprint hash differs and the runner rebuilds the world (~30 min)
# instead of reusing the existing incremental artifacts.
RUSTFLAGS="${RUSTFLAGS:-}${RUSTFLAGS:+ }$SANCOV_FLAGS --cfg fuzzing"
export RUSTFLAGS
TRIPLE=$(rustc -vV | sed -n 's/^host: //p')
if [ -z "$TRIPLE" ]; then
    echo "cannot determine host triple via rustc -vV" >&2
    echo 2 > "$RC_FILE"
    exit 2
fi
BUILD_CMD=(cargo build --manifest-path "$REPO_ROOT/fuzz/Cargo.toml" --target "$TRIPLE" --release --config profile.release.debug=true --bin "$TARGET")
if command -v lockf >/dev/null 2>&1; then
    if lockf -k "${CARGO_LOCKF_PATH:-/tmp/cass-cargo.lock}" "${BUILD_CMD[@]}" >"$OUT_DIR/build.txt" 2>&1; then
        :
    else
        echo "cargo fuzz build failed; see $OUT_DIR/build.txt" >&2
        echo 2 > "$RC_FILE"
        exit 2
    fi
else
    if "${BUILD_CMD[@]}" >"$OUT_DIR/build.txt" 2>&1; then
        :
    else
        echo "cargo fuzz build failed; see $OUT_DIR/build.txt" >&2
        echo 2 > "$RC_FILE"
        exit 2
    fi
fi

# --- locate the built binary (cargo-fuzz's target layout moved across
# versions: fuzz/target/<triple>/release|debug/<target> or fuzz/target/...) --
BIN=""
NEWEST=""
while IFS= read -r candidate; do
    if [ -z "$NEWEST" ] || [ "$candidate" -nt "$NEWEST" ]; then
        NEWEST=$candidate
    fi
done < <(find "$REPO_ROOT/fuzz/target" -type f -name "$TARGET" -perm -111 2>/dev/null)
BIN=$NEWEST
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "cannot locate built fuzz binary for $TARGET" >&2
    echo 2 > "$RC_FILE"
    exit 2
fi
echo "binary: $BIN" >&2

# --- run with plateau polling ---------------------------------------------
# The clock starts here, after the build: the plateau window, the max-secs
# cap and WALL_SECS measure fuzzing only, so a cold ~30 min build neither
# eats the budget nor counts as a coverage plateau.
START_TS=$(date +%s)
: > "$LOG"
"$BIN" \
    -fork="$FORKS" \
    -max_total_time="$MAX_SECS" \
    -rss_limit_mb=4096 \
    -detect_leaks=0 \
    -max_len=128 \
    -print_final_stats=1 \
    -artifact_prefix="$CRASH_DIR/" \
    "$WORK_CORPUS" \
    >>"$LOG" 2>&1 &
FUZZ_PID=$!

stop_fuzzer() {
    # libFuzzer treats SIGINT as a graceful stop and merges fork corpora.
    kill -INT "$FUZZ_PID" 2>/dev/null
    for _ in $(seq 1 30); do
        kill -0 "$FUZZ_PID" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL "$FUZZ_PID" 2>/dev/null
    wait "$FUZZ_PID" 2>/dev/null
}

poll_log() {
    # Best (cov, ft) pair seen anywhere in the log so far.
    grep -oE 'cov: [0-9]+ ft: [0-9]+' "$LOG" 2>/dev/null \
        | tail -1 || true
}

BEST="none"
LAST_IMPROVED_TS=$START_TS
STOP_REASON=""
while kill -0 "$FUZZ_PID" 2>/dev/null; do
    sleep 30
    CURRENT=$(poll_log)
    if [ -n "$CURRENT" ] && [ "$CURRENT" != "$BEST" ]; then
        BEST=$CURRENT
        LAST_IMPROVED_TS=$(date +%s)
    fi
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    if [ "$ELAPSED" -ge "$MAX_SECS" ]; then
        STOP_REASON="max-secs"
        break
    fi
    if [ $((NOW - LAST_IMPROVED_TS)) -ge "$PLATEAU_SECS" ]; then
        STOP_REASON="coverage-plateau"
        stop_fuzzer
        break
    fi
done
if [ -z "$STOP_REASON" ]; then
    # The fuzzer went down on its own: either libFuzzer's own -max_total_time
    # fired before our 30s poller noticed, or it exited early (fork-mode
    # parents exit as soon as a worker reports a crash).
    NOW=$(date +%s)
    if [ $((NOW - START_TS)) -ge "$MAX_SECS" ]; then
        STOP_REASON="max-secs"
    else
        STOP_REASON="fuzzer-exited-early"
    fi
fi
wait "$FUZZ_PID" 2>/dev/null
FUZZ_RC=$?
END_TS=$(date +%s)
WALL_SECS=$((END_TS - START_TS))

# --- summary ---------------------------------------------------------------
COV=$(grep -oE 'cov: [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)
FT=$(grep -oE 'ft: [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)
# Best-effort executed-units count: fork mode's log shape varies across
# libFuzzer versions. Try the final-stats counter, then the largest #N pulse.
EXECS=$(grep -oE 'stat::number_of_executed_units: [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+' || true)
if [ -z "$EXECS" ]; then
    # Fork-mode pulses look like `#503355: cov: ...` (colon directly after
    # the count), so do not require whitespace after the digits.
    EXECS=$(grep -oE '^#[0-9]+' "$LOG" | grep -oE '[0-9]+' | sort -n | tail -1)
fi
if [ -z "$EXECS" ]; then
    EXECS="n/a"
fi
CRASHES=$(ls -1 "$CRASH_DIR" 2>/dev/null | wc -l | tr -d ' ')
if [ "$CRASHES" -gt 0 ]; then
    RC=1
    STOP_REASON="$STOP_REASON+crashes"
else
    # libFuzzer exits non-zero when a fork run is cut short by SIGINT (the
    # graceful plateau stop) or by its own -max_total_time; both are clean
    # campaign stops. Anything else that exits non-zero is surfaced as-is.
    if [ "$FUZZ_RC" -eq 0 ]; then
        RC=0
    elif [ "$STOP_REASON" = "max-secs" ] || [ "$STOP_REASON" = "coverage-plateau" ]; then
        RC=0
    else
        RC=$FUZZ_RC
    fi
fi

SUMMARY="target=$TARGET execs=$EXECS cov=$COV ft=$FT crashes=$CRASHES stop=$STOP_REASON wall_secs=$WALL_SECS forks=$FORKS corpus=$WORK_CORPUS"
echo "$SUMMARY" | tee "$SUMMARY_FILE"
echo "$RC" > "$RC_FILE"
exit "$RC"
