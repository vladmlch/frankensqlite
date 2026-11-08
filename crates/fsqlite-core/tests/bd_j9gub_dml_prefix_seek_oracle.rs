//! bd-j9gub: DELETE/UPDATE with equality conjuncts covering a PREFIX of an
//! index's key columns must seek that index instead of full-scanning, with the
//! affected row set identical to stock SQLite's.
//!
//! * [`bd_j9gub_dml_prefix_seek_oracle`] — fixture-driven differential against
//!   stock SQLite (rusqlite): affected rows, `changes()`, RETURNING, triggers,
//!   AUTOINCREMENT, NULL probes, empty matches, numbered/anonymous
//!   placeholders and literals, TEXT-affinity keys, trailing-DESC indexes,
//!   non-prefix scan controls; memory and persistent databases.
//! * [`bd_j9gub_dml_prefix_seek_eqp_matches_stock`] — `EXPLAIN QUERY PLAN`
//!   renders the seek the emitted program performs, class-matching stock.
//! * [`bd_j9gub_dml_prefix_seek_must_not_walk_the_table`] — opcode-level
//!   keeper: the table cursor is never walked and the index read cursor is
//!   positioned by `SeekGE`.
//! * [`bd_j9gub_dml_prefix_seek_scaling`] — per-statement cost must not scale
//!   with table size (ignored; run with `--ignored --nocapture`).
#![recursion_limit = "512"]

use std::time::{Duration, Instant};

use fsqlite_core::connection::Connection;
use fsqlite_types::SqliteValue;
use rusqlite::types::Value;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    setup: Vec<String>,
    #[serde(default)]
    verify: Vec<String>,
    mutations: Vec<Mutation>,
}

#[derive(Deserialize)]
struct Mutation {
    sql: String,
    params: Vec<serde_json::Value>,
    changes: usize,
}

fn oracle_param(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::String(text) => Value::Text(text.clone()),
        serde_json::Value::Number(number) => number.as_i64().map_or_else(
            || Value::Real(number.as_f64().expect("finite numeric fixture")),
            Value::Integer,
        ),
        other => panic!("unsupported fixture parameter: {other}"),
    }
}

fn frank_value(value: &Value) -> SqliteValue {
    match value {
        Value::Null => SqliteValue::Null,
        Value::Integer(value) => SqliteValue::Integer(*value),
        Value::Real(value) => SqliteValue::Float(*value),
        Value::Text(value) => SqliteValue::from(value.as_str()),
        Value::Blob(value) => SqliteValue::Blob(value.clone().into()),
    }
}

async fn compare_query(
    frank: &Connection,
    stock: &rusqlite::Connection,
    sql: &str,
    params: &[Value],
    context: &str,
) {
    let bound: Vec<_> = params.iter().map(frank_value).collect();
    let mut actual: Vec<Vec<SqliteValue>> = frank
        .query_with_params(sql, &bound)
        .await
        .unwrap_or_else(|error| panic!("{context}: {sql}: {error}"))
        .iter()
        .map(|row| row.values().to_vec())
        .collect();
    let mut statement = stock.prepare(sql).expect("oracle prepare");
    let width = statement.column_count();
    let mut expected: Vec<Vec<SqliteValue>> = statement
        .query_map(rusqlite::params_from_iter(params.iter()), |row| {
            (0..width)
                .map(|column| row.get::<_, Value>(column).map(|value| frank_value(&value)))
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .expect("oracle query")
        .collect::<rusqlite::Result<_>>()
        .expect("oracle rows");
    // RETURNING order is unspecified. Keep duplicate rows, but compare as a
    // multiset so an index-ordered candidate scan may differ from rowid order.
    actual.sort_by_cached_key(|row| format!("{row:?}"));
    expected.sort_by_cached_key(|row| format!("{row:?}"));
    assert_eq!(actual, expected, "{context}: {sql}");
}

/// The composite-prefix schema shared by the plan-level tests: `a` leads both
/// indexes, so `WHERE a = ?` pins a one-column PREFIX of `t_ab(a, b)`.
const PREFIX_SETUP: &[&str] = &[
    "CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, n INTEGER, v TEXT)",
    "CREATE INDEX t_ab ON t(a,b)",
    "CREATE INDEX t_an ON t(a,n)",
    "INSERT INTO t(id,a,b,n,v) VALUES (1,1,1,7,'s'),(2,1,2,8,'s'),(3,1,9,7,'s'),(4,2,1,7,'s'),(5,NULL,1,7,'s'),(6,1,NULL,8,'s')",
];

async fn execute_setup(frank: &Connection, stock: &rusqlite::Connection, setup: &[&str]) {
    for sql in setup {
        frank.execute(sql).await.expect("FrankenSQLite fixture");
        stock.execute_batch(sql).expect("oracle fixture");
    }
}

#[test]
fn bd_j9gub_dml_prefix_seek_oracle() {
    asupersync::test_utils::run_test(|| async {
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("bd_j9gub_dml_prefix_seek_cases.json"))
                .expect("fixtures");
        assert!(!cases.is_empty(), "fixture file must not be empty");
        for case in &cases {
            for persistent in [false, true] {
                let dir = tempfile::tempdir().expect("temporary database directory");
                let path = if persistent {
                    dir.path()
                        .join("prefix.sqlite")
                        .to_string_lossy()
                        .into_owned()
                } else {
                    ":memory:".to_owned()
                };
                let frank = Connection::open(&path).await.expect("open FrankenSQLite");
                let stock = rusqlite::Connection::open_in_memory().expect("open oracle");
                let context = format!("{}; persistent={persistent}", case.name);
                for sql in &case.setup {
                    frank.execute(sql).await.expect("FrankenSQLite fixture");
                    stock.execute_batch(sql).expect("oracle fixture");
                }
                for mutation in &case.mutations {
                    let params: Vec<_> = mutation.params.iter().map(oracle_param).collect();
                    compare_query(&frank, &stock, &mutation.sql, &params, &context).await;
                    let expected_changes: i64 = stock
                        .query_row("SELECT changes()", [], |row| row.get(0))
                        .expect("oracle changes");
                    assert_eq!(
                        expected_changes,
                        i64::try_from(mutation.changes).expect("small fixture count"),
                        "{context}: fixture changed unexpectedly"
                    );
                    compare_query(&frank, &stock, "SELECT changes()", &[], &context).await;
                    for verify in &case.verify {
                        compare_query(&frank, &stock, verify, &[], &context).await;
                    }
                    compare_query(&frank, &stock, "PRAGMA integrity_check", &[], &context).await;
                }
                frank.close().await.expect("close FrankenSQLite");
                if persistent {
                    let reopened = Connection::open(&path).await.expect("reopen FrankenSQLite");
                    for verify in &case.verify {
                        compare_query(&reopened, &stock, verify, &[], &context).await;
                    }
                    compare_query(
                        &reopened,
                        &stock,
                        "PRAGMA integrity_check",
                        &[],
                        &context,
                    )
                    .await;
                    reopened.close().await.expect("close reopened database");
                }
            }
        }
    });
}

async fn frank_eqp_detail(frank: &Connection, sql: &str, params: &[SqliteValue]) -> String {
    let rows = frank
        .query_with_params(&format!("EXPLAIN QUERY PLAN {sql}"), params)
        .await
        .unwrap_or_else(|error| panic!("EXPLAIN QUERY PLAN {sql}: {error}"));
    rows.iter()
        .map(|row| row.values())
        .find_map(|values| match &values[3] {
            SqliteValue::Text(detail)
                if detail.starts_with("SCAN ") || detail.starts_with("SEARCH ") =>
            {
                Some(detail.to_string())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no plan row for {sql}: {rows:?}"))
}

fn stock_eqp_detail(stock: &rusqlite::Connection, sql: &str, placeholders: usize) -> String {
    let mut statement = stock
        .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
        .expect("stock EXPLAIN QUERY PLAN prepare");
    // EXPLAIN QUERY PLAN never reads the bound values, but rusqlite still
    // requires the statement's full parameter count to be supplied.
    let dummies = vec![Value::Integer(0); placeholders];
    let mut rows = statement
        .query(rusqlite::params_from_iter(dummies))
        .expect("stock plan rows");
    let mut detail = None;
    while let Ok(Some(row)) = rows.next() {
        let text: String = row.get(3).expect("stock plan detail");
        if text.starts_with("SCAN ") || text.starts_with("SEARCH ") {
            detail = Some(text);
        }
    }
    detail.unwrap_or_else(|| panic!("stock has no plan row for {sql}"))
}

#[test]
fn bd_j9gub_dml_prefix_seek_eqp_matches_stock() {
    asupersync::test_utils::run_test(|| async {
        let frank = Connection::open(":memory:").await.expect("open FrankenSQLite");
        let stock = rusqlite::Connection::open_in_memory().expect("open oracle");
        execute_setup(&frank, &stock, PREFIX_SETUP).await;
        // (sql, placeholder count, expected FrankenSQLite detail). Index choice
        // is engine-specific — compare the SEARCH/SCAN class against stock and
        // the exact detail only against our own deterministic plan.
        let shapes: &[(&str, usize, &str)] = &[
            (
                "DELETE FROM t WHERE a=3",
                0,
                "SEARCH t USING INDEX t_ab (a=?)",
            ),
            (
                "DELETE FROM t WHERE a=?1",
                1,
                "SEARCH t USING INDEX t_ab (a=?)",
            ),
            (
                "UPDATE t SET n=n+1 WHERE a=?1",
                1,
                "SEARCH t USING INDEX t_ab (a=?)",
            ),
            (
                "DELETE FROM t WHERE a=?1 AND b>?2",
                2,
                "SEARCH t USING INDEX t_ab (a=?)",
            ),
            (
                "DELETE FROM t WHERE a=3 AND b=4",
                0,
                "SEARCH t USING INDEX t_ab (a=? AND b=?)",
            ),
            (
                "UPDATE t SET n=n+1 WHERE a=?1 AND b=?2",
                2,
                "SEARCH t USING INDEX t_ab (a=? AND b=?)",
            ),
            (
                "DELETE FROM t WHERE a=3 AND n=7",
                0,
                "SEARCH t USING INDEX t_an (a=? AND n=?)",
            ),
            ("DELETE FROM t WHERE b=3", 0, "SCAN t (DELETE)"),
            ("DELETE FROM t WHERE n=5", 0, "SCAN t (DELETE)"),
            (
                "UPDATE t SET n=n+1 WHERE b=3",
                0,
                "SCAN t (UPDATE)",
            ),
        ];
        for (sql, placeholders, expected_detail) in shapes {
            let dummies = vec![SqliteValue::Integer(0); *placeholders];
            let detail = frank_eqp_detail(&frank, sql, &dummies).await;
            assert_eq!(detail, *expected_detail, "EXPLAIN QUERY PLAN {sql}");
            let stock_detail = stock_eqp_detail(&stock, sql, *placeholders);
            let frank_searches = detail.starts_with("SEARCH t USING INDEX");
            let stock_searches = stock_detail.starts_with("SEARCH t USING");
            assert_eq!(
                frank_searches, stock_searches,
                "{sql}: access class diverged: fsqlite={detail:?} stock={stock_detail:?}"
            );
        }
        frank.close().await.expect("close FrankenSQLite");
    });
}

fn integer(value: &SqliteValue) -> i64 {
    match value {
        SqliteValue::Integer(value) => *value,
        other => panic!("expected EXPLAIN integer, got {other:?}"),
    }
}

fn text(value: &SqliteValue) -> &str {
    match value {
        SqliteValue::Text(value) => value.as_str(),
        other => panic!("expected EXPLAIN text, got {other:?}"),
    }
}

async fn root_page(connection: &Connection, name: &str) -> i64 {
    let rows = connection
        .query(&format!(
            "SELECT rootpage FROM sqlite_master WHERE type IN ('table','index') AND name='{name}'"
        ))
        .await
        .unwrap_or_else(|error| panic!("rootpage of {name}: {error}"));
    integer(&rows[0].values()[0])
}

/// Populate `t` with `rows` rows in `rows / block` contiguous `a`-blocks of
/// `block` rows each — the shape of the motivating workload (delete one
/// conversation out of many).
async fn populate_blocks(connection: &Connection, rows: i64, block: i64) {
    connection
        .execute("CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, n INTEGER, v TEXT)")
        .await
        .expect("create blocked table");
    connection
        .execute("CREATE INDEX t_ab ON t(a,b)")
        .await
        .expect("create t_ab");
    connection
        .execute("CREATE INDEX t_an ON t(a,n)")
        .await
        .expect("create t_an");
    connection.execute("BEGIN").await.expect("begin population");
    {
        let insert = connection
            .prepare("INSERT INTO t(a,b,n,v) VALUES(?,?,?,'x')")
            .await
            .expect("prepare population");
        for key in 0..rows {
            insert
                .execute_with_params(&[
                    SqliteValue::Integer(key / block),
                    SqliteValue::Integer(key),
                    SqliteValue::Integer(key % block),
                ])
                .await
                .expect("populate");
        }
    }
    connection.execute("COMMIT").await.expect("commit population");
}

#[test]
fn bd_j9gub_dml_prefix_seek_must_not_walk_the_table() {
    asupersync::test_utils::run_test(|| async {
        let dir = tempfile::tempdir().expect("temporary database directory");
        let path = dir
            .path()
            .join("plan.sqlite")
            .to_string_lossy()
            .into_owned();
        let connection = Connection::open(&path).await.expect("open database");
        populate_blocks(&connection, 256, 32).await;
        let table_root = root_page(&connection, "t").await;
        let index_root = root_page(&connection, "t_ab").await;
        let params = [SqliteValue::Integer(7)];
        // bd-j9gub is a DML change; these are the covered DML shapes. (A
        // `SELECT v FROM t WHERE a=?` on this composite schema still
        // Rewind-walks the table in fsqlite today — SELECT composite-prefix
        // routing is a separate, pre-existing planner-side gap, out of this
        // bead's DML scope and recorded in the report.)
        for sql in [
            "UPDATE t SET v=v+1 WHERE a=?",
            "DELETE FROM t WHERE a=?",
            "DELETE FROM t WHERE a=? AND b>?",
        ] {
            let bound = if sql.contains("b>?") {
                vec![SqliteValue::Integer(7), SqliteValue::Integer(100)]
            } else {
                params.to_vec()
            };
            let rows = connection
                .query_with_params(&format!("EXPLAIN {sql}"), &bound)
                .await
                .unwrap_or_else(|error| panic!("EXPLAIN {sql}: {error}"));
            let table_cursors: Vec<_> = rows
                .iter()
                .map(|row| row.values())
                .filter(|row| {
                    matches!(text(&row[1]), "OpenRead" | "OpenWrite") && integer(&row[3]) == table_root
                })
                .map(|row| integer(&row[2]))
                .collect();
            assert!(!table_cursors.is_empty(), "missing table cursor: {sql}");
            for row in &rows {
                let op = row.values();
                assert!(
                    !table_cursors.contains(&integer(&op[2]))
                        || !matches!(text(&op[1]), "Rewind" | "Last" | "Next" | "Prev"),
                    "bd-j9gub: table walk in {sql}: {op:?}"
                );
            }
            let seek_cursors: Vec<_> = rows
                .iter()
                .map(|row| row.values())
                .filter(|row| matches!(text(&row[1]), "OpenRead") && integer(&row[3]) == index_root)
                .map(|row| integer(&row[2]))
                .collect();
            assert!(!seek_cursors.is_empty(), "missing t_ab read cursor: {sql}");
            assert!(
                rows.iter().any(|row| {
                    let op = row.values();
                    text(&op[1]) == "SeekGE" && seek_cursors.contains(&integer(&op[2]))
                }),
                "bd-j9gub: missing prefix seek on t_ab in {sql}"
            );
        }
        connection.close().await.expect("close plan database");
    });
}

/// (prefix-seek UPDATE hit, prefix-seek DELETE miss, non-prefix scan control)
async fn measure_trio(
    connection: &Connection,
    block: i64,
    scan_hits: i64,
) -> (Duration, Duration, Duration) {
    // The engine reports change counts as usize; the table geometry is i64.
    let block_changed = usize::try_from(block).expect("block size fits usize");
    let scan_changed = usize::try_from(scan_hits).expect("scan hits fit usize");
    let seeked = connection
        .prepare("UPDATE t SET v=v+1 WHERE a=?")
        .await
        .expect("prepare prefix-seek UPDATE");
    let miss = connection
        .prepare("DELETE FROM t WHERE a=?")
        .await
        .expect("prepare prefix-seek DELETE");
    let scanned = connection
        .prepare("UPDATE t SET v=v+1 WHERE n=?")
        .await
        .expect("prepare scan UPDATE");
    let hit_params = [SqliteValue::Integer(0)];
    let miss_params = [SqliteValue::Integer(-1)];
    let scan_params = [SqliteValue::Integer(0)];
    for _ in 0..8 {
        assert_eq!(
            seeked
                .execute_with_params(&hit_params)
                .await
                .expect("warm seek"),
            block_changed
        );
        assert_eq!(
            miss.execute_with_params(&miss_params).await.expect("warm miss"),
            0
        );
        assert_eq!(
            scanned
                .execute_with_params(&scan_params)
                .await
                .expect("warm scan"),
            scan_changed
        );
    }
    let mut seek_times = Vec::new();
    let mut miss_times = Vec::new();
    let mut scan_times = Vec::new();
    for sample in 0..9 {
        let mut workloads = [
            (&seeked, hit_params.as_slice(), block_changed, &mut seek_times),
            (&miss, miss_params.as_slice(), 0, &mut miss_times),
            (
                &scanned,
                scan_params.as_slice(),
                scan_changed,
                &mut scan_times,
            ),
        ];
        // Rotate start order to avoid consistently favoring a cache state.
        workloads.rotate_left(sample % 3);
        for (statement, params, changed, samples) in workloads {
            let start = Instant::now();
            for _ in 0..16 {
                assert_eq!(
                    statement.execute_with_params(params).await.expect("timed DML"),
                    changed
                );
            }
            samples.push(start.elapsed());
        }
    }
    seek_times.sort_unstable();
    miss_times.sort_unstable();
    scan_times.sort_unstable();
    (seek_times[4], miss_times[4], scan_times[4])
}

#[test]
#[ignore = "bd-j9gub: wall-clock scaling probe; run with --ignored --nocapture (dev profile suffices — no LTO per machine rule)"]
fn bd_j9gub_dml_prefix_seek_scaling() {
    asupersync::test_utils::run_test(|| async {
        let mut measurements = Vec::new();
        const BLOCK: i64 = 32;
        for rows in [1_600_i64, 19_200] {
            let dir = tempfile::tempdir().expect("temporary database directory");
            let path = dir
                .path()
                .join("scaling.sqlite")
                .to_string_lossy()
                .into_owned();
            let connection = Connection::open(&path).await.expect("open scaling database");
            populate_blocks(&connection, rows, BLOCK).await;
            // Exclude population, preparation, warm-up and commit/fsync costs.
            // Every `a`-block stays 32 rows as the table grows 12x: a
            // prefix-seek statement's cost must stay flat; the non-prefix scan
            // control walks the whole table and is printed for contrast only.
            connection.execute("BEGIN").await.expect("begin measurements");
            let trio = measure_trio(&connection, BLOCK, rows / BLOCK).await;
            connection
                .execute("ROLLBACK")
                .await
                .expect("rollback measurements");
            eprintln!(
                "rows={rows}, 16-execution medians: prefix_update_hit={:?}, prefix_delete_miss={:?}, scan_control={:?}",
                trio.0, trio.1, trio.2
            );
            measurements.push(trio);
            connection.close().await.expect("close scaling database");
        }
        let (small_hit, small_miss, small_scan) = measurements[0];
        let (large_hit, large_miss, large_scan) = measurements[1];
        for (name, small, large) in [
            ("prefix UPDATE hit", small_hit, large_hit),
            ("prefix DELETE miss", small_miss, large_miss),
        ] {
            assert!(
                large <= small.saturating_mul(4),
                "bd-j9gub: {name} grew more than 4x for 12x rows; small={small:?}, large={large:?}"
            );
        }
        eprintln!(
            "scan control grew {:.1}x for 12x rows (printed, not asserted)",
            large_scan.as_secs_f64() / small_scan.as_secs_f64().max(f64::EPSILON)
        );
    });
}
