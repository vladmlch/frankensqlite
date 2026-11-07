//! Range seek + residual filter: a range on an indexed column or on the rowid seeks
//! (index `SeekGE` walk, or a bounded rowid walk) and re-applies the REST of the WHERE
//! per row — instead of declining to a full table scan the moment the WHERE carries
//! any other conjunct.
//!
//! Scope 1 (index): `WHERE k >= ? AND k < ? AND <other conjuncts>` walks the
//! single-column index and applies the WHOLE WHERE as a per-row residual, mirroring the
//! equality path's residual mode. Scope 2 (rowid): `WHERE id > ?1 AND other > ?2` walks
//! the [lower, upper] rowid slice with PLACEHOLDER bounds (the residual walk previously
//! required integer literals) and re-applies the whole WHERE per row.
//!
//! Every accepted shape is proven three ways: bit-identical row set to stock SQLite
//! (rusqlite), bit-identical to frank's own forced full scan (`NOT INDEXED`) where the
//! query is un-ordered, and an EXPLAIN-opcode seek assertion (`IdxRowid` for the index
//! walk — a plain table scan never emits it; `SeekGT`/`SeekGE` with no `Rewind` for the
//! rowid walk). EXPLAIN QUERY PLAN prints SCAN even for rowid seeks, so the opcodes are
//! the signal. Decline shapes (DESC-only index, nondeterministic ORDER BY, DISTINCT)
//! assert the opposite: they must keep the scan AND stay bit-identical.

// Fixture row counters; precision loss is irrelevant to the assertions.
#![allow(clippy::cast_precision_loss)]
#![recursion_limit = "512"]

use fsqlite::Connection;
use fsqlite_types::SqliteValue;

fn render(v: &SqliteValue) -> String {
    match v {
        SqliteValue::Null => "NULL".to_owned(),
        SqliteValue::Integer(n) => n.to_string(),
        SqliteValue::Float(f) => format!("{f}"),
        SqliteValue::Text(s) => format!("'{s}'"),
        SqliteValue::Blob(b) => format!(
            "X'{}'",
            b.iter().map(|x| format!("{x:02X}")).collect::<String>()
        ),
    }
}

fn render_rusqlite(v: &rusqlite::types::Value) -> String {
    match v {
        rusqlite::types::Value::Null => "NULL".to_owned(),
        rusqlite::types::Value::Integer(x) => x.to_string(),
        rusqlite::types::Value::Real(f) => format!("{f}"),
        rusqlite::types::Value::Text(s) => format!("'{s}'"),
        rusqlite::types::Value::Blob(b) => format!(
            "X'{}'",
            b.iter().map(|x| format!("{x:02X}")).collect::<String>()
        ),
    }
}

async fn frank_rows(conn: &Connection, sql: &str) -> Vec<Vec<String>> {
    conn.query(sql)
        .await
        .unwrap_or_else(|e| panic!("frank `{sql}`: {e}"))
        .iter()
        .map(|row| row.values().iter().map(render).collect())
        .collect()
}

async fn frank_bound(conn: &Connection, sql: &str, params: &[SqliteValue]) -> Vec<Vec<String>> {
    let stmt = conn
        .prepare(sql)
        .await
        .unwrap_or_else(|e| panic!("frank prepare `{sql}`: {e}"));
    let rows = stmt
        .query_with_params(params)
        .await
        .unwrap_or_else(|e| panic!("frank bind `{sql}`: {e}"));
    rows.iter()
        .map(|row| row.values().iter().map(render).collect())
        .collect()
}

fn sqlite_rows(conn: &rusqlite::Connection, sql: &str) -> Vec<Vec<String>> {
    let mut stmt = conn.prepare(sql).unwrap();
    let n = stmt.column_count();
    stmt.query_map([], |row| {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(render_rusqlite(
                &row.get_unwrap::<_, rusqlite::types::Value>(i),
            ));
        }
        Ok(out)
    })
    .unwrap()
    .map(Result::unwrap)
    .collect()
}

fn sqlite_bound(
    conn: &rusqlite::Connection,
    sql: &str,
    params: &[SqliteValue],
) -> Vec<Vec<String>> {
    let rp: Vec<rusqlite::types::Value> = params
        .iter()
        .map(|v| match v {
            SqliteValue::Null => rusqlite::types::Value::Null,
            SqliteValue::Integer(n) => rusqlite::types::Value::Integer(*n),
            SqliteValue::Float(f) => rusqlite::types::Value::Real(*f),
            SqliteValue::Text(s) => rusqlite::types::Value::Text(s.to_string()),
            SqliteValue::Blob(b) => rusqlite::types::Value::Blob(b.to_vec()),
        })
        .collect();
    let mut stmt = conn.prepare(sql).unwrap();
    let n = stmt.column_count();
    stmt.query_map(rusqlite::params_from_iter(rp), |row| {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(render_rusqlite(
                &row.get_unwrap::<_, rusqlite::types::Value>(i),
            ));
        }
        Ok(out)
    })
    .unwrap()
    .map(Result::unwrap)
    .collect()
}

fn sorted(mut rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    rows.sort();
    rows
}

/// EXPLAIN opcodes of `sql` (the opcode-name column).
async fn opcodes(conn: &Connection, sql: &str) -> Vec<String> {
    conn.query(&format!("EXPLAIN {sql}"))
        .await
        .unwrap_or_else(|e| panic!("EXPLAIN `{sql}`: {e}"))
        .iter()
        .filter_map(|row| match row.values().get(1) {
            Some(SqliteValue::Text(op)) => Some(op.to_string()),
            _ => None,
        })
        .collect()
}

/// The index-range walk always dereferences index entries with `IdxRowid` (rowid tables),
/// which a plain table scan never emits.
fn uses_index(ops: &[String]) -> bool {
    ops.iter().any(|o| o == "IdxRowid")
}

/// A bounded rowid walk seeks the table b-tree directly (`SeekGT`/`SeekGE`); on a table
/// with no secondary index those opcodes can only be the rowid range seek. A full scan
/// `Rewind`s instead.
fn rowid_seeks(ops: &[String]) -> bool {
    ops.iter().any(|o| o == "SeekGT" || o == "SeekGE") && !ops.iter().any(|o| o == "Rewind")
}

async fn setup_range_tables() -> (Connection, rusqlite::Connection) {
    let f = Connection::open(":memory:").await.expect("open frank");
    let r = rusqlite::Connection::open_in_memory().expect("open rusqlite");
    let mut stmts = vec![
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, w TEXT, v TEXT);".to_owned(),
        "CREATE INDEX idx_t_k ON t(k);".to_owned(),
        "CREATE INDEX idx_t_w ON t(w);".to_owned(),
    ];
    for i in 1..=600_i64 {
        let k = (i % 17) - 4; // -4..12, duplicates, negatives
        let w = if i % 23 == 0 {
            "NULL".to_owned()
        } else {
            format!("'r{:02}'", i % 11)
        };
        let v = if i % 19 == 0 {
            "NULL".to_owned()
        } else {
            format!("'v{}'", i % 7)
        };
        stmts.push(format!("INSERT INTO t VALUES ({i}, {k}, {w}, {v});"));
    }
    // NULL in the indexed column too.
    stmts.push("INSERT INTO t VALUES (700, NULL, 'r00', 'v1');".to_owned());
    for s in &stmts {
        f.execute(s)
            .await
            .unwrap_or_else(|e| panic!("frank `{s}`: {e}"));
        r.execute_batch(s)
            .unwrap_or_else(|e| panic!("rusqlite `{s}`: {e}"));
    }
    (f, r)
}

/// Scope 1, literal bounds: every range + residual shape must be bit-identical to BOTH
/// stock SQLite and frank's own forced full scan (`NOT INDEXED`), and must walk the index.
#[test]
fn index_range_residual_literals_match_sqlite_and_forced_scan() {
    asupersync::test_utils::run_test(|| async {
        let (f, r) = setup_range_tables().await;
        let shapes = [
            // The issue-draft shape: range plus an IS NOT NULL residual.
            "SELECT id, v FROM t WHERE k >= 2 AND k < 9 AND v IS NOT NULL",
            // Equality residual on an unindexed column.
            "SELECT id FROM t WHERE k > 2 AND k < 8 AND v = 'v3'",
            // Multi-conjunct residual.
            "SELECT id, k FROM t WHERE k BETWEEN -2 AND 3 AND v = 'v2' AND w IS NOT NULL",
            // IS NULL residual (i % 19 == 0 rows have NULL v).
            "SELECT id FROM t WHERE k > 0 AND k <= 4 AND v IS NULL",
            // Residual false everywhere -> empty result.
            "SELECT id FROM t WHERE k BETWEEN 1 AND 5 AND v = 'zzz'",
            // One-sided LOWER range plus residual.
            "SELECT id FROM t WHERE k > 9 AND v = 'v5'",
            // Residual referencing the range column redundantly.
            "SELECT id FROM t WHERE k >= 1 AND k < 6 AND k <> 3",
            // Text column range with residual (BINARY collation, idx_t_w).
            "SELECT id, w FROM t WHERE w >= 'r05' AND w < 'r09' AND v IS NOT NULL",
        ];
        for sql in shapes {
            let forced = sql.replace(" FROM t ", " FROM t NOT INDEXED ");
            let seek = frank_rows(&f, sql).await;
            let scan = frank_rows(&f, &forced).await;
            assert_eq!(
                sorted(seek.clone()),
                sorted(scan),
                "seek row set must equal the forced full-scan row set for `{sql}`"
            );
            let stock = sqlite_rows(&r, sql);
            assert_eq!(
                sorted(seek.clone()),
                sorted(stock),
                "seek row set diverged from stock SQLite for `{sql}`"
            );
            let ops = opcodes(&f, sql).await;
            assert!(
                uses_index(&ops),
                "index range + residual must walk the index for `{sql}`; ops = {ops:?}"
            );
            assert!(
                !ops.iter().any(|o| o == "Rewind"),
                "index range + residual must not full-scan for `{sql}`; ops = {ops:?}"
            );
        }

        // Upper-only ranges (`k <= 5` — no lower bound): the walk Rewinds the INDEX cursor
        // and walks to the upper bound, so the no-Rewind assertion does NOT apply here (it
        // guards the lower-bounded shapes above). `IdxRowid` still separates the walk from a
        // table scan, and the row set must match both references.
        for sql in [
            "SELECT id, v FROM t WHERE k <= 5 AND v IS NOT NULL",
            "SELECT id FROM t WHERE w < 'r05' AND v = 'v3'",
        ] {
            let forced = sql.replace(" FROM t ", " FROM t NOT INDEXED ");
            let seek = frank_rows(&f, sql).await;
            assert_eq!(
                sorted(seek.clone()),
                sorted(frank_rows(&f, &forced).await),
                "upper-only seek row set must equal the forced full-scan row set for `{sql}`"
            );
            assert_eq!(
                sorted(seek.clone()),
                sorted(sqlite_rows(&r, sql)),
                "upper-only seek row set diverged from stock SQLite for `{sql}`"
            );
            let ops = opcodes(&f, sql).await;
            assert!(
                uses_index(&ops),
                "upper-only range + residual must walk the index for `{sql}`; ops = {ops:?}"
            );
        }

        // Aggregate range + residual keeps its (pre-existing) aggregate seek path correct.
        let sql = "SELECT COUNT(*) FROM t WHERE k > 2 AND k < 8 AND v = 'v3'";
        let forced = sql.replace(" FROM t ", " FROM t NOT INDEXED ");
        assert_eq!(
            frank_rows(&f, sql).await,
            frank_rows(&f, &forced).await,
            "aggregate range residual must equal the forced full scan"
        );
        assert_eq!(
            frank_rows(&f, sql).await,
            sqlite_rows(&r, sql),
            "aggregate range residual diverged from stock SQLite"
        );
    });
}

/// Scope 1, placeholder bounds: numbered `?N`, anonymous `?` (the connection numbers the
/// slots), every bind class (NULL, blob, text, real), swapped bound order, the residual
/// placeholder before the bounds in text order, and LIMIT/OFFSET — each bit-identical to
/// stock SQLite with the seek engaged.
#[test]
fn index_range_residual_placeholder_bounds_match_sqlite() {
    asupersync::test_utils::run_test(|| async {
        let (f, r) = setup_range_tables().await;

        // The accepted placeholder shapes must walk the index.
        for sql in [
            "SELECT id, v FROM t WHERE k >= ?1 AND k < ?2 AND v IS NOT NULL",
            "SELECT id, v FROM t WHERE k >= ? AND k < ? AND v IS NOT NULL",
            "SELECT id FROM t WHERE k < ?1 AND k >= ?2 AND v = ?3",
            "SELECT id FROM t WHERE v = ?1 AND k > ?2 AND k < ?3",
            "SELECT id FROM t WHERE k BETWEEN ?1 AND ?2 AND v = ?3 LIMIT ?4",
        ] {
            let ops = opcodes(&f, sql).await;
            assert!(
                uses_index(&ops),
                "placeholder range + residual must walk the index for `{sql}`; ops = {ops:?}"
            );
        }

        // Numbered two-bound range + numbered residual: full value matrix per bound.
        let vals = [
            SqliteValue::Integer(-4),
            SqliteValue::Integer(2),
            SqliteValue::Integer(9),
            SqliteValue::Float(2.5),
            SqliteValue::Text("3".into()),
            SqliteValue::Text("abc".into()),
            SqliteValue::Null,
            SqliteValue::Blob(vec![1, 2, 3].into()),
        ];
        let sql = "SELECT id, k, v FROM t WHERE k >= ?1 AND k < ?2 AND v = ?3";
        for a in &vals {
            for b in &vals {
                let params = [a.clone(), b.clone(), SqliteValue::Text("v3".into())];
                assert_eq!(
                    sorted(frank_bound(&f, sql, &params).await),
                    sorted(sqlite_bound(&r, sql, &params)),
                    "numbered range residual diverged for `{sql}` with {a:?}..{b:?}"
                );
            }
        }
        // Residual bind classes: NULL (never equal) and BLOB (text column never equals blob).
        for res in [SqliteValue::Null, SqliteValue::Blob(vec![1, 2, 3].into())] {
            let params = [
                SqliteValue::Integer(2),
                SqliteValue::Integer(9),
                res.clone(),
            ];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "numbered range residual diverged for residual bind {res:?}"
            );
        }

        // Anonymous `?` form — the connection canonicalizes slots in text order.
        let sql = "SELECT id, v FROM t WHERE k >= ? AND k < ? AND v IS NOT NULL";
        for a in &vals {
            for b in &vals {
                let params = [a.clone(), b.clone()];
                assert_eq!(
                    sorted(frank_bound(&f, sql, &params).await),
                    sorted(sqlite_bound(&r, sql, &params)),
                    "anonymous range residual diverged for params {a:?}..{b:?}"
                );
            }
        }

        // TEXT column range (BINARY collation) with placeholder bounds.
        let sql = "SELECT id, w FROM t WHERE w >= ?1 AND w < ?2 AND v IS NOT NULL";
        let wvals = [
            SqliteValue::Text("r02".into()),
            SqliteValue::Text("r08".into()),
            SqliteValue::Text("r00".into()),
            SqliteValue::Text("r99".into()),
            SqliteValue::Text("abc".into()),
            SqliteValue::Null,
            SqliteValue::Blob(vec![0x62, 0x62].into()),
        ];
        for a in &wvals {
            for b in &wvals {
                let params = [a.clone(), b.clone()];
                assert_eq!(
                    sorted(frank_bound(&f, sql, &params).await),
                    sorted(sqlite_bound(&r, sql, &params)),
                    "text-column range residual diverged for {a:?}..{b:?}"
                );
            }
        }

        // Swapped bound order (`k < ?1 AND k >= ?2`): each slot binds to ITS bound.
        let sql = "SELECT id FROM t WHERE k < ?1 AND k >= ?2 AND v = ?3";
        for (a, b) in [
            (SqliteValue::Integer(9), SqliteValue::Integer(2)),
            (SqliteValue::Integer(3), SqliteValue::Integer(-4)),
            (SqliteValue::Integer(2), SqliteValue::Integer(2)),
            (SqliteValue::Null, SqliteValue::Integer(4)),
        ] {
            let params = [a.clone(), b.clone(), SqliteValue::Text("v3".into())];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "swapped-order range residual diverged for {a:?}, {b:?}"
            );
        }

        // Residual placeholder BEFORE the bounds in text order + LIMIT/OFFSET.
        // ORDER BY k, id is deterministic in the index walk, so exact comparison holds.
        // (A NULL LIMIT is NOT in the matrix: stock rejects `LIMIT ?` bound to NULL with
        // "datatype mismatch" — verified against sqlite3 3.54.0 — and frank matches.)
        let sql = "SELECT id, v FROM t WHERE v = ?1 AND k > ?2 AND k < ?3 ORDER BY k, id LIMIT ?4 OFFSET ?5";
        for (limit, offset) in [
            (SqliteValue::Integer(3), SqliteValue::Integer(2)),
            (SqliteValue::Integer(0), SqliteValue::Integer(0)),
            (SqliteValue::Integer(-1), SqliteValue::Integer(0)),
        ] {
            let params = [
                SqliteValue::Text("v2".into()),
                SqliteValue::Integer(-3),
                SqliteValue::Integer(6),
                limit.clone(),
                offset.clone(),
            ];
            assert_eq!(
                frank_bound(&f, sql, &params).await,
                sqlite_bound(&r, sql, &params),
                "ordered limited range residual diverged for limit={limit:?} offset={offset:?}"
            );
        }

        // NULL LOWER bound: empty result, matching stock (the probe's IsNull short-circuit).
        let sql = "SELECT id FROM t WHERE k > ?1 AND k < ?2 AND v IS NOT NULL";
        let params = [SqliteValue::Null, SqliteValue::Integer(5)];
        let frank = frank_bound(&f, sql, &params).await;
        assert!(
            frank.is_empty(),
            "NULL lower bound must yield the empty result, got {frank:?}"
        );
        assert_eq!(
            frank,
            sqlite_bound(&r, sql, &params),
            "NULL lower bound diverged from stock"
        );
    });
}

/// Scope 1 declines: a nondeterministic ORDER BY or DISTINCT keeps the full scan (and its
/// explicit sort/dedup) and stays bit-identical; a deterministic ORDER BY is served by the
/// walk in exact order.
#[test]
fn index_range_residual_order_by_and_distinct_declines() {
    asupersync::test_utils::run_test(|| async {
        let (f, r) = setup_range_tables().await;

        // ORDER BY v (not the range column): the walk cannot provide v order, so the seek
        // must decline to the scan + explicit sort. Total order via the id tiebreaker.
        let sql = "SELECT id FROM t WHERE k >= 2 AND k < 9 AND v IS NOT NULL ORDER BY v, id";
        let ops = opcodes(&f, sql).await;
        assert!(
            !uses_index(&ops),
            "nondeterministic ORDER BY must decline the range residual seek; ops = {ops:?}"
        );
        assert_eq!(
            frank_rows(&f, sql).await,
            sqlite_rows(&r, sql),
            "nondeterministic ORDER BY must stay bit-identical in order"
        );

        // ORDER BY k, id (range column + rowid): deterministic in the walk — exact match.
        let sql = "SELECT id, k FROM t WHERE k >= 2 AND k < 9 AND v IS NOT NULL ORDER BY k, id";
        let ops = opcodes(&f, sql).await;
        assert!(
            uses_index(&ops),
            "deterministic ORDER BY (range col, rowid) must be served by the walk; \
             ops = {ops:?}"
        );
        assert_eq!(
            frank_rows(&f, sql).await,
            sqlite_rows(&r, sql),
            "deterministic ORDER BY diverged in order"
        );

        // DISTINCT: declines to the scan path and stays bit-identical.
        let sql = "SELECT DISTINCT k FROM t WHERE k >= 2 AND k < 9 AND v IS NOT NULL";
        let ops = opcodes(&f, sql).await;
        assert!(
            !uses_index(&ops),
            "DISTINCT must decline the range residual seek; ops = {ops:?}"
        );
        assert_eq!(
            sorted(frank_rows(&f, sql).await),
            sorted(sqlite_rows(&r, sql)),
            "DISTINCT decline diverged"
        );
    });
}

/// Scope 1 decline: a DESC-only index must not serve the ascending range residual — the
/// query keeps the full scan and stays bit-identical to stock.
#[test]
fn index_range_residual_desc_index_declines() {
    asupersync::test_utils::run_test(|| async {
        let f = Connection::open(":memory:").await.expect("open frank");
        let r = rusqlite::Connection::open_in_memory().expect("open rusqlite");
        for s in [
            "CREATE TABLE d (id INTEGER PRIMARY KEY, k INTEGER, v TEXT);",
            "CREATE INDEX idx_d_k ON d(k DESC);",
        ] {
            f.execute(s)
                .await
                .unwrap_or_else(|e| panic!("frank `{s}`: {e}"));
            r.execute_batch(s)
                .unwrap_or_else(|e| panic!("rusqlite `{s}`: {e}"));
        }
        for i in 1..=200_i64 {
            let s = format!("INSERT INTO d VALUES ({i}, {}, 'v{}');", i % 9, i % 5);
            f.execute(&s)
                .await
                .unwrap_or_else(|e| panic!("frank `{s}`: {e}"));
            r.execute_batch(&s)
                .unwrap_or_else(|e| panic!("rusqlite `{s}`: {e}"));
        }
        let sql = "SELECT id FROM d WHERE k > 2 AND k < 7 AND v = 'v3'";
        let ops = opcodes(&f, sql).await;
        assert!(
            !uses_index(&ops),
            "a DESC index must not serve the ascending range residual; ops = {ops:?}"
        );
        assert_eq!(
            sorted(frank_rows(&f, sql).await),
            sorted(sqlite_rows(&r, sql)),
            "DESC-index decline must stay bit-identical"
        );
    });
}

/// The issue-draft shape exactly: a UNIQUE TEXT key (automatic index), a BLOB value,
/// `key >= ? AND key < ? AND value IS NOT NULL` — the motivating 0.7-1.2s full scan.
#[test]
fn unique_text_key_range_residual_matches_sqlite() {
    asupersync::test_utils::run_test(|| async {
        let f = Connection::open(":memory:").await.expect("open frank");
        let r = rusqlite::Connection::open_in_memory().expect("open rusqlite");
        for s in ["CREATE TABLE kv (key TEXT UNIQUE, value BLOB);"] {
            f.execute(s)
                .await
                .unwrap_or_else(|e| panic!("frank `{s}`: {e}"));
            r.execute_batch(s)
                .unwrap_or_else(|e| panic!("rusqlite `{s}`: {e}"));
        }
        for i in 1..=300_i64 {
            let blob = if i % 29 == 0 {
                "NULL".to_owned()
            } else {
                format!("x'{}{}'", i % 10, i % 7)
            };
            let s = format!("INSERT INTO kv VALUES ('key{i:04}', {blob});");
            f.execute(&s)
                .await
                .unwrap_or_else(|e| panic!("frank `{s}`: {e}"));
            r.execute_batch(&s)
                .unwrap_or_else(|e| panic!("rusqlite `{s}`: {e}"));
        }
        let sql = "SELECT key, value FROM kv WHERE key >= ? AND key < ? AND value IS NOT NULL";
        let ops = opcodes(&f, sql).await;
        assert!(
            uses_index(&ops),
            "the unique-text-key range + residual must walk the autoindex; ops = {ops:?}"
        );
        for (a, b) in [
            ("key0100".to_owned(), "key0200".to_owned()),
            ("key0000".to_owned(), "key9999".to_owned()),
            ("key0250".to_owned(), "key0250".to_owned()), // empty
            ("key0290".to_owned(), "key0299".to_owned()), // all-NULL residual region
        ] {
            let params = [
                SqliteValue::Text(a.as_str().into()),
                SqliteValue::Text(b.as_str().into()),
            ];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "kv range residual diverged for {a:?}..{b:?}"
            );
        }
        // ORDER BY key (unique index -> deterministic) with LIMIT: exact order.
        let sql = "SELECT key FROM kv WHERE key > ? AND key < ? AND value IS NOT NULL ORDER BY key LIMIT ?";
        let params = [
            SqliteValue::Text("key0050".into()),
            SqliteValue::Text("key0250".into()),
            SqliteValue::Integer(40),
        ];
        assert_eq!(
            frank_bound(&f, sql, &params).await,
            sqlite_bound(&r, sql, &params),
            "ordered limited kv range residual diverged"
        );
    });
}

async fn setup_rowid_table() -> (Connection, rusqlite::Connection) {
    let f = Connection::open(":memory:").await.expect("open frank");
    let r = rusqlite::Connection::open_in_memory().expect("open rusqlite");
    let mut stmts = vec!["CREATE TABLE m (id INTEGER PRIMARY KEY, a INTEGER, v TEXT);".to_owned()];
    for i in 1..=600_i64 {
        let a = if i % 17 == 0 {
            "NULL".to_owned()
        } else {
            format!("{}", i % 13)
        };
        let v = if i % 19 == 0 {
            "NULL".to_owned()
        } else {
            format!("'v{}'", i % 7)
        };
        stmts.push(format!("INSERT INTO m VALUES ({i}, {a}, {v});"));
    }
    for s in &stmts {
        f.execute(s)
            .await
            .unwrap_or_else(|e| panic!("frank `{s}`: {e}"));
        r.execute_batch(s)
            .unwrap_or_else(|e| panic!("rusqlite `{s}`: {e}"));
    }
    (f, r)
}

/// Scope 2: a rowid range with PLACEHOLDER bounds plus a residual walks the [lower, upper]
/// slice (`SeekGT`/`SeekGE` on the table b-tree) instead of scanning — numbered and
/// anonymous forms, swapped bound order, BETWEEN, every bound class — each bit-identical
/// to stock SQLite.
#[test]
fn rowid_range_placeholder_residual_matches_sqlite() {
    asupersync::test_utils::run_test(|| async {
        let (f, r) = setup_rowid_table().await;

        // The accepted shapes must seek the table b-tree (no Rewind anywhere).
        for sql in [
            "SELECT id, a, v FROM m WHERE id > ?1 AND a > ?2",
            "SELECT id, v FROM m WHERE id > ? AND a > ?",
            "SELECT id FROM m WHERE id < ?1 AND id >= ?2 AND v = ?3",
            "SELECT id FROM m WHERE id BETWEEN ?1 AND ?2 AND a IS NULL",
        ] {
            let ops = opcodes(&f, sql).await;
            assert!(
                rowid_seeks(&ops),
                "placeholder rowid range + residual must seek the table b-tree for \
                 `{sql}`; ops = {ops:?}"
            );
        }

        let bounds = [
            (SqliteValue::Integer(100), SqliteValue::Integer(200)),
            (SqliteValue::Integer(590), SqliteValue::Integer(599)),
            (SqliteValue::Integer(400), SqliteValue::Integer(2)), // inverted -> empty
            (SqliteValue::Null, SqliteValue::Integer(50)),        // NULL lower -> empty
            (SqliteValue::Integer(100), SqliteValue::Null),       // NULL upper -> empty
            (SqliteValue::Float(99.5), SqliteValue::Float(150.5)), // real bounds
            // Text '100' coerces to 100 via the rowid comparison affinity.
            (SqliteValue::Text("100".into()), SqliteValue::Integer(150)),
            // INTEGER < BLOB: `id > blob` matches nothing.
            (
                SqliteValue::Blob(vec![1, 2].into()),
                SqliteValue::Integer(150),
            ),
            // ...and `id < blob` matches everything.
            (
                SqliteValue::Integer(5),
                SqliteValue::Blob(vec![1, 2].into()),
            ),
        ];
        // Numbered bounds + residual on an unindexed column.
        let sql = "SELECT id, a, v FROM m WHERE id > ?1 AND id < ?2 AND a > ?3";
        for (lo, hi) in &bounds {
            let params = [lo.clone(), hi.clone(), SqliteValue::Integer(6)];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "numbered rowid range residual diverged for {lo:?}..{hi:?}"
            );
        }
        // A NULL residual bind (`a > NULL` never true) must yield the empty set.
        let params = [
            SqliteValue::Integer(100),
            SqliteValue::Integer(200),
            SqliteValue::Null,
        ];
        let frank = frank_bound(&f, sql, &params).await;
        assert!(
            frank.is_empty(),
            "`a > NULL` must yield no rows, got {frank:?}"
        );
        assert_eq!(
            frank,
            sqlite_bound(&r, sql, &params),
            "NULL residual bind diverged from stock"
        );

        // Anonymous form (connection canonicalization): issue-draft shape.
        let sql = "SELECT id, v FROM m WHERE id > ? AND a > ?";
        for (lo, _) in &bounds {
            let params = [lo.clone(), SqliteValue::Integer(4)];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "anonymous rowid range residual diverged for lower={lo:?}"
            );
        }

        // Swapped bound text order: each slot binds to ITS bound.
        let sql = "SELECT id FROM m WHERE id < ?2 AND id > ?1 AND v = ?3";
        for (lo, hi) in &bounds {
            let params = [lo.clone(), hi.clone(), SqliteValue::Text("v3".into())];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "swapped-order rowid range residual diverged for {lo:?}..{hi:?}"
            );
        }

        // BETWEEN with placeholder bounds + IS NULL residual.
        let sql = "SELECT id, a FROM m WHERE id BETWEEN ?1 AND ?2 AND a IS NULL";
        for (lo, hi) in &bounds {
            let params = [lo.clone(), hi.clone()];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "BETWEEN rowid range residual diverged for {lo:?}..{hi:?}"
            );
        }

        // Literal bounds with an anonymous residual `?`: the pre-existing residual walk
        // keeps its seek and its textual slot numbering.
        let sql = "SELECT id, v FROM m WHERE id > 500 AND v = ?";
        let params = [SqliteValue::Text("v5".into())];
        let ops = opcodes(&f, sql).await;
        assert!(
            rowid_seeks(&ops),
            "literal-bounds rowid residual must keep its seek; ops = {ops:?}"
        );
        assert_eq!(
            sorted(frank_bound(&f, sql, &params).await),
            sorted(sqlite_bound(&r, sql, &params)),
            "literal-bounds rowid residual diverged"
        );

        // Upper-only rowid range (`id < ?1` — no lower bound): the walk Rewinds the table
        // cursor and stops at the bound, which is opcode-indistinguishable from a full scan,
        // so this is differential-only (correctness of the placeholder upper bound).
        let sql = "SELECT id FROM m WHERE id < ?1 AND a > ?2";
        for (hi, res) in [
            (SqliteValue::Integer(50), SqliteValue::Integer(6)),
            (SqliteValue::Integer(599), SqliteValue::Integer(2)),
            (SqliteValue::Float(100.5), SqliteValue::Integer(1)),
            (SqliteValue::Text("150".into()), SqliteValue::Integer(6)),
        ] {
            let params = [hi.clone(), res.clone()];
            assert_eq!(
                sorted(frank_bound(&f, sql, &params).await),
                sorted(sqlite_bound(&r, sql, &params)),
                "upper-only rowid range residual diverged for upper={hi:?} residual={res:?}"
            );
        }

        // ORDER BY id stays exactly ordered (walk order or explicit sort both satisfy it).
        let sql = "SELECT id, a FROM m WHERE id > ?1 AND id < ?2 AND a > ?3 ORDER BY id";
        let params = [
            SqliteValue::Integer(300),
            SqliteValue::Integer(340),
            SqliteValue::Integer(2),
        ];
        assert_eq!(
            frank_bound(&f, sql, &params).await,
            sqlite_bound(&r, sql, &params),
            "ordered rowid range residual diverged"
        );
    });
}

/// Row count of a bound query, without rendering — the timing probe below measures plan
/// cost, not string formatting.
async fn count_bound(conn: &Connection, sql: &str, params: &[SqliteValue]) -> usize {
    let stmt = conn
        .prepare(sql)
        .await
        .unwrap_or_else(|e| panic!("prepare `{sql}`: {e}"));
    stmt.query_with_params(params)
        .await
        .unwrap_or_else(|e| panic!("bind `{sql}`: {e}"))
        .len()
}

/// Timing evidence for both scopes on a 500k-row in-memory table. Ignored by default;
/// run explicitly:
/// `cargo test -p fsqlite-e2e --test range_seek_residual_oracle_e2e timing -- --ignored --nocapture`
///
/// The incumbent full scan is measured IN THE SAME BINARY (same process, same table):
/// for the index scope via `NOT INDEXED` — the exact plan the pre-change code chose for
/// any `range AND residual`; for the rowid scope via `id > ?1 + 0` — semantically
/// identical to `id > ?1`, but the range extractor declines the computed bound, which is
/// exactly the full scan the pre-change code ran for `id > ?1 AND a > ?2`. The opcode
/// assertions verify the seek sides actually seek and the incumbent sides actually scan,
/// so the printed ratio can never silently measure seek-vs-seek. Row-set equality is
/// asserted; TIMING is printed, never asserted (no flaky speed gates).
#[test]
#[ignore = "timing evidence: 500k-row fixture; run explicitly with --ignored --nocapture"]
fn range_seek_residual_timing_500k() {
    asupersync::test_utils::run_test(|| async {
        let f = Connection::open(":memory:").await.expect("open frank");
        f.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER, v TEXT);")
            .await
            .expect("create big");
        f.execute("CREATE INDEX idx_big_k ON big(k);")
            .await
            .expect("create idx_big_k");
        const ROWS: i64 = 500_000;
        const BATCH: usize = 500;
        let mut batch_start = 1_i64;
        while batch_start <= ROWS {
            let mut sql = String::with_capacity(BATCH * 40 + 32);
            sql.push_str("INSERT INTO big VALUES ");
            for i in batch_start..(batch_start + BATCH as i64) {
                // ~1% NULL v so the residual actually filters.
                let v = if i % 97 == 0 {
                    "NULL".to_owned()
                } else {
                    format!("'v{}'", i % 100)
                };
                if i > batch_start {
                    sql.push_str(", ");
                }
                sql.push_str(&format!("({i}, {i}, {}, {v})", i % 100));
            }
            sql.push(';');
            f.execute(&sql)
                .await
                .unwrap_or_else(|e| panic!("load: {e}"));
            batch_start += BATCH as i64;
        }
        let stmt = f
            .prepare("SELECT COUNT(*) FROM big")
            .await
            .expect("count fixture prepare");
        let rows = stmt
            .query_with_params(&[])
            .await
            .expect("count fixture query");
        let loaded = match rows.first().and_then(|row| row.values().first()) {
            Some(SqliteValue::Integer(n)) => *n,
            other => panic!("fixture COUNT(*) returned {other:?}"),
        };
        assert_eq!(loaded, ROWS, "fixture row count");

        let reps = 3_usize;
        let mut timings: Vec<(f64, f64)> = Vec::new(); // (incumbent_scan_s, seek_s) per rep

        // ── Scope 1: index range + residual (the issue-draft shape) ──────────────
        let seek_sql = "SELECT id, v FROM big WHERE k >= ? AND k < ? AND v IS NOT NULL";
        let scan_sql = "SELECT id, v FROM big NOT INDEXED WHERE k >= ? AND k < ? AND v IS NOT NULL";
        let params = [SqliteValue::Integer(400_000), SqliteValue::Integer(400_500)];
        let ops = opcodes(&f, seek_sql).await;
        assert!(
            uses_index(&ops),
            "timing: the index seek side must walk the index; ops = {ops:?}"
        );
        let ops = opcodes(&f, scan_sql).await;
        assert!(
            !uses_index(&ops),
            "timing: the NOT INDEXED side must full-scan; ops = {ops:?}"
        );
        let n_seek = count_bound(&f, seek_sql, &params).await;
        let n_scan = count_bound(&f, scan_sql, &params).await;
        assert_eq!(n_seek, n_scan, "timing: index seek/scan row sets differ");
        count_bound(&f, seek_sql, &params).await; // warm both paths
        count_bound(&f, scan_sql, &params).await;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            count_bound(&f, scan_sql, &params).await;
            let scan_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            count_bound(&f, seek_sql, &params).await;
            let seek_s = t.elapsed().as_secs_f64();
            timings.push((scan_s, seek_s));
        }
        let mut sorted_t: Vec<(f64, f64)> = timings.clone();
        sorted_t.sort_by(|a, b| a.0.total_cmp(&b.0));
        let med_scan = sorted_t[sorted_t.len() / 2].0;
        sorted_t.sort_by(|a, b| a.1.total_cmp(&b.1));
        let med_seek = sorted_t[sorted_t.len() / 2].1;
        println!(
            "TIMING scope1 index range+residual (500k rows, {n_seek} matching): \
             full scan (NOT INDEXED) median {med_scan:.4}s vs range walk median {med_seek:.4}s \
             -> speedup {:.0}x  [reps: {timings:?}]",
            med_scan / med_seek.max(1e-9),
        );

        // ── Scope 2: rowid range (placeholder bounds) + residual ─────────────────
        timings.clear();
        let seek_sql = "SELECT id, v FROM big WHERE id > ?1 AND a > ?2";
        // The computed bound `?1 + 0` is semantically identical but not extractable —
        // the pre-change plan (full scan) as the live incumbent.
        let scan_sql = "SELECT id, v FROM big WHERE id > ?1 + 0 AND a > ?2";
        let params = [SqliteValue::Integer(499_000), SqliteValue::Integer(50)];
        let ops = opcodes(&f, seek_sql).await;
        assert!(
            rowid_seeks(&ops),
            "timing: the rowid seek side must seek the table b-tree; ops = {ops:?}"
        );
        let ops = opcodes(&f, scan_sql).await;
        assert!(
            ops.iter().any(|o| o == "Rewind"),
            "timing: the computed-bound side must full-scan; ops = {ops:?}"
        );
        let n_seek = count_bound(&f, seek_sql, &params).await;
        let n_scan = count_bound(&f, scan_sql, &params).await;
        assert_eq!(n_seek, n_scan, "timing: rowid seek/scan row sets differ");
        count_bound(&f, seek_sql, &params).await;
        count_bound(&f, scan_sql, &params).await;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            count_bound(&f, scan_sql, &params).await;
            let scan_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            count_bound(&f, seek_sql, &params).await;
            let seek_s = t.elapsed().as_secs_f64();
            timings.push((scan_s, seek_s));
        }
        let mut sorted_t: Vec<(f64, f64)> = timings.clone();
        sorted_t.sort_by(|a, b| a.0.total_cmp(&b.0));
        let med_scan = sorted_t[sorted_t.len() / 2].0;
        sorted_t.sort_by(|a, b| a.1.total_cmp(&b.1));
        let med_seek = sorted_t[sorted_t.len() / 2].1;
        println!(
            "TIMING scope2 rowid range(placeholder)+residual (500k rows, {n_seek} matching): \
             full scan (computed bound) median {med_scan:.4}s vs slice walk median {med_seek:.4}s \
             -> speedup {:.1}x  [reps: {timings:?}]",
            med_scan / med_seek.max(1e-9),
        );
    });
}
