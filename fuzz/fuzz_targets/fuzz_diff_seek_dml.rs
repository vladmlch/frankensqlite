#![no_main]

//! Differential fuzz target: FrankenSQLite vs stock SQLite (rusqlite, bundled
//! C SQLite) over the seek/DML query shapes the fork's commits optimise
//! (db5d846e DML prefix seeks, aed53a33 rowid range placeholder bounds,
//! e4c17486 index range with residual conjuncts, 97ba5802 grouped residual
//! prefixes, bd-7orjo rowid-driven UPDATE, bd-gtjlv param prefix seeks):
//!
//!   1. single-column index ranges with residual conjuncts, including a
//!      UNIQUE TEXT key (`WHERE k >= ? AND k < ? AND v IS NOT NULL`);
//!   2. rowid range seeks with numbered placeholders (`WHERE id > ?1 AND
//!      id < ?2`), also ?N reuse, mixed literal/param bounds, inclusive and
//!      exclusive bounds, empty/inverted ranges, NULL and mixed-type params;
//!   3. DELETE/UPDATE whose equality conjuncts pin an index prefix (single
//!      and composite indexes, extra residual conjuncts, RETURNING,
//!      AUTOINCREMENT, a trigger) — comparing affected rows, `changes()`,
//!      RETURNING rows and the full final table state;
//!   4. SELECTs with ORDER BY / LIMIT over the same indexes.
//!
//! Every input decodes to a small schema + rows + one statement from those
//! shapes. The statement runs on both engines against identical databases;
//! ANY observable divergence panics:
//!
//!   * success/error asymmetry, or both-fail with a different SQLite
//!     primary error code (class comparison via `FrankenError::error_code()`
//!     vs rusqlite's primary code — message wording is normalised away,
//!     classes are not);
//!   * result rows — element-wise when the template's ORDER BY fixes a
//!     total order, else as a sorted multiset;
//!   * `SELECT changes()` after the statement, after the probe INSERT;
//!   * every table dump (ordered by the INTEGER PRIMARY KEY), plus
//!     `sqlite_sequence` for the AUTOINCREMENT schema;
//!   * per-row setup INSERT outcomes (engines must agree which rows a
//!     UNIQUE constraint rejects).
//!
//! Determinism: one fresh in-memory database per engine per input, no
//! threads spawned by the harness, no wall-clock, no randomness beyond the
//! input bytes (all "random" choices are `byte % N` picks from fixed pools).
//! The asupersync runtime is a thread-local current-thread runtime reused
//! across inputs (the same `RuntimeBuilder::current_thread()` the engine's
//! own test utils use).
//!
//! Seed corpus: derived from
//! `crates/fsqlite-core/tests/bd_j9gub_dml_prefix_seek_cases.json`.

use std::future::Future;

use asupersync::prelude::RuntimeBuilder;
use fsqlite_core::connection::Connection as FrankConnection;
use fsqlite_error::FrankenError;
use fsqlite_types::SqliteValue;
use libfuzzer_sys::fuzz_target;
use rusqlite::Connection as StockConnection;
use rusqlite::types::Value as StockValue;

// ---------------------------------------------------------------------------
// Value pool
// ---------------------------------------------------------------------------

/// One entry of the fixed value pool. Values are chosen to collide (duplicate
/// integers, NULLs, empty strings), to sit on affinity edges (int/real/text
/// in the "wrong" column), and to hug the i64 limits.
#[derive(Clone, Copy, Debug)]
enum PoolVal {
    Null,
    Int(i64),
    Real(f64),
    Text(&'static str),
    Blob(&'static [u8]),
}

const POOL: &[PoolVal] = &[
    PoolVal::Null,
    // integers: small colliding values, negatives, i64 edges
    PoolVal::Int(0),
    PoolVal::Int(1),
    PoolVal::Int(2),
    PoolVal::Int(3),
    PoolVal::Int(5),
    PoolVal::Int(7),
    PoolVal::Int(8),
    PoolVal::Int(9),
    PoolVal::Int(10),
    PoolVal::Int(20),
    PoolVal::Int(42),
    PoolVal::Int(-1),
    PoolVal::Int(-4),
    PoolVal::Int(17),
    PoolVal::Int(i64::MIN),
    PoolVal::Int(i64::MAX),
    PoolVal::Int(i64::MAX - 1),
    // reals: affinity edges and magnitude edges
    PoolVal::Real(1.5),
    PoolVal::Real(-2.5),
    PoolVal::Real(0.0),
    PoolVal::Real(1.0e300),
    PoolVal::Real(0.1),
    // texts: empty, case pairs, numeric-looking, fixture values
    PoolVal::Text(""),
    PoolVal::Text("a"),
    PoolVal::Text("A"),
    PoolVal::Text("ab"),
    PoolVal::Text("b"),
    PoolVal::Text("p"),
    PoolVal::Text("q"),
    PoolVal::Text("s"),
    PoolVal::Text("v1"),
    PoolVal::Text("v2"),
    PoolVal::Text("v3"),
    PoolVal::Text("x"),
    PoolVal::Text("y"),
    PoolVal::Text("z"),
    PoolVal::Text("k1"),
    PoolVal::Text("k2"),
    PoolVal::Text("new"),
    PoolVal::Text("1"),
    PoolVal::Text("10"),
    PoolVal::Text("r00"),
    PoolVal::Text("0.5"),
    // blobs
    PoolVal::Blob(&[0x00]),
    PoolVal::Blob(&[0x6b, 0x32]),
];

fn pool_at(byte: u8) -> &'static PoolVal {
    &POOL[usize::from(byte) % POOL.len()]
}

fn pool_frank(value: &PoolVal) -> SqliteValue {
    match value {
        PoolVal::Null => SqliteValue::Null,
        PoolVal::Int(n) => SqliteValue::Integer(*n),
        PoolVal::Real(f) => SqliteValue::Float(*f),
        PoolVal::Text(s) => SqliteValue::from(*s),
        PoolVal::Blob(bytes) => SqliteValue::Blob(bytes.to_vec().into()),
    }
}

fn pool_stock(value: &PoolVal) -> StockValue {
    match value {
        PoolVal::Null => StockValue::Null,
        PoolVal::Int(n) => StockValue::Integer(*n),
        PoolVal::Real(f) => StockValue::Real(*f),
        PoolVal::Text(s) => StockValue::Text((*s).to_owned()),
        PoolVal::Blob(bytes) => StockValue::Blob(bytes.to_vec()),
    }
}

fn pool_show(value: &PoolVal) -> String {
    match value {
        PoolVal::Null => "NULL".to_owned(),
        PoolVal::Int(n) => n.to_string(),
        PoolVal::Real(f) => format!("{f}"),
        PoolVal::Text(s) => format!("'{s}'"),
        PoolVal::Blob(bytes) => format!("x'{}'", hex(bytes)),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Canonical comparison values
// ---------------------------------------------------------------------------

/// Storage-class-exact comparison value. Reals compare bitwise (the harness
/// never computes floats, so engines must produce identical bits); texts
/// compare exact bytes (invalid UTF-8 preserved via `as_bytes_direct`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CanonVal {
    Null,
    Int(i64),
    Real(u64),
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

impl CanonVal {
    fn show(&self) -> String {
        match self {
            Self::Null => "NULL".to_owned(),
            Self::Int(n) => n.to_string(),
            Self::Real(bits) => format!("{} /*bits={bits}*/", f64::from_bits(*bits)),
            Self::Text(bytes) => format!("'{}'", String::from_utf8_lossy(bytes)),
            Self::Blob(bytes) => format!("x'{}'", hex(bytes)),
        }
    }
}

fn canon_frank(value: &SqliteValue) -> CanonVal {
    match value {
        SqliteValue::Null => CanonVal::Null,
        SqliteValue::Integer(n) => CanonVal::Int(*n),
        SqliteValue::Float(f) => CanonVal::Real(f.to_bits()),
        SqliteValue::Text(text) => CanonVal::Text(text.as_bytes_direct().to_vec()),
        SqliteValue::Blob(bytes) => CanonVal::Blob(bytes.to_vec()),
    }
}

fn canon_stock(value: &StockValue) -> CanonVal {
    match value {
        StockValue::Null => CanonVal::Null,
        StockValue::Integer(n) => CanonVal::Int(*n),
        StockValue::Real(f) => CanonVal::Real(f.to_bits()),
        StockValue::Text(text) => CanonVal::Text(text.as_bytes().to_vec()),
        StockValue::Blob(bytes) => CanonVal::Blob(bytes.clone()),
    }
}

fn rows_show(rows: &[Vec<CanonVal>]) -> String {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(CanonVal::show)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

// ---------------------------------------------------------------------------
// Error classes
// ---------------------------------------------------------------------------

/// FrankenSQLite error class: its own SQLite-primary-code mapping.
fn frank_class(error: &FrankenError) -> i64 {
    i64::from(error.error_code() as i32)
}

/// Stock SQLite error class: the primary code (low byte of the extended
/// code) of the underlying `SQLITE_ERROR`-family failure. rusqlite-side
/// wrapping errors (never produced by this harness's statements in practice)
/// map to the generic `SQLITE_ERROR` class 1.
fn stock_class(error: &rusqlite::Error) -> i64 {
    match error {
        rusqlite::Error::SqliteFailure(ffi_error, _) => i64::from(ffi_error.extended_code & 0xff),
        _ => 1,
    }
}

const CLASS_NAMES: &[(i64, &str)] = &[
    (0, "OK"),
    (1, "SQLITE_ERROR"),
    (2, "SQLITE_INTERNAL"),
    (4, "SQLITE_ABORT"),
    (5, "SQLITE_BUSY"),
    (6, "SQLITE_LOCKED"),
    (8, "SQLITE_READONLY"),
    (11, "SQLITE_CORRUPT"),
    (13, "SQLITE_FULL"),
    (17, "SQLITE_SCHEMA"),
    (18, "SQLITE_TOOBIG"),
    (19, "SQLITE_CONSTRAINT"),
    (20, "SQLITE_MISMATCH"),
    (25, "SQLITE_RANGE"),
];

fn class_name(code: i64) -> String {
    CLASS_NAMES
        .iter()
        .find(|(key, _)| *key == code)
        .map_or_else(|| format!("SQLITE_{code}"), |(_, name)| (*name).to_owned())
}

// ---------------------------------------------------------------------------
// Schemas and statement templates
// ---------------------------------------------------------------------------

/// A statement template. `params` is the number of DISTINCT numbered
/// placeholders (?1 .. ?N, consecutive; reuse via repeated ?N costs one
/// bind). `ordered` is true only when the ORDER BY ends in a unique key so
/// the result order is total; otherwise rows compare as a sorted multiset.
struct Template {
    sql: &'static str,
    params: usize,
    ordered: bool,
}

const fn t(sql: &'static str, params: usize, ordered: bool) -> Template {
    Template {
        sql,
        params,
        ordered,
    }
}

/// A fixed schema: DDL, the prepared row INSERT (explicit id for rowid
/// tables, engine-assigned id for the AUTOINCREMENT table), the probe INSERT
/// (the same statement with fresh pool values — exposes rowid/AUTOINCREMENT
/// allocation in the dumps), ordered dumps, and the template list.
struct Schema {
    name: &'static str,
    ddl: &'static [&'static str],
    insert_sql: &'static str,
    explicit_id: bool,
    row_cols: usize,
    dumps: &'static [&'static str],
    templates: &'static [Template],
}

/// S0 — composite int indexes (bd-j9gub `t(a,b)` / `t(a,n)`), plus rowid-range
/// shapes and the non-prefix scan control.
const S0: Schema = Schema {
    name: "s0_composite_int",
    ddl: &[
        "CREATE TABLE t(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, n INTEGER, v TEXT)",
        "CREATE INDEX t_ab ON t(a,b)",
        "CREATE INDEX t_an ON t(a,n)",
    ],
    insert_sql: "INSERT INTO t(id,a,b,n,v) VALUES(?,?,?,?,?)",
    explicit_id: true,
    row_cols: 4,
    dumps: &["SELECT id,a,b,n,v FROM t ORDER BY id"],
    templates: &[
        t("DELETE FROM t WHERE a = ?1", 1, false),
        t(
            "UPDATE t SET v = ?3 WHERE a = ?1 AND b > ?2 RETURNING id, v",
            3,
            false,
        ),
        t("DELETE FROM t WHERE a = ?1 AND b = ?2", 2, false),
        t(
            "UPDATE t SET n = n + 1 WHERE a = ?1 AND n = ?2 AND b IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id, a, b FROM t WHERE a = ?1 AND b > ?2 AND v IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id, a, b, n, v FROM t WHERE a = ?1 ORDER BY id LIMIT ?2",
            2,
            true,
        ),
        t("DELETE FROM t WHERE a = ?1 AND b > ?2 AND b < ?3", 3, false),
        t(
            "SELECT id FROM t WHERE a >= ?1 AND a < ?2 AND v IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id, a FROM t WHERE a BETWEEN ?1 AND ?2 AND b IS NOT NULL ORDER BY a, id",
            2,
            true,
        ),
        t(
            "DELETE FROM t WHERE a = ?1 AND b >= ?2 AND b <= ?3",
            3,
            false,
        ),
        t("UPDATE t SET b = ?2 WHERE a = ?1 RETURNING id, b", 2, false),
        t(
            "SELECT id, b, n FROM t WHERE a = ?1 AND b = ?2 AND n > ?3 AND v IS NOT NULL",
            3,
            false,
        ),
        t("DELETE FROM t WHERE b = ?1 AND n = ?2", 2, false),
        t(
            "SELECT id, a, b FROM t WHERE id > ?1 AND id < ?2 AND a IS NOT NULL",
            2,
            false,
        ),
        t("DELETE FROM t WHERE id >= ?1 AND id < ?2", 2, false),
        t("UPDATE t SET v = ?3 WHERE id > ?1 AND id <= ?2", 3, false),
        t(
            "SELECT id, a FROM t WHERE id BETWEEN ?1 AND ?2 AND n IS NOT NULL ORDER BY id DESC LIMIT ?3",
            3,
            true,
        ),
        t(
            "SELECT id, a, b, n FROM t WHERE a = ?1 AND b > ?2 AND b < ?3 ORDER BY b, id",
            3,
            true,
        ),
        t(
            "DELETE FROM t WHERE a = ?1 AND b = ?2 RETURNING id",
            2,
            false,
        ),
        t("UPDATE t SET v = ?2 WHERE a = ?1", 2, false),
        t("DELETE FROM t WHERE n = ?1", 1, false),
        t("UPDATE t SET v = ?2 WHERE b = ?1", 2, false),
        t("DELETE FROM t WHERE b = ?1", 1, false),
        t("UPDATE t SET v = ?2 WHERE n = ?1", 2, false),
    ],
};

/// S1 — single-column int + text indexes (the e2e range-seek residual shape).
const S1: Schema = Schema {
    name: "s1_single_int_text",
    ddl: &[
        "CREATE TABLE t(id INTEGER PRIMARY KEY, k INTEGER, w TEXT, v TEXT)",
        "CREATE INDEX idx_t_k ON t(k)",
        "CREATE INDEX idx_t_w ON t(w)",
    ],
    insert_sql: "INSERT INTO t(id,k,w,v) VALUES(?,?,?,?)",
    explicit_id: true,
    row_cols: 3,
    dumps: &["SELECT id,k,w,v FROM t ORDER BY id"],
    templates: &[
        t(
            "SELECT id,k,v FROM t WHERE k >= ?1 AND k < ?2 AND v IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k > ?1 AND k <= ?2 AND w IS NOT NULL AND w != ?3",
            3,
            false,
        ),
        t(
            "SELECT id,w,v FROM t WHERE w >= ?1 AND w < ?2 AND v IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k BETWEEN ?1 AND ?2 AND v = ?3",
            3,
            false,
        ),
        t(
            "SELECT id,k,w,v FROM t WHERE k >= ?1 AND k < ?2 AND v IS NOT NULL ORDER BY k, id",
            2,
            true,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k >= ?1 AND k < ?2 AND v IS NOT NULL ORDER BY k DESC, id DESC LIMIT ?3",
            3,
            true,
        ),
        t(
            "SELECT id,w FROM t WHERE w > ?1 AND w <= ?2 ORDER BY w, id LIMIT ?3",
            3,
            true,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k IN (?1, ?2, ?3) AND v IS NOT NULL",
            3,
            false,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k = ?1 AND v IS NOT NULL",
            1,
            false,
        ),
        t("DELETE FROM t WHERE k >= ?1 AND k < ?2", 2, false),
        t("UPDATE t SET v = ?3 WHERE k = ?1 AND w = ?2", 3, false),
        t(
            "SELECT id,k FROM t WHERE k > ?1 AND k < ?2 AND k <> ?3",
            3,
            false,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k >= ?1 AND k <= ?1 AND v IS NOT NULL",
            1,
            false,
        ),
        t(
            "DELETE FROM t WHERE k = ?1 AND w = ?2 RETURNING id",
            2,
            false,
        ),
        t("DELETE FROM t WHERE k = ?1", 1, false),
        // ?2, not ?3: a skipped index (?1, ?3) makes SQLite's declared
        // parameter count 3 while only 2 slots carry values, which no
        // positional bind API can express (rusqlite and frankensqlite both
        // require one value per declared index) — the engines then disagree
        // only in their API-layer misuse error class, which is not a SQL
        // semantics divergence. Renumbered to the contiguous form the S3
        // sibling already uses; the value-to-slot mapping is unchanged.
        t("UPDATE t SET v = ?2 WHERE k = ?1 RETURNING id, v", 2, false),
        t(
            "SELECT id,k,v FROM t WHERE id > ?1 AND id <= ?2 AND k IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id,k,v FROM t WHERE id >= ?1 AND id < 100 AND v = ?2",
            2,
            false,
        ),
        t("SELECT id,k,v FROM t WHERE id > ?1 AND id <= ?1", 1, false),
        t("DELETE FROM t WHERE id >= ?1 AND id < ?2", 2, false),
        t("UPDATE t SET v = ?3 WHERE id > ?1 AND id < ?2", 3, false),
        t(
            "SELECT id FROM t WHERE id > ?1 AND id < ?2 ORDER BY id DESC LIMIT ?3",
            3,
            true,
        ),
        t("UPDATE t SET k = ?2 WHERE id = ?1", 2, false),
        t(
            "SELECT id,k,v FROM t WHERE id > ?1 AND id < ?2 ORDER BY id",
            2,
            true,
        ),
        t("UPDATE t SET w = ?2 WHERE k = ?1", 2, false),
    ],
};

/// S2 — UNIQUE TEXT key (`key TEXT UNIQUE`) plus a value index: text range
/// seeks, unique-key equality DML, RETURNING.
const S2: Schema = Schema {
    name: "s2_unique_text",
    ddl: &[
        "CREATE TABLE kv(id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT)",
        "CREATE INDEX idx_kv_value ON kv(value)",
    ],
    insert_sql: "INSERT INTO kv(id,key,value) VALUES(?,?,?)",
    explicit_id: true,
    row_cols: 2,
    dumps: &["SELECT id,key,value FROM kv ORDER BY id"],
    templates: &[
        t(
            "SELECT id,key,value FROM kv WHERE key >= ?1 AND key < ?2 AND value IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id,key,value FROM kv WHERE key > ?1 AND key <= ?2",
            2,
            false,
        ),
        t("SELECT id,key FROM kv WHERE key = ?1", 1, false),
        t(
            "SELECT id,key,value FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key, id",
            2,
            true,
        ),
        t(
            "SELECT id,value FROM kv WHERE value = ?1 AND key > ?2",
            2,
            false,
        ),
        t("DELETE FROM kv WHERE key = ?1", 1, false),
        t(
            "UPDATE kv SET value = ?2 WHERE key >= ?1 AND key < ?3",
            3,
            false,
        ),
        t(
            "SELECT id,key,value FROM kv WHERE key >= ?1 AND key < ?1 AND value IS NOT NULL",
            1,
            false,
        ),
        t(
            "SELECT id,key,value FROM kv WHERE key >= ?1 ORDER BY key, id LIMIT ?2",
            2,
            true,
        ),
        t(
            "DELETE FROM kv WHERE key > ?1 AND key < ?2 RETURNING id, key",
            2,
            false,
        ),
        t(
            "SELECT id,value FROM kv WHERE value >= ?1 AND value < ?2 AND key IS NOT NULL",
            2,
            false,
        ),
        t(
            "UPDATE kv SET value = value || '!' WHERE key = ?1 RETURNING id",
            1,
            false,
        ),
        t(
            "SELECT id,key,value FROM kv WHERE id > ?1 AND id < ?2 ORDER BY id",
            2,
            true,
        ),
        t("DELETE FROM kv WHERE id >= ?1 AND id <= ?2", 2, false),
        t("DELETE FROM kv WHERE key = ?1 AND value = ?2", 2, false),
        t("DELETE FROM kv WHERE key = ?1 AND value > ?2", 2, false),
    ],
};

/// S3 — AUTOINCREMENT rowid allocation, an AFTER DELETE trigger, RETURNING,
/// and the `sqlite_sequence` dump.
const S3: Schema = Schema {
    name: "s3_autoinc_trigger",
    ddl: &[
        "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, k INTEGER, v TEXT)",
        "CREATE INDEX idx_t_k ON t(k)",
        "CREATE TABLE log(id INTEGER PRIMARY KEY, tag TEXT, n INTEGER)",
        "CREATE TRIGGER trg_del AFTER DELETE ON t FOR EACH ROW BEGIN INSERT INTO log(tag, n) VALUES('del', OLD.k); END",
    ],
    insert_sql: "INSERT INTO t(k,v) VALUES(?,?)",
    explicit_id: false,
    row_cols: 2,
    dumps: &[
        "SELECT id,k,v FROM t ORDER BY id",
        "SELECT id,tag,n FROM log ORDER BY id",
        "SELECT name, seq FROM sqlite_sequence ORDER BY name",
    ],
    templates: &[
        t(
            "SELECT id,k,v FROM t WHERE k >= ?1 AND k < ?2 AND v IS NOT NULL",
            2,
            false,
        ),
        t("DELETE FROM t WHERE k = ?1", 1, false),
        t(
            "DELETE FROM t WHERE k >= ?1 AND k < ?2 RETURNING id, k, v",
            2,
            false,
        ),
        t("UPDATE t SET v = ?2 WHERE k = ?1 RETURNING id, v", 2, false),
        t("SELECT id,k,v FROM t WHERE id > ?1 AND id < ?2", 2, false),
        t(
            "SELECT id,k,v FROM t WHERE id >= ?1 AND id < ?2 ORDER BY id",
            2,
            true,
        ),
        t("DELETE FROM t WHERE id > ?1 AND id <= ?2", 2, false),
        t("UPDATE t SET k = ?2 WHERE id = ?1", 2, false),
        t(
            "SELECT id,k,v FROM t WHERE id BETWEEN ?1 AND ?2 AND k IS NOT NULL ORDER BY id DESC LIMIT ?3",
            3,
            true,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k > ?1 AND k <= ?2 AND v IS NOT NULL ORDER BY k DESC, id DESC",
            2,
            true,
        ),
        t(
            "SELECT id,k,v FROM t WHERE k >= ?1 AND k < ?2 AND v IS NOT NULL LIMIT ?3",
            3,
            false,
        ),
        t("UPDATE t SET v = ?3 WHERE id > ?1 AND id < ?2", 3, false),
        t("UPDATE t SET v = ?3 WHERE k = ?1 AND v = ?2", 3, false),
    ],
};

/// S4 — trailing-DESC and leading-DESC indexes (bd-j9gub case 8).
const S4: Schema = Schema {
    name: "s4_desc_index",
    ddl: &[
        "CREATE TABLE d(id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE INDEX d_ab ON d(a, b DESC)",
        "CREATE INDEX d_ba ON d(b DESC, a)",
    ],
    insert_sql: "INSERT INTO d(id,a,b) VALUES(?,?,?)",
    explicit_id: true,
    row_cols: 2,
    dumps: &["SELECT id,a,b FROM d ORDER BY id"],
    templates: &[
        t("DELETE FROM d WHERE a = ?1", 1, false),
        t("UPDATE d SET b = b + 1 WHERE a = ?1", 1, false),
        t("DELETE FROM d WHERE b = ?1", 1, false),
        t(
            "SELECT id,a,b FROM d WHERE a = ?1 AND b > ?2 AND b < ?3",
            3,
            false,
        ),
        t(
            "SELECT id,a,b FROM d WHERE a >= ?1 AND a < ?2 AND b IS NOT NULL",
            2,
            false,
        ),
        t(
            "SELECT id,a FROM d WHERE a = ?1 ORDER BY b DESC, id DESC",
            1,
            true,
        ),
        t(
            "DELETE FROM d WHERE a = ?1 AND b = ?2 RETURNING id",
            2,
            false,
        ),
        t(
            "SELECT id,b FROM d WHERE b >= ?1 AND b < ?2 ORDER BY b DESC, id DESC LIMIT ?3",
            3,
            true,
        ),
        t(
            "SELECT id,a,b FROM d WHERE id > ?1 AND id <= ?2 AND a IS NOT NULL",
            2,
            false,
        ),
    ],
};

const SCHEMAS: &[Schema] = &[S0, S1, S2, S3, S4];

/// Maximum generated rows per input. Small on purpose: collisions and edge
/// values matter more than volume for divergence detection, and the per-input
/// statement count drives exec/s.
const MAX_ROWS: usize = 16;

// ---------------------------------------------------------------------------
// Input decoding
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn next(&mut self) -> u8 {
        let byte = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        byte
    }
}

struct Case {
    schema: &'static Schema,
    rows: Vec<Vec<&'static PoolVal>>,
    template: &'static Template,
    params: Vec<&'static PoolVal>,
    probe: Vec<&'static PoolVal>,
    input_hex: String,
}

fn decode(data: &[u8]) -> Case {
    let mut cursor = Cursor { data, pos: 0 };
    let schema = &SCHEMAS[usize::from(cursor.next()) % SCHEMAS.len()];
    let row_count = usize::from(cursor.next()) % (MAX_ROWS + 1);
    let mut rows = Vec::with_capacity(row_count);
    for _ in 0..row_count {
        rows.push(
            (0..schema.row_cols)
                .map(|_| pool_at(cursor.next()))
                .collect(),
        );
    }
    let template = &schema.templates[usize::from(cursor.next()) % schema.templates.len()];
    let params = (0..template.params)
        .map(|_| pool_at(cursor.next()))
        .collect();
    let mut probe: Vec<&'static PoolVal> = (0..schema.row_cols)
        .map(|_| pool_at(cursor.next()))
        .collect();
    // The probe insert binds the SAME insert_sql as the row inserts. For
    // explicit-id schemas that SQL has one more placeholder than row_cols,
    // and the row path binds an explicit id. The probe instead binds NULL
    // for the id column on purpose: both engines then assign it themselves,
    // so the ordered dumps compare engine rowid / AUTOINCREMENT allocation.
    // (Binding only row_cols params would make both engines reject the
    // statement with API-layer error classes that differ by construction —
    // rusqlite pre-checks the parameter count, frankensqlite fails in the
    // binder — which is harness misuse, not an engine divergence.)
    if schema.explicit_id {
        probe.insert(0, &POOL[0]);
    }
    Case {
        schema,
        rows,
        template,
        params,
        probe,
        input_hex: hex(data),
    }
}

impl Case {
    /// Human-readable reproduction script: run the DDL + INSERTs, then the
    /// mutation, on any SQLite engine.
    fn describe(&self) -> String {
        let mut text = String::new();
        text.push_str(&format!("input hex: {}\n", self.input_hex));
        text.push_str(&format!("schema: {}\n", self.schema.name));
        for ddl in self.schema.ddl {
            text.push_str(&format!("{ddl};\n"));
        }
        for (index, row) in self.rows.iter().enumerate() {
            let rendered: Vec<_> = row.iter().map(|value| pool_show(value)).collect();
            if self.schema.explicit_id {
                text.push_str(&format!(
                    "INSERT: id={}, {}\n",
                    index + 1,
                    rendered.join(", ")
                ));
            } else {
                text.push_str(&format!("INSERT: {}\n", rendered.join(", ")));
            }
        }
        text.push_str(&format!("statement: {}\n", self.template.sql));
        let params: Vec<_> = self.params.iter().map(|value| pool_show(value)).collect();
        text.push_str(&format!("params: [{}]\n", params.join(", ")));
        let probe: Vec<_> = self.probe.iter().map(|value| pool_show(value)).collect();
        text.push_str(&format!("probe insert: [{}]\n", probe.join(", ")));
        text
    }

    fn panic(&self, aspect: &str, detail: &str) -> ! {
        panic!(
            "DIFFERENTIAL DIVERGENCE [{aspect}]\n{detail}\n{}",
            self.describe()
        );
    }
}

// ---------------------------------------------------------------------------
// Engine drivers
// ---------------------------------------------------------------------------

async fn frank_rows(
    conn: &FrankConnection,
    sql: &str,
    params: &[SqliteValue],
) -> Result<Vec<Vec<CanonVal>>, FrankenError> {
    let rows = conn.query_with_params(sql, params).await?;
    Ok(rows
        .iter()
        .map(|row| row.values().iter().map(canon_frank).collect())
        .collect())
}

fn stock_rows(
    conn: &StockConnection,
    sql: &str,
    params: &[StockValue],
) -> Result<Vec<Vec<CanonVal>>, rusqlite::Error> {
    let mut statement = conn.prepare(sql)?;
    let width = statement.column_count();
    let mut rows = statement.query(rusqlite::params_from_iter(params.iter()))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let mut rendered = Vec::with_capacity(width);
        for column in 0..width {
            rendered.push(canon_stock(&row.get::<_, StockValue>(column)?));
        }
        out.push(rendered);
    }
    Ok(out)
}

/// Compare one statement's outcome on both engines: class on error, rows on
/// success (element-wise when `ordered`, else as a sorted multiset).
async fn compare_statement(
    case: &Case,
    frank: &FrankConnection,
    stock: &StockConnection,
    aspect: &str,
    sql: &str,
    frank_params: &[SqliteValue],
    stock_params: &[StockValue],
    ordered: bool,
) {
    let frank_outcome = frank_rows(frank, sql, frank_params).await;
    let stock_outcome = stock_rows(stock, sql, stock_params);
    match (frank_outcome, stock_outcome) {
        (Ok(mut frank_result), Ok(mut stock_result)) => {
            if !ordered {
                frank_result.sort();
                stock_result.sort();
            }
            if frank_result != stock_result {
                case.panic(
                    aspect,
                    &format!(
                        "{sql}\n  frankensqlite rows: [{}]\n  stock sqlite rows:   [{}]",
                        rows_show(&frank_result),
                        rows_show(&stock_result)
                    ),
                );
            }
        }
        (Err(frank_error), Err(stock_error)) => {
            let (frank_code, stock_code) = (frank_class(&frank_error), stock_class(&stock_error));
            if frank_code != stock_code {
                case.panic(
                    aspect,
                    &format!(
                        "{sql}\n  both engines failed with DIFFERENT error classes:\n  \
                         frankensqlite: {} [{frank_code}] {frank_error}\n  \
                         stock sqlite:  {} [{stock_code}] {stock_error}",
                        class_name(frank_code),
                        class_name(stock_code)
                    ),
                );
            }
        }
        (Err(frank_error), Ok(stock_result)) => case.panic(
            aspect,
            &format!(
                "{sql}\n  frankensqlite FAILED ({}) but stock sqlite succeeded with rows: [{}]",
                frank_error,
                rows_show(&stock_result)
            ),
        ),
        (Ok(frank_result), Err(stock_error)) => case.panic(
            aspect,
            &format!(
                "{sql}\n  stock sqlite FAILED ({stock_error}) but frankensqlite succeeded with rows: [{}]",
                rows_show(&frank_result)
            ),
        ),
    }
}

/// Compare a prepared INSERT's outcome on both engines.
async fn compare_insert(
    case: &Case,
    aspect: &str,
    frank: &mut fsqlite_core::connection::PreparedStatement<'_>,
    frank_params: &[SqliteValue],
    stock: &mut rusqlite::Statement<'_>,
    stock_params: &[StockValue],
) {
    let frank_outcome = frank.execute_with_params(frank_params).await;
    let stock_outcome = stock.execute(rusqlite::params_from_iter(stock_params.iter()));
    match (frank_outcome, stock_outcome) {
        (Ok(frank_count), Ok(stock_count)) => {
            if frank_count != stock_count {
                case.panic(
                    aspect,
                    &format!(
                        "INSERT affected-row counts differ: frankensqlite={frank_count}, stock={stock_count}"
                    ),
                );
            }
        }
        (Err(frank_error), Err(stock_error)) => {
            let (frank_code, stock_code) = (frank_class(&frank_error), stock_class(&stock_error));
            if frank_code != stock_code {
                case.panic(
                    aspect,
                    &format!(
                        "INSERT failed with DIFFERENT error classes:\n  \
                         frankensqlite: {} [{frank_code}] {frank_error}\n  \
                         stock sqlite:  {} [{stock_code}] {stock_error}",
                        class_name(frank_code),
                        class_name(stock_code),
                    ),
                );
            }
        }
        (Err(frank_error), Ok(_)) => {
            case.panic(
                aspect,
                &format!("INSERT: frankensqlite failed but stock succeeded: {frank_error}"),
            );
        }
        (Ok(_), Err(stock_error)) => {
            case.panic(
                aspect,
                &format!("INSERT: stock failed but frankensqlite succeeded: {stock_error}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The differential case
// ---------------------------------------------------------------------------

async fn run_case(data: &[u8]) {
    let case = decode(data);
    let schema = case.schema;

    let frank = FrankConnection::open(":memory:")
        .await
        .expect("open frankensqlite :memory:");
    let stock = StockConnection::open_in_memory().expect("open stock :memory:");

    // DDL — fixed, known-good on both engines; a failure here is a loud
    // harness/engine divergence, never silently skipped.
    for ddl in schema.ddl {
        let frank_outcome = frank.execute(ddl).await;
        let stock_outcome = stock.execute(ddl, []);
        match (frank_outcome, stock_outcome) {
            (Ok(_), Ok(_)) => {}
            (Err(frank_error), Err(stock_error)) => {
                let (frank_code, stock_code) =
                    (frank_class(&frank_error), stock_class(&stock_error));
                if frank_code != stock_code {
                    case.panic(
                        "ddl",
                        &format!(
                            "{ddl}\n  DDL failed with different classes: frank={} [{frank_code}], stock={} [{stock_code}]",
                            class_name(frank_code),
                            class_name(stock_code)
                        ),
                    );
                }
            }
            (Err(frank_error), Ok(_)) => {
                case.panic(
                    "ddl",
                    &format!("{ddl}\n  DDL: frankensqlite failed, stock ok: {frank_error}"),
                );
            }
            (Ok(_), Err(stock_error)) => {
                case.panic(
                    "ddl",
                    &format!("{ddl}\n  DDL: stock failed, frankensqlite ok: {stock_error}"),
                );
            }
        }
    }

    // Rows: prepared INSERT, one execution per row, outcomes compared per
    // row (a UNIQUE reject must happen on both engines).
    let mut frank_insert = frank
        .prepare(schema.insert_sql)
        .await
        .expect("frankensqlite prepare row insert");
    let mut stock_insert = stock
        .prepare(schema.insert_sql)
        .expect("stock prepare row insert");
    for (index, row) in case.rows.iter().enumerate() {
        let mut frank_params: Vec<SqliteValue> = Vec::with_capacity(schema.row_cols + 1);
        let mut stock_params: Vec<StockValue> = Vec::with_capacity(schema.row_cols + 1);
        if schema.explicit_id {
            frank_params.push(SqliteValue::Integer(index as i64 + 1));
            stock_params.push(StockValue::Integer(index as i64 + 1));
        }
        for value in row {
            frank_params.push(pool_frank(value));
            stock_params.push(pool_stock(value));
        }
        compare_insert(
            &case,
            &format!("row insert {index}"),
            &mut frank_insert,
            &frank_params,
            &mut stock_insert,
            &stock_params,
        )
        .await;
    }

    // The mutation / SELECT under test.
    let frank_params: Vec<SqliteValue> =
        case.params.iter().map(|value| pool_frank(value)).collect();
    let stock_params: Vec<StockValue> = case.params.iter().map(|value| pool_stock(value)).collect();
    compare_statement(
        &case,
        &frank,
        &stock,
        "statement",
        case.template.sql,
        &frank_params,
        &stock_params,
        case.template.ordered,
    )
    .await;

    // changes() after the statement.
    compare_statement(
        &case,
        &frank,
        &stock,
        "changes()",
        "SELECT changes()",
        &[],
        &[],
        true,
    )
    .await;

    // Probe INSERT: exposes engine rowid / AUTOINCREMENT allocation in the
    // dumps (and UNIQUE behaviour for the unique-key schema).
    let mut frank_probe: Vec<SqliteValue> = Vec::with_capacity(schema.row_cols);
    let mut stock_probe: Vec<StockValue> = Vec::with_capacity(schema.row_cols);
    for value in &case.probe {
        frank_probe.push(pool_frank(value));
        stock_probe.push(pool_stock(value));
    }
    compare_insert(
        &case,
        "probe insert",
        &mut frank_insert,
        &frank_probe,
        &mut stock_insert,
        &stock_probe,
    )
    .await;

    // changes() after the probe.
    compare_statement(
        &case,
        &frank,
        &stock,
        "changes() after probe",
        "SELECT changes()",
        &[],
        &[],
        true,
    )
    .await;

    // Final table state (plus trigger log and sqlite_sequence where present).
    for dump in schema.dumps {
        compare_statement(&case, &frank, &stock, "final state", dump, &[], &[], true).await;
    }

    frank.close().await.expect("close frankensqlite connection");
    // rusqlite::Statement's Drop keeps the connection borrow alive to scope
    // end; drop it explicitly before the connection so the borrow ends first.
    drop(stock_insert);
    drop(stock);
}

// ---------------------------------------------------------------------------
// Runtime plumbing
// ---------------------------------------------------------------------------

thread_local! {
    /// One current-thread asupersync runtime per fuzzing thread, reused
    /// across inputs (the engine's futures are polled by the caller's
    /// executor; this is the same builder its own test utils use).
    static RUNTIME: asupersync::prelude::Runtime =
        RuntimeBuilder::current_thread()
            .build()
            .expect("build asupersync current-thread runtime");
}

fn block_on<F: Future>(future: F) -> F::Output {
    RUNTIME.with(|runtime| runtime.block_on(future))
}

fuzz_target!(|data: &[u8]| {
    block_on(run_case(data));
});
