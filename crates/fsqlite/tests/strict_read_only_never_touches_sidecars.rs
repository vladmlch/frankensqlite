//! The strict read-only open mode (`SQLITE_OPEN_STRICT_READ_ONLY`,
//! `Connection::open_strict_read_only`, `compat::open_strict_read_only`)
//! never creates or modifies any sidecar of the database family.
//!
//! 0.4.13's plain read-only open rebuilds an existing-but-uninitialized
//! `-shm` in place (GH#528) — the exact write that makes an external store
//! unsafe to inspect. Every recovery class here therefore asserts the whole
//! directory's bytes AND mtimes are identical before and after a strict
//! open, and that no file appears or disappears, which covers `-shm`,
//! `-wal`, `-journal`, the MVCC companion (`.fsqlite-shm`), and `-wal-fec`
//! at once.
//!
//! A strict open answers `BusyRecovery` for every WAL-index state that
//! needs recovery to be read, exactly like 0.4.12's plain read-only pager
//! (the brief's preferred "refuse" answer): zeroed and empty `-shm`
//! (uninitialized, the classes plain 0.4.13 rebuilds), torn and garbage
//! `-shm` (initialized-but-invalid, already refused by plain 0.4.13), and
//! a missing `-shm` beside a WAL with committed frames (never created by
//! any read-only pager). The two never-write reads stay available: a valid
//! `-shm` is used as-is, and a header-less WAL without a `-shm` reads the
//! main file (GH#515).
//!
//! The last test pins plain read-only to its 0.4.13 behavior, so a strict
//! regression cannot be hidden by changing the plain path instead.

#![cfg(all(unix, feature = "native"))]
#![recursion_limit = "512"]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use fsqlite::compat::{OpenFlags, open_strict_read_only, open_with_flags};
use fsqlite::{Connection, FrankenError};
use fsqlite_types::SqliteValue;

const ROWS: i64 = 12;
/// One full WAL-index region, the unit the stock first attacher allocates.
const SHM_REGION_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WalShape {
    /// Committed frames stay in the `-wal`; only a WAL-index read sees them.
    Frames,
    /// `wal_checkpoint(TRUNCATE)` settled every frame into the main file and
    /// left the `-wal` at zero bytes (a header-less WAL).
    Checkpointed,
}

impl WalShape {
    fn name(self) -> &'static str {
        match self {
            Self::Frames => "frames",
            Self::Checkpointed => "checkpointed",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShmClass {
    /// No `-shm` at all.
    Absent,
    /// One full region of zeroes: present but never initialized.
    Zeroed,
    /// A zero-byte `-shm`.
    Empty,
    /// Header copy 1's checksum word flipped, so the two copies disagree.
    Torn,
    /// One full region of `0xA5`: initialized but invalid.
    Garbage,
}

impl ShmClass {
    fn name(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Zeroed => "zeroed",
            Self::Empty => "empty",
            Self::Torn => "torn",
            Self::Garbage => "garbage",
        }
    }
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    name.into()
}

fn expected_rows() -> Vec<(i64, String)> {
    (1..=ROWS).map(|id| (id, format!("issue-{id}"))).collect()
}

async fn frank_rows(conn: &Connection, context: &str) -> Vec<(i64, String)> {
    conn.query("SELECT id, v FROM t ORDER BY id")
        .await
        .unwrap_or_else(|error| panic!("{context}: read-only query failed: {error}"))
        .iter()
        .map(|row| match row.values() {
            [SqliteValue::Integer(id), SqliteValue::Text(v)] => (*id, v.to_string()),
            other => panic!("{context}: unexpected row {other:?}"),
        })
        .collect()
}

/// Every file in the store's directory with its bytes and mtime.
type DirectoryFingerprint = BTreeMap<String, (Vec<u8>, SystemTime)>;

fn directory_fingerprint(dir: &Path) -> DirectoryFingerprint {
    let mut entries = DirectoryFingerprint::new();
    for entry in
        std::fs::read_dir(dir).unwrap_or_else(|error| panic!("read the store directory: {error}"))
    {
        let entry = entry.expect("directory entry");
        let path = entry.path();
        let name = entry.file_name().into_string().expect("UTF-8 file name");
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read {name} for the fingerprint: {error}"));
        let mtime = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or_else(|error| panic!("mtime of {name}: {error}"));
        entries.insert(name, (bytes, mtime));
    }
    entries
}

fn assert_directory_unchanged(dir: &Path, before: &DirectoryFingerprint, context: &str) {
    let after = directory_fingerprint(dir);
    assert_eq!(
        after, *before,
        "{context}: no file in the store's directory may be created, deleted, \
         rewritten, or even re-mtime-d by a strict read-only open"
    );
}

/// Build the fixture store and leave its sidecars on disk.
async fn write_fixture(dir: &Path, shape: WalShape) -> PathBuf {
    let db = dir.join("store.db");
    let writer = Connection::open(db.to_str().unwrap())
        .await
        .expect("open the fixture writer");
    for sql in [
        "PRAGMA journal_mode=WAL",
        "PRAGMA wal_autocheckpoint=0",
        "CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT NOT NULL)",
    ] {
        writer
            .execute(sql)
            .await
            .unwrap_or_else(|error| panic!("fixture: {sql} failed: {error}"));
    }
    for id in 1..=ROWS {
        writer
            .execute(&format!("INSERT INTO t VALUES ({id}, 'issue-{id}')"))
            .await
            .expect("fixture insert");
    }
    if shape == WalShape::Checkpointed {
        writer
            .execute("PRAGMA wal_checkpoint(TRUNCATE)")
            .await
            .expect("fixture checkpoint settles every frame into the main file");
    }
    writer
        .close_without_checkpoint()
        .await
        .expect("close the fixture writer without checkpointing");
    if shape == WalShape::Checkpointed {
        // A stock writer's `wal_checkpoint(TRUNCATE)` leaves its WAL at zero
        // bytes; this engine's leaves a bare 32-byte header behind, so
        // truncate to the stock shape the field reports (GH#515's
        // header-less WAL, which a reader must read as "no WAL").
        std::fs::write(sidecar(&db, "-wal"), [])
            .expect("leave the checkpointed -wal at zero bytes");
    }
    let wal = sidecar(&db, "-wal");
    let shm = sidecar(&db, "-shm");
    assert!(wal.exists(), "fixture leaves its -wal for {shape:?}");
    assert!(
        shm.exists(),
        "fixture leaves its -shm for {shape:?} to damage"
    );
    if shape == WalShape::Checkpointed {
        assert_eq!(
            std::fs::metadata(&wal).unwrap().len(),
            0,
            "the truncated checkpoint leaves a header-less zero-byte -wal"
        );
    }
    db
}

/// Damage the fixture's own synthetic `-shm` inside its private tempdir.
fn damage_shm(db: &Path, class: ShmClass) {
    let shm = sidecar(db, "-shm");
    match class {
        ShmClass::Absent => {
            std::fs::remove_file(&shm).expect("remove the fixture's own -shm");
        }
        ShmClass::Zeroed => {
            let zeros = vec![0u8; SHM_REGION_BYTES];
            std::fs::write(&shm, zeros).expect("rewrite the fixture -shm as one zeroed region");
        }
        ShmClass::Empty => {
            std::fs::write(&shm, []).expect("truncate the fixture -shm to zero bytes");
        }
        ShmClass::Torn => {
            let mut bytes = std::fs::read(&shm).expect("read the fixture -shm to tear it");
            assert!(
                bytes.len() >= 2 * 48,
                "the fixture -shm carries both header copies"
            );
            // Byte 32 is inside copy 1's checksum words (bytes 32..40);
            // flipping a bit there makes the copies disagree while leaving
            // copy 1's isInit byte (12) initialized, so the index is torn
            // rather than uninitialized.
            bytes[32] ^= 0x01;
            std::fs::write(&shm, bytes).expect("write the torn fixture -shm");
        }
        ShmClass::Garbage => {
            let garbage = vec![0xA5u8; SHM_REGION_BYTES];
            std::fs::write(&shm, garbage).expect("rewrite the fixture -shm as one garbage region");
        }
    }
}

/// Open through the compat flags surface with the strict bit.
async fn strict_open(db: &Path) -> Result<Connection, FrankenError> {
    open_with_flags(
        db.to_str().unwrap(),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_STRICT_READ_ONLY,
    )
    .await
}

/// Every class that a strict open must refuse with `BusyRecovery` — the
/// whole recovery taxonomy a plain 0.4.13 read-only pager would either
/// rebuild or refuse, minus the never-write reads that stay available.
fn refusal_classes() -> Vec<(WalShape, ShmClass)> {
    let mut classes = Vec::new();
    for shape in [WalShape::Frames, WalShape::Checkpointed] {
        for class in [
            ShmClass::Absent,
            ShmClass::Zeroed,
            ShmClass::Empty,
            ShmClass::Torn,
            ShmClass::Garbage,
        ] {
            classes.push((shape, class));
        }
    }
    // GH#515: a header-less WAL without a `-shm` reads the main file.
    classes
        .retain(|&(shape, class)| !(shape == WalShape::Checkpointed && class == ShmClass::Absent));
    classes
}

#[test]
fn strict_read_only_reads_a_valid_store_without_touching_any_sidecar() {
    asupersync::test_utils::run_test(|| async {
        for shape in [WalShape::Frames, WalShape::Checkpointed] {
            let dir = tempfile::TempDir::new().unwrap();
            let db = write_fixture(dir.path(), shape).await;
            let before = directory_fingerprint(dir.path());
            let context = format!("{} WAL, valid -shm", shape.name());
            let reader = strict_open(&db)
                .await
                .unwrap_or_else(|error| panic!("{context}: strict open failed: {error}"));
            assert_eq!(
                frank_rows(&reader, &context).await,
                expected_rows(),
                "{context}"
            );
            assert!(
                matches!(
                    reader
                        .execute("INSERT INTO t VALUES (99, 'forbidden')")
                        .await,
                    Err(FrankenError::ReadOnly)
                ),
                "{context}: a strict reader must refuse writes"
            );
            reader.close().await.expect("close the strict reader");
            assert_directory_unchanged(dir.path(), &before, &context);
        }
    });
}

#[test]
fn strict_read_only_is_reachable_from_every_documented_entry_point() {
    asupersync::test_utils::run_test(|| async {
        let dir = tempfile::TempDir::new().unwrap();
        let db = write_fixture(dir.path(), WalShape::Frames).await;

        // The core constructor re-exported through the facade, with a valid
        // index and with a refused one.
        let before = directory_fingerprint(dir.path());
        let reader = Connection::open_strict_read_only(db.to_str().unwrap())
            .await
            .expect("core strict constructor opens a valid store");
        let rows = frank_rows(&reader, "core constructor, valid -shm").await;
        assert_eq!(rows, expected_rows());
        reader.close().await.expect("close");
        assert_directory_unchanged(dir.path(), &before, "core constructor, valid -shm");

        // The compat convenience function.
        let reader = open_strict_read_only(db.to_str().unwrap())
            .await
            .expect("compat strict convenience opens a valid store");
        let rows = frank_rows(&reader, "compat convenience, valid -shm").await;
        assert_eq!(rows, expected_rows());
        reader.close().await.expect("close");
        assert_directory_unchanged(dir.path(), &before, "compat convenience, valid -shm");

        // Both must refuse a store whose index needs recovery, not silently
        // take the plain 0.4.13 rebuild path.
        damage_shm(&db, ShmClass::Zeroed);
        let before = directory_fingerprint(dir.path());
        match Connection::open_strict_read_only(db.to_str().unwrap()).await {
            Err(FrankenError::BusyRecovery) => {}
            Err(other) => {
                panic!("core constructor, zeroed -shm: expected BusyRecovery, got {other}")
            }
            Ok(_) => panic!(
                "core constructor, zeroed -shm: the strict open succeeded instead of refusing"
            ),
        }
        assert_directory_unchanged(dir.path(), &before, "core constructor, zeroed -shm");
        match open_strict_read_only(db.to_str().unwrap()).await {
            Err(FrankenError::BusyRecovery) => {}
            Err(other) => {
                panic!("compat convenience, zeroed -shm: expected BusyRecovery, got {other}")
            }
            Ok(_) => panic!(
                "compat convenience, zeroed -shm: the strict open succeeded instead of refusing"
            ),
        }
        assert_directory_unchanged(dir.path(), &before, "compat convenience, zeroed -shm");
    });
}

#[test]
fn strict_read_only_refuses_every_wal_index_recovery_class_without_touching_any_file() {
    asupersync::test_utils::run_test(|| async {
        for (shape, class) in refusal_classes() {
            let dir = tempfile::TempDir::new().unwrap();
            let db = write_fixture(dir.path(), shape).await;
            damage_shm(&db, class);
            let before = directory_fingerprint(dir.path());
            let context = format!("{} WAL, {} -shm", shape.name(), class.name());
            let started = Instant::now();
            let outcome = strict_open(&db).await;
            match outcome {
                Err(FrankenError::BusyRecovery) => {}
                Err(other) => panic!("{context}: expected BusyRecovery, got {other}"),
                Ok(_) => panic!("{context}: the strict open succeeded instead of refusing"),
            }
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "{context}: the refusal must not ride the 30 s busy bootstrap retry"
            );
            assert_directory_unchanged(dir.path(), &before, &context);
        }
    });
}

#[test]
fn strict_read_only_keeps_the_gh515_headerless_main_file_read() {
    asupersync::test_utils::run_test(|| async {
        let dir = tempfile::TempDir::new().unwrap();
        let db = write_fixture(dir.path(), WalShape::Checkpointed).await;
        damage_shm(&db, ShmClass::Absent);
        let before = directory_fingerprint(dir.path());
        let context = "checkpointed WAL, absent -shm";
        let reader = strict_open(&db).await.unwrap_or_else(|error| {
            panic!("{context}: GH#515 main-file read must stay available: {error}")
        });
        assert_eq!(
            frank_rows(&reader, context).await,
            expected_rows(),
            "{context}"
        );
        reader.close().await.expect("close the strict reader");
        assert_directory_unchanged(dir.path(), &before, context);
        assert!(
            !sidecar(&db, "-shm").exists(),
            "{context}: no -shm may be created"
        );
    });
}

#[test]
fn plain_read_only_keeps_the_gh528_rebuild_and_its_fail_closed_boundaries() {
    asupersync::test_utils::run_test(|| async {
        // Valid index, both shapes: readable with no writes at all.
        for shape in [WalShape::Frames, WalShape::Checkpointed] {
            let dir = tempfile::TempDir::new().unwrap();
            let db = write_fixture(dir.path(), shape).await;
            let before = directory_fingerprint(dir.path());
            let context = format!("plain, {} WAL, valid -shm", shape.name());
            let reader = open_with_flags(db.to_str().unwrap(), OpenFlags::SQLITE_OPEN_READ_ONLY)
                .await
                .unwrap_or_else(|error| panic!("{context}: open failed: {error}"));
            assert_eq!(
                frank_rows(&reader, &context).await,
                expected_rows(),
                "{context}"
            );
            reader.close().await.expect("close the plain reader");
            assert_directory_unchanged(dir.path(), &before, &context);
        }

        // Missing `-shm`: a WAL with committed frames is refused (no reader
        // may create the index), a header-less WAL still reads the main
        // file (GH#515), and neither writes anything.
        for (shape, readable) in [(WalShape::Frames, false), (WalShape::Checkpointed, true)] {
            let dir = tempfile::TempDir::new().unwrap();
            let db = write_fixture(dir.path(), shape).await;
            damage_shm(&db, ShmClass::Absent);
            let before = directory_fingerprint(dir.path());
            let context = format!("plain, {} WAL, absent -shm", shape.name());
            let outcome =
                open_with_flags(db.to_str().unwrap(), OpenFlags::SQLITE_OPEN_READ_ONLY).await;
            if readable {
                let reader =
                    outcome.unwrap_or_else(|error| panic!("{context}: open failed: {error}"));
                assert_eq!(
                    frank_rows(&reader, &context).await,
                    expected_rows(),
                    "{context}"
                );
                reader.close().await.expect("close the plain reader");
            } else {
                match outcome {
                    Err(FrankenError::BusyRecovery) => {}
                    Err(other) => panic!("{context}: expected BusyRecovery, got {other}"),
                    Ok(_) => panic!("{context}: the open succeeded instead of refusing"),
                }
            }
            assert_directory_unchanged(dir.path(), &before, &context);
            assert!(
                !sidecar(&db, "-shm").exists(),
                "{context}: no -shm may be created"
            );
        }

        // Uninitialized index (GH#528): plain 0.4.13 rebuilds it in place,
        // which is exactly the write the strict mode exists to refuse.
        for shape in [WalShape::Frames, WalShape::Checkpointed] {
            for class in [ShmClass::Zeroed, ShmClass::Empty] {
                let dir = tempfile::TempDir::new().unwrap();
                let db = write_fixture(dir.path(), shape).await;
                damage_shm(&db, class);
                let before = directory_fingerprint(dir.path());
                let context = format!("plain, {} WAL, {} -shm", shape.name(), class.name());
                let reader =
                    open_with_flags(db.to_str().unwrap(), OpenFlags::SQLITE_OPEN_READ_ONLY)
                        .await
                        .unwrap_or_else(|error| panic!("{context}: open failed: {error}"));
                assert_eq!(
                    frank_rows(&reader, &context).await,
                    expected_rows(),
                    "{context}"
                );
                reader.close().await.expect("close the plain reader");
                let after = directory_fingerprint(dir.path());
                assert_eq!(
                    after.keys().collect::<Vec<_>>(),
                    before.keys().collect::<Vec<_>>(),
                    "{context}: no file may be created"
                );
                for (name, entry) in &after {
                    let prior = &before[name.as_str()];
                    if name.ends_with("-shm") {
                        assert_ne!(
                            entry.0, prior.0,
                            "{context}: the GH#528 rebuild must rewrite the -shm in place"
                        );
                        assert!(
                            fsqlite_wal::wal_index::parse_shm_header(&entry.0)
                                .unwrap()
                                .is_some(),
                            "{context}: the rebuilt index has a valid header"
                        );
                    } else {
                        assert_eq!(
                            entry, prior,
                            "{context}: {name} must stay byte- and mtime-identical"
                        );
                    }
                }
            }
        }

        // Initialized-but-invalid index: fail-closed for plain 0.4.13 too.
        for shape in [WalShape::Frames, WalShape::Checkpointed] {
            for class in [ShmClass::Torn, ShmClass::Garbage] {
                let dir = tempfile::TempDir::new().unwrap();
                let db = write_fixture(dir.path(), shape).await;
                damage_shm(&db, class);
                let before = directory_fingerprint(dir.path());
                let context = format!("plain, {} WAL, {} -shm", shape.name(), class.name());
                let outcome =
                    open_with_flags(db.to_str().unwrap(), OpenFlags::SQLITE_OPEN_READ_ONLY).await;
                match outcome {
                    Err(FrankenError::BusyRecovery) => {}
                    Err(other) => panic!("{context}: expected BusyRecovery, got {other}"),
                    Ok(_) => panic!("{context}: the open succeeded instead of refusing"),
                }
                assert_directory_unchanged(dir.path(), &before, &context);
            }
        }
    });
}
