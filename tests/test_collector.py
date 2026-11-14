#!/usr/bin/env python3
"""End-to-end correctness tests for the collector.

Runs entirely on the local filesystem + SQLite (no Docker needed):

* partial (unterminated) tails are held back until the newline arrives
* rotation by rename: old inode keeps draining, new file starts at offset 0
* collector exit at arbitrary points resumes without gaps or duplicates
* identical content lines are stored as separate rows
* committed data only is visible to queries (transactional)
* FastAPI query/pagination/progress endpoints work
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.request

# Make `app` importable when run as `python tests/test_collector.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "collector"))

from app import storage  # noqa: E402
from app.ingest import CollectorLoop, drain_file, poll_once  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def jline(path: str, status: int, ts: str = "2026-11-13T10:00:00Z") -> str:
    return json.dumps({"time": ts, "path": path, "status": status,
                       "method": "GET"}) + "\n"


def fresh_env():
    base = tempfile.mkdtemp(prefix="collector-test-")
    log_dir = os.path.join(base, "logs")
    db_path = os.path.join(base, "data", "warehouse.db")
    os.makedirs(log_dir)
    os.makedirs(os.path.dirname(db_path))
    conn = storage.connect(db_path)
    storage.init_db(conn)
    return base, log_dir, db_path, conn


def test_drain_partial_tail():
    print("\n[1] 未换行尾部必须等补齐后入仓")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        with open(f, "w") as fh:
            fh.write(jline("/a", 200))
            fh.write('{"time":"x","path":"/partial","status":5')  # no newline

        r = poll_once(conn, log_dir)
        check("first poll ingests only the complete line", r["ingested"] == 1)
        st = storage.all_states(conn)[0]
        check("offset stops before partial tail",
              st["offset"] == len(jline("/a", 200)),
              f"offset={st['offset']}")
        n = conn.execute("SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("warehouse contains 1 row", n == 1)

        # complete the line
        with open(f, "a") as fh:
            fh.write('00,"method":"GET"}\n')
        r = poll_once(conn, log_dir)
        check("second poll ingests the now-completed line", r["ingested"] == 1)
        rows = conn.execute("SELECT path, status FROM log_events ORDER BY id").fetchall()
        check("both rows present with correct content",
              [dict(x) for x in rows] == [
                  {"path": "/a", "status": 200},
                  {"path": "/partial", "status": 500}],
              str([dict(x) for x in rows]))
        st = storage.all_states(conn)[0]
        check("offset reaches EOF", st["offset"] == os.path.getsize(f))
    finally:
        conn.close()
        shutil.rmtree(base)


def test_rotation_rename():
    print("\n[2] rename 轮转：旧 inode 续采、新文件独立偏移")
    base, log_dir, db_path, conn = fresh_env()
    try:
        cur = os.path.join(log_dir, "access.log")
        with open(cur, "w") as fh:
            for i in range(5):
                fh.write(jline(f"/old/{i}", 200))
        poll_once(conn, log_dir)
        check("5 lines ingested pre-rotation",
              conn.execute("SELECT COUNT(*) c FROM log_events").fetchone()["c"] == 5)

        old_inode = os.stat(cur).st_ino
        archive = os.path.join(log_dir, "access.log.20261113T100000Z.log")
        os.rename(cur, archive)
        # An open fd on the archive keeps its inode allocated, exactly like
        # nginx workers that still hold the renamed log file.
        arch_fh = open(archive, "a")
        # old inode receives 2 late lines (in-flight requests)
        arch_fh.write(jline("/late/1", 503))
        arch_fh.write(jline("/late/2", 500))
        arch_fh.flush()
        # nginx reopens a brand new file at the same path
        with open(cur, "w") as fh:
            fh.write(jline("/new/1", 404))

        r = poll_once(conn, log_dir)
        check("3 lines ingested across both inodes", r["ingested"] == 3,
              f"got {r['ingested']}")

        states = {(s["inode"]): s for s in storage.all_states(conn)}
        check("two distinct inodes tracked", len(states) == 2)
        old_state = states[old_inode]
        check("old inode state keeps archive source_code",
              old_state["source_code"].startswith("access.log.20261113"),
              old_state["source_code"])
        check("old inode drained to its EOF",
              old_state["offset"] == os.path.getsize(archive))
        new_inode = os.stat(cur).st_ino
        check("new inode differs from old", new_inode != old_inode)
        check("new inode offset is its own size (starts at 0, not old offset)",
              states[new_inode]["offset"] == os.path.getsize(cur))

        rows = conn.execute(
            "SELECT source_code, path FROM log_events ORDER BY id"
        ).fetchall()
        srcs = {x["source_code"] for x in rows}
        check("rows carry both source file codes",
              srcs == {"access.log", "access.log.20261113T100000Z.log"}, str(srcs))
        check("late lines landed against the archive code",
              all(x["source_code"].startswith("access.log.2026")
                  for x in rows if x["path"].startswith("/late")))
        arch_fh.close()
    finally:
        conn.close()
        shutil.rmtree(base)


def test_duplicate_content_kept():
    print("\n[3] 内容相同的两行分别保留")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        line = jline("/same", 500)
        with open(f, "w") as fh:
            fh.write(line * 4)
        poll_once(conn, log_dir)
        rows = conn.execute(
            "SELECT byte_offset FROM log_events ORDER BY byte_offset"
        ).fetchall()
        check("4 rows for 4 identical lines", len(rows) == 4)
        offsets = [r["byte_offset"] for r in rows]
        check("each row has a distinct byte offset",
              offsets == [0, len(line), 2 * len(line), 3 * len(line)],
              str(offsets))
    finally:
        conn.close()
        shutil.rmtree(base)


def test_resume_no_gap_no_dup():
    print("\n[4] 任意时刻退出后自动续采：不漏不重")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        total = 500
        with open(f, "w") as fh:
            for i in range(total):
                fh.write(jline(f"/p/{i}", 200 if i % 7 else 500))

        # Simulate many crash/restart cycles while draining.
        seen = 0
        for cycle in range(20):
            conn2 = storage.connect(db_path)
            r = poll_once(conn2, log_dir)
            seen += r["ingested"]
            conn2.close()  # hard close = "process exit"
            if seen >= total:
                break
        check(f"all {total} lines ingested across restarts", seen == total,
              f"seen={seen}")

        # Re-polling changes nothing: offsets are committed.
        r = poll_once(conn, log_dir)
        check("idle poll ingests zero duplicates", r["ingested"] == 0)
        n = conn.execute("SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("exactly one row per physical line", n == total, f"n={n}")
        check("unique position constraint holds",
              conn.execute(
                  "SELECT COUNT(*) c FROM (SELECT DISTINCT dev,inode,source_code,byte_offset FROM log_events)"
              ).fetchone()["c"] == total)
        paths = [r["path"] for r in conn.execute(
            "SELECT path FROM log_events ORDER BY id").fetchall()]
        check("order/content matches the source file exactly",
              paths == [f"/p/{i}" for i in range(total)])
    finally:
        conn.close()
        shutil.rmtree(base)


def test_background_loop_and_rotation_interleave():
    print("\n[5] 后台循环采集与轮转交错")
    base, log_dir, db_path, conn = fresh_env()
    try:
        loop_conn = storage.connect(db_path)
        loop = CollectorLoop(loop_conn, log_dir, interval=0.05)
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        cur = os.path.join(log_dir, "access.log")
        written = 0
        archives = 0
        fh = open(cur, "w")
        archive_handles = []  # keep rotated inodes alive, as nginx workers do
        for round_no in range(6):
            for i in range(30):
                fh.write(jline(f"/r{round_no}/line{i}", 200))
                written += 1
            fh.flush()
            time.sleep(0.05)
            # rotate mid-stream
            fh.close()
            archives += 1
            arch_path = os.path.join(log_dir, f"access.log.rot{round_no}.log")
            os.rename(cur, arch_path)
            # Holding the fd keeps the old inode from being recycled by the
            # filesystem — equivalent to nginx workers finishing in-flight
            # requests against the renamed file.
            archive_handles.append(open(arch_path, "a"))
            # late write into the old inode
            archive_handles[-1].write(jline(f"/r{round_no}/late", 599))
            archive_handles[-1].flush()
            written += 1
            fh = open(cur, "w")
            time.sleep(0.05)
        fh.close()

        deadline = time.time() + 10
        while time.time() < deadline:
            got = loop_conn.execute(
                "SELECT COUNT(*) c FROM log_events").fetchone()["c"]
            if got == written:
                break
            time.sleep(0.1)
        loop.request_stop()
        t.join(timeout=5)
        got = loop_conn.execute(
            "SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("every line captured under interleaved rotation",
              got == written, f"got={got} written={written}")
        check("late 599 lines captured from archive inodes",
              loop_conn.execute(
                  "SELECT COUNT(*) c FROM log_events WHERE status=599"
              ).fetchone()["c"] == 6)
        # every archive fully drained
        bad = [dict(s) for s in storage.all_states(loop_conn)
               if s["offset"] != os.path.getsize(s["path"])]
        check("all tracked offsets equal current file sizes", not bad, str(bad))
        # Guaranteed properties, independent of the filesystem's inode
        # allocation policy: every archive is tracked as a distinct source
        # and every source is drained exactly to its current size. On the
        # Docker volume nginx holds every rotated fd open, so inodes are never
        # recycled and the live file additionally appears as "access.log".
        codes = {s["source_code"] for s in storage.all_states(loop_conn)}
        expected_archives = {f"access.log.rot{r}.log" for r in range(6)}
        check("all 6 archives tracked as distinct sources",
              expected_archives.issubset(codes), f"codes={sorted(codes)}")
        for h in archive_handles:
            h.close()
        loop_conn.close()
    finally:
        conn.close()
        shutil.rmtree(base)


def test_malformed_quarantine():
    print("\n[6] 坏行隔离且不阻断后续行")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        with open(f, "w") as fh:
            fh.write(jline("/ok", 200))
            fh.write("{this is not json}\n")
            fh.write(jline("/ok2", 500))
        r = poll_once(conn, log_dir)
        check("2 good + 1 malformed reported",
              r["ingested"] == 2 and r["malformed"] == 1, str(r))
        check("pipeline continued past the bad line",
              conn.execute(
                  "SELECT path FROM log_events ORDER BY id"
              ).fetchall()[-1]["path"] == "/ok2")
        check("bad line preserved with its byte offset",
              conn.execute(
                  "SELECT byte_offset, error FROM malformed_lines"
              ).fetchone()["byte_offset"] == len(jline("/ok", 200)))
    finally:
        conn.close()
        shutil.rmtree(base)


def test_api():
    print("\n[7] FastAPI：进度、分页、按路径/状态码查询")
    # Environment must be configured BEFORE importing app.main, because its
    # path constants are resolved at import time.
    base = tempfile.mkdtemp(prefix="collector-api-")
    log_dir = os.path.join(base, "logs")
    db_path = os.path.join(base, "data", "warehouse.db")
    os.makedirs(log_dir)
    os.makedirs(os.path.dirname(db_path))
    os.environ["DB_PATH"] = db_path
    os.environ["LOG_DIR"] = log_dir
    os.environ["POLL_INTERVAL"] = "0.05"

    import uvicorn
    from app.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

        f = os.path.join(log_dir, "access.log")
        with open(f, "w") as fh:
            for i in range(25):
                fh.write(jline("/api/orders", 200))
            for i in range(5):
                fh.write(jline("/api/orders", 500))
            for i in range(3):
                fh.write(jline("/api/payments", 503))

        deadline = time.time() + 10
        while time.time() < deadline:
            h = json.load(urllib.request.urlopen(
                "http://127.0.0.1:8765/health"))
            if h["counts"]["events"] == 33:
                break
            time.sleep(0.1)
        check("background ingestion visible via /health",
              h["counts"]["events"] == 33, str(h["counts"]))
        check("health reports collector thread running, not stopping",
              h["collector_running"] is True
              and h["collector_stopping"] is False, str(h))

        prog = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/ingest/progress"))
        check("progress shows file complete (offset == size)",
              prog["files"][0]["complete"] is True
              and prog["files"][0]["offset"] == prog["files"][0]["size"],
              str(prog["files"][0]))

        page = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/events?limit=10&offset=0"))
        check("pagination metadata correct",
              page["total"] == 33 and len(page["items"]) == 10
              and page["next_offset"] == 10, str({k: page[k] for k in
              ("total", "next_offset")}))

        page2 = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/events?limit=10&offset=30"))
        check("last page has 3 rows and no next_offset",
              len(page2["items"]) == 3 and page2["next_offset"] is None)

        errs = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/events?status_min=500&status_max=599"))
        check("status range query finds all 8 failures",
              errs["total"] == 8 and all(i["status"] >= 500 for i in errs["items"]),
              f"total={errs['total']}")

        s503 = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/events?status=503"))
        check("exact status=503 query", s503["total"] == 3)

        by_path = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/events?path=/api/orders&limit=100"))
        check("path filter returns 30 order events",
              by_path["total"] == 30
              and all(i["path"] == "/api/orders" for i in by_path["items"]))

        both = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8765/events?path=/api/orders&status=500"))
        check("path + status combined filter", both["total"] == 5)

        one = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:8765/events/{page['items'][0]['id']}"))
        check("single event returns raw JSON line",
              json.loads(one["raw_line"])["path"] == "/api/orders")

        code = urllib.request.urlopen(
            "http://127.0.0.1:8765/events/999999").getcode()
    except urllib.error.HTTPError as e:
        code = e.code
        check("unknown event id -> 404", code == 404, str(code))
    else:
        check("unknown event id -> 404", False, f"got {code}")
    finally:
        server.should_exit = True
        t.join(timeout=5)
        shutil.rmtree(base)


def test_offset_math_after_partial():
    print("\n[8] 偏移量计算：部分行补齐后字节位置精确")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        l1 = jline("/x", 200)
        with open(f, "w") as fh:
            fh.write(l1)
            fh.write('{"incomplete":')
        poll_once(conn, log_dir)
        tail_part = 'true}\n'
        with open(f, "a") as fh:
            fh.write(tail_part)
        poll_once(conn, log_dir)
        rows = conn.execute(
            "SELECT byte_offset, raw_line FROM log_events ORDER BY id").fetchall()
        check("second line offset = length of first line",
              rows[1]["byte_offset"] == len(l1.encode()),
              str(rows[1]["byte_offset"]))
        st = storage.all_states(conn)[0]
        check("final offset equals file size",
              st["offset"] == os.path.getsize(f),
              f"{st['offset']} != {os.path.getsize(f)}")
    finally:
        conn.close()
        shutil.rmtree(base)


def test_multibyte_offsets_and_chunk_boundaries():
    print("\n[9] 多字节 UTF-8：字节偏移精确、跨块字符不损坏")
    import app.ingest as ingest_mod
    base, log_dir, db_path, conn = fresh_env()
    # Force tiny reads so multi-byte chars and newlines straddle chunks.
    original_chunk = ingest_mod.READ_CHUNK
    ingest_mod.READ_CHUNK = 7
    try:
        f = os.path.join(log_dir, "access.log")
        lines = [
            jline("/unicafé/✓", 200),   # é (2 bytes) and ✓ (3 bytes)
            jline("/ascii/normal", 404),
            jline("/中文/路径", 500),
            jline("/emoji/😀/end", 503),
        ]
        with open(f, "wb") as fh:
            for ln in lines:
                fh.write(ln.encode("utf-8"))
        poll_once(conn, log_dir)
        rows = conn.execute(
            "SELECT byte_offset, path, raw_line FROM log_events ORDER BY id"
        ).fetchall()
        check("all 4 multibyte lines ingested", len(rows) == 4)
        raw = open(f, "rb").read()
        expected_offsets, cursor = [], 0
        for ln in lines:
            expected_offsets.append(cursor)
            cursor += len(ln.encode("utf-8"))
        check("byte offsets match true byte positions",
              [r["byte_offset"] for r in rows] == expected_offsets,
              str([r["byte_offset"] for r in rows]))
        check("committed offset equals exact file size in bytes",
              storage.all_states(conn)[0]["offset"] == len(raw))
        # raw lines round-trip exactly (no replacement chars introduced)
        check("raw content preserved byte-for-byte",
              [r["raw_line"] for r in rows] ==
              [ln.rstrip("\n") for ln in lines])
        check("unicode paths normalised correctly",
              [r["path"] for r in rows] ==
              ["/unicafé/✓", "/ascii/normal", "/中文/路径", "/emoji/😀/end"],
              str([r["path"] for r in rows]))
    finally:
        ingest_mod.READ_CHUNK = original_chunk
        conn.close()
        shutil.rmtree(base)


def test_stop_finishes_at_line_boundary():
    print("\n[10] 停机交接：停止信号后当前轮在完整行+事务边界收尾")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        lines = [jline(f"/stop/{i}", 200 if i % 3 else 500) for i in range(200)]
        with open(f, "w") as fh:
            fh.writelines(lines)

        import app.ingest as ingest_mod
        original_chunk = ingest_mod.READ_CHUNK
        # Small chunks: the stop flag is consulted between chunks, so the
        # pass must halt almost immediately after the flag flips.
        ingest_mod.READ_CHUNK = 256
        try:
            calls = {"n": 0}

            def stop_after_first_chunk():
                calls["n"] += 1
                # call 1: poll_once file-loop guard -> keep going
                # call 2: drain_file before chunk 1 -> keep going
                # call 3: drain_file before chunk 2 -> stop
                return calls["n"] > 2

            r = poll_once(conn, log_dir, should_stop=stop_after_first_chunk)
        finally:
            ingest_mod.READ_CHUNK = original_chunk

        check("pass reports it stopped early", r["stopped_early"] is True)
        check("some lines committed, the rest deferred",
              0 < r["ingested"] < 200, f"ingested={r['ingested']}")
        st = storage.all_states(conn)[0]
        boundary = sum(len(lines[i].encode()) for i in range(r["ingested"]))
        check("committed offset sits exactly on a complete-line boundary",
              st["offset"] == boundary,
              f"offset={st['offset']} boundary={boundary}")
        n = conn.execute("SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("warehouse rows == committed lines", n == r["ingested"])

        # Resume: a normal pass picks up every remaining line exactly once.
        r2 = poll_once(conn, log_dir)
        check("resume ingests exactly the remainder",
              r2["ingested"] == 200 - r["ingested"])
        n = conn.execute("SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("no line lost, none duplicated", n == 200, f"n={n}")
        paths = [row["path"] for row in conn.execute(
            "SELECT path FROM log_events ORDER BY id").fetchall()]
        check("final row order matches the source file",
              paths == [f"/stop/{i}" for i in range(200)])
    finally:
        conn.close()
        shutil.rmtree(base)


def test_drain_capped_at_scanned_size():
    print("\n[11] 轮次有界：扫描后追加的字节留给下一轮，持续追加也能收尾")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = os.path.join(log_dir, "access.log")
        with open(f, "w") as fh:
            fh.writelines(jline(f"/snap/{i}", 200) for i in range(10))
        size_snapshot = os.path.getsize(f)
        # Bytes appended AFTER the directory snapshot must not extend the
        # current pass — this is what makes every pass terminate even while
        # the source file keeps growing.
        with open(f, "a") as fh:
            fh.writelines(jline(f"/late/{i}", 200) for i in range(5))

        drained = drain_file(f, 0, size_snapshot)
        check("drain stops at the scanned size",
              drained.next_offset == size_snapshot,
              f"{drained.next_offset} != {size_snapshot}")
        check("only the 10 snapshot lines returned",
              len(drained.complete_lines) == 10,
              str(len(drained.complete_lines)))
        drained2 = drain_file(f, drained.next_offset, os.path.getsize(f))
        check("late bytes are picked up from the committed offset next pass",
              len(drained2.complete_lines) == 5
              and drained2.next_offset == os.path.getsize(f))
    finally:
        conn.close()
        shutil.rmtree(base)


def test_loop_thread_exits_promptly_on_stop():
    print("\n[12] 采集线程：持续追加中停止 -> 线程退出、不再开新轮、重启续采不漏不重")
    base, log_dir, db_path, conn = fresh_env()
    try:
        loop_conn = storage.connect(db_path)
        loop = CollectorLoop(loop_conn, log_dir, interval=0.05)
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        f = os.path.join(log_dir, "access.log")
        stop_writing = threading.Event()
        written = {"n": 0}

        def writer():
            # The source file keeps being appended while shutdown begins —
            # the maintenance-window scenario from the incident report.
            with open(f, "a") as fh:
                while not stop_writing.is_set():
                    fh.write(jline(f"/live/{written['n']}", 200))
                    written["n"] += 1
                    fh.flush()
                    time.sleep(0.005)

        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        try:
            time.sleep(0.5)  # several passes run under continuous appends
            polls_before = loop.total_polls
            loop.request_stop()
            t.join(timeout=10)
            check("collector thread exited after stop request",
                  not t.is_alive())
            check("no new pass started after stop",
                  loop.total_polls <= polls_before + 1,
                  f"before={polls_before} after={loop.total_polls}")
        finally:
            stop_writing.set()
            wt.join(timeout=5)

        # The connection outlives the thread: main.py closes it only after
        # the collector thread has exited, so this read is always safe.
        n_committed = loop_conn.execute(
            "SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("committed rows satisfy the exactly-once position constraint",
              loop_conn.execute(
                  "SELECT COUNT(*) c FROM (SELECT DISTINCT dev,inode,source_code,byte_offset FROM log_events)"
              ).fetchone()["c"] == n_committed)

        # Restart-style resume from the committed position: nothing lost.
        poll_once(loop_conn, log_dir)
        total = loop_conn.execute(
            "SELECT COUNT(*) c FROM log_events").fetchone()["c"]
        check("resume after stop loses no line and adds no duplicate",
              total == written["n"],
              f"total={total} written={written['n']}")
        loop_conn.close()
    finally:
        conn.close()
        shutil.rmtree(base)


def test_manual_poll_rejected_after_stop():
    print("\n[13] 关闭开始后：手动采集返回 409，不再启动新轮次")
    # app.main was already imported by test_api, so its path constants are
    # fixed; only the shared `state` needs to point at this test's objects.
    base, log_dir, db_path, conn = fresh_env()
    try:
        from fastapi import HTTPException
        from app.main import poll_now, state

        loop = CollectorLoop(conn, log_dir, interval=0.05)
        state.update(conn=conn, loop=loop)
        try:
            r = poll_now()
            check("manual poll runs while the service is up",
                  r["ingested"] == 0 and r["stopped_early"] is False)
            loop.request_stop()
            try:
                poll_now()
                check("manual poll rejected once shutdown begins", False,
                      "no exception raised")
            except HTTPException as exc:
                check("manual poll rejected once shutdown begins",
                      exc.status_code == 409, f"status={exc.status_code}")
        finally:
            state.clear()
    finally:
        conn.close()
        shutil.rmtree(base)


if __name__ == "__main__":
    test_drain_partial_tail()
    test_rotation_rename()
    test_duplicate_content_kept()
    test_resume_no_gap_no_dup()
    test_background_loop_and_rotation_interleave()
    test_malformed_quarantine()
    test_api()
    test_offset_math_after_partial()
    test_multibyte_offsets_and_chunk_boundaries()
    test_stop_finishes_at_line_boundary()
    test_drain_capped_at_scanned_size()
    test_loop_thread_exits_promptly_on_stop()
    test_manual_poll_rejected_after_stop()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
