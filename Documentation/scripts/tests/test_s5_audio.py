"""Sound on the presentation path (S5): the recorder's audio records, the format's audio
lines, the reader that holds a version 2 stream to them and to the audio devices' own
record of the run, and the player's mixer that plays them, held without a GPU, a window,
an audio device or game data. Each is a measuring instrument or feeds one (AGENTS.md rule
5), so every check has a case that must fail it:

  * the reader (tools/s5/stream_dump.py), on small version 2 streams made here byte by
    byte: a whole one reads clean to literal counts; each malformation (an AUDIO record
    in a version 1 stream, a blob its line does not give, a PCM blob of another length,
    bytes outside their buffer's store, a matrix that is not its STATE's, a STATE of
    another object, a re-anchoring call's STATE anchored elsewhere, a submit whose
    frames do not end at its STATE's queued_end, an argc that is not the table's, a call
    on a buffer nothing made, a clock that goes back, a header table that is not the
    format's lines or STATE sizes that are not its structs', an Unlock's store_hash where
    no Lock closed or none where the last did) is a problem, and under the
    SA2_STREAM_DUMP_BREAK naming its check the same stream reads clean; a version 2
    header without its audio table, and a version 1 header with one, are not streams;
    the counts file's per-op counts and store watch are held to the stream's own records;
    and the records and counts are held to the audio devices' own record of the run
    (hold_audio_to_run: a dropped record, a failed call recorded, an answer, an out word,
    a guest thread or a clock that is not the device's, counts that are not its calls,
    and a summary that kept too few records to walk a device's log); the reader's matrix
    bound is the recorder's and the format's.
  * the player's reader (host/player/stream.cpp), through the renderer and a driver
    built here: a version 2 header whose STATE sizes are not the format's, whose audio
    table is not whole, or whose version, schema and audio table disagree is refused
    at open (the renderer exits 3) and at open_live, and the format's own open.
  * the mixer (host/player/mixer.cpp, built here with g++ as sa2_audio_render, the
    renderer host/player/build.sh builds), on literal streams: a 48 kHz buffer plays its
    bytes one frame per output sample and wraps when it loops; one that does not loop
    halts at its end; a 24 kHz one is interpolated halfway between its frames, and so is
    a voice at half rate; mono, 8-bit and float buffers play their frames; volume and pan
    are DirectSound's decibels; a six-channel voice is mixed by its level matrix and runs
    dry at the end of what it was given, and a four-channel one with none plays its first
    two; a cursor, status or voice state the mixer's clock does not give is a problem, so
    are a store that does not hash to the guest's, a store of another length than its
    buffer and a record before the one before it; the WAV is the format it says. Each
    MIXER_BREAK_* build fails the check that names it.
  * the recorder (host/sa2_present_rec.c), on the runtime's own DirectSound and XAudio2
    devices with no game (AUDIO_HARNESS_C below, written out and built with the pinned
    zig from runtime/sa2_d3d9.c, runtime/sa2_dsound.c, runtime/sa2_xaudio2.c and, but for
    the control, the recorder and the -wrap flags a presentation build links it with),
    in two scenarios: the guest sees the same with the recorder off, on, and on with
    SA2_REC_AUDIO=1, every buffer's and voice's record included; without audio the
    stream is version 1 with no audio table, class or count, the same bytes with
    SA2_REC_AUDIO unset and 0; any other value, a break of an audio device's model, and
    an argument count that is not the recorder's (at start-up or at the registration of
    DirectSound's thunks) stop the stream; counting alone counts the calls that failed;
    with audio every record reads clean against the format's lines, the counts file and
    the devices' own record, every PCM blob is the bytes the guest wrote, every STATE the
    device's after the call, every store_hash the guest's store, and the mixer's clock
    gives every cursor and voice state the device gave the guest; a guest write outside
    a Lock is caught by the store watch and, when an Unlock follows, by the mixer.
    REC_BREAK_AUPCM, _AUSTATE, _AUCURSOR, _AUDROP and _AUNOWATCH each fail the check that
    names them, and the recorder's object reaches the audio devices' table getters and
    no cursor helper (tools/s5/objcheck.py).
  * the runner (tools/s5/present_run.py --audio): SA2_REC_AUDIO=1 and nothing else in its
    command, paths of its own, refused with --no-trace, and a stream of the version the
    run asked for from a recorder that wrapped every `audio` line; audio_counts fails on
    each of its terms alone; check_stream, on the harness's streams, fails a stream of
    the version the run did not ask for, and REC_BREAK_AUDROP's on the devices' record
    alone; and stream_dump --run fails that stream the same way.

The registered boot and Metal Harbor runs with SA2_REC_AUDIO=1 are the evidence phase's
(tools/s5/present_run.py --audio); nothing here runs the game.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "s5"))
sys.path.insert(0, str(ROOT / "tools" / "s4"))
import objcheck as OC  # noqa: E402
import stream_dump as SD  # noqa: E402

STREAM_H = ROOT / "host" / "sa2_stream.h"
PLAYER = ROOT / "host" / "player"
RECORDER = ROOT / "host" / "sa2_present_rec.c"
ZIG = ROOT / "private" / "predecessor-tools" / "zig" / "zig-x86_64-linux-0.15.2" / "zig"
NO_ARG = 0xFFFF
W32 = struct.Struct("<I")


def pad4(n: int) -> int:
    return (n + 3) & ~3


def f32(x: float) -> float:
    """x rounded to float32, as the mixer's arithmetic rounds it."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def fnv1a64(data: bytes) -> int:
    """host/sa2_stream.h's store_hash: 64-bit FNV-1a, one byte at a time."""
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def store_hash(data: bytes) -> tuple[int, int, bytes]:
    """The HASH blob an Unlock that closes its store's last Lock carries."""
    return (SD.HASH, NO_ARG, struct.pack("<Q", fnv1a64(data)))


# ---------------------------------------------------------------- literal streams

OPS = SD.doc_table(STREAM_H.read_text())[2]           # op -> (NAME, iface, method, argc, aops)
OP = {name: op for op, (name, *_r) in OPS.items()}
D3D9, DEV, RT, DSURF = 0x03200000, 0x03200010, 0x03200020, 0x03200030
DSOUND, BUF, BUF2, ENGINE, MASTER, SOURCE = 0x03250000, 0x03250040, 0x03250080, 0x03273000, 0x03273040, 0x03273080
MEM = 0x03300000
T0 = 5_000_000_000                 # the first FRAME's virtual time: past 2^32, for the u64


def header(version: int = 2, **over) -> dict:
    """A header with the device's Present and CreateDevice, Direct3DCreate9, and (version
    2) the format's own audio table."""
    h = {"schema": SD.SCHEMAS[version], "writer": "test_s5_audio.py",
         "meta": json.dumps({"build": "test"}), "object_base": D3D9, "object_stride": 16,
         "interfaces": ["IDirect3D9", "IDirect3DDevice9"],
         "methods": [[0, 16, "CreateDevice", 6], [1, 17, "Present", 4]],
         "rules": [[0, 16, "pre_pp 4; out 5 4; device"], [1, 17, "pre_rect 0; pre_rect 1; pre_rgn 3; frame"]],
         "entries": [[12, "Direct3DCreate9", 0, "object_eax"]],
         "note_classes": ["blob-left-out"]}
    if version == 2:
        h["audio"] = {"ops": [[op, n, i, m, a, o] for op, (n, i, m, a, o) in sorted(OPS.items())],
                      "ds_state_bytes": 84, "xa_state_bytes": 176}
    h.update(over)
    return h


def make(records: list[tuple[int, bytes]], *, version: int = 2, head: dict | None = None) -> bytes:
    text = json.dumps(head if head is not None else header(version)).encode()
    hb = 16 + pad4(len(text))
    out = bytearray(b"SA2D3D9S" + struct.pack("<II", version, hb) + text + b" " * (hb - 16 - len(text)))
    for i, (kind, payload) in enumerate(records):
        out += struct.pack("<III", kind, 12 + len(payload), i) + payload
    return bytes(out)


def blobs(items) -> bytes:
    out = struct.pack("<I", len(items))
    for role, arg, data in items:
        out += struct.pack("<HHI", role, arg, len(data)) + data + b"\0" * (pad4(len(data)) - len(data))
    return out


def call(t, slot, this, args, eax=0, items=()) -> tuple[int, bytes]:
    return (1, struct.pack("<HHIII", t, slot, this, 0, len(args)) + struct.pack(f"<{len(args)}I", *args)
            + struct.pack("<I", eax) + blobs(list(items)))


def frame(us: int, n: int = 0) -> tuple[int, bytes]:
    return (3, struct.pack("<IIII", us & 0xFFFFFFFF, us >> 32, n, 0))


def present(us: int, n: int = 0) -> list[tuple[int, bytes]]:
    return [call(1, 17, DEV, [0, 0, 0, 0]), frame(us, n)]


def audio(name: str, this: int, us: int, args, eax: int = 0, items=(), api: int | None = None,
          argc: int | None = None) -> tuple[int, bytes]:
    op = OP[name]
    iface = OPS[op][1]
    a = list(args)
    return (6, struct.pack("<HHIIIII", api if api is not None else SD.AUDIO_IFACES[iface], op, this, 0,
                           us & 0xFFFFFFFF, us >> 32, len(a) if argc is None else argc)
            + struct.pack(f"<{len(a)}I", *a) + struct.pack("<I", eax) + blobs(list(items)))


def start() -> list[tuple[int, bytes]]:
    """Direct3DCreate9, CreateDevice, and the first Present at T0."""
    pp = struct.pack("<14I", 64, 32, 21, 1, 0, 0, 1, 0, 1, 1, 75, 0, 0, 0)
    return [(2, struct.pack("<II", 12, 0) + struct.pack("<I", D3D9) + blobs([])),
            call(0, 16, D3D9, [0, 1, 0x7e000000, 0x40, 0x03004000, 0x03004100],
                 items=[(1, 4, pp), (2, 5, W32.pack(DEV)), (5, NO_ARG, struct.pack("<3I", DEV, RT, DSURF))]),
            *present(T0, 1)]


END = (5, b"")


def ds_state(**f) -> bytes:
    """sa2s_ds_state: a 48 kHz 16-bit stereo buffer of 32 bytes (8 frames), by default."""
    d = dict(buffer=BUF, index=0, flags=0xE0, bytes=32, primary=0, format_tag=1, channels=2,
             samples_per_sec=48000, avg_bytes_per_sec=192000, block_align=4, bits_per_sample=16,
             playing=0, looping=0, halted=0, released=0, anchor_pos=0, anchor_us=T0, mem=0, mem_len=0,
             write_lead=8)
    d.update(f)
    us = d.pop("anchor_us")
    d["anchor_us_lo"], d["anchor_us_hi"] = us & 0xFFFFFFFF, us >> 32
    return SD.DS_STATE.pack(*(d[k] for k in SD.DS_FIELDS))


def xa_state(**f) -> bytes:
    """sa2s_xa_state: a six-channel float source voice at 48 kHz, ratio 1, by default."""
    d = dict(voice=SOURCE, index=1, kind=2, engine=ENGINE, flags=0, destroyed=0, details_flags=0,
             input_channels=0, input_sample_rate=0, format_tag=0xFFFE, channels=6, samples_per_sec=48000,
             avg_bytes_per_sec=1152000, block_align=24, bits_per_sample=32, valid_bits=32, channel_mask=0x3F,
             max_ratio_bits=0x40000000, ratio_bits=0x3F800000, output=MASTER, matrix_src=0, matrix_dst=0,
             running=0, queued=0, anchor_us=T0, anchor_frame=0, played=0, queued_end=0)
    m = list(f.pop("matrix", [0] * 12))
    d.update(f)
    for k in range(12):
        d[f"matrix{k}"] = m[k]
    for k in ("anchor_us", "anchor_frame", "played", "queued_end"):
        v = d.pop(k)
        d[f"{k}_lo"], d[f"{k}_hi"] = v & 0xFFFFFFFF, v >> 32
    return SD.XA_STATE.pack(*(d[k] for k in SD.XA_FIELDS))


STATE = SD.STATE
OUT, IN, PCM, HASH = SD.OUT, SD.IN, SD.PCM, SD.HASH
# Eight stereo frames: frame k is L = (k + 1) * 1024, R = -(k + 1) * 1024, so as floats
# (k + 1) / 32 and -(k + 1) / 32, exact.
FRAMES8 = b"".join(struct.pack("<hh", (k + 1) * 1024, -(k + 1) * 1024) for k in range(8))


def ds_buffer(us: int = T0, *, looping: int = 1, avg: int = 192000, volume: int | None = None,
              pan: int | None = None, data: bytes = FRAMES8) -> list[tuple[int, bytes]]:
    """A buffer made, filled through a Lock, optionally retimed and set, and played at us."""
    st = dict(mem=MEM, mem_len=len(data), bytes=len(data))
    recs = [audio("DS_CREATE_BUFFER", DSOUND, us, [0x03004200, 0x03004104, 0],
                  items=[(OUT, 1, W32.pack(BUF)), (STATE, 1, ds_state(bytes=len(data)))]),
            audio("DS_LOCK", BUF, us, [0, len(data), 0x03004108, 0x0300410C, 0x03004110, 0x03004114, 0],
                  items=[(OUT, 2, W32.pack(MEM)), (OUT, 3, W32.pack(len(data))), (OUT, 4, W32.pack(0)),
                         (OUT, 5, W32.pack(0)), (STATE, NO_ARG, ds_state(**st))]),
            audio("DS_UNLOCK", BUF, us, [MEM, len(data), 0, 0],
                  items=[(PCM, 0, data), (STATE, NO_ARG, ds_state(**st)), store_hash(data)])]
    if avg != 192000:
        recs.append(audio("DS_SET_FREQUENCY", BUF, us, [avg // 4],
                          items=[(STATE, NO_ARG, ds_state(**st, avg_bytes_per_sec=avg))]))
    if volume is not None:
        recs.append(audio("DS_SET_VOLUME", BUF, us, [volume & 0xFFFFFFFF],
                          items=[(STATE, NO_ARG, ds_state(**st, avg_bytes_per_sec=avg))]))
    if pan is not None:
        recs.append(audio("DS_SET_PAN", BUF, us, [pan & 0xFFFFFFFF],
                          items=[(STATE, NO_ARG, ds_state(**st, avg_bytes_per_sec=avg))]))
    recs.append(audio("DS_PLAY", BUF, us, [0, 0, looping],
                      items=[(STATE, NO_ARG, ds_state(**st, avg_bytes_per_sec=avg, playing=1, looping=looping,
                                                      anchor_us=us))]))
    return recs


# Levels: L takes FL, half of FC and three quarters of BL; R the mirror; LFE a quarter.
LEVELS = [1.0, 0.0, 0.5, 0.25, 0.75, 0.0, 0.0, 1.0, 0.5, 0.25, 0.0, 0.75]


def xa_frames(n: int) -> list[list[float]]:
    """n six-channel frames, frame k channel c = (k + 1) / 32 + c / 256: exact in float32,
    and mixed by LEVELS to well inside [-1, 1]."""
    return [[(k + 1) / 32 + c / 256 for c in range(6)] for k in range(n)]


def xa_voice(us: int = T0, frames: int = 4, *, matrix: bool = True) -> list[tuple[int, bytes]]:
    """A mastering voice, a source voice with the level matrix, `frames` frames submitted,
    and Start at us."""
    m = [fbits(v) for v in LEVELS] if matrix else [0] * 12
    ms, md = (6, 2) if matrix else (0, 0)
    data = b"".join(struct.pack("<6f", *f) for f in xa_frames(frames))
    xb = struct.pack("<9I", 0, len(data), 0x03004400, 0, 0, 0, 0, 0, 0)
    after = dict(matrix=m, matrix_src=ms, matrix_dst=md, queued=1, queued_end=frames)
    recs = [audio("XA_CREATE_MASTER", ENGINE, us, [0x01AF0EDC, 0, 48000, 0, 0, 0],
                  items=[(OUT, 0, W32.pack(MASTER)),
                         (STATE, 0, xa_state(voice=MASTER, index=0, kind=1, input_channels=2,
                                             input_sample_rate=48000, format_tag=0, channels=0,
                                             samples_per_sec=0, avg_bytes_per_sec=0, block_align=0,
                                             bits_per_sample=0, valid_bits=0, channel_mask=0, output=0))]),
            audio("XA_CREATE_SOURCE", ENGINE, us, [0x03004148, 0x03004280, 0, 0x40000000, 0, 0, 0],
                  items=[(OUT, 0, W32.pack(SOURCE)), (STATE, 0, xa_state())])]
    if matrix:
        recs.append(audio("XA_SET_MATRIX", SOURCE, us, [MASTER, 6, 2, 0x0089E4A8, 0],
                          items=[(IN, 3, struct.pack("<12I", *m)),
                                 (STATE, NO_ARG, xa_state(matrix=m, matrix_src=6, matrix_dst=2))]))
    recs += [audio("XA_SUBMIT", SOURCE, us, [0x03004300, 0],
                   items=[(IN, 0, xb), (PCM, 0, data), (STATE, NO_ARG, xa_state(**after))]),
             audio("XA_START", SOURCE, us, [0, 0],
                   items=[(STATE, NO_ARG, xa_state(**after, running=1))])]
    return recs


def scan(tmp_path: Path, data: bytes, brk: str | None = None, **kw) -> dict:
    p = tmp_path / "s.sa2s"
    p.write_bytes(data)
    old = os.environ.pop("SA2_STREAM_DUMP_BREAK", None)
    if brk:
        os.environ["SA2_STREAM_DUMP_BREAK"] = brk
    try:
        return SD.scan(p, want_range=(0, 1000), **kw)
    finally:
        os.environ.pop("SA2_STREAM_DUMP_BREAK", None)
        if old is not None:
            os.environ["SA2_STREAM_DUMP_BREAK"] = old


def whole() -> list[tuple[int, bytes]]:
    return start() + ds_buffer() + xa_voice() + present(T0 + 500, 2) + [END]


def doc() -> tuple:
    """The format's lines for what these streams' headers carry: the two device rules,
    Direct3DCreate9's entry, and every audio line."""
    rules, entries, aud = SD.doc_table(STREAM_H.read_text())
    keep = {("IDirect3D9", "CreateDevice"), ("IDirect3DDevice9", "Present")}
    return {k: v for k, v in rules.items() if k in keep}, {12: entries[12]}, aud


# ---------------------------------------------------------------- the reader

def test_a_whole_version_2_stream_reads_to_its_counts(tmp_path):
    s = scan(tmp_path, make(whole()), doc=doc())
    assert s["problems"] == []
    assert s["version"] == 2 and s["kinds"]["audio"]["records"] == 9
    assert s["audio"] == {"records": 9, "ops": {"DS_CREATE_BUFFER": 1, "DS_LOCK": 1, "DS_PLAY": 1, "DS_UNLOCK": 1,
                                                "XA_CREATE_MASTER": 1, "XA_CREATE_SOURCE": 1, "XA_SET_MATRIX": 1,
                                                "XA_START": 1, "XA_SUBMIT": 1},
                          "pcm_bytes": 32 + 96, "checked": 9, "buffers": 1, "voices": 2, "stores": 1,
                          "store_watch": {"every_frames": 64, "set": 1, "checks": 0, "bytes_hashed": 0,
                                          "watched_at_exit": 1}}
    assert s["blobs"]["pcm"] == {"blobs": 2, "bytes": 128} and s["blobs"]["state"]["blobs"] == 9
    assert s["blobs"]["hash"] == {"blobs": 1, "bytes": 8}
    # the same records under a version 1 header are a kind that version does not have
    v1 = scan(tmp_path, make(whole(), version=1))
    assert v1["audio"] is None and "kinds this reader does not know in a version 1 stream: [6]" in v1["problems"][-1]


def watched() -> list[tuple[int, bytes]]:
    """whole()'s buffer and voice, then 64 Presents more: the 64th FRAME checks the store
    its Unlock set the watch on, and a Lock after it checks it again and ends the watch."""
    st = dict(mem=MEM, mem_len=32, playing=1, looping=1)
    lock = audio("DS_LOCK", BUF, T0 + 700, [0, 8, 0x03004108, 0x0300410C, 0x03004110, 0x03004114, 0],
                 items=[(OUT, 2, W32.pack(MEM)), (OUT, 3, W32.pack(8)), (OUT, 4, W32.pack(0)),
                        (OUT, 5, W32.pack(0)), (STATE, NO_ARG, ds_state(**st))])
    return (start() + ds_buffer() + xa_voice() + [r for k in range(64) for r in present(T0 + 10 * (k + 1), k + 2)]
            + [lock, END])


def test_the_audio_counts_are_held_to_the_recorders(tmp_path):
    s = scan(tmp_path, make(watched()))
    assert s["problems"] == [] and s["frames"] == 65
    # set at the Unlock; checked at FRAME 64 and at the Lock, which ends the watch
    assert s["audio"]["store_watch"] == {"every_frames": 64, "set": 1, "checks": 2, "bytes_hashed": 64,
                                         "watched_at_exit": 0}
    counts = {"buffer_watch": s["buffer_watch"], "methods": [], "entries": [], "presents": 65, "notes": {},
              "audio": {"pcm_bytes": 128, "ops": [{"name": n, "recorded": c} for n, c in s["audio"]["ops"].items()],
                        "store_watch": dict(s["audio"]["store_watch"], changed=0)}}
    counts["methods"] = [{"type": m["type"], "slot": m["slot"], "calls": m["calls"]} for m in s["methods"]]
    counts["entries"] = [{"name": "Direct3DCreate9", "calls": 1}]
    assert SD.compare_counts(s, counts) == []
    off = json.loads(json.dumps(counts))
    off["audio"]["ops"][0]["recorded"] += 1
    off["audio"]["pcm_bytes"] = 4
    off["audio"]["store_watch"]["checks"] = 1
    assert SD.compare_counts(s, off) == [
        "audio DS_CREATE_BUFFER: 1 records in the stream, 2 recorded in the counts file",
        "audio: 128 PCM bytes in the stream, 4 in the counts file",
        "audio store_watch.checks: 1 in the counts file, 2 from the stream's own Locks, Unlocks and FRAMEs"]
    os.environ["SA2_STREAM_DUMP_BREAK"] = "auwatch"
    try:
        assert SD.compare_counts(s, off)[2:] == []
    finally:
        os.environ.pop("SA2_STREAM_DUMP_BREAK", None)
    del off["audio"]["store_watch"]
    assert SD.compare_counts(s, off)[2:] == ["the counts file has no audio store_watch to hold to the stream's"]
    del off["audio"]
    assert SD.compare_counts(s, off) == ["the stream has audio and the counts file has no audio counts"]


def two_locks(first_hash: bool = False, second: bytes | None = None) -> list[tuple[int, bytes]]:
    """A buffer locked twice before either Lock is unlocked: the first Unlock leaves a
    Lock open on the store and carries no hash (with first_hash, one anyway); the second
    closes the last and carries the store's (`second`, its 8 bytes, in its place)."""
    st = dict(mem=MEM, mem_len=32)

    def lock(off):
        return audio("DS_LOCK", BUF, T0, [off, 16, 0x03004108, 0x0300410C, 0x03004110, 0x03004114, 0],
                     items=[(OUT, 2, W32.pack(MEM + off)), (OUT, 3, W32.pack(16)), (OUT, 4, W32.pack(0)),
                            (OUT, 5, W32.pack(0)), (STATE, NO_ARG, ds_state(**st))])
    u1 = [(PCM, 0, FRAMES8[:16]), (STATE, NO_ARG, ds_state(**st))] + ([store_hash(FRAMES8)] if first_hash else [])
    u2 = [(PCM, 0, FRAMES8[16:]), (STATE, NO_ARG, ds_state(**st))]
    u2 += [store_hash(FRAMES8)] if second is None else ([(HASH, NO_ARG, second)] if second else [])
    return (start() + ds_buffer()[:1] + [lock(0), lock(16),
                                         audio("DS_UNLOCK", BUF, T0, [MEM, 16, 0, 0], items=u1),
                                         audio("DS_UNLOCK", BUF, T0, [MEM + 16, 16, 0, 0], items=u2)]
            + ds_buffer()[-1:] + present(T0 + 500, 2) + [END])


def test_an_unlock_carries_the_store_hash_when_it_closes_the_last_lock(tmp_path):
    s = scan(tmp_path, make(two_locks()))
    assert s["problems"] == [] and s["audio"]["store_watch"]["set"] == 1 and s["blobs"]["hash"]["blobs"] == 1
    for build, needle in ((two_locks(first_hash=True), "blobs its line does not give: [('hash', 65535, 8)]"),
                          (two_locks(second=b""), "no hash blob of the call, which its line gives"),
                          (two_locks(second=b"\0" * 4), "the hash blob of the call is 4 bytes, and its line gives 8")):
        got = scan(tmp_path, make(build))["problems"]
        assert any(needle in p for p in got), got
        assert scan(tmp_path, make(build), "aurules")["problems"] == []


def test_the_header_names_the_formats_state_sizes(tmp_path):
    head = header()
    head["audio"]["ds_state_bytes"] = 80
    got = scan(tmp_path, make(whole(), head=head))["problems"]
    assert got == ["the header's audio ds_state_bytes is 80, and host/sa2_stream.h's struct is 84 bytes"], got
    assert scan(tmp_path, make(whole(), head=head), "autable")["problems"] == []


def replace(records, i, rec):
    out = list(records)
    out[i] = rec
    return out


W = whole()
I_UNLOCK = next(i for i, r in enumerate(W) if r[0] == 6 and struct.unpack_from("<H", r[1], 2)[0] == OP["DS_UNLOCK"])
I_MATRIX = next(i for i, r in enumerate(W) if r[0] == 6 and struct.unpack_from("<H", r[1], 2)[0] == OP["XA_SET_MATRIX"])
I_PLAY = next(i for i, r in enumerate(W) if r[0] == 6 and struct.unpack_from("<H", r[1], 2)[0] == OP["DS_PLAY"])
I_SUBMIT = next(i for i, r in enumerate(W) if r[0] == 6 and struct.unpack_from("<H", r[1], 2)[0] == OP["XA_SUBMIT"])
UNLOCKED = ds_state(mem=MEM, mem_len=32)

MALFORMED = [
    ("aurules", lambda: replace(W, I_UNLOCK, audio("DS_UNLOCK", BUF, T0, [MEM, 32, 0, 0], items=[(PCM, 0, FRAMES8)])),
     "(DS_UNLOCK, IDirectSoundBuffer8::Unlock): no state blob of the call, which its line gives"),
    ("aurules", lambda: replace(W, I_UNLOCK, audio("DS_UNLOCK", BUF, T0, [MEM, 32, 0, 0],
                                                   items=[(PCM, 0, FRAMES8[:28]), (STATE, NO_ARG, UNLOCKED)])),
     "the pcm blob of arg 0 is 28 bytes, and its line gives 32"),
    ("aurules", lambda: replace(W, I_UNLOCK, audio("DS_UNLOCK", BUF, T0, [MEM + 8, 32, 0, 0],
                                                   items=[(PCM, 0, FRAMES8), (STATE, NO_ARG, UNLOCKED)])),
     "32 bytes at 0x03300008 are not inside the buffer's store 0x03300000+32"),
    ("aurules", lambda: replace(W, I_UNLOCK, audio("DS_UNLOCK", BUF, T0, [MEM, 32, 0, 0],
                                                   items=[(PCM, 0, FRAMES8), (STATE, NO_ARG, ds_state(buffer=BUF2))])),
     "its STATE describes 0x03250080, and the call names 0x03250040"),
    ("aurules", lambda: replace(W, I_MATRIX, audio("XA_SET_MATRIX", SOURCE, T0, [MASTER, 6, 2, 0x0089E4A8, 0],
                                                   items=[(IN, 3, struct.pack("<12I", *([0] * 12))),
                                                          (STATE, NO_ARG, xa_state(matrix=[fbits(v) for v in LEVELS],
                                                                                   matrix_src=6, matrix_dst=2))])),
     "the level matrix the call passed is not the one its STATE holds"),
    ("argc", lambda: replace(W, I_PLAY, audio("DS_PLAY", BUF, T0, [0, 0], items=[(STATE, NO_ARG, UNLOCKED)])),
     "2 arguments, and the header's audio table says 3"),
    ("auobjects", lambda: replace(W, I_PLAY, audio("DS_PLAY", BUF2, T0, [0, 0, 1],
                                                   items=[(STATE, NO_ARG, ds_state(buffer=BUF2))])),
     "buffer 0x03250080 is one no earlier call made"),
    ("auclock", lambda: replace(W, I_PLAY, audio("DS_PLAY", BUF, T0 - 1, [0, 0, 1],
                                                 items=[(STATE, NO_ARG, ds_state(mem=MEM, mem_len=32,
                                                                                 anchor_us=T0 - 1))])),
     "goes back: virtual_us 4999999999 after 5000000000"),
    # a call that re-anchors, its STATE anchored elsewhere
    ("aurules", lambda: replace(W, I_PLAY, audio("DS_PLAY", BUF, T0, [0, 0, 1],
                                                 items=[(STATE, NO_ARG, ds_state(mem=MEM, mem_len=32,
                                                                                 anchor_us=T0 - 3))])),
     "the call re-anchors, and its STATE's anchor_us 4999999997 is not its virtual_us 5000000000"),
    # a submit whose frames do not end at its STATE's queued_end
    ("aurules", lambda: replace(W, I_SUBMIT, audio("XA_SUBMIT", SOURCE, T0, [0x03004300, 0],
                                                   items=[(IN, 0, struct.pack("<9I", 0, 96, 0x03004400, 0, 0, 0, 0, 0, 0)),
                                                          (PCM, 0, bytes(96)),
                                                          (STATE, NO_ARG, xa_state(matrix=[fbits(v) for v in LEVELS],
                                                                                   matrix_src=6, matrix_dst=2,
                                                                                   queued=1, queued_end=5))])),
     "its STATE's queued_end is 5, and the 4 frames it submits end at 4"),
    # a matrix past 64 x 64 carries no blob
    ("aurules", lambda: replace(W, I_MATRIX, audio("XA_SET_MATRIX", SOURCE, T0, [MASTER, 65, 2, 0x0089E4A8, 0],
                                                   items=[(IN, 3, bytes(65 * 2 * 4)),
                                                          (STATE, NO_ARG, xa_state())])),
     "blobs its line does not give: [('in', 3, 520)]"),
]


@pytest.mark.parametrize("brk,build,needle", MALFORMED, ids=[f"{m[0]}-{i}" for i, m in enumerate(MALFORMED)])
def test_each_audio_malformation_is_a_problem_and_its_break_hides_it(tmp_path, brk, build, needle):
    data = make(build())
    assert any(needle in p for p in scan(tmp_path, data)["problems"]), scan(tmp_path, data)["problems"]
    assert scan(tmp_path, data, brk)["problems"] == []


def test_the_header_audio_table_is_held_to_the_format_lines(tmp_path):
    head = header()
    head["audio"]["ops"][4][5] = "pcm 0 1; ds_state; store_hash"     # DS_UNLOCK without its second segment
    data = make(whole(), head=head)
    got = scan(tmp_path, data, doc=doc())["problems"]
    assert got == ["audio op 5: the header says ('DS_UNLOCK', 'IDirectSoundBuffer8', 'Unlock', 4, "
                   "'pcm 0 1; ds_state; store_hash'), the format's line ('DS_UNLOCK', 'IDirectSoundBuffer8', "
                   "'Unlock', 4, 'pcm 0 1; pcm 2 3; ds_state; store_hash')"], got
    assert scan(tmp_path, data, "autable", doc=doc())["problems"] == []
    # and both ways: a line the header does not carry
    head = header()
    head["audio"]["ops"] = head["audio"]["ops"][:-1]
    got = scan(tmp_path, make(whole(), head=head), doc=doc())["problems"]
    assert got == ["audio op 28: the header says None, the format's line ('XA_SET_RATIO', "
                   "'IXAudio2SourceVoice', 'SetFrequencyRatio', 2, 'xa_state')"], got


@pytest.mark.parametrize("version,head,why", [
    (2, header(1, schema=SD.SCHEMA_AUDIO), "a version 2 header without an audio table"),
    (1, header(2, schema=SD.SCHEMA), "a version 1 header with an audio table"),
    (2, header(2, schema=SD.SCHEMA), "not sa2-d3d9-stream/v2 (version 2)"),
    (3, header(2), "version 3; this reader reads 1 and 2"),
])
def test_a_header_that_is_not_its_versions_is_not_a_stream(tmp_path, version, head, why):
    p = tmp_path / "x.sa2s"
    p.write_bytes(make([END], version=version, head=head))
    with pytest.raises(SD.StreamError) as e:
        SD.scan(p)
    assert why in str(e.value)


def test_the_format_names_every_op_once_and_the_reader_reads_its_grammar():
    text = STREAM_H.read_text()
    assert sorted(OPS) == list(range(1, 29))
    for op, (name, iface, method, argc, aops) in OPS.items():
        assert SD.AUDIO_IFACES[iface] in (1, 2) and name.startswith(("DS_", "XA_"))
        assert name.startswith("DS_") == (SD.AUDIO_IFACES[iface] == 1)
        assert SD.aops_text(SD.parse_aops(aops)) == aops
    for bad in ("pcm 0", "frob 1", "ds_state 1"):
        with pytest.raises(ValueError):
            SD.parse_aops(bad)
    with pytest.raises(ValueError, match="audio op 5 has two lines"):
        SD.doc_table(text + "\n *   audio 5 DS_UNLOCK IDirectSoundBuffer8::Unlock 4   ds_state\n")


def test_the_matrix_bound_is_the_recorders_and_the_formats():
    """An `in` op carries no blob past a count of 64 (the malformation case above): the
    reader's MATRIX_MAX is the recorder's AU_MATRIX_MAX and the number the format's line
    gives, as the watch period is the recorder's (test_s5_stream.py)."""
    assert re.search(r"#define AU_MATRIX_MAX (\d+)u", RECORDER.read_text()).group(1) == str(SD.MATRIX_MAX)
    assert re.search(r"none when args\[B\] or args\[C\] is over (\d+)", STREAM_H.read_text()).group(1) == \
        str(SD.MATRIX_MAX)


def device_row(seq: int, iface: str, method: str, this: int, args, ret: int = 0, out=(0, 0), thread=None,
               us: int = T0, answered: str = "from-guest") -> dict:
    """One row of a device's call log, in run.json's words (harness/crt_probe.c)."""
    row = {"seq": seq, "virtual_us": us, "interface": iface, "method": method, "this": f"0x{this:08x}",
           "args": [f"0x{a:08x}" for a in args], "ret": f"0x{ret:08x}", "out": [f"0x{o:08x}" for o in out],
           "answered": answered}
    if thread is not None:
        row["thread"] = thread
    return row


def device_run() -> dict:
    """The audio devices' own record of the run whole() stands for: each device's call log
    with a call no op records (QueryInterface) and a call that failed (a SetVolume below
    DSBVOLUME_MIN) among the calls the stream holds, and its per-method calls."""
    ds = [device_row(1, "IDirectSound8", "CreateSoundBuffer", DSOUND, [0x03004200, 0x03004104, 0], out=(BUF, 0)),
          device_row(2, "IDirectSoundBuffer8", "QueryInterface", BUF, [0x03004000, 0x03004004], ret=0x80004002),
          device_row(3, "IDirectSoundBuffer8", "Lock", BUF, [0, 32, 0x03004108, 0x0300410C, 0x03004110,
                                                             0x03004114, 0], out=(MEM, 32)),
          device_row(4, "IDirectSoundBuffer8", "SetVolume", BUF, [0xFFFFD8EF], ret=0x80070057),
          device_row(5, "IDirectSoundBuffer8", "Unlock", BUF, [MEM, 32, 0, 0]),
          device_row(6, "IDirectSoundBuffer8", "Play", BUF, [0, 0, 1])]
    xa = [device_row(1, "IXAudio2", "CreateMasteringVoice", ENGINE, [0x01AF0EDC, 0, 48000, 0, 0, 0],
                     out=(MASTER, 0), thread=0),
          device_row(2, "IXAudio2", "CreateSourceVoice", ENGINE, [0x03004148, 0x03004280, 0, 0x40000000, 0, 0, 0],
                     out=(SOURCE, 0), thread=0),
          device_row(3, "IXAudio2SourceVoice", "SetOutputMatrix", SOURCE, [MASTER, 6, 2, 0x0089E4A8, 0], thread=0),
          device_row(4, "IXAudio2SourceVoice", "SubmitSourceBuffer", SOURCE, [0x03004300, 0], out=(1, 0), thread=0),
          device_row(5, "IXAudio2SourceVoice", "Start", SOURCE, [0, 0], thread=0)]

    def methods(rows):
        n: dict = {}
        for r in rows:
            n[(r["interface"], r["method"])] = n.get((r["interface"], r["method"]), 0) + 1
        return [{"interface": i, "name": m, "calls": c} for (i, m), c in n.items()]
    return {"dsound": {"method_calls": len(ds), "calls": ds, "methods": methods(ds)},
            "xaudio2": {"method_calls": len(xa), "log_dropped": 0, "calls": xa, "methods": methods(xa)}}


def device_counts(s: dict) -> dict:
    """The recorder's counts file for whole(): every op entered, returned and recorded as
    the stream holds it, and the failed SetVolume."""
    ops = [{"name": n, "entered": s["audio"]["ops"].get(n, 0), "returned": s["audio"]["ops"].get(n, 0),
            "recorded": s["audio"]["ops"].get(n, 0), "failed": 0} for _op, (n, *_r) in sorted(OPS.items())]
    vol = next(r for r in ops if r["name"] == "DS_SET_VOLUME")
    vol.update(entered=1, returned=1, failed=1)
    return {"audio": {"ops": ops}}


def held_literal(tmp_path: Path, records, run: dict, counts: dict | None) -> list[str]:
    s = scan(tmp_path, make(records), keep_audio=SD.audio_keep(run))
    assert s["problems"] == [], s["problems"]
    return SD.hold_audio_to_run(s, run, counts)


def test_the_audio_records_are_held_to_the_devices_own_record(tmp_path):
    s = scan(tmp_path, make(whole()))
    counts = device_counts(s)
    assert held_literal(tmp_path, whole(), device_run(), counts) == []
    # an Unlock the recorder dropped and counted as failed: the stream and the counts agree,
    # and the device's log says the call succeeded
    dropped = [r for i, r in enumerate(whole()) if i != I_UNLOCK]
    c = json.loads(json.dumps(counts))
    next(r for r in c["audio"]["ops"] if r["name"] == "DS_UNLOCK").update(recorded=0, failed=1)
    assert held_literal(tmp_path, dropped, device_run(), c) == [
        "the dsound device's call 5 (IDirectSoundBuffer8::Unlock on 0x03250040) is not the stream's next dsound "
        "record, seq 6: the record is DS_PLAY; its arguments are ['0x0', '0x0', '0x1'], the device's "
        "['0x3300000', '0x20', '0x0', '0x0']"]
    # a record of a call the device failed, as a recorder that misjudged the answer makes it
    run = device_run()
    run["dsound"]["calls"][5]["ret"] = "0x88780032"
    failed = replace(whole(), I_PLAY, audio("DS_PLAY", BUF, T0, [0, 0, 1], eax=0x88780032,
                                            items=[(STATE, NO_ARG, ds_state(mem=MEM, mem_len=32))]))
    assert held_literal(tmp_path, failed, run, counts) == [
        "record seq 7 (DS_PLAY) is the dsound device's call 6, which answered 0x88780032: a call that failed "
        "has no record"]
    # a record whose answer is not the device's
    assert held_literal(tmp_path, whole(), run, counts) == [
        "the dsound device's log holds every call it answered, and 1 of the stream's 4 dsound records are none "
        "of them",
        "audio DS_PLAY: 1 records, and the dsound device's log holds 0 calls of it that succeeded",
        "audio DS_PLAY: 0 failed in the counts file, and the dsound device's log holds 1 that failed"]
    # the device's words against the record's: an out word, a guest thread, a clock
    run = device_run()
    run["dsound"]["calls"][2]["out"][0] = "0x03300004"
    run["xaudio2"]["calls"][3]["thread"] = 1
    run["xaudio2"]["calls"][4]["virtual_us"] = T0 + 1
    assert held_literal(tmp_path, whole(), run, counts) == [
        "the dsound device's call 3 (IDirectSoundBuffer8::Lock on 0x03250040) is not the stream's next dsound "
        "record, seq 5: its out0 word is 0x03300000, the device's 0x03300004",
        "the xaudio2 device's call 4 (IXAudio2SourceVoice::SubmitSourceBuffer on 0x03273080) is not the "
        "stream's next xaudio2 record, seq 11: its guest thread is 0, the device's 1"]
    run["xaudio2"]["calls"][3]["thread"] = 0
    assert held_literal(tmp_path, whole(), run, counts)[1:] == [
        "the xaudio2 device's call 5 (IXAudio2SourceVoice::Start on 0x03273080) is not the stream's next "
        "xaudio2 record, seq 12: its virtual_us is 5000000000, the device's 5000000001"]
    # the counts against the device: a call that did not enter, did not come back, was
    # neither recorded nor failed, or failed where the device's log says it succeeded
    c = json.loads(json.dumps(counts))
    ops = {r["name"]: r for r in c["audio"]["ops"]}
    ops["DS_PLAY"]["entered"] = 2
    ops["DS_LOCK"]["returned"] = 0
    ops["XA_START"]["recorded"] = 0
    ops["DS_SET_VOLUME"]["failed"] = 0
    ops["DS_SET_VOLUME"]["recorded"] = 1
    assert held_literal(tmp_path, whole(), device_run(), c) == [
        "audio DS_LOCK: 1 calls entered the recorder and 0 came back",
        "audio DS_LOCK: 1 recorded and 0 failed of the 0 that came back",
        "audio DS_PLAY: 2 calls entered the recorder, and the dsound device answered 1 calls of "
        "IDirectSoundBuffer8::Play",
        "audio DS_PLAY: 2 calls entered the recorder and 1 came back",
        "audio DS_SET_VOLUME: 0 failed in the counts file, and the dsound device's log holds 1 that failed",
        "audio XA_START: 0 recorded and 0 failed of the 1 that came back"]
    # a record the device's whole log does not hold
    extra = whole()[:I_PLAY + 1] + [whole()[I_PLAY]] + whole()[I_PLAY + 1:]
    assert held_literal(tmp_path, extra, device_run(), None) == [
        "audio DS_PLAY: 2 records in the stream, and the dsound device answered 1 calls of IDirectSoundBuffer8::Play",
        "the dsound device's log holds every call it answered, and 1 of the stream's 5 dsound records are none "
        "of them",
        "audio DS_PLAY: 2 records, and the dsound device's log holds 1 calls of it that succeeded"]
    # the same past the end of a log that does not hold every call: only the walk's window
    run = device_run()
    run["dsound"]["method_calls"] = 7
    run["dsound"]["methods"] = [dict(m, calls=2) if m["name"] == "Play" else m for m in run["dsound"]["methods"]]
    assert held_literal(tmp_path, extra, run, None) == []
    del run["dsound"]
    assert held_literal(tmp_path, whole(), run, None) == [
        "the stream holds 4 dsound records, and the run's record has no dsound device"]
    os.environ["SA2_STREAM_DUMP_BREAK"] = "auheld"
    try:
        assert SD.hold_audio_to_run(s, run, None) == []
    finally:
        os.environ.pop("SA2_STREAM_DUMP_BREAK", None)
    assert SD.hold_audio_to_run(scan(tmp_path, make(whole(), version=1)), device_run(), None) == []


def test_a_log_the_summary_kept_too_few_records_to_walk_is_a_problem(tmp_path):
    """The walk needs the AUDIO heads scan kept (keep_audio, audio_keep(run)). Past a log
    cut short it is the only check of the records: an Unlock the recorder dropped and
    counted as failed agrees with every per-op count. A summary that kept fewer of a
    device's records than its log has calls, or none (one read back from --json, which
    leaves the heads out), fails for that device rather than pass a log it never walked."""
    run = device_run()
    for key in ("dsound", "xaudio2"):
        run[key]["method_calls"] += 1                   # each log cut short: the walk is all there is
    dropped = [r for i, r in enumerate(whole()) if i != I_UNLOCK]
    counts = device_counts(scan(tmp_path, make(whole())))
    next(r for r in counts["audio"]["ops"] if r["name"] == "DS_UNLOCK").update(recorded=0, failed=1)
    assert held_literal(tmp_path, dropped, run, counts) == [
        "the dsound device's call 5 (IDirectSoundBuffer8::Unlock on 0x03250040) is not the stream's next dsound "
        "record, seq 6: the record is DS_PLAY; its arguments are ['0x0', '0x0', '0x1'], the device's "
        "['0x3300000', '0x20', '0x0', '0x0']"]

    def unwalked(ds_kept: int, xa_kept: int) -> list[str]:
        return [f"the dsound device's log holds 6 calls, and the summary kept the heads of {ds_kept} of the "
                f"stream's 3 dsound records to walk it against (scan's keep_audio, which audio_keep(run) "
                f"gives): the log is not walked",
                f"the xaudio2 device's log holds 5 calls, and the summary kept the heads of {xa_kept} of the "
                f"stream's 5 xaudio2 records to walk it against (scan's keep_audio, which audio_keep(run) "
                f"gives): the log is not walked"]
    assert SD.hold_audio_to_run(scan(tmp_path, make(dropped)), run, counts) == unwalked(0, 0)
    assert SD.hold_audio_to_run(scan(tmp_path, make(dropped), keep_audio=2), run, counts) == unwalked(2, 2)
    s = scan(tmp_path, make(dropped), keep_audio=SD.audio_keep(run))
    del s["audio_heads"]
    assert SD.hold_audio_to_run(s, run, counts) == unwalked(0, 0)
    # a device whose records were all kept is walked, however few they are: none here
    s = scan(tmp_path, make([r for r in whole() if r[0] != 6 or struct.unpack_from("<H", r[1])[0] == 1]),
             keep_audio=SD.audio_keep(run))
    assert s["problems"] == [] and SD.hold_audio_to_run(s, run, None) == [
        "the xaudio2 device's call 1 (IXAudio2::CreateMasteringVoice on 0x03273000) answered 0x00000000, and the "
        "stream holds no xaudio2 record for it"]


# ---------------------------------------------------------------- the mixer

@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    """A function (break) -> host/player/mixer.cpp built as sa2_audio_render (-DSA2_MIXER_MAIN,
    as host/player/build.sh builds it), with MIXER_BREAK_<break> when one is named. The
    system g++ and libzstd, which build.sh uses, are required: without them this fails
    rather than skips, since a skip would pass the gate with the mixer unheld."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.fail("g++ is not on PATH: the mixer cannot be built")
    d = tmp_path_factory.mktemp("mixer")
    made: dict[str, Path] = {}

    def get(brk: str = "") -> Path:
        if brk not in made:
            exe = d / f"sa2_audio_render{'.' + brk if brk else ''}"
            cmd = [gxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                   "-DSA2_MIXER_MAIN", f"-I{ROOT / 'host'}", "-o", str(exe), str(PLAYER / "mixer.cpp"),
                   str(PLAYER / "stream.cpp"), str(PLAYER / "sha256.cpp"), "-lzstd"]
            if brk:
                cmd.insert(1, f"-DMIXER_BREAK_{brk}")
            done = subprocess.run(cmd, capture_output=True, text=True)
            assert done.returncode == 0, done.stderr
            made[brk] = exe
        return made[brk]
    return get


def read_wav(path: Path) -> tuple[dict, list[tuple[float, float]]]:
    """The renderer's WAV: its fmt and fact fields, and its stereo float frames."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    chunks, p = {}, 12
    while p + 8 <= len(data):
        cid, n = data[p:p + 4], struct.unpack_from("<I", data, p + 4)[0]
        chunks[cid] = data[p + 8:p + 8 + n]
        p += 8 + n + (n & 1)
    tag, ch, rate, bps, block, bits, cb = struct.unpack("<HHIIHHH", chunks[b"fmt "])
    fmt = {"riff": struct.unpack_from("<I", data, 4)[0], "tag": tag, "channels": ch, "rate": rate,
           "bytes_per_sec": bps, "block": block, "bits": bits, "cb": cb,
           "fact": struct.unpack("<I", chunks[b"fact"])[0], "data": len(chunks[b"data"]), "file": len(data)}
    vals = struct.unpack(f"<{len(chunks[b'data']) // 4}f", chunks[b"data"])
    return fmt, list(zip(vals[0::2], vals[1::2]))


def render(exe: Path, tmp_path: Path, records: list[tuple[int, bytes]], name: str = "s",
           extra: tuple[str, ...] = ()) -> tuple[int, dict, list]:
    """The stream of `records` (version 2) rendered: exit status, the JSON, the frames."""
    src = tmp_path / f"{name}.sa2s"
    src.write_bytes(make(records))
    wav, js = tmp_path / f"{name}.wav", tmp_path / f"{name}.json"
    done = subprocess.run([str(exe), str(src), "--wav", str(wav), "--json", str(js), *extra],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode in (0, 1), done.stderr
    return done.returncode, json.loads(js.read_text()), read_wav(wav)[1]


def ds_value(k: int) -> tuple[float, float]:
    return ((k % 8) + 1) / 32, -((k % 8) + 1) / 32


def test_a_48k_buffer_plays_one_frame_per_sample_and_wraps_when_it_loops(renderer, tmp_path):
    rc, js, got = render(renderer(), tmp_path, start() + ds_buffer() + present(T0 + 500, 2) + [END])
    assert rc == 0 and js["audio"]["problems"]["count"] == 0
    assert got == [ds_value(n) for n in range(24)]               # 500 us at 48 kHz: 24 samples, 3 laps
    assert js["audio"]["samples"] == 24 and js["audio"]["origin_us"] == T0 and js["first_frame_us"] == T0
    assert js["audio"]["ds"]["resampled_samples"] == 0
    level = js["audio"]["ds"]["level"]
    assert level["peak"] == 0.25
    # the report prints nine significant digits
    assert math.isclose(level["rms"], math.sqrt(sum(((n % 8) + 1) ** 2 for n in range(24)) / 24) / 32, rel_tol=1e-8)
    # a looping buffer that does not wrap plays its 8 frames once
    _, _, broken = render(renderer("NOWRAP"), tmp_path, start() + ds_buffer() + present(T0 + 500, 2) + [END])
    assert broken[:8] == got[:8] and broken[8:] != got[8:]


def test_a_buffer_that_does_not_loop_halts_at_its_end(renderer, tmp_path):
    rc, js, got = render(renderer(), tmp_path, start() + ds_buffer(looping=0) + present(T0 + 500, 2) + [END])
    assert rc == 0 and got == [ds_value(n) for n in range(8)] + [(0.0, 0.0)] * 16
    assert js["audio"]["ds"]["playing_at_end"] == 0


@pytest.mark.parametrize("rate", [8000, 11025, 22050, 32000, 44100, 48000, 96000, 192000, 1073741823])
@pytest.mark.parametrize("looping", [0, 1])
@pytest.mark.parametrize("offset_us", [0, 7, 21])
def test_fractional_clock_survives_block_boundaries_and_multiple_wraps(renderer, tmp_path, rate, looping, offset_us):
    recs = start() + ds_buffer(us=T0 + offset_us, avg=rate * 4, looping=looping)
    for frame, elapsed in enumerate([137, 502, 1001, 1987, 3227], 2):
        recs += present(T0 + elapsed, frame)
    rc, js, got = render(renderer(), tmp_path, recs + [END])
    assert rc == 0 and js["audio"]["problems"]["count"] == 0
    want = []
    for n in range((3227 * 48000 + 999999) // 1000000):
        elapsed = 125 * n - offset_us * 6
        position = max(0, elapsed) * rate
        if elapsed < 0 or (not looping and position >= 8 * 6000000):
            want.append((0.0, 0.0))
            continue
        whole, fraction = divmod(position, 6000000)
        frame = whole % 8
        a = ds_value(frame)
        b = ds_value((frame + 1) % 8 if looping else min(frame + 1, 7))
        want.append(a if rate == 48000 else tuple(f32(x + (y - x) * (fraction / 6000000)) for x, y in zip(a, b)))
    assert got == want


def test_a_24k_buffer_is_interpolated_halfway_between_its_frames(renderer, tmp_path):
    recs = start() + ds_buffer(avg=96000) + present(T0 + 500, 2) + [END]
    rc, js, got = render(renderer(), tmp_path, recs)

    def want(n: int) -> tuple[float, float]:
        a, b = ds_value(n // 2), ds_value(n // 2 + 1)            # the next frame of a loop wraps
        return (a if n % 2 == 0 else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

    assert rc == 0 and got == [want(n) for n in range(24)]
    assert got[15] == (4.5 / 32, -4.5 / 32)                       # frame 7 and frame 0, halfway
    assert js["audio"]["ds"]["resampled_samples"] == 24
    for brk in ("NOLERP", "CLOCK"):                               # no interpolation; SetFrequency ignored
        _, _, broken = render(renderer(brk), tmp_path, recs, brk)
        assert broken != got, brk


def test_volume_and_pan_are_directsounds_hundredths_of_a_decibel(renderer, tmp_path):
    recs = start() + ds_buffer(volume=-600, pan=2000) + present(T0 + 500, 2) + [END]
    rc, _, got = render(renderer(), tmp_path, recs)
    gl, gr = f32(10 ** (-600 / 2000) * 10 ** (-2000 / 2000)), f32(10 ** (-600 / 2000))
    assert rc == 0 and got == [(f32(ds_value(n)[0] * gl), f32(ds_value(n)[1] * gr)) for n in range(24)]
    assert math.isclose(got[0][1] / ds_value(0)[1], 0.501187, rel_tol=1e-6)    # -6 dB
    _, _, broken = render(renderer("GAIN"), tmp_path, recs, "gain")
    assert broken == [ds_value(n) for n in range(24)]


def test_a_six_channel_voice_is_mixed_by_its_level_matrix_and_runs_dry(renderer, tmp_path):
    recs = start() + xa_voice(frames=4) + present(T0 + 125, 2) + [END]
    rc, js, got = render(renderer(), tmp_path, recs)

    def mix(x: list[float]) -> tuple[float, float]:
        l = r = 0.0
        for c in range(6):
            l, r = f32(l + f32(LEVELS[c] * x[c])), f32(r + f32(LEVELS[6 + c] * x[c]))
        return l, r

    assert rc == 0 and got == [mix(x) for x in xa_frames(4)] + [(0.0, 0.0)] * 2
    x = [1 / 32 + c / 256 for c in range(6)]
    assert got[0] == (x[0] + 0.5 * x[2] + 0.25 * x[3] + 0.75 * x[4], x[1] + 0.5 * x[2] + 0.25 * x[3] + 0.75 * x[5])
    xa = js["audio"]["xa"]
    assert (xa["submits"], xa["submit_bytes"], xa["starved_samples"], xa["default_matrix_samples"]) == (1, 96, 2, 0)
    _, js2, broken = render(renderer("MATRIX"), tmp_path, recs, "matrix")
    assert broken != got and js2["audio"]["xa"]["default_matrix_samples"] == 4


# Eight 16-bit stereo frames at full scale and below it.
LOUD8 = b"".join(struct.pack("<hh", v, -v) for v in (32767, 30000, 20000, 0, -20000, -30000, -32767, -10000))


def test_live_play_turns_a_mix_past_full_scale_down_and_the_wav_keeps_the_clamp(renderer, tmp_path):
    """The game's 5.1 fold passes full scale in loud stretches. Live play sets the limiter
    (host/player/player.cpp, a live stream): the gain drops to hold the mix under 0.98 and
    nothing is clamped. Without it, as every receipt's WAV is rendered, the mix is clamped."""
    recs = start() + ds_buffer(data=LOUD8) + xa_voice(frames=4) + present(T0 + 500, 2) + [END]
    rc, js, clamped = render(renderer(), tmp_path, recs, "clamp")
    assert rc == 0 and js["audio"]["mix"]["clipped_samples"] > 0 and clamped[0][0] == 1.0
    assert js["audio"]["mix"]["limiter"] == {"on": False, "limited_frames": 0, "least_gain": 1}
    rc, jl, limited = render(renderer(), tmp_path, recs, "limit", ("--limit",))
    mix = jl["audio"]["mix"]
    assert rc == 0 and mix["clipped_samples"] == 0 and mix["level"] == js["audio"]["mix"]["level"]
    assert mix["limiter"]["on"] and mix["limiter"]["limited_frames"] == len(limited) == 24
    assert max(abs(v) for f in limited for v in f) <= f32(0.98) + 1e-6
    assert 0.85 < mix["limiter"]["least_gain"] < 0.9          # 0.98 / 1.0967, the first frame
    for a, b in zip(limited, clamped):                         # turned down, never turned over
        assert all(abs(x) <= abs(y) + 1e-6 and x * y >= 0 for x, y in zip(a, b))
    # a mix inside full scale is the same with the limiter
    quiet = start() + ds_buffer() + present(T0 + 500, 2) + [END]
    assert render(renderer(), tmp_path, quiet, "q1")[2] == render(renderer(), tmp_path, quiet, "q2", ("--limit",))[2]


def checks(play: int, write: int, status: int, queued: int, played: int) -> list[tuple[int, bytes]]:
    """A buffer and a voice, then at T0 + 250 the answers a GetCurrentPosition, a GetStatus
    and a GetState gave."""
    st = dict(mem=MEM, mem_len=32, playing=1, looping=1)
    xs = dict(matrix=[fbits(v) for v in LEVELS], matrix_src=6, matrix_dst=2, queued=1, queued_end=4, running=1)
    us = T0 + 250
    return (start() + ds_buffer() + xa_voice(frames=4)
            + [audio("DS_GET_POSITION", BUF, us, [0x03004118, 0x0300411C],
                     items=[(OUT, 0, W32.pack(play)), (OUT, 1, W32.pack(write)), (STATE, NO_ARG, ds_state(**st))]),
               audio("DS_GET_STATUS", BUF, us, [0x03004130],
                     items=[(OUT, 0, W32.pack(status)), (STATE, NO_ARG, ds_state(**st))]),
               audio("XA_GET_STATE", SOURCE, us, [0x03004380],
                     items=[(OUT, 0, struct.pack("<IIQ", 0, queued, played)),
                            (STATE, NO_ARG, xa_state(**dict(xs, played=4, queued=0)))])]
            + present(T0 + 500, 2) + [END])


def test_every_answer_the_device_gave_is_held_to_the_mixers_clock(renderer, tmp_path):
    """At 250 us the buffer (192000 bytes per second, 32 bytes, looping) is at byte 48 mod 32
    = 16, its write cursor 8 bytes on, playing and looping; the voice (48 kHz, four frames
    queued) has played min(12, 4) = 4 frames and has nothing queued."""
    rc, js, _ = render(renderer(), tmp_path, checks(16, 24, 5, 0, 4))
    au = js["audio"]
    assert rc == 0 and au["problems"]["count"] == 0, au["problems"]
    assert (au["ds"]["cursor_checks"], au["ds"]["status_checks"], au["xa"]["state_checks"]) == (2, 1, 1)
    for wrong, needle in (((20, 24, 5, 0, 4), "play cursor was 20 to the guest; the mixer's clock gives 16"),
                          ((16, 16, 5, 0, 4), "write cursor was 16 to the guest; the mixer's clock gives 24"),
                          ((16, 24, 1, 0, 4), "status was 0x1 to the guest; the mixer's clock gives 0x5"),
                          ((16, 24, 5, 1, 3), "answered 1 queued, 3 played; the mixer's clock gives 0, 4")):
        rc, js, _ = render(renderer(), tmp_path, checks(*wrong), "wrong")
        assert rc == 1 and any(needle in p for p in js["audio"]["problems"]["first"]), js["audio"]["problems"]


def test_a_record_the_mixer_cannot_place_is_a_problem(renderer, tmp_path):
    outside = start() + ds_buffer()[:2] + [
        audio("DS_UNLOCK", BUF, T0, [MEM + 8, 32, 0, 0],
              items=[(PCM, 0, FRAMES8), (STATE, NO_ARG, ds_state(mem=MEM, mem_len=32))])] + present(T0 + 500) + [END]
    rc, js, _ = render(renderer(), tmp_path, outside)
    assert rc == 1 and "32 bytes at 0x03300008 are outside buffer 0x03250040's store" in js["audio"]["problems"]["first"][0]
    unknown = start() + [audio("DS_PLAY", BUF, T0, [0, 0, 1], items=[(STATE, NO_ARG, ds_state())])] + present(T0 + 500) + [END]
    rc, js, _ = render(renderer(), tmp_path, unknown, "unknown")
    assert rc == 1 and "buffer 0x03250040, which no record made" in js["audio"]["problems"]["first"][0]


def test_a_store_that_is_not_the_guests_is_a_problem(renderer, tmp_path):
    """The store the mixer rebuilds from the PCM must hash, at the Unlock that closes its
    last Lock, to the recorder's hash of the guest's; an Unlock that should carry one and
    does not is a problem too, and so is a store whose length is not its buffer's."""
    rc, js, _ = render(renderer(), tmp_path, two_locks())
    assert rc == 0 and js["audio"]["ds"]["store_checks"] == 1 and js["audio"]["ds"]["store_mismatches"] == 0
    rc, js, _ = render(renderer(), tmp_path, two_locks(second=struct.pack("<Q", fnv1a64(FRAMES8[:16] + bytes(16)))),
                       "other")
    assert rc == 1 and js["audio"]["ds"]["store_mismatches"] == 1
    assert ("buffer 0x03250040's store 0x03300000, rebuilt from the stream's PCM, does not hash to the guest's "
            "store after the Unlock") in js["audio"]["problems"]["first"][0]
    rc, js, _ = render(renderer(), tmp_path, two_locks(second=b""), "none")
    assert rc == 1 and "buffer 0x03250040's Unlock closes its store's last Lock and carries no 8-byte store_hash" \
        in js["audio"]["problems"]["first"][0]
    for second in (struct.pack("<Q", 7), b""):                       # a mixer that does not hold the hash
        rc, js, _ = render(renderer("NOHASH"), tmp_path, two_locks(second=second), "nohash")
        assert rc == 0 and js["audio"]["ds"]["store_checks"] == 0
    longer = [audio("DS_CREATE_BUFFER", DSOUND, T0, [0x03004200, 0x03004104, 0],
                    items=[(OUT, 1, W32.pack(BUF)), (STATE, 1, ds_state(mem=MEM, mem_len=64))])]
    rc, js, _ = render(renderer(), tmp_path, start() + longer + present(T0 + 500, 2) + [END], "longer")
    assert rc == 1 and js["audio"]["problems"]["first"] == [
        "seq 4 audio op 1: buffer 0x03250040's store 0x03300000 is 64 bytes and the buffer 32: the runtime makes a "
        "store its buffer's length"] and js["audio"]["ds"]["stores"] == 0


def test_a_record_before_the_one_before_it_is_a_problem(renderer, tmp_path):
    late = replace(W, I_PLAY, audio("DS_PLAY", BUF, T0 - 1, [0, 0, 1],
                                    items=[(STATE, NO_ARG, ds_state(mem=MEM, mem_len=32, anchor_us=T0 - 1))]))
    rc, js, _ = render(renderer(), tmp_path, late)
    assert rc == 1 and js["audio"]["late_records"] == 1
    assert js["audio"]["problems"]["first"] == [
        "a record at virtual 4999999999 us after one at 5000000000 us: the mix cannot go back"]


def ds_format(fmt: dict, data: bytes) -> list[tuple[int, bytes]]:
    """A 48 kHz buffer of the format `fmt` (sa2s_ds_state fields), filled with `data`
    through a Lock and played looping at T0."""
    st = dict(fmt, mem=MEM, mem_len=len(data), bytes=len(data))
    return [audio("DS_CREATE_BUFFER", DSOUND, T0, [0x03004200, 0x03004104, 0],
                  items=[(OUT, 1, W32.pack(BUF)), (STATE, 1, ds_state(**dict(fmt, bytes=len(data))))]),
            audio("DS_LOCK", BUF, T0, [0, len(data), 0x03004108, 0x0300410C, 0x03004110, 0x03004114, 0],
                  items=[(OUT, 2, W32.pack(MEM)), (OUT, 3, W32.pack(len(data))), (OUT, 4, W32.pack(0)),
                         (OUT, 5, W32.pack(0)), (STATE, NO_ARG, ds_state(**st))]),
            audio("DS_UNLOCK", BUF, T0, [MEM, len(data), 0, 0],
                  items=[(PCM, 0, data), (STATE, NO_ARG, ds_state(**st)), store_hash(data)]),
            audio("DS_PLAY", BUF, T0, [0, 0, 1], items=[(STATE, NO_ARG, ds_state(**dict(st, playing=1, looping=1)))])]


@pytest.mark.parametrize("name,fmt,data,frame", [
    ("mono 16-bit", dict(channels=1, block_align=2, avg_bytes_per_sec=96000),
     b"".join(struct.pack("<h", (k + 1) * 2048) for k in range(8)), lambda k: ((k + 1) / 16, (k + 1) / 16)),
    ("stereo 8-bit", dict(bits_per_sample=8, block_align=2, avg_bytes_per_sec=96000),
     bytes(b for k in range(8) for b in (128 + 8 * (k + 1), 128 - 8 * (k + 1))),
     lambda k: ((k + 1) / 16, -(k + 1) / 16)),
    ("stereo float", dict(format_tag=3, bits_per_sample=32, block_align=8, avg_bytes_per_sec=384000),
     b"".join(struct.pack("<ff", (k + 1) / 64, -(k + 1) / 128) for k in range(8)),
     lambda k: ((k + 1) / 64, -(k + 1) / 128)),
])
def test_each_format_the_mixer_decodes_plays_its_frames(renderer, tmp_path, name, fmt, data, frame):
    """Mono 16-bit (CRI's first buffer), 8-bit and float at 48 kHz: one frame per sample, a
    mono frame on both channels, an 8-bit sample about 128, a float as it is."""
    rc, js, got = render(renderer(), tmp_path, start() + ds_format(fmt, data) + present(T0 + 500, 2) + [END])
    assert rc == 0 and js["audio"]["problems"]["count"] == 0, js["audio"]["problems"]
    assert got == [frame(n % 8) for n in range(24)]


def test_a_voice_at_half_rate_is_interpolated_halfway_between_its_frames(renderer, tmp_path):
    """A frequency ratio of 0.5: sample n is at stream frame n / 2, so every odd sample is
    halfway between two frames, each frame's channels mixed by the level matrix."""
    m = [fbits(v) for v in LEVELS]
    half = audio("XA_SET_RATIO", SOURCE, T0, [fbits(0.5), 0],
                 items=[(STATE, NO_ARG, xa_state(matrix=m, matrix_src=6, matrix_dst=2, queued=1, queued_end=4,
                                                 running=1, ratio_bits=fbits(0.5)))])
    rc, js, got = render(renderer(), tmp_path, start() + xa_voice(frames=4) + [half] + present(T0 + 125, 2) + [END])
    xs = xa_frames(4)

    def mix(x: list[float]) -> tuple[float, float]:
        l = r = 0.0
        for c in range(6):
            l, r = f32(l + f32(LEVELS[c] * x[c])), f32(r + f32(LEVELS[6 + c] * x[c]))
        return l, r

    want = [mix(xs[n // 2] if n % 2 == 0 else [f32(a + (b - a) * 0.5) for a, b in zip(xs[n // 2], xs[n // 2 + 1])])
            for n in range(6)]
    assert rc == 0 and got == want and js["audio"]["xa"]["resampled_samples"] == 6
    assert got[1] != got[0] and got[1] != got[2]


def test_a_voice_of_four_channels_with_no_matrix_plays_its_first_two(renderer, tmp_path):
    fmt = dict(channels=4, block_align=16, avg_bytes_per_sec=768000, channel_mask=0x33)
    data = b"".join(struct.pack("<4f", (k + 1) / 32, -(k + 1) / 32, 0.5, 0.25) for k in range(3))
    after = dict(fmt, queued=1, queued_end=3)
    recs = [audio("XA_CREATE_SOURCE", ENGINE, T0, [0x03004148, 0x03004280, 0, 0x40000000, 0, 0, 0],
                  items=[(OUT, 0, W32.pack(SOURCE)), (STATE, 0, xa_state(**fmt))]),
            audio("XA_SUBMIT", SOURCE, T0, [0x03004300, 0],
                  items=[(IN, 0, struct.pack("<9I", 0, len(data), 0x03004400, 0, 0, 0, 0, 0, 0)), (PCM, 0, data),
                         (STATE, NO_ARG, xa_state(**after))]),
            audio("XA_START", SOURCE, T0, [0, 0], items=[(STATE, NO_ARG, xa_state(**after, running=1))])]
    rc, js, got = render(renderer(), tmp_path, start() + recs + present(T0 + 62, 2) + [END])   # 3 samples
    assert rc == 0 and got == [((k + 1) / 32, -(k + 1) / 32) for k in range(3)]
    assert js["audio"]["xa"]["default_matrix_samples"] == 3


def test_the_wav_is_stereo_float_at_48k_with_its_sizes(renderer, tmp_path):
    render(renderer(), tmp_path, start() + ds_buffer() + present(T0 + 500, 2) + [END])
    fmt, frames = read_wav(tmp_path / "s.wav")
    assert fmt == {"riff": 50 + 24 * 8, "tag": 3, "channels": 2, "rate": 48000, "bytes_per_sec": 384000,
                   "block": 8, "bits": 32, "cb": 0, "fact": 24, "data": 192, "file": 58 + 192}
    assert len(frames) == 24


# ---------------------------------------------------------------- the player's reader

# host/player/stream.cpp's two ways in, on one stream file: a C++ program this test
# writes out and builds (the test's own fixture, not a tool). The player and the renderer
# open a file with Sa2StreamReader::open; the live player opens a descriptor with
# open_live, which holds the version, the schema and the audio table again on its own.
OPEN_DRIVER_CPP = r"""// test_s5_audio.py's open driver. Prints one line per way in:
//   file <ok|io|malformed>\t<error>   Sa2StreamReader::open
//   live <status>\t<detail>           Sa2StreamReader::open_live, the file's bytes written
//                                     to a socket pair whose writing end is then closed
#include "stream.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    {
        Sa2StreamReader rd;
        std::string err;
        const int rc = rd.open(argv[1], &err);
        std::printf("file %s\t%s\n", rc == Sa2StreamReader::OPEN_OK ? "ok" : rc == Sa2StreamReader::OPEN_IO ? "io"
                                     : "malformed", err.c_str());
    }
    std::vector<char> bytes;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    std::fclose(f);
    int video[2], cancel[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, video) != 0 || pipe(cancel) != 0) return 2;
    if (write(video[1], bytes.data(), bytes.size()) != ssize_t(bytes.size())) return 2;
    close(video[1]);
    static const char* const names[] = {"Ready", "End", "Cancelled", "Malformed", "IoFailure"};
    Sa2StreamReader live;
    const Sa2ReadOutcome o = live.open_live(video[0], cancel[0], nullptr, nullptr);
    std::printf("live %s\t%s\n", names[int(o.status)], o.detail.c_str());
    close(cancel[1]);
    return 0;
}
"""


@pytest.fixture(scope="module")
def opener(tmp_path_factory):
    """A function (stream path) -> {"file": (outcome, error), "live": (status, detail)}:
    OPEN_DRIVER_CPP built with host/player/stream.cpp by the system g++, which the
    renderer needs too (a skip would pass the gate with the reader unheld)."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.fail("g++ is not on PATH: the player's reader cannot be built")
    d = tmp_path_factory.mktemp("opener")
    src, exe = d / "open_driver.cpp", d / "open_driver"
    src.write_text(OPEN_DRIVER_CPP)
    done = subprocess.run([gxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                           f"-I{ROOT / 'host'}", f"-I{PLAYER}", "-o", str(exe), str(src), str(PLAYER / "stream.cpp"),
                           str(PLAYER / "sha256.cpp"), "-lzstd"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    def get(path: Path) -> dict:
        out = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        lines = [line.split(" ", 1) for line in out.stdout.splitlines()]
        return {way: tuple(rest.split("\t", 1)) for way, rest in lines}
    return get


def header_with(change) -> dict:
    head = header()
    change(head["audio"])
    return head


STATE_SIZES = ("header JSON \"audio\" does not give ds_state_bytes 84 and xa_state_bytes 176, the sizes of "
               "host/sa2_stream.h's STATE structs")
# (name, version, header, the error open gives after the path, open_live's detail when not the same)
PLAYER_REFUSALS = [
    ("a ds STATE size that is not the struct's", 2, lambda: header_with(lambda a: a.update(ds_state_bytes=80)),
     STATE_SIZES, None),
    ("no xa STATE size", 2, lambda: header_with(lambda a: a.pop("xa_state_bytes")), STATE_SIZES, None),
    ("an audio table with no ops", 2, lambda: header_with(lambda a: a.pop("ops")),
     "header JSON \"audio\" has no \"ops\" table", None),
    ("an audio row of five", 2, lambda: header_with(lambda a: a["ops"][0].pop()),
     "header JSON \"audio\" row 0 is not [op, NAME, interface, method, argc, aops]", None),
    ("an op named twice", 2, lambda: header_with(lambda a: a["ops"].__setitem__(1, a["ops"][0])),
     "header JSON \"audio\" names op 1 twice", None),
    ("version 3", 3, header, "stream version 3; this reader reads versions 1 and 2 only",
     "unsupported live stream version"),
    ("version 2 with version 1's schema", 2, lambda: header(2, schema=SD.SCHEMA),
     "stream version 2 with schema \"sa2-d3d9-stream/v1\"; that version's schema is \"sa2-d3d9-stream/v2\"",
     "unsupported live stream schema"),
    ("version 2 without an audio table", 2, lambda: header(1, schema=SD.SCHEMA_AUDIO),
     "a version 2 header without an \"audio\" table", "live stream audio table does not match its version"),
    ("version 1 with an audio table", 1, lambda: header(2, schema=SD.SCHEMA),
     "a version 1 header with an \"audio\" table", "live stream audio table does not match its version"),
]


@pytest.mark.parametrize("name,version,head,why,live_why", PLAYER_REFUSALS, ids=[r[0] for r in PLAYER_REFUSALS])
def test_the_players_reader_refuses_a_header_that_is_not_its_versions(renderer, opener, tmp_path, name, version,
                                                                     head, why, live_why):
    """host/player/stream.cpp opens a version 2 stream only with the format's STATE sizes,
    a whole audio table, and its version, schema and audio table in agreement. Each
    refusal through open, as the renderer exits on it (3, a stream refused at open, as
    the player does), and through open_live."""
    src = tmp_path / "h.sa2s"
    src.write_bytes(make([END], version=version, head=head()))
    done = subprocess.run([str(renderer()), str(src), "--wav", str(tmp_path / "h.wav")], capture_output=True,
                          text=True, timeout=60)
    assert (done.returncode, done.stderr) == (3, f"{src}: {why}\n"), name
    assert opener(src) == {"file": ("malformed", f"{src}: {why}"), "live": ("Malformed", live_why or why)}, name


def test_the_players_reader_opens_the_formats_own_headers(opener, tmp_path):
    for version in (1, 2):
        src = tmp_path / f"v{version}.sa2s"
        src.write_bytes(make([END], version=version))
        assert opener(src) == {"file": ("ok", ""), "live": ("Ready", "")}, version


# ---------------------------------------------------------------- the recorder

# The recorder's audio path on the runtime's own devices, with no game: a C program
# this test writes out and builds (the harness is the test's own fixture, not a tool).
AUDIO_HARNESS_C = r"""/* The recorder's audio path (host/sa2_present_rec.c, SA2_REC_AUDIO=1) on the runtime's
 * own DirectSound and XAudio2 devices (runtime/sa2_dsound.c, runtime/sa2_xaudio2.c) and
 * the recording device (runtime/sa2_d3d9.c), with no game: made-up calls through the
 * devices' vtables, as lifted code makes them, with Presents between them to move the
 * virtual clock. Documentation/scripts/tests/test_s5_audio.py writes this file, builds
 * it with the pinned zig with and without the recorder and the -Wl,-wrap flags a
 * presentation build links it with, runs it, and holds the guest's view and the stream.
 *
 *   usage: rec_audio_harness --out PATH [--scenario main|extended|unlocks] [--stray] [--late-argc]
 *
 * The runtime pieces the devices and the recorder call -- guest memory, the runtime
 * arena, the guest heap, the function table, the CPU state, the trap, the scheduler's
 * clock and frame hook, the window's numbers, the device profile and the import table
 * -- are stubs below over a host buffer standing for guest addresses 0x00800000 ..
 * 0x04400000, which holds the three literal addresses XAudio2's calls must name
 * (0x01AF0EDC, 0x01AF0EE0, 0x0089E4A8). Every byte handed to a device is made up here.
 *
 * The main scenario: Direct3DCreate9 and CreateDevice; DirectSound's device, its
 * cooperative level, the primary buffer (no format until SetFormat gives it CRI's 48 kHz
 * stereo, played looping), a 4800-byte 48 kHz 16-bit stereo buffer filled through a
 * Lock, retimed to 44100 Hz and played looping; both cursors read; a Lock that wraps,
 * its two segments filled; volume and pan, and a volume the device refuses; a duplicate
 * played once to its end; Stop and SetCurrentPosition; then XAudio2's engine, mastering
 * voice, a six-channel float source voice with its level matrix, two buffers submitted
 * around a Start, GetState, a frequency ratio of 1.25, GetState, Stop, a flush and the
 * teardown. Presents between them advance the clock by 16666 us each.
 *
 * The extended scenario (the paths the main one does not take): the primary again; a
 * 1920-byte buffer duplicated before its first Lock (a store of its own) and after it
 * (sharing its original's); a Lock of the whole buffer (DSBLOCK_ENTIREBUFFER); the
 * shared duplicate locked, written, retimed to 24 kHz, played once to its end, rewound
 * and played again; a Lock from the write cursor that wraps; the buffer retimed to
 * 32 kHz while it plays and back two Presents later; the other duplicate filled, set
 * and played; a refused volume; three buffers submitted, a ratio of 1.5, a flush that
 * keeps the head buffer it had started; and 64 Presents more, so the store watch checks
 * at a FRAME.
 *
 * The unlocks scenario (a live session's store_hash, SA2_REC_STORE_HASH_EVERY): the primary,
 * then one 256-byte buffer locked and unlocked 130 times, a quarter at a time (the k-th Lock,
 * from 1, takes bytes (k - 1) % 4 * 64 to 64 more and writes them with seed k), a Present after
 * every eighth Unlock, and Play.
 *
 * --stray writes two bytes into a store outside any Lock, as a guest could: in the main
 * scenario after the first Unlock (the next Lock and Unlock come after it), in the
 * extended one after the last Unlock of its own-store duplicate (only the watch's FRAME
 * check comes after it), in the unlocks one at byte 200 after the 64th Unlock (the 65th to
 * 67th lock other quarters, and the 68th rewrites it). --late-argc gives IDirectSound8::CreateSoundBuffer two
 * arguments (SA2_DSOUND_BREAK=argc) after the first Present, when the recorder has
 * started and before DirectSound's thunks are registered.
 *
 * --out writes what the guest saw: every call's EAX, how far ESP moved and a digest of
 * the stand-in register file after it; the function table's registrations; a digest of
 * all of guest memory; the devices' own counters and a digest of every buffer's and
 * voice's record; the addresses and answers the test needs to read the stream by; and,
 * as harness/crt_probe.c writes them into a run's record, the two devices' per-method
 * calls and call logs ("run").
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu_state.h"
#include "function_table.h"
#include "guest_memory.h"
#include "remill_runtime.h"
#include "sa2_d3d9.h"
#include "sa2_devprofile.h"
#include "sa2_dsound.h"
#include "sa2_imports.h"
#include "sa2_threads.h"
#include "sa2_trap.h"
#include "sa2_window.h"
#include "sa2_xaudio2.h"

/* ------------------------------------------------------------ the stubs */

#define GUEST_LO   0x00800000u
#define GUEST_HI   0x04400000u
#define ARENA      0x03300000u     /* sa2_guest_alloc: the runtime arena */
#define HEAP_LO    0x04000000u     /* the guest heap */
#define STACK_TOP  0x030eff00u
#define SCRATCH    0x03004000u
#define CALLER     0x00401000u

static uint8_t *g_mem;
int sa2_peek(uint32_t addr, uint32_t len, void *out) {
    if (addr < GUEST_LO || addr >= GUEST_HI || len > GUEST_HI - addr) return SA2_E_RANGE;
    memcpy(out, g_mem + (addr - GUEST_LO), len);
    return SA2_OK;
}
int sa2_poke(uint32_t addr, uint32_t len, const void *in) {
    if (addr < GUEST_LO || addr >= GUEST_HI || len > GUEST_HI - addr) return SA2_E_RANGE;
    memcpy(g_mem + (addr - GUEST_LO), in, len);
    return SA2_OK;
}
int sa2_guest_fill(uint32_t addr, int byte, uint32_t len) {
    if (addr < GUEST_LO || addr >= GUEST_HI || len > GUEST_HI - addr) return SA2_E_RANGE;
    memset(g_mem + (addr - GUEST_LO), byte, len);
    return SA2_OK;
}
int sa2_guest_valid(uint32_t addr, uint32_t len) {
    return addr >= GUEST_LO && addr < GUEST_HI && len <= GUEST_HI - addr;
}
static uint32_t g_arena = ARENA;
uint32_t sa2_guest_alloc(uint32_t len, uint32_t align) {
    if (!align) align = 1u;
    const uint32_t at = (g_arena + align - 1u) & ~(align - 1u);
    if (len > HEAP_LO - at) return 0;
    g_arena = at + len;
    return at;
}
static uint32_t g_heap = HEAP_LO;
uint32_t sa2_heap_alloc_owned(uint32_t owner, uint32_t bytes) {
    (void)owner;
    const uint32_t at = (g_heap + 15u) & ~15u;
    if (bytes > GUEST_HI - at) return 0;
    g_heap = at + bytes;
    return at;
}
void sa2_heap_free_owned(uint32_t owner, uint32_t payload) { (void)owner; (void)payload; }

static uint64_t fnv1a_more(uint64_t h, const void *p, size_t n) {
    for (size_t i = 0; i < n; i++) { h ^= ((const uint8_t *)p)[i]; h *= 0x100000001b3ull; }
    return h;
}
#define FNV_BASIS 0xcbf29ce484222325ull

/* The function table: the three devices' thunk regions, and a digest of what each
 * registration hands the table -- the address, the name and the kind. */
#define D3_N (SA2_D3D9_MAX_TYPES * SA2_D3D9_MAX_SLOTS)
#define DS_N (SA2_DS_MAX_TYPES * SA2_DS_MAX_SLOTS)
#define XA_N (SA2_XA2_MAX_TYPES * SA2_XA2_MAX_SLOTS)
static sa2_lifted_fn g_d3_fn[D3_N], g_ds_fn[DS_N], g_xa_fn[XA_N];
static uint32_t g_ft_regs;
static uint64_t g_ft_fnv = FNV_BASIS;
static sa2_lifted_fn *slot_of(uint32_t va) {
    if (va >= SA2_D3D9_THUNK_BASE && va < SA2_D3D9_THUNK_END)
        return &g_d3_fn[(va - SA2_D3D9_THUNK_BASE) / SA2_D3D9_THUNK_STRIDE];
    if (va >= SA2_DS_THUNK_BASE && va < SA2_DS_THUNK_END)
        return &g_ds_fn[(va - SA2_DS_THUNK_BASE) / SA2_DS_THUNK_STRIDE];
    if (va >= SA2_XA2_THUNK_BASE && va < SA2_XA2_THUNK_END)
        return &g_xa_fn[(va - SA2_XA2_THUNK_BASE) / SA2_XA2_THUNK_STRIDE];
    return NULL;
}
int sa2_ft_add(uint32_t va, sa2_lifted_fn fn, const char *name, uint8_t kind) {
    sa2_lifted_fn *s = slot_of(va);
    if (!s) return -1;
    *s = fn;
    g_ft_regs++;
    g_ft_fnv = fnv1a_more(g_ft_fnv, &va, sizeof va);
    g_ft_fnv = fnv1a_more(g_ft_fnv, &kind, sizeof kind);
    g_ft_fnv = fnv1a_more(g_ft_fnv, name ? name : "", name ? strlen(name) + 1u : 1u);
    return 0;
}

sa2_lifted_fn sa2_ft_lookup(uint32_t va) {
    if (va >= SA2_D3D9_THUNK_BASE && va < SA2_D3D9_THUNK_END) {
        if ((va - SA2_D3D9_THUNK_BASE) % SA2_D3D9_THUNK_STRIDE) return NULL;
    } else if (va >= SA2_DS_THUNK_BASE && va < SA2_DS_THUNK_END) {
        if ((va - SA2_DS_THUNK_BASE) % SA2_DS_THUNK_STRIDE) return NULL;
    } else if (va >= SA2_XA2_THUNK_BASE && va < SA2_XA2_THUNK_END) {
        if ((va - SA2_XA2_THUNK_BASE) % SA2_XA2_THUNK_STRIDE) return NULL;
    } else return NULL;
    const sa2_lifted_fn *s = slot_of(va);
    return s ? *s : NULL;
}

typedef struct { uint32_t gpr[SA2_GPR_COUNT]; uint16_t cw; } probe_state;
uint32_t sa2_rt_get_gpr(const void *state, int idx) { return ((const probe_state *)state)->gpr[idx]; }
void sa2_rt_set_gpr(void *state, int idx, uint32_t value) { ((probe_state *)state)->gpr[idx] = value; }
uint16_t sa2_rt_get_cw(const void *state) { return ((const probe_state *)state)->cw; }
void sa2_rt_set_cw(void *state, uint16_t cw) { ((probe_state *)state)->cw = cw; }
void sa2_rt_set_host_round(int mode) { (void)mode; }

uint64_t sa2_sched_budget_us(void) { return 0; }
int sa2_sched_budget_hit(void) { return 0; }
static uint64_t g_frames, g_virtual_us = 5000000000ull;   /* past 2^32, for the u64 */
uint64_t sa2_sched_frames(void) { return g_frames; }
uint64_t sa2_virtual_us(void) { return g_virtual_us; }
int sa2_sched_current(void) { return 0; }
void sa2_sched_frame_boundary(void) {
    g_frames++;
    g_virtual_us += 16666u;
}
void sa2_window_stats(sa2_wnd_stats *out) {
    memset(out, 0, sizeof *out);
    out->polls = 7u;
}
uint32_t sa2_window_screen_w(void) { return 1280u; }
uint32_t sa2_window_screen_h(void) { return 720u; }

static char g_note[1024];
void sa2_trap_note(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_note, sizeof g_note, fmt, ap);
    va_end(ap);
}
void sa2_trap_raise(uint32_t status, uint32_t exit_pc) {
    fprintf(stderr, "rec_audio_harness: an unexpected trap %u at %#010x: %s\n", status, exit_pc,
            g_note);
    exit(2);
}

/* The default device profile: a stereo master at the game's 48 kHz. */
int sa2_devprofile_is_default(void) { return 1; }
const char *sa2_devprofile_source(void) { return "default"; }
uint32_t sa2_dp_xaudio2_master_flags(void) { return 0u; }
uint32_t sa2_dp_xaudio2_master_channels(void) { return 2u; }
uint32_t sa2_dp_xaudio2_master_rate(void) { return 48000u; }
uint32_t sa2_import_refused(const char *dll, const char *name, uint32_t code) {
    (void)dll; (void)name;
    return code;
}
const sa2_import_info *sa2_import_by_name(const char *dll, const char *name) {
    (void)dll; (void)name;
    return NULL;
}

/* ------------------------------------------------------------ calls */

#define MAX_CALLS 160
static uint64_t g_answers[MAX_CALLS][3];
static int g_ncalls;
static int g_stray, g_late_argc;

static void pk32(uint32_t a, uint32_t v) { sa2_poke(a, 4, &v); }
static uint32_t peek32(uint32_t a) {
    uint32_t v = 0;
    sa2_peek(a, 4, &v);
    return v;
}

static uint32_t call_on(uint32_t self, uint32_t slot, uint32_t argc, const uint32_t *argv) {
    const uint32_t thunk = peek32(peek32(self) + 4u * slot);
    probe_state st;
    memset(&st, 0, sizeof st);
    const uint32_t esp0 = STACK_TOP - 4u * (argc + 2u);
    st.gpr[SA2_ESP] = esp0;
    pk32(esp0, CALLER);
    pk32(esp0 + 4u, self);
    for (uint32_t i = 0; i < argc; i++) pk32(esp0 + 8u + 4u * i, argv[i]);
    sa2_lifted_fn *s = slot_of(thunk);
    if (!s || !*s) {
        fprintf(stderr, "rec_audio_harness: slot %u of %#010x names %#010x, which no thunk "
                        "answers\n", slot, self, thunk);
        exit(2);
    }
    (*s)(&st, thunk, NULL);
    if (g_ncalls < MAX_CALLS) {
        g_answers[g_ncalls][0] = st.gpr[SA2_EAX];
        g_answers[g_ncalls][1] = st.gpr[SA2_ESP] - esp0;
        g_answers[g_ncalls][2] = fnv1a_more(FNV_BASIS, &st, sizeof st);
        g_ncalls++;
    }
    return st.gpr[SA2_EAX];
}
#define CALL(self, slot, ...) \
    call_on(self, slot, (uint32_t)(sizeof((uint32_t[]){__VA_ARGS__}) / sizeof(uint32_t)), \
            (const uint32_t[]){__VA_ARGS__})
#define CALL0(self, slot) call_on(self, slot, 0, NULL)

static void put_words(uint32_t at, const uint32_t *w, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) pk32(at + 4u * i, w[i]);
}
static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/* The DirectSound pattern: 16-bit stereo, frame f of the buffer (byte 4f) holds
 * L = 11 f + seed and R = -7 f - seed, as int16. */
static void put_ds(uint32_t at, uint32_t buf_off, uint32_t n, int seed) {
    for (uint32_t i = 0; i < n; i += 2u) {
        const uint32_t b = buf_off + i, f = b / 4u;
        const int16_t v = (b % 4u) == 0u ? (int16_t)(11 * (int)f + seed) : (int16_t)(-7 * (int)f - seed);
        sa2_poke(at + i, 2, &v);
    }
}
/* The XAudio2 pattern: six float channels, frame i channel c = ((i + seed) % 64 - 32) / 64
 * + c / 128, every value exact in float32. */
static void put_xa(uint32_t at, uint32_t frames, uint32_t seed) {
    for (uint32_t i = 0; i < frames; i++)
        for (uint32_t c = 0; c < 6u; c++)
            pk32(at + 24u * i + 4u * c,
                 fbits((float)((int)((i + seed) % 64u) - 32) / 64.0f + (float)c / 128.0f));
}
/* Two bytes a guest writes into a store outside any Lock. */
static void stray(uint32_t at) {
    const uint16_t v = 0x7777u;
    sa2_poke(at, 2, &v);
}

/* Addresses of the made-up inputs, fixed so the test can name them. */
#define PP       (SCRATCH + 0x000u)
#define OUTS     (SCRATCH + 0x100u)          /* one dword per out-parameter */
#define DESC     (SCRATCH + 0x200u)
#define WFX      (SCRATCH + 0x240u)
#define FMT      (SCRATCH + 0x280u)
#define XB1      (SCRATCH + 0x300u)
#define XB2      (SCRATCH + 0x340u)
#define VSTATE   (SCRATCH + 0x380u)
#define GUIDS    (SCRATCH + 0x3c0u)
#define PDESC    (SCRATCH + 0x400u)          /* the primary buffer's DSBUFFERDESC */
#define XB3      (SCRATCH + 0x440u)
#define DATA1    (SCRATCH + 0x1000u)         /* 512 frames of 24 bytes */
#define DATA2    (SCRATCH + 0x4000u)         /* 1024 frames */
#define DATA3    (SCRATCH + 0x10000u)        /* 1800 frames */
#define PP_MASTER  0x01AF0EDCu
#define DETAILS    0x01AF0EE0u
#define MATRIX     0x0089E4A8u

static uint32_t g_dev;
static void present(void) { CALL(g_dev, 17u, 0u, 0u, 0u, 0u); }

static FILE *g_out;

static void start_d3d9(void) {
    const uint32_t pp[14] = {1280u, 720u, 21u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 75u, 0u, 0u, 0u};
    put_words(PP, pp, 14);
    const uint32_t d3d9 = sa2_d3d9_make_d3d9(CALLER);
    CALL(d3d9, 16u, 0u, 1u, 0x7e000000u, 0x40u, PP, OUTS + 0u);   /* CreateDevice */
    g_dev = peek32(OUTS + 0u);
    present();
}

/* The primary buffer: made with no format, given CRI's 48 kHz stereo 16-bit by
 * SetFormat, played looping. */
static uint32_t make_primary(uint32_t ds) {
    const uint32_t wfx[5] = {0x00020001u, 48000u, 192000u, 0x00100004u, 0u};
    put_words(WFX, wfx, 5);
    const uint32_t pdesc[9] = {36u, 0x00000001u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};   /* DSBCAPS_PRIMARYBUFFER */
    put_words(PDESC, pdesc, 9);
    CALL(ds, 3u, PDESC, OUTS + 76u, 0u);                          /* CreateSoundBuffer */
    const uint32_t prim = peek32(OUTS + 76u);
    CALL(prim, 14u, WFX);                                         /* SetFormat */
    CALL(prim, 12u, 0u, 0u, 1u);                                  /* Play, DSBPLAY_LOOPING */
    return prim;
}

/* XAudio2's engine, its mastering voice and its details, and a six-channel float source
 * voice with the level matrix. */
static uint32_t g_eng, g_master;
static uint32_t make_voice(void) {
    const uint8_t clsid[16] = {0x49,0x9B,0xDA,0x3E,0x85,0x20,0x8B,0x49,0x9B,0xB2,0x39,0xA6,0x77,0x84,0x93,0xDE};
    const uint8_t iid[16] = {0x58,0x1F,0xCF,0x8B,0xE7,0x9F,0x83,0x45,0x8A,0xC6,0xE2,0xAD,0xC4,0x65,0xC8,0xBB};
    sa2_poke(GUIDS, 16, clsid);
    sa2_poke(GUIDS + 16u, 16, iid);
    probe_state st;
    memset(&st, 0, sizeof st);
    const uint32_t esp = STACK_TOP - 0x40u;
    st.gpr[SA2_ESP] = esp;
    put_words(esp, (const uint32_t[]){CALLER, GUIDS, 0u, 1u, GUIDS + 16u, OUTS + 68u}, 6);
    uint32_t handled = 0;
    const uint32_t hr = sa2_xaudio2_cocreate(&st, clsid, iid, OUTS + 68u, &handled);
    if (hr || !handled) {
        fprintf(stderr, "rec_audio_harness: CoCreateInstance answered %#x\n", hr);
        exit(2);
    }
    g_eng = peek32(OUTS + 68u);
    CALL(g_eng, 5u, 0u, 0xFFFFFFFFu);                             /* Initialize */
    CALL(g_eng, 10u, PP_MASTER, 0u, 48000u, 0u, 0u, 0u);          /* CreateMasteringVoice */
    g_master = peek32(PP_MASTER);
    CALL(g_master, 0u, DETAILS);                                  /* GetVoiceDetails */
    const uint32_t fmt[6] = {0x0006FFFEu, 48000u, 1152000u, 0x00200018u, 0x00200016u, 0x3Fu};
    put_words(FMT, fmt, 6);
    const uint8_t sub[16] = {0x03,0x00,0x00,0x00,0x00,0x00,0x10,0x00,0x80,0x00,0x00,0xAA,0x00,0x38,0x9B,0x71};
    sa2_poke(FMT + 24u, 16, sub);
    CALL(g_eng, 8u, OUTS + 72u, FMT, 0u, 0x40000000u, 0u, 0u, 0u);  /* CreateSourceVoice */
    const uint32_t src = peek32(OUTS + 72u);
    const float levels[12] = {1.0f, 0.0f, 0.5f, 0.25f, 0.75f, 0.0f,
                              0.0f, 1.0f, 0.5f, 0.25f, 0.0f, 0.75f};
    for (uint32_t k = 0; k < 12u; k++) pk32(MATRIX + 4u * k, fbits(levels[k]));
    CALL(src, 16u, g_master, 6u, 2u, MATRIX, 0u);                 /* SetOutputMatrix */
    return src;
}

static void scenario_main(void) {
    start_d3d9();

    /* DirectSound */
    if (g_late_argc) {
        setenv("SA2_DSOUND_BREAK", "argc", 1);
        sa2_dsound_config_from_env();
    }
    const uint32_t ds = sa2_dsound_make_device(CALLER);
    CALL(ds, 6u, 0x00001234u, 2u);                                /* SetCooperativeLevel */
    const uint32_t prim = make_primary(ds);
    const uint32_t desc[9] = {36u, 0x000000E0u, 4800u, 0u, WFX, 0u, 0u, 0u, 0u};
    put_words(DESC, desc, 9);
    CALL(ds, 3u, DESC, OUTS + 4u, 0u);                            /* CreateSoundBuffer */
    const uint32_t buf = peek32(OUTS + 4u);
    CALL(buf, 11u, 0u, 4800u, OUTS + 8u, OUTS + 12u, OUTS + 16u, OUTS + 20u, 0u);   /* Lock */
    const uint32_t p1 = peek32(OUTS + 8u), n1 = peek32(OUTS + 12u);
    put_ds(p1, 0u, n1, 3);
    CALL(buf, 19u, p1, n1, 0u, 0u);                               /* Unlock */
    if (g_stray) stray(p1 + 2000u);
    CALL(buf, 17u, 44100u);                                       /* SetFrequency */
    CALL(buf, 12u, 0u, 0u, 1u);                                   /* Play, DSBPLAY_LOOPING */
    present();
    present();
    CALL(buf, 4u, OUTS + 24u, OUTS + 28u);                        /* GetCurrentPosition */
    CALL(prim, 4u, OUTS + 80u, OUTS + 84u);                       /* the primary's */
    CALL(buf, 11u, 4000u, 1600u, OUTS + 32u, OUTS + 36u, OUTS + 40u, OUTS + 44u, 0u);
    const uint32_t q1 = peek32(OUTS + 32u), m1 = peek32(OUTS + 36u);
    const uint32_t q2 = peek32(OUTS + 40u), m2 = peek32(OUTS + 44u);
    put_ds(q1, 4000u, m1, 5);
    put_ds(q2, 0u, m2, 5);
    CALL(buf, 19u, q1, m1, q2, m2);                               /* Unlock, two segments */
    CALL(buf, 15u, (uint32_t)-600);                               /* SetVolume */
    CALL(buf, 15u, (uint32_t)-10001);                             /* SetVolume, refused */
    CALL(buf, 16u, (uint32_t)-2000);                              /* SetPan */
    present();
    CALL(buf, 9u, OUTS + 48u);                                    /* GetStatus */
    CALL(ds, 5u, buf, OUTS + 52u);                                /* DuplicateSoundBuffer */
    const uint32_t dup = peek32(OUTS + 52u);
    CALL(dup, 12u, 0u, 0u, 0u);                                   /* Play, once */
    present();
    present();
    CALL(dup, 9u, OUTS + 56u);                                    /* GetStatus: it ran out */
    CALL(dup, 4u, OUTS + 60u, OUTS + 64u);                        /* GetCurrentPosition */
    CALL0(buf, 18u);                                              /* Stop */
    CALL(buf, 13u, 400u);                                         /* SetCurrentPosition */
    CALL0(dup, 2u);                                               /* Release */

    /* XAudio2 */
    const uint32_t src = make_voice();
    put_xa(DATA1, 512u, 0u);
    put_words(XB1, (const uint32_t[]){0u, 512u * 24u, DATA1, 0u, 0u, 0u, 0u, 0u, 0u}, 9);
    CALL(src, 21u, XB1, 0u);                                      /* SubmitSourceBuffer */
    CALL(src, 19u, 0u, 0u);                                       /* Start */
    present();
    CALL(src, 25u, VSTATE);                                       /* GetState: it ran dry */
    put_xa(DATA2, 1024u, 7u);
    put_words(XB2, (const uint32_t[]){0u, 1024u * 24u, DATA2, 0u, 0u, 0u, 0u, 0u, 0u}, 9);
    CALL(src, 21u, XB2, 0u);
    CALL(src, 26u, fbits(1.25f), 0u);                             /* SetFrequencyRatio */
    present();
    CALL(src, 25u, VSTATE);
    CALL(src, 20u, 0u, 0u);                                       /* Stop */
    CALL0(src, 22u);                                              /* FlushSourceBuffers */
    CALL0(src, 18u);                                              /* DestroyVoice */
    CALL0(g_master, 18u);
    CALL0(g_eng, 2u);                                             /* Release */
    present();

    fprintf(g_out, " \"objects\": {\"device\": %u, \"dsound\": %u, \"primary\": %u, \"buffer\": %u, "
                   "\"duplicate\": %u, \"engine\": %u, \"master\": %u, \"source\": %u},\n",
            g_dev, ds, prim, buf, dup, g_eng, g_master, src);
    fprintf(g_out, " \"locks\": [[%u, %u], [%u, %u, %u, %u]],\n", p1, n1, q1, m1, q2, m2);
    fprintf(g_out, " \"data\": [%u, %u],\n", DATA1, DATA2);
}

static void scenario_unlocks(void) {
    start_d3d9();
    const uint32_t ds = sa2_dsound_make_device(CALLER);
    CALL(ds, 6u, 0x00001234u, 2u);                                /* SetCooperativeLevel */
    const uint32_t prim = make_primary(ds);
    const uint32_t desc[9] = {36u, 0x000000E0u, 256u, 0u, WFX, 0u, 0u, 0u, 0u};
    put_words(DESC, desc, 9);
    CALL(ds, 3u, DESC, OUTS + 4u, 0u);                            /* CreateSoundBuffer */
    const uint32_t buf = peek32(OUTS + 4u);
    uint32_t mem = 0;
    for (uint32_t k = 1; k <= 130u; k++) {
        const uint32_t off = (k - 1u) % 4u * 64u;
        CALL(buf, 11u, off, 64u, OUTS + 8u, OUTS + 12u, OUTS + 16u, OUTS + 20u, 0u);   /* Lock */
        const uint32_t p = peek32(OUTS + 8u), n = peek32(OUTS + 12u);
        mem = p - off;
        put_ds(p, off, n, (int)k);
        CALL(buf, 19u, p, n, 0u, 0u);                             /* Unlock */
        if (g_stray && k == 64u) stray(mem + 200u);
        if (k % 8u == 0u) present();
    }
    CALL(buf, 12u, 0u, 0u, 1u);                                   /* Play, DSBPLAY_LOOPING */
    present();
    fprintf(g_out, " \"objects\": {\"device\": %u, \"dsound\": %u, \"primary\": %u, \"buffer\": %u, "
                   "\"store\": %u},\n", g_dev, ds, prim, buf, mem);
}

static void scenario_extended(void) {
    start_d3d9();
    const uint32_t ds = sa2_dsound_make_device(CALLER);
    CALL(ds, 6u, 0x00001234u, 2u);                                /* SetCooperativeLevel */
    const uint32_t prim = make_primary(ds);
    const uint32_t desc[9] = {36u, 0x000000E0u, 1920u, 0u, WFX, 0u, 0u, 0u, 0u};
    put_words(DESC, desc, 9);
    CALL(ds, 3u, DESC, OUTS + 4u, 0u);                            /* CreateSoundBuffer */
    const uint32_t a = peek32(OUTS + 4u);
    CALL(ds, 5u, a, OUTS + 52u);                                  /* D1: before A's first Lock */
    const uint32_t d1 = peek32(OUTS + 52u);
    CALL(a, 11u, 0u, 0u, OUTS + 8u, OUTS + 12u, OUTS + 16u, OUTS + 20u, 2u);   /* Lock, ENTIREBUFFER */
    const uint32_t pa = peek32(OUTS + 8u), na = peek32(OUTS + 12u);
    put_ds(pa, 0u, na, 9);
    CALL(a, 19u, pa, na, 0u, 0u);                                 /* Unlock */
    CALL(ds, 5u, a, OUTS + 56u);                                  /* D2: after it, sharing A's store */
    const uint32_t d2 = peek32(OUTS + 56u);
    CALL(d2, 11u, 960u, 480u, OUTS + 24u, OUTS + 28u, OUTS + 32u, OUTS + 36u, 0u);
    const uint32_t pd = peek32(OUTS + 24u), nd = peek32(OUTS + 28u);
    put_ds(pd, 960u, nd, 4);
    CALL(d2, 19u, pd, nd, 0u, 0u);                                /* Unlock: the shared store */
    CALL(d2, 17u, 24000u);                                        /* SetFrequency on the duplicate */
    CALL(a, 12u, 0u, 0u, 1u);                                     /* Play A, looping */
    CALL(d2, 12u, 0u, 0u, 0u);                                    /* Play D2, once */
    present();
    CALL(a, 11u, 0u, 1000u, OUTS + 40u, OUTS + 44u, OUTS + 60u, OUTS + 64u, 1u);   /* FROMWRITECURSOR */
    const uint32_t q1 = peek32(OUTS + 40u), m1 = peek32(OUTS + 44u);
    const uint32_t q2 = peek32(OUTS + 60u), m2 = peek32(OUTS + 64u);
    put_ds(q1, q1 - pa, m1, 6);
    put_ds(q2, 0u, m2, 6);
    CALL(a, 19u, q1, m1, q2, m2);                                 /* Unlock, two segments */
    CALL(a, 17u, 32000u);                                         /* SetFrequency while it plays */
    present();
    present();
    CALL(a, 4u, OUTS + 80u, OUTS + 84u);                          /* GetCurrentPosition A */
    CALL(a, 17u, 0u);                                             /* back to its own: DSBFREQUENCY_ORIGINAL */
    CALL(d2, 9u, OUTS + 88u);                                     /* GetStatus D2: it ran out */
    CALL(d2, 13u, 0u);                                            /* SetCurrentPosition 0 */
    CALL(d2, 12u, 0u, 0u, 0u);                                    /* the halted buffer, again */
    CALL(d1, 11u, 0u, 0u, OUTS + 92u, OUTS + 96u, OUTS + 100u, OUTS + 104u, 2u);   /* D1: its own store */
    const uint32_t p1 = peek32(OUTS + 92u), n1 = peek32(OUTS + 96u);
    put_ds(p1, 0u, n1, 2);
    CALL(d1, 19u, p1, n1, 0u, 0u);
    CALL(d1, 15u, (uint32_t)-1200);                               /* SetVolume */
    CALL(d1, 16u, 800u);                                          /* SetPan */
    CALL(d1, 12u, 0u, 0u, 1u);                                    /* Play D1, looping */
    CALL(a, 15u, (uint32_t)-10001);                               /* SetVolume, refused */
    if (g_stray) stray(p1 + 100u);
    present();
    CALL(d1, 4u, OUTS + 108u, OUTS + 112u);                       /* GetCurrentPosition D1 */
    CALL(a, 9u, OUTS + 116u);                                     /* GetStatus A */
    CALL(prim, 4u, OUTS + 120u, OUTS + 124u);                     /* the primary's */

    /* XAudio2: three buffers, a ratio, a flush that keeps the head it had started */
    const uint32_t src = make_voice();
    const uint32_t xbs[3] = {XB1, XB2, XB3};
    for (uint32_t k = 0; k < 3u; k++) {
        put_xa(DATA3 + k * 600u * 24u, 600u, 1u + k);
        put_words(xbs[k], (const uint32_t[]){0u, 600u * 24u, DATA3 + k * 600u * 24u, 0u, 0u, 0u, 0u, 0u, 0u}, 9);
        CALL(src, 21u, xbs[k], 0u);                               /* SubmitSourceBuffer */
    }
    CALL(src, 19u, 0u, 0u);                                       /* Start */
    present();
    CALL(src, 26u, fbits(1.5f), 0u);                              /* SetFrequencyRatio */
    CALL(src, 25u, VSTATE);                                       /* GetState */
    CALL0(src, 22u);                                              /* FlushSourceBuffers: the head stays */
    CALL(src, 25u, VSTATE);
    present();
    CALL(src, 25u, VSTATE);                                       /* it ran dry */
    CALL(src, 20u, 0u, 0u);                                       /* Stop */
    CALL0(src, 18u);                                              /* DestroyVoice */
    CALL0(g_master, 18u);
    CALL0(g_eng, 2u);                                             /* Release */
    for (int k = 0; k < 64; k++) present();                       /* the store watch's FRAME check */

    fprintf(g_out, " \"objects\": {\"device\": %u, \"dsound\": %u, \"primary\": %u, \"buffer\": %u, "
                   "\"own\": %u, \"shared\": %u, \"engine\": %u, \"master\": %u, \"source\": %u},\n",
            g_dev, ds, prim, a, d1, d2, g_eng, g_master, src);
    fprintf(g_out, " \"locks\": [[%u, %u], [%u, %u], [%u, %u, %u, %u], [%u, %u]],\n", pa, na, pd, nd, q1, m1,
            q2, m2, p1, n1);
}

/* A digest of every buffer's and every voice's record, field by field. */
static uint64_t mix64(uint64_t h, uint64_t v) { return fnv1a_more(h, &v, sizeof v); }
static uint64_t buffers_fnv(void) {
    uint64_t h = FNV_BASIS;
    for (uint32_t i = 0; i < sa2_dsound_buffer_count(); i++) {
        const sa2_ds_buffer *b = sa2_dsound_buffer_at(i);
        const uint64_t f[] = {b->addr, b->create_caller, b->flags, b->bytes, b->primary, b->format_tag,
                              b->channels, b->samples_per_sec, b->avg_bytes_per_sec, b->block_align,
                              b->bits_per_sample, b->format_from_guest, b->write_lead, b->playing,
                              b->looping, b->play_calls, b->stop_calls, b->lock_calls, b->unlock_calls,
                              b->setpos_calls, b->getpos_calls, b->mem, b->mem_len, b->anchor_us,
                              b->anchor_pos, b->last_play, b->last_write, b->last_read_us, b->bytes_locked,
                              b->halted, b->released};
        for (size_t k = 0; k < sizeof f / sizeof f[0]; k++) h = mix64(h, f[k]);
    }
    return h;
}
static uint64_t voices_fnv(void) {
    uint64_t h = FNV_BASIS;
    for (uint32_t i = 0; i < sa2_xaudio2_voice_count(); i++) {
        const sa2_xa2_voice *v = sa2_xaudio2_voice_at(i);
        const uint64_t f[] = {v->addr, v->engine, v->create_caller, v->kind, v->flags, v->destroyed,
                              v->destroy_caller, v->details_flags, v->input_channels, v->input_sample_rate,
                              v->details_calls, v->format_tag, v->channels, v->samples_per_sec,
                              v->avg_bytes_per_sec, v->block_align, v->bits_per_sample, v->cb_size,
                              v->valid_bits, v->channel_mask, v->max_ratio_bits, v->ratio_bits, v->output,
                              v->matrix_src, v->matrix_dst, v->matrix_calls, v->running, v->anchor_us,
                              v->anchor_frame, v->played, v->queued_end, v->queued, v->max_queued, v->submits,
                              v->completed, v->flushed, v->dropped_at_destroy, v->submitted_bytes,
                              v->start_calls, v->stop_calls, v->flush_calls, v->getstate_calls, v->ratio_calls,
                              v->starvations, v->last_queued, v->last_samples, v->last_state_us};
        for (size_t k = 0; k < sizeof f / sizeof f[0]; k++) h = mix64(h, f[k]);
        for (uint32_t k = 0; k < 12u; k++) h = mix64(h, v->matrix[k]);
    }
    return h;
}

/* The two devices' per-method calls and call logs, as harness/crt_probe.c writes them
 * into a run's record ("dsound" and "xaudio2"), with the answer class as "refused" or
 * "answered". */
static void put_args(const uint32_t *a, uint32_t n) {
    for (uint32_t k = 0; k < n; k++) fprintf(g_out, "%s\"0x%08x\"", k ? ", " : "", a[k]);
}
static void run_record(void) {
    sa2_ds_stats ds;
    sa2_dsound_stats_get(&ds);
    sa2_xa2_stats xa;
    sa2_xaudio2_stats_get(&xa);
    fprintf(g_out, " \"run\": {\"dsound\": {\"method_calls\": %u, \"calls\": [", ds.method_calls);
    for (uint32_t i = 0; i < sa2_dsound_log_count(); i++) {
        const sa2_ds_call *c = sa2_dsound_log_at(i);
        const char *nm = sa2_dsound_method_name(c->type, c->slot);
        fprintf(g_out, "%s\n  {\"seq\": %u, \"virtual_us\": %llu, \"interface\": \"%s\", \"method\": \"%s\", "
                       "\"slot\": %u, \"this\": \"0x%08x\", \"args\": [", i ? "," : "", c->seq,
                (unsigned long long)c->virtual_us, sa2_dsound_type_name(c->type), nm ? nm : "<unnamed slot>",
                c->slot, c->self);
        put_args(c->arg, c->argc);
        fprintf(g_out, "], \"ret\": \"0x%08x\", \"out\": [\"0x%08x\", \"0x%08x\"], \"answered\": \"%s\"}",
                c->ret, c->out0, c->out1, c->answered ? "answered" : "refused");
    }
    fprintf(g_out, "], \"methods\": [");
    int n = 0;
    for (uint32_t t = 0; t < sa2_dsound_type_count(); t++)
        for (uint32_t k = 0; k < SA2_DS_MAX_SLOTS; k++) {
            const char *nm = sa2_dsound_method_name(t, k);
            if (!nm) continue;
            const sa2_ds_slot_stat *st = sa2_dsound_slot_stat_at(t, k);
            fprintf(g_out, "%s\n  {\"interface\": \"%s\", \"name\": \"%s\", \"slot\": %u, \"argc\": %d, "
                           "\"calls\": %u}", n++ ? "," : "", sa2_dsound_type_name(t), nm, k,
                    sa2_dsound_method_argc(t, k), st ? st->calls : 0u);
        }
    fprintf(g_out, "]},\n \"xaudio2\": {\"method_calls\": %u, \"log_dropped\": %u, \"calls\": [",
            xa.method_calls, xa.log_dropped);
    for (uint32_t i = 0; i < sa2_xaudio2_log_count(); i++) {
        const sa2_xa2_call *c = sa2_xaudio2_log_at(i);
        const char *nm = sa2_xaudio2_method_name(c->type, c->slot);
        fprintf(g_out, "%s\n  {\"seq\": %u, \"virtual_us\": %llu, \"thread\": %u, \"interface\": \"%s\", "
                       "\"method\": \"%s\", \"slot\": %u, \"this\": \"0x%08x\", \"args\": [", i ? "," : "",
                c->seq, (unsigned long long)c->virtual_us, c->thread, sa2_xaudio2_type_name(c->type),
                nm ? nm : "<unnamed slot>", c->slot, c->self);
        put_args(c->arg, c->argc);
        fprintf(g_out, "], \"ret\": \"0x%08x\", \"out\": [\"0x%08x\", \"0x%08x\"], \"answered\": \"%s\"}",
                c->ret, c->out0, c->out1, c->answered ? "answered" : "refused");
    }
    fprintf(g_out, "], \"methods\": [");
    n = 0;
    for (uint32_t t = 0; t < sa2_xaudio2_type_count(); t++)
        for (uint32_t k = 0; k < SA2_XA2_MAX_SLOTS; k++) {
            const char *nm = sa2_xaudio2_method_name(t, k);
            if (!nm) continue;
            const sa2_xa2_slot_stat *st = sa2_xaudio2_slot_stat_at(t, k);
            fprintf(g_out, "%s\n  {\"interface\": \"%s\", \"name\": \"%s\", \"slot\": %u, \"argc\": %d, "
                           "\"calls\": %u}", n++ ? "," : "", sa2_xaudio2_type_name(t), nm, k,
                    sa2_xaudio2_method_argc(t, k), st ? st->calls : 0u);
        }
    fprintf(g_out, "]}},\n");
}

int main(int argc, char **argv) {
    const char *out_path = NULL, *which = "main";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--scenario") && i + 1 < argc) which = argv[++i];
        else if (!strcmp(argv[i], "--stray")) g_stray = 1;
        else if (!strcmp(argv[i], "--late-argc")) g_late_argc = 1;
        else out_path = NULL, i = argc;
    }
    if (!out_path || (strcmp(which, "main") && strcmp(which, "extended") && strcmp(which, "unlocks"))) {
        fprintf(stderr, "usage: rec_audio_harness --out PATH [--scenario main|extended|unlocks] [--stray] "
                        "[--late-argc]\n");
        return 2;
    }
    g_mem = (uint8_t *)calloc(1, GUEST_HI - GUEST_LO);
    if (!g_mem) return 2;
    setenv("SA2_D3D9_DEVICE", "1", 1);
    setenv("SA2_DSOUND_DEVICE", "1", 1);
    setenv("SA2_XAUDIO2_DEVICE", "1", 1);
    sa2_d3d9_config_from_env();
    sa2_dsound_config_from_env();
    sa2_xaudio2_config_from_env();
    g_out = fopen(out_path, "w");
    if (!g_out) return 2;
    fprintf(g_out, "{\n");
    if (!strcmp(which, "main")) scenario_main();
    else if (!strcmp(which, "unlocks")) scenario_unlocks();
    else scenario_extended();

    sa2_ds_stats ds;
    sa2_dsound_stats_get(&ds);
    sa2_xa2_stats xa;
    sa2_xaudio2_stats_get(&xa);
    run_record();
    fprintf(g_out, " \"answers\": [");
    for (int i = 0; i < g_ncalls; i++)
        fprintf(g_out, "%s[%llu, %llu, \"%016llx\"]", i ? ", " : "",
                (unsigned long long)g_answers[i][0], (unsigned long long)g_answers[i][1],
                (unsigned long long)g_answers[i][2]);
    fprintf(g_out, "],\n \"function_table\": [%u, \"%016llx\"],\n", g_ft_regs,
            (unsigned long long)g_ft_fnv);
    fprintf(g_out, " \"guest_fnv\": \"%016llx\",\n",
            (unsigned long long)fnv1a_more(FNV_BASIS, g_mem, GUEST_HI - GUEST_LO));
    fprintf(g_out, " \"records_fnv\": [\"%016llx\", \"%016llx\"],\n", (unsigned long long)buffers_fnv(),
            (unsigned long long)voices_fnv());
    fprintf(g_out, " \"dsound\": [%u, %u, %u, %u, %u, %u, %u, %u, %u],\n", ds.method_calls,
            ds.buffers, ds.objects_built, ds.from_guest, ds.clock, ds.noop, ds.cursor_reads,
            ds.cursor_wraps, ds.guest_bytes_allocated);
    fprintf(g_out, " \"xaudio2\": [%u, %u, %u, %u, %u, %llu, %llu, %u]\n}\n", xa.method_calls,
            xa.submits, xa.buffers_completed, xa.getstate_calls, xa.starvations,
            (unsigned long long)xa.submitted_bytes, (unsigned long long)xa.frames_played,
            xa.objects_built);
    fclose(g_out);
    return 0;
}
"""


def zig_env(local: Path) -> dict:
    return dict(os.environ, ZIG_LOCAL_CACHE_DIR=str(local),
                ZIG_GLOBAL_CACHE_DIR=os.environ.get(
                    "ZIG_GLOBAL_CACHE_DIR", str(ROOT / "private" / "generated" / "zig-cache-tests")))


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """A function (variant) -> the harness built with the pinned zig: "control" (no
    recorder, no wraps), "rec", or a REC_BREAK_* name. Without the pinned zig this fails,
    since a skip would pass the gate with the recorder unheld."""
    if not ZIG.is_file():
        pytest.fail(f"the pinned zig is not present: {ZIG}")
    d = tmp_path_factory.mktemp("rec-audio-harness")
    src = d / "rec_audio_harness.c"
    src.write_text(AUDIO_HARNESS_C)
    env = zig_env(d / "zc")
    rt = ROOT / "runtime"
    base = [str(ZIG), "cc", "-target", "x86_64-linux-musl", "-O1", "-Wall", "-Wextra", "-Werror",
            "-I", str(rt), "-I", str(ROOT / "host"), str(src), str(rt / "sa2_d3d9.c"), str(rt / "sa2_guest_call.c"),
            str(rt / "sa2_dsound.c"), str(rt / "sa2_xaudio2.c")]
    made: dict[str, Path] = {}

    def get(variant: str) -> Path:
        if variant not in made:
            exe = d / f"rec_audio_harness.{variant}"
            cmd = list(base)
            if variant != "control":
                cmd += [str(RECORDER)] + [f"-Wl,-wrap={w}" for w in OC.WRAPS]
                if variant != "rec":
                    cmd.append(f"-D{variant}")
            done = subprocess.run(cmd + ["-o", str(exe)], capture_output=True, text=True, env=env)
            assert done.returncode == 0, done.stderr
            made[variant] = exe
        return made[variant]
    return get


def run_harness(exe: Path, tmp_path: Path, name: str, *, audio: str | None = "1", trace: bool = True,
                counts: bool = True, scenario: str = "main", flags: tuple[str, ...] = (),
                env_extra: dict | None = None) -> dict:
    """One run: what the guest saw, and where the stream and counts went."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SA2_REC_", "SA2_D3D9", "SA2_D3DX", "SA2_DSOUND", "SA2_XAUDIO2", "SA2_STREAM_"))}
    tr, cn = tmp_path / f"{name}.sa2s", tmp_path / f"{name}.counts.json"
    if trace:
        env.update(SA2_REC_TRACE=str(tr), SA2_REC_META='{"build": "rec_audio_harness"}')
    if counts:
        env["SA2_REC_COUNTS"] = str(cn)
    if audio is not None:
        env["SA2_REC_AUDIO"] = audio
    env.update(env_extra or {})
    out = tmp_path / f"{name}.json"
    done = subprocess.run([str(exe), "--out", str(out), "--scenario", scenario, *flags], capture_output=True,
                          text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    seen = json.loads(out.read_text())
    seen["trace"], seen["counts"] = tr, cn
    return seen


def guest_view(seen: dict) -> dict:
    """What the guest can see: every answer (EAX, ESP and a digest of the stand-in
    register file after the call), the function table's registrations, all of guest
    memory, the two devices' own counters, and every buffer's and voice's record."""
    return {k: seen[k] for k in ("answers", "function_table", "guest_fnv", "records_fnv", "dsound", "xaudio2")}


def ds_pattern(buf_off: int, n: int, seed: int) -> bytes:
    """The harness's DirectSound bytes (put_ds): byte b of the buffer is in frame b / 4,
    L = 11 f + seed and R = -7 f - seed as int16."""
    out = bytearray()
    for i in range(0, n, 2):
        b = buf_off + i
        f = b // 4
        v = 11 * f + seed if b % 4 == 0 else -7 * f - seed
        out += struct.pack("<h", ((v + 0x8000) & 0xFFFF) - 0x8000)
    return bytes(out)


def xa_pattern(frames: int, seed: int) -> bytes:
    """The harness's XAudio2 frames (put_xa): channel c of frame i = ((i + seed) % 64 - 32)
    / 64 + c / 128."""
    return b"".join(struct.pack("<f", ((i + seed) % 64 - 32) / 64 + c / 128)
                    for i in range(frames) for c in range(6))


def audio_records(path: Path) -> dict[str, list[dict]]:
    """The stream's AUDIO records by op name, each with its blobs whole."""
    s = SD.scan(path, want_range=(0, 1 << 30))
    out: dict[str, list[dict]] = {}
    for r in s["range"]:
        if r["kind"] != "audio":
            continue
        r["data"] = {(ROLE, (None if a == NO_ARG else a)): d
                     for ROLE, a, d in SD.blobs_of(path, r["seq"])}
        out.setdefault(OPS[r["op"]][0], []).append(r)
    return out


def state(r: dict, fields, arg=None) -> dict:
    data = r["data"][(STATE, arg)]
    st = (SD.DS_STATE if fields is SD.DS_FIELDS else SD.XA_STATE).unpack(data)
    return dict(zip(fields, st))


def pcm_is_the_guests(path: Path, seen: dict) -> bool:
    """Every PCM blob of the stream is the bytes the harness wrote: the first Unlock the
    whole 4800-byte buffer (seed 3), the second the wrapped Lock's 800 bytes at 4000 and
    800 at 0 (seed 5), and the two submits 512 and 1024 frames (seeds 0 and 7)."""
    recs = audio_records(path)
    (p1, n1), (q1, m1, q2, m2) = seen["locks"]
    u1, u2 = recs["DS_UNLOCK"]
    s1, s2 = recs["XA_SUBMIT"]
    return (u1["data"][(PCM, 0)] == ds_pattern(0, n1, 3) and u2["data"][(PCM, 0)] == ds_pattern(4000, m1, 5)
            and u2["data"][(PCM, 2)] == ds_pattern(0, m2, 5) and s1["data"][(PCM, 0)] == xa_pattern(512, 0)
            and s2["data"][(PCM, 0)] == xa_pattern(1024, 7))


def held(seen: dict, counts: bool = True) -> tuple[dict, list[str], list[str]]:
    """The run's stream read (scan, keeping the AUDIO heads the device logs need), its
    counts held to it, and both held to the devices' own record of the run."""
    s = SD.scan(seen["trace"], doc=SD.doc_table(STREAM_H.read_text()), keep_audio=SD.audio_keep(seen["run"]))
    c = json.loads(seen["counts"].read_text())
    return s, SD.compare_counts(s, c), SD.hold_audio_to_run(s, seen["run"], c if counts else None)


def render_harness(renderer, tmp_path: Path, seen: dict, name: str) -> tuple[int, dict, str]:
    wav, js = tmp_path / f"{name}.wav", tmp_path / f"{name}.audio.json"
    done = subprocess.run([str(renderer()), str(seen["trace"]), "--wav", str(wav), "--json", str(js)],
                          capture_output=True, text=True, timeout=60)
    return done.returncode, json.loads(js.read_text())["audio"], done.stderr


def test_the_guest_sees_the_same_with_audio_recorded(build, tmp_path):
    control = guest_view(run_harness(build("control"), tmp_path, "control", trace=False, counts=False, audio=None))
    for name, audio_env in (("on", "1"), ("unset", None), ("zero", "0"), ("bad", "yes")):
        assert guest_view(run_harness(build("rec"), tmp_path, name, audio=audio_env)) == control, name
    assert len(control["answers"]) == 50 and control["function_table"][0] == 16 * 128 + 2 * 32 + 3 * 32
    # the paths the main scenario does not take, and a guest write outside a Lock
    for scenario, flags in (("extended", ()), ("main", ("--stray",)), ("extended", ("--stray",))):
        ctl = run_harness(build("control"), tmp_path, "ctl", trace=False, counts=False, audio=None,
                          scenario=scenario, flags=flags)
        assert guest_view(run_harness(build("rec"), tmp_path, "rec", scenario=scenario, flags=flags)) == \
            guest_view(ctl), (scenario, flags)
    assert len(ctl["answers"]) == 121


def test_without_audio_the_stream_is_the_version_1_stream(build, tmp_path):
    unset = run_harness(build("rec"), tmp_path, "unset", audio=None)
    zero = run_harness(build("rec"), tmp_path, "zero", audio="0")
    data = unset["trace"].read_bytes()
    assert data == zero["trace"].read_bytes() and unset["counts"].read_bytes() == zero["counts"].read_bytes()
    head, _ = SD.read_header(data)
    assert struct.unpack_from("<I", data, 8)[0] == 1 and head["schema"] == SD.SCHEMA and "audio" not in head
    assert "audio" not in head["note_classes"]
    counts = json.loads(unset["counts"].read_text())
    assert "audio" not in counts and "audio" not in counts["stream_records"] and "audio" not in counts["notes"]
    assert set(counts["stream_blobs"]) == {"1", "2", "3", "4", "5", "6"}
    s = SD.scan(unset["trace"], doc=SD.doc_table(STREAM_H.read_text()))
    assert s["problems"] == [] and s["audio"] is None and "audio" not in s["kinds"]
    assert s["frames"] == 9 and s["method_calls"] == 10
    assert SD.hold_audio_to_run(s, unset["run"], counts) == []          # nothing to hold


def test_a_setting_that_is_not_0_or_1_stops_the_stream(build, tmp_path):
    bad = run_harness(build("rec"), tmp_path, "bad", audio="yes")
    assert not bad["trace"].exists()
    status = Path(f"{bad['trace']}.status").read_text()
    assert status.startswith("disabled: SA2_REC_AUDIO=yes: 1 records audio, 0 or unset does not"), status
    # counting only, the same: the counting goes on, without audio, and says why
    only = run_harness(build("rec"), tmp_path, "only", audio="yes", trace=False)
    counts = json.loads(only["counts"].read_text())
    why = "SA2_REC_AUDIO=yes: 1 records audio, 0 or unset does not"
    assert counts["counting"] is True and counts["why"] == why and counts["stream"] is None
    assert "audio" not in counts and counts["method_calls"] == 10
    assert Path(f"{only['counts']}.status").read_text() == f"closed: counting only, without audio: {why}\n"


def test_counting_only_counts_the_audio_calls_that_failed(build, tmp_path):
    """Without a stream the audio calls are counted as with one: entered, returned, and
    failed by their answer (the main scenario's refused SetVolume), none recorded."""
    only = run_harness(build("rec"), tmp_path, "only", trace=False)
    counts = json.loads(only["counts"].read_text())
    ops = {r["name"]: r for r in counts["audio"]["ops"]}
    assert counts["why"] == "" and ops["DS_SET_VOLUME"]["failed"] == 1 and ops["DS_SET_VOLUME"]["returned"] == 2
    assert all(r["entered"] == r["returned"] and not r["recorded"] for r in ops.values())
    assert sum(r["failed"] for r in ops.values()) == 1


def test_a_device_whose_counts_are_not_the_recorders_stops_the_stream(build, tmp_path):
    """runtime/sa2_dsound.c's own break, SA2_DSOUND_BREAK=argc, declares CreateSoundBuffer
    with two arguments: the recorder, written for three, records nothing rather than read
    a dword the device does not pop. Held at start-up when the break is set from the
    start, and at the registration of DirectSound's thunks when it comes after the
    recorder started (--late-argc)."""
    early = run_harness(build("rec"), tmp_path, "argc", env_extra={"SA2_DSOUND_BREAK": "argc"})
    status = Path(f"{early['trace']}.status").read_text()
    assert not early["trace"].exists() and status == (
        "disabled: the audio device gives IDirectSound8::CreateSoundBuffer 2 arguments, and this "
        "recorder was written for 3\n"), status
    late = run_harness(build("rec"), tmp_path, "late", flags=("--late-argc",))
    status = Path(f"{late['trace']}.status").read_text()
    assert status == ("disabled: the audio device gives IDirectSound8::CreateSoundBuffer 2 arguments at its "
                      "registration, and this recorder was written for 3\n"), status
    s = SD.scan(late["trace"])
    assert s["audio"] == {"records": 0, "ops": {}, "pcm_bytes": 0, "checked": 0, "buffers": 0, "voices": 0,
                          "stores": 0, "store_watch": {"every_frames": 64, "set": 0, "checks": 0,
                                                       "bytes_hashed": 0, "watched_at_exit": 0}}
    assert "the stream has no END record" in s["problems"][-1]


@pytest.mark.parametrize("brk", ["SA2_DSOUND_BREAK=frozen", "SA2_XAUDIO2_BREAK=noconsume"])
def test_a_break_of_an_audio_device_stops_the_stream(build, tmp_path, brk):
    """A break changes an audio device's model, as SA2_D3D9_BREAK does the recording
    device's: with SA2_REC_AUDIO=1 the stream stops rather than describe a device nobody
    meant to present, and the counting goes on."""
    k, v = brk.split("=")
    seen = run_harness(build("rec"), tmp_path, "brk", env_extra={k: v})
    status = Path(f"{seen['trace']}.status").read_text()
    assert not seen["trace"].exists() and status == (
        f"disabled: {brk} is set, and the recorder records no audio under a break of an audio device's "
        f"model\n"), status
    assert json.loads(seen["counts"].read_text())["counting"] is True


def test_the_audio_stream_holds_every_call_as_the_guest_made_it(build, tmp_path):
    on = run_harness(build("rec"), tmp_path, "on")
    s, mismatch, against_run = held(on)
    assert s["problems"] == [] and s["version"] == 2 and mismatch == [] and against_run == []
    counts = json.loads(on["counts"].read_text())
    assert s["audio"]["ops"] == {
        "DS_COOPERATIVE": 1, "DS_CREATE_BUFFER": 2, "DS_DUPLICATE": 1, "DS_GET_POSITION": 3,
        "DS_GET_STATUS": 2, "DS_LOCK": 2, "DS_PLAY": 3, "DS_RELEASE": 1, "DS_SET_FORMAT": 1, "DS_SET_FREQUENCY": 1,
        "DS_SET_PAN": 1, "DS_SET_POSITION": 1, "DS_SET_VOLUME": 1, "DS_STOP": 1, "DS_UNLOCK": 2,
        "XA_CREATE_MASTER": 1, "XA_CREATE_SOURCE": 1, "XA_DESTROY": 1, "XA_FLUSH": 1, "XA_GET_STATE": 2,
        "XA_MASTER_DESTROY": 1, "XA_MASTER_DETAILS": 1, "XA_SET_MATRIX": 1, "XA_SET_RATIO": 1, "XA_START": 1,
        "XA_STOP": 1, "XA_SUBMIT": 2}
    # every op entered and returned as often as the device answered it, each call recorded
    # or failed: the one SetVolume the device refused (-10001 is below DSBVOLUME_MIN)
    ops = {r["name"]: r for r in counts["audio"]["ops"]}
    assert all(r["entered"] == r["returned"] == r["recorded"] + r["failed"] and r["recorded"] == s["audio"]["ops"].get(n, 0)
               for n, r in ops.items())
    assert {n: r["failed"] for n, r in ops.items() if r["failed"]} == {"DS_SET_VOLUME": 1}
    assert counts["audio"]["thunks_wrapped"] == 28 and counts["stream_records"]["audio"] == 37
    # the store watch: each Unlock that closed the store's last Lock set it, the second Lock
    # checked the 4800 bytes once, and the store was watched at exit
    assert counts["audio"]["store_watch"] == {"every_frames": 64, "set": 2, "checks": 1, "bytes_hashed": 4800,
                                              "watched_at_exit": 1, "changed": 0}
    assert pcm_is_the_guests(on["trace"], on)
    recs = audio_records(on["trace"])
    obj = on["objects"]
    (p1, n1), (q1, m1, q2, m2) = on["locks"]
    unlock = recs["DS_UNLOCK"][1]
    st = state(unlock, SD.DS_FIELDS)
    assert (q1 - st["mem"], m1, q2 - st["mem"], m2) == (4000, 800, 0, 800) and st["mem_len"] == 4800
    # each Unlock's store_hash is the store the guest wrote: the first Unlock's the 4800
    # bytes of seed 3, the second's those with the wrapped Lock's 1600 bytes of seed 5 in
    store = bytearray(ds_pattern(0, 4800, 3))
    assert recs["DS_UNLOCK"][0]["data"][(HASH, None)] == struct.pack("<Q", fnv1a64(bytes(store)))
    store[4000:4800], store[0:800] = ds_pattern(4000, 800, 5), ds_pattern(0, 800, 5)
    assert unlock["data"][(HASH, None)] == struct.pack("<Q", fnv1a64(bytes(store)))
    assert state(recs["DS_SET_FREQUENCY"][0], SD.DS_FIELDS)["avg_bytes_per_sec"] == 44100 * 4
    play = state(recs["DS_PLAY"][1], SD.DS_FIELDS)
    assert (play["buffer"], play["playing"], play["looping"]) == (obj["buffer"], 1, 1)
    assert play["anchor_us_lo"] | play["anchor_us_hi"] << 32 == recs["DS_PLAY"][1]["virtual_us"]
    prim = state(recs["DS_SET_FORMAT"][0], SD.DS_FIELDS)
    assert (prim["buffer"], prim["primary"], prim["bytes"], prim["avg_bytes_per_sec"]) == (obj["primary"], 1, 96000, 192000)
    dup = state(recs["DS_DUPLICATE"][0], SD.DS_FIELDS, arg=1)
    assert dup["buffer"] == obj["duplicate"] and dup["mem"] == st["mem"]       # the original's store, shared
    master = state(recs["XA_CREATE_MASTER"][0], SD.XA_FIELDS, arg=0)
    assert (master["voice"], master["kind"], master["input_channels"], master["input_sample_rate"]) == \
        (obj["master"], 1, 2, 48000)
    assert recs["XA_SET_MATRIX"][0]["data"][(IN, 3)] == struct.pack(
        "<12f", 1.0, 0.0, 0.5, 0.25, 0.75, 0.0, 0.0, 1.0, 0.5, 0.25, 0.0, 0.75)
    assert state(recs["XA_SET_RATIO"][0], SD.XA_FIELDS)["ratio_bits"] == fbits(1.25)
    sub = state(recs["XA_SUBMIT"][0], SD.XA_FIELDS)
    assert (sub["queued"], sub["queued_end_lo"], sub["running"]) == (1, 512, 0)
    assert struct.unpack("<9I", recs["XA_SUBMIT"][1]["data"][(IN, 0)])[1:3] == (1024 * 24, on["data"][1])


def test_the_extended_scenario_holds_and_plays_by_the_devices_own_clock(build, renderer, tmp_path):
    """The paths the main scenario does not take: a duplicate made before its original's
    first Lock (a store of its own) and one after (sharing the original's, locked and
    retimed), DSBLOCK_ENTIREBUFFER, a Lock from the write cursor that wraps, SetFrequency
    on a playing buffer, a halted buffer rewound and played again, three submits, a ratio
    of 1.5 and a flush that keeps the head; and 64 Presents more, so the store watch
    checks at a FRAME."""
    ext = run_harness(build("rec"), tmp_path, "ext", scenario="extended")
    s, mismatch, against_run = held(ext)
    assert s["problems"] == [] and mismatch == [] and against_run == []
    counts = json.loads(ext["counts"].read_text())
    assert s["audio"]["records"] == 46 and s["frames"] == 71 and s["audio"]["stores"] == 2
    # set at the four Unlocks; checked at the two Locks of the watched shared store and at
    # FRAME 64 (both stores); both watched at exit
    assert counts["audio"]["store_watch"] == {"every_frames": 64, "set": 4, "checks": 4, "bytes_hashed": 4 * 1920,
                                              "watched_at_exit": 2, "changed": 0}
    recs = audio_records(ext["trace"])
    obj = ext["objects"]
    own = state(recs["DS_DUPLICATE"][0], SD.DS_FIELDS, arg=1)
    shared = state(recs["DS_DUPLICATE"][1], SD.DS_FIELDS, arg=1)
    first = state(recs["DS_UNLOCK"][0], SD.DS_FIELDS)
    assert (own["buffer"], own["mem"]) == (obj["own"], 0) and (shared["buffer"], shared["mem"]) == \
        (obj["shared"], first["mem"])
    (pa, na), (pd, nd), (q1, m1, q2, m2), (p1, n1) = ext["locks"]
    assert (na, pd - pa, nd, q1 - pa, m1, q2 - pa, m2, n1) == (1920, 960, 480, 1272, 648, 0, 352, 1920)
    assert state(recs["DS_UNLOCK"][3], SD.DS_FIELDS)["mem"] == p1 != pa
    rc, au, err = render_harness(renderer, tmp_path, ext, "ext")
    assert rc == 0 and au["problems"]["count"] == 0, au["problems"]
    ds, xa = au["ds"], au["xa"]
    assert (ds["cursor_checks"], ds["status_checks"], xa["state_checks"]) == (6, 2, 3)
    assert (ds["patches"], ds["store_checks"], ds["store_mismatches"], ds["stores"]) == (5, 4, 0, 2)
    # resampled: the shared duplicate at 24 kHz, played twice to its end (2 x 1920 bytes at
    # 96000 bytes/s, 960 samples each), and the buffer at 32 kHz from FRAME 2 to FRAME 4,
    # samples 800 .. 2399 (sample n is at n / 48000 s from FRAME 1, a FRAME every 16666 us)
    assert ds["resampled_samples"] == 2 * 960 + 1600 and ds["samples_without_store"] == 0
    assert (xa["submits"], xa["flushed_frames"]) == (3, 600)


def test_the_mixer_plays_the_harness_by_the_devices_own_clock(build, renderer, tmp_path):
    """Every cursor, status and voice state the devices answered the harness's guest is
    the one the mixer's clock gives at that instant, from the records before it, and its
    store is the guest's at each Unlock that closed the store's last Lock."""
    on = run_harness(build("rec"), tmp_path, "on")
    rc, au, _ = render_harness(renderer, tmp_path, on, "on")
    assert rc == 0 and au["problems"]["count"] == 0, au["problems"]
    assert (au["ds"]["cursor_checks"], au["ds"]["status_checks"], au["xa"]["state_checks"]) == (6, 2, 2)
    assert (au["ds"]["patches"], au["ds"]["store_checks"], au["ds"]["store_mismatches"]) == (3, 2, 0)
    # at ratio 1.25 the voice plays 999 of the second buffer's 1024 frames in one Present;
    # the flush drops the other 25
    assert au["xa"]["submits"] == 2 and au["xa"]["flushed_frames"] == 25 and au["ds"]["primary_buffers"] == 1
    frames = read_wav(tmp_path / "on.wav")[1]
    # sample 0 is at the first FRAME, where the buffer starts playing at frame 0 (seed 3)
    assert frames[0] == (3 / 32768, -3 / 32768) and len(frames) == au["samples"] == 6400


def test_a_write_outside_a_lock_is_caught(build, renderer, tmp_path):
    """--stray writes a store between an Unlock and the next Lock. The main scenario's
    Lock after it sees the change (store_watch.changed, an "audio" note after its
    record), and the Unlock after that carries the hash of a store the PCM does not
    rebuild. The extended scenario writes after the store's last Unlock: only the watch's
    check at FRAME 64 sees it."""
    main = run_harness(build("rec"), tmp_path, "stray", flags=("--stray",))
    s = SD.scan(main["trace"])
    counts = json.loads(main["counts"].read_text())
    assert counts["audio"]["store_watch"]["changed"] == 1 and counts["notes"]["audio"] == 1
    assert s["notes"][0].endswith("audio: the store at 0x03300000 (4800 bytes) changed after its last Unlock, "
                                  "seen at DS_LOCK on 0x03250080")
    rc, au, _ = render_harness(renderer, tmp_path, main, "stray")
    assert rc == 1 and au["ds"]["store_mismatches"] == 1
    assert "store 0x03300000, rebuilt from the stream's PCM, does not hash to the guest's store" in au["problems"]["first"][0]
    ext = run_harness(build("rec"), tmp_path, "ext-stray", scenario="extended", flags=("--stray",))
    counts = json.loads(ext["counts"].read_text())
    assert counts["audio"]["store_watch"]["changed"] == 1
    assert SD.scan(ext["trace"])["notes"] == [
        "audio: the store at 0x03300780 (1920 bytes) changed after its last Unlock, seen at frame 64"]
    rc, au, _ = render_harness(renderer, tmp_path, ext, "ext-stray")
    assert rc == 0 and au["ds"]["store_mismatches"] == 0           # no Unlock after it: the watch alone saw it


def test_without_the_store_watch_only_the_unlocks_hash_holds_the_store(build, renderer, tmp_path):
    """SA2_REC_AUDIO_WATCH=0, a live session's play (host/launch, host/android): no store is
    hashed again after its Unlock. The stream is the watched run's byte for byte, each Unlock's
    store_hash in it; the counts file's store_watch counts the Unlocks that set the watch and no
    check, so stream_dump's own count of the checks (compare_counts) differs by exactly those;
    a write outside a Lock is still caught by the next Unlock's hash in the mixer, and one after
    the store's last Unlock, which only the watch saw, is not."""
    off = {"SA2_REC_AUDIO_WATCH": "0"}
    on = run_harness(build("rec"), tmp_path, "won", scenario="extended")
    ext = run_harness(build("rec"), tmp_path, "woff", scenario="extended", env_extra=off)
    assert ext["trace"].read_bytes() == on["trace"].read_bytes()
    counts = json.loads(ext["counts"].read_text())
    assert counts["audio"]["store_watch"] == {"every_frames": 64, "set": 4, "checks": 0, "bytes_hashed": 0,
                                              "watched_at_exit": 2, "changed": 0}
    s, mismatch, against_run = held(ext)
    assert s["problems"] == [] and against_run == []
    assert mismatch == [
        "audio store_watch.checks: 0 in the counts file, 4 from the stream's own Locks, Unlocks and FRAMEs",
        "audio store_watch.bytes_hashed: 0 in the counts file, 7680 from the stream's own Locks, Unlocks and FRAMEs"]
    main = run_harness(build("rec"), tmp_path, "woff-stray", flags=("--stray",), env_extra=off)
    counts = json.loads(main["counts"].read_text())
    assert counts["audio"]["store_watch"]["changed"] == 0 and not counts["notes"].get("audio")
    rc, au, _ = render_harness(renderer, tmp_path, main, "woff-stray")
    assert rc == 1 and au["ds"]["store_mismatches"] == 1
    late = run_harness(build("rec"), tmp_path, "woff-ext-stray", scenario="extended", flags=("--stray",),
                       env_extra=off)
    assert json.loads(late["counts"].read_text())["audio"]["store_watch"]["changed"] == 0
    assert SD.scan(late["trace"])["notes"] == []
    rc, au, _ = render_harness(renderer, tmp_path, late, "woff-ext-stray")
    assert rc == 0 and au["ds"]["store_mismatches"] == 0


def without_hashes(data: bytes, unlocks: set[int]) -> bytes:
    """The stream `data` with the HASH blob taken out of each DS_UNLOCK record whose index (from
    0, in stream order) is in `unlocks`, and each such record's size made its own again."""
    hb = struct.unpack_from("<I", data, 12)[0]
    out, p, k = bytearray(data[:hb]), hb, 0
    while p < len(data):
        kind, size = struct.unpack_from("<II", data, p)
        rec = data[p:p + size]
        if kind == 6 and struct.unpack_from("<H", rec, 14)[0] == OP["DS_UNLOCK"]:
            if k in unlocks:
                argc = struct.unpack_from("<I", rec, 32)[0]
                at = 36 + 4 * argc + 4                       # the blob count, after the args and EAX
                n, q, kept = struct.unpack_from("<I", rec, at)[0], at + 4, []
                for _ in range(n):
                    role, _arg, ln = struct.unpack_from("<HHI", rec, q)
                    end = q + 8 + pad4(ln)
                    if role != HASH:
                        kept.append(rec[q:end])
                    q = end
                body = rec[12:at] + struct.pack("<I", len(kept)) + b"".join(kept) + rec[q:]
                rec = struct.pack("<II", kind, 12 + len(body)) + rec[8:12] + body
            k += 1
        out += rec
        p += size
    return bytes(out)


LIVE_HASH = {"SA2_REC_AUDIO_WATCH": "0", "SA2_REC_STORE_HASH_EVERY": "64"}     # host/launch's play


def test_a_live_sessions_recorder_hashes_the_first_closed_store_and_every_64th(build, tmp_path):
    """SA2_REC_STORE_HASH_EVERY=64, a live session's play (host/launch, host/android): of the 130
    Unlocks that close their store's last Lock, the 1st, 65th and 129th carry the store_hash, the
    every-hash run's own; the stream is otherwise that run's byte for byte, and the counts file
    says how many closed and how many went without. stream_dump, which holds every such Unlock
    to its hash, reports each of the others; so does the renderer (the mixer test below)."""
    every = run_harness(build("rec"), tmp_path, "every", scenario="unlocks", env_extra={"SA2_REC_AUDIO_WATCH": "0"})
    live = run_harness(build("rec"), tmp_path, "live", scenario="unlocks", env_extra=LIVE_HASH)
    ue, ul = audio_records(every["trace"])["DS_UNLOCK"], audio_records(live["trace"])["DS_UNLOCK"]
    assert len(ue) == len(ul) == 130
    assert [k + 1 for k, r in enumerate(ue) if (HASH, None) in r["data"]] == list(range(1, 131))
    assert [k + 1 for k, r in enumerate(ul) if (HASH, None) in r["data"]] == [1, 65, 129]
    assert all(ul[k]["data"][(HASH, None)] == ue[k]["data"][(HASH, None)] for k in (0, 64, 128))
    data = every["trace"].read_bytes()
    assert without_hashes(data, set(range(130)) - {0, 64, 128}) == live["trace"].read_bytes() != data
    au = json.loads(live["counts"].read_text())["audio"]
    assert au["store_hash"] == {"every": 64, "closes": 130, "not_hashed": 127}
    assert au["store_watch"]["set"] == 3 and au["store_watch"]["checks"] == 0
    assert "store_hash" not in json.loads(every["counts"].read_text())["audio"]
    got = SD.scan(live["trace"])["problems"]
    assert len(got) == 127 and all("no hash blob of the call, which its line gives" in p for p in got), got[:3]
    assert SD.scan(every["trace"])["problems"] == []
    # 1 and anything that is not 1 to 65536 in decimal hash every one
    for value in ("1", "0", "65537", "64x", "-64", " 64", ""):
        other = run_harness(build("rec"), tmp_path, "other", scenario="unlocks",
                            env_extra={"SA2_REC_AUDIO_WATCH": "0", "SA2_REC_STORE_HASH_EVERY": value})
        assert other["trace"].read_bytes() == data, value


def test_the_live_mixer_holds_the_hashes_a_live_stream_carries(build, renderer, tmp_path):
    """A live session's mixer (store_check_every 64, player.cpp's kLiveStoreCheckEvery; the
    renderer's --store-check-every 64) holds each store_hash a live stream carries and counts the
    127 closing Unlocks without one as not hashed, and holds an every-hash stream at the same
    three; a store the guest wrote outside a Lock before the 65th Unlock is still caught there.
    A replay's mixer, the renderer's default, takes every closing Unlock without a hash as a
    mismatch."""
    live = run_harness(build("rec"), tmp_path, "live", scenario="unlocks", env_extra=LIVE_HASH)
    every = run_harness(build("rec"), tmp_path, "every", scenario="unlocks", env_extra={"SA2_REC_AUDIO_WATCH": "0"})

    def render_one(seen: dict, name: str, *extra: str) -> tuple[int, dict]:
        js = tmp_path / f"{name}.audio.json"
        done = subprocess.run([str(renderer()), str(seen["trace"]), "--wav", str(tmp_path / f"{name}.wav"),
                               "--json", str(js), *extra], capture_output=True, text=True, timeout=60)
        assert done.returncode in (0, 1), done.stderr
        return done.returncode, json.loads(js.read_text())["audio"]

    def store(au: dict) -> tuple:
        ds = au["ds"]
        return ds["store_checks"], ds["store_mismatches"], ds["stores_not_hashed"], ds["store_check_every"]
    rc, au = render_one(live, "live", "--store-check-every", "64")
    assert rc == 0 and au["problems"]["count"] == 0 and store(au) == (3, 0, 127, 64), au["problems"]
    rc, au = render_one(every, "every-live", "--store-check-every", "64")
    assert rc == 0 and store(au) == (3, 0, 127, 64)
    rc, au = render_one(every, "every")
    assert rc == 0 and store(au) == (130, 0, 0, 1)
    rc, au = render_one(live, "live-replay")
    assert rc == 1 and store(au) == (130, 127, 0, 1)
    assert au["problems"]["count"] == 127 and all(
        "Unlock closes its store's last Lock and carries no 8-byte store_hash" in p for p in au["problems"]["first"])
    # a write outside a Lock, after the 64th Unlock: the live mixer sees it at the 65th's hash,
    # and a replay's at the 65th to 67th, until the 68th rewrites the bytes
    stray = run_harness(build("rec"), tmp_path, "live-stray", scenario="unlocks", flags=("--stray",),
                        env_extra=LIVE_HASH)
    rc, au = render_one(stray, "live-stray", "--store-check-every", "64")
    assert rc == 1 and store(au) == (3, 1, 127, 64)
    assert au["problems"]["first"][0].endswith(
        f"buffer 0x{stray['objects']['buffer']:08x}'s store 0x{stray['objects']['store']:08x}, rebuilt from the "
        f"stream's PCM, does not hash to the guest's store after the Unlock (0x"
        f"{struct.unpack('<Q', audio_records(stray['trace'])['DS_UNLOCK'][64]['data'][(HASH, None)])[0]:016x}): "
        f"the stream does not hold what the guest wrote")
    stray_every = run_harness(build("rec"), tmp_path, "every-stray", scenario="unlocks", flags=("--stray",),
                              env_extra={"SA2_REC_AUDIO_WATCH": "0"})
    rc, au = render_one(stray_every, "every-stray")
    assert rc == 1 and store(au) == (130, 3, 0, 1)
    # the NOHASH mixer holds nothing: the check above is the mixer's
    nohash = subprocess.run([str(renderer("NOHASH")), str(stray["trace"]), "--wav", str(tmp_path / "nh.wav"),
                             "--store-check-every", "64"], capture_output=True, text=True, timeout=60)
    assert nohash.returncode == 0, nohash.stderr


# REC_BREAK_AUDROP's main scenario held to the devices' own record: the stream lacks the
# first Unlock, and its counts file agrees with it (the Unlock counted as failed).
AUDROP_HELD = ("the dsound device's call 7 (IDirectSoundBuffer8::Unlock on 0x03250080) is not the stream's next "
               "dsound record, seq 10: the record is DS_SET_FREQUENCY; its arguments are ['0xac44'], the device's "
               "['0x3300000', '0x12c0', '0x0', '0x0']")


def test_each_audio_break_of_the_recorder_fails_its_check(build, renderer, tmp_path):
    control = guest_view(run_harness(build("control"), tmp_path, "control", trace=False, counts=False, audio=None))
    # the bytes copied from past the guest's pointer: the stream reads clean and is not the
    # guest's, and the store it rebuilds does not hash to the guest's
    pcm = run_harness(build("REC_BREAK_AUPCM"), tmp_path, "aupcm")
    assert SD.scan(pcm["trace"])["problems"] == [] and not pcm_is_the_guests(pcm["trace"], pcm)
    rc, au, _ = render_harness(renderer, tmp_path, pcm, "aupcm")
    assert rc == 1 and au["ds"]["store_mismatches"] == 2
    # the state from before the call: the mixer's clock disagrees with the devices' answers
    st = run_harness(build("REC_BREAK_AUSTATE"), tmp_path, "austate")
    done = subprocess.run([str(renderer()), str(st["trace"]), "--wav", str(tmp_path / "austate.wav")],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and "the mixer's clock gives" in done.stderr
    # a cursor helper: the device's own counters move, so the guest's view differs
    cur = run_harness(build("REC_BREAK_AUCURSOR"), tmp_path, "aucursor")
    assert [k for k, v in guest_view(cur).items() if v != control[k]] == ["dsound"]
    # every Unlock counted as failed and not recorded: the stream reads clean, the counts
    # file agrees with it, and only the devices' own record says what is missing
    drop = run_harness(build("REC_BREAK_AUDROP"), tmp_path, "audrop")
    s, mismatch, against_run = held(drop)
    assert s["problems"] == [] and mismatch == [] and "DS_UNLOCK" not in s["audio"]["ops"]
    assert against_run == [AUDROP_HELD], against_run
    os.environ["SA2_STREAM_DUMP_BREAK"] = "auheld"
    try:
        assert held(drop)[2] == []
    finally:
        os.environ.pop("SA2_STREAM_DUMP_BREAK", None)
    # the same with each device's log cut short, as a run's is past its 4,096 calls: the
    # walk still names the drop, and a summary that kept no record to walk it against
    # (scan's default keep_audio) fails rather than pass
    short = json.loads(json.dumps(drop["run"]))
    for key in ("dsound", "xaudio2"):
        short[key]["method_calls"] += 1
    counts = json.loads(drop["counts"].read_text())
    assert SD.hold_audio_to_run(s, short, counts) == [AUDROP_HELD]
    assert SD.hold_audio_to_run(SD.scan(drop["trace"]), short, counts) == [
        f"the {key} device's log holds {calls} calls, and the summary kept the heads of 0 of the stream's "
        f"{records} {key} records to walk it against (scan's keep_audio, which audio_keep(run) gives): the log "
        f"is not walked" for key, calls, records in (("dsound", 24, 21), ("xaudio2", 16, 14))]
    # the store watch never checks: the stream's own Locks and FRAMEs say it should have
    nowatch = run_harness(build("REC_BREAK_AUNOWATCH"), tmp_path, "aunowatch", scenario="extended")
    s, mismatch, _ = held(nowatch)
    assert mismatch == ["audio store_watch.checks: 0 in the counts file, 4 from the stream's own Locks, Unlocks and FRAMEs",
                        "audio store_watch.bytes_hashed: 0 in the counts file, 7680 from the stream's own Locks, "
                        "Unlocks and FRAMEs"], mismatch
    os.environ["SA2_STREAM_DUMP_BREAK"] = "auwatch"
    try:
        assert held(nowatch)[1] == []
    finally:
        os.environ.pop("SA2_STREAM_DUMP_BREAK", None)


@pytest.fixture(scope="module")
def production_object(tmp_path_factory):
    """A function (defines) -> host/sa2_present_rec.c compiled as tools/s4/link_s4.py
    compiles a presentation build's hook (present_run.compile_c)."""
    if not ZIG.is_file():
        pytest.fail(f"the pinned zig is not present: {ZIG}")
    import present_run as PR
    d = tmp_path_factory.mktemp("rec-audio-object")

    def get(*defines: str) -> Path:
        o = d / f"rec{''.join(defines)}.o"
        done = PR.compile_c(RECORDER.resolve(), o, defines, env=zig_env(d / "zc"))
        assert done["rc"] == 0, done["stderr"]
        return o
    return get


AUDIO_GETTERS = {"sa2_dsound_type_count", "sa2_dsound_type_name", "sa2_dsound_method_name",
                 "sa2_dsound_method_argc", "sa2_dsound_buffer_count", "sa2_dsound_buffer_at",
                 "sa2_xaudio2_type_count", "sa2_xaudio2_type_name", "sa2_xaudio2_method_name",
                 "sa2_xaudio2_method_argc", "sa2_xaudio2_voice_count", "sa2_xaudio2_voice_at"}


def test_the_recorder_object_reaches_the_audio_tables_and_no_cursor_helper(production_object):
    got = OC.check(production_object())
    assert got["ok"] is True, got
    assert AUDIO_GETTERS <= OC.ALLOWED_UNDEFINED <= OC.undefined_symbols(production_object().read_bytes())
    assert not {"sa2_dsound_buffer_cursors", "sa2_xaudio2_voice_now"} & OC.ALLOWED_UNDEFINED
    bad = OC.check(production_object("REC_BREAK_AUCURSOR"))
    assert bad["ok"] is False and bad["not_allowed"] == ["sa2_dsound_buffer_cursors"]
    for brk in ("REC_BREAK_AUDROP", "REC_BREAK_AUNOWATCH"):    # the store watch and drop builds hold too
        assert OC.check(production_object(brk))["ok"] is True, brk


# ---------------------------------------------------------------- the runner

def test_the_runner_records_audio_only_when_asked():
    """tools/s5/present_run.py --audio: SA2_REC_AUDIO=1 in the run's settings, and the
    recorder's own setting to the settings check, as the other three are; without it the
    run's command is what it was."""
    import present_run as PR
    out = ROOT / "private" / "generated" / "s5" / "present" / "tree" / "boot"
    p = PR.paths(out, ROOT / "private" / "generated" / "s5" / "trace" / "boot.sa2s")
    plain_cmd, plain = PR.command("present-boot.tree", p, {}, {"build": "present-boot.tree"})
    cmd, spec = PR.command("present-boot.tree", p, {}, {"build": "present-boot.tree"}, audio=True)
    assert "SA2_REC_AUDIO" not in plain and spec.split("|") == (
        plain.split("|")[:-1] + ["SA2_REC_AUDIO=1", plain.split("|")[-1]])
    assert [a for a in cmd if a != spec] == [a for a in plain_cmd if a != plain]
    mine = {"SA2_REC_COUNTS": "c", "SA2_REC_META": "m", "SA2_REC_AUDIO": "1"}
    assert PR.spec_problems(mine, {}, "out") == []
    assert PR.spec_problems(dict(mine), {"SA2_REC_AUDIO": "1"}, "out") == [
        "SA2_REC_AUDIO is in the registered run's settings"]
    assert "SA2_REC_AUDIO" in PR.RECORDER_VARS and PR.clean_env().get("SA2_REC_AUDIO") is None


def test_the_runner_keeps_a_run_with_audio_apart_and_holds_it_to_its_version(capsys):
    """present_run.py --audio: its own default paths, so its version 2 stream is never
    renamed over a plain run's trace; refused with --no-trace; and its stream must be the
    version the run asked for, from a recorder that wrapped every `audio` line, with an
    AUDIO record at least, as a plain run's must be version 1."""
    import present_run as PR
    gen, tag = ROOT / "private" / "generated" / "s5", PR.tree_tag()
    assert PR.default_paths("boot", False, False, None, None) == (gen / "present" / tag / "boot",
                                                                   gen / "trace" / "boot.sa2s")
    assert PR.default_paths("boot", False, True, None, None) == (gen / "present" / tag / "boot-audio",
                                                                  gen / "trace" / "boot-audio.sa2s")
    assert PR.default_paths("metalharbor", True, False, None, None) == (
        gen / "present" / tag / "metalharbor-counts", None)
    assert PR.default_paths("boot", False, True, Path("x/o"), Path("x/t.sa2s")) == (ROOT / "x/o", ROOT / "x/t.sa2s")
    assert PR.main(["--audio", "--no-trace"]) == 2
    assert "--no-trace writes none" in capsys.readouterr().err
    v1, v2 = {"version": 1, "audio": None}, {"version": 2, "audio": {"records": 3}}
    wrapped = {"audio": {"thunks_wrapped": 28}}
    assert PR.audio_stream_problems(v2, wrapped, {"audio": True}) == []
    assert PR.audio_stream_problems(v1, {}, {"build": "b"}) == [] == PR.audio_stream_problems(v1, {}, None)
    assert PR.audio_stream_problems(v1, {}, {"audio": True}) == [
        "the run asked for audio (--audio), and its stream is version 1",
        "the recorder wrapped None audio thunks, and host/sa2_stream.h has 28 `audio` lines"]
    assert PR.audio_stream_problems({"version": 2, "audio": {"records": 0}}, {"audio": {"thunks_wrapped": 27}},
                                    {"audio": True}) == [
        "the run asked for audio (--audio), and its stream holds no AUDIO record",
        "the recorder wrapped 27 audio thunks, and host/sa2_stream.h has 28 `audio` lines"]
    assert PR.audio_stream_problems(v2, wrapped, {"build": "b"}) == [
        "the run did not ask for audio, and its stream is version 2"]


GOOD_AUDIO_COUNTS = {"thunks_wrapped": 28, "ops": [{"name": "DS_LOCK", "entered": 2, "returned": 2},
                                                   {"name": "XA_SUBMIT", "entered": 1, "returned": 1}],
                     "store_watch": {"every_frames": 64, "set": 1, "checks": 1, "bytes_hashed": 32,
                                     "watched_at_exit": 0, "changed": 0}}


AUDIO_COUNTS_BROKEN = [
    ("an audio line not wrapped", lambda a: a.update(thunks_wrapped=27), []),
    ("a call that did not come back", lambda a: a["ops"][1].update(returned=0), ["XA_SUBMIT"]),
    ("no store watch", lambda a: a.pop("store_watch"), []),
    ("a store watch that is not a table", lambda a: a.update(store_watch=0), []),
    ("a store that changed after its Unlock", lambda a: a["store_watch"].update(changed=1), []),
    ("a store watch with no count of changes", lambda a: a["store_watch"].pop("changed"), []),
]


@pytest.mark.parametrize("name,change,unfinished", AUDIO_COUNTS_BROKEN, ids=[c[0] for c in AUDIO_COUNTS_BROKEN])
def test_the_audio_counts_fail_on_each_term_alone(name, change, unfinished):
    """present_run.audio_counts, one term at a time: each alone fails it (check_counts makes
    it part of its own verdict; test_s5_stream.py holds that)."""
    import present_run as PR
    assert PR.audio_counts({"audio": GOOD_AUDIO_COUNTS}) == {
        "thunks_wrapped": [28, 28], "unfinished": [], "store_watch": GOOD_AUDIO_COUNTS["store_watch"], "ok": True}
    bad = json.loads(json.dumps(GOOD_AUDIO_COUNTS))
    change(bad)
    got = PR.audio_counts({"audio": bad})
    assert got["ok"] is False and got["unfinished"] == unfinished, name


def test_the_stream_check_holds_an_audio_run_to_its_version_and_to_the_devices(build, tmp_path, monkeypatch):
    """present_run.check_stream on the audio harness's streams, with the d3d9 holding
    (stream_dump.hold_to_run: the harness's run record has no d3d9 device to hold to)
    stood in for. A run whose meta asked for audio holds. A version 2 stream from a run
    whose meta did not ask for it, and a version 1 stream from one whose meta did, fail on
    that alone (audio_stream_problems). REC_BREAK_AUDROP's stream, which reads clean and
    agrees with its counts file, fails on the devices' own record alone
    (hold_audio_to_run)."""
    import present_run as PR
    monkeypatch.setattr(SD, "hold_to_run", lambda summary, run, rows=None: [])
    asked, plain = {"build": "rec_audio_harness", "audio": True}, {"build": "rec_audio_harness"}
    clean = {"problems": [], "counts_mismatch": [], "notes_not_accepted": {}, "held_to_run": [],
             "meta_problems": [], "audio_problems": []}

    def check(variant: str, name: str, meta: dict, audio: str | None = "1") -> dict:
        seen = run_harness(build(variant), tmp_path, name, audio=audio, env_extra={"SA2_REC_META": json.dumps(meta)})
        p = PR.paths(tmp_path / f"{name}-out", seen["trace"])
        p["out"].mkdir()
        shutil.copyfile(seen["counts"], p["counts"])
        return PR.check_stream(p, seen["trace"], seen["run"], meta)

    got = check("rec", "asked", asked)
    assert {k: got[k] for k in clean} == clean and got["ok"] is True, got
    assert (got["version"], got["audio"]["records"], got["status"].startswith("closed:")) == (2, 37, True)
    got = check("rec", "unasked", plain)
    assert {k: got[k] for k in clean} == dict(clean, audio_problems=[
        "the run did not ask for audio, and its stream is version 2"]) and got["ok"] is False
    got = check("rec", "v1", asked, audio=None)
    assert {k: got[k] for k in clean} == dict(clean, audio_problems=[
        "the run asked for audio (--audio), and its stream is version 1",
        "the recorder wrapped None audio thunks, and host/sa2_stream.h has 28 `audio` lines"]) and got["ok"] is False
    got = check("REC_BREAK_AUDROP", "audrop", asked)
    assert {k: got[k] for k in clean} == dict(clean, held_to_run=[AUDROP_HELD]) and got["ok"] is False


def test_stream_dump_run_holds_the_audio_to_the_devices(build, tmp_path, monkeypatch, capsys):
    """tools/s5/stream_dump.py --counts --run holds a version 2 stream's records and counts
    to the audio devices' own record (hold_audio_to_run) beside the d3d9 holding, stood in
    for as above: the harness's stream passes, and REC_BREAK_AUDROP's, whose counts agree
    with it, fails on the devices' record alone."""
    monkeypatch.setattr(SD, "hold_to_run", lambda summary, run, rows=None: [])
    for variant, name, want in (("rec", "on", []), ("REC_BREAK_AUDROP", "audrop", [AUDROP_HELD])):
        seen = run_harness(build(variant), tmp_path, name)
        run, dump = tmp_path / f"{name}.run.json", tmp_path / f"{name}.dump.json"
        run.write_text(json.dumps(seen["run"]))
        rc = SD.main([str(seen["trace"]), "--format", str(STREAM_H), "--counts", str(seen["counts"]),
                      "--run", str(run), "--json", str(dump)])
        got = json.loads(dump.read_text())
        assert (rc, got["problems"], got["counts_problems"], got["run_problems"]) == (
            1 if want else 0, [], [], want), name
    assert f"PROBLEM {AUDROP_HELD}\n" in capsys.readouterr().err
