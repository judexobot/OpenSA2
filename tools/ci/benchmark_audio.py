#!/usr/bin/env python3
"""Compare the real mixer's speed and WAV bytes against a commit, using synthetic PCM."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_ref):
        parser.error("--baseline-ref must be a full commit SHA")
    out = ROOT / "private/generated/ci/audio-benchmark"
    out.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.check_output(["git", "show", f"{args.baseline_ref}:host/player/mixer.cpp"], cwd=ROOT)
    (out / "baseline.cpp").write_bytes(baseline)
    spec = importlib.util.spec_from_file_location("audio_fixtures", ROOT / "Documentation/scripts/tests/test_s5_audio.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    records = fixtures.start()
    for index in range(16):
        fixtures.BUF = 0x03250040 + index * 0x40
        fixtures.MEM = 0x04000000 + index * 0x20000
        records += fixtures.ds_buffer(avg=44100 * 4, volume=-2600, data=fixtures.FRAMES8 * 4096)
    for frame in range(1, 1801):
        records += fixtures.present(fixtures.T0 + frame * 1000000 // 60, frame + 1)
    stream = out / "synthetic.sa2s"
    stream.write_bytes(fixtures.make(records + [fixtures.END]))
    executables = {}
    for name, source in (("baseline", out / "baseline.cpp"), ("updated", ROOT / "host/player/mixer.cpp")):
        exe = out / name
        subprocess.run(["g++", "-std=c++17", "-O2", "-DSA2_MIXER_MAIN", f"-I{ROOT / 'host'}",
                        f"-I{ROOT / 'host/player'}", str(source), str(ROOT / "host/player/stream.cpp"),
                        str(ROOT / "host/player/sha256.cpp"), "-lzstd", "-o", str(exe)], check=True)
        executables[name] = exe
    timings = {name: [] for name in executables}
    for repetition in range(5):
        order = list(executables) if repetition % 2 == 0 else list(reversed(executables))
        for name in order:
            start = time.perf_counter()
            subprocess.run([str(executables[name]), str(stream), "--wav", str(out / f"{name}.wav")],
                           check=True, capture_output=True, timeout=120)
            if repetition:  # Warm up both executables before measuring.
                timings[name].append(time.perf_counter() - start)
    hashes = {name: hashlib.sha256((out / f"{name}.wav").read_bytes()).hexdigest() for name in executables}
    if hashes["baseline"] != hashes["updated"]:
        raise RuntimeError("the optimized mixer changed the synthetic benchmark's WAV")
    medians = {name: statistics.median(values) for name, values in timings.items()}
    result = {"scope": "synthetic 16-voice 44.1 kHz mixer; not gameplay or device FPS",
              "machine": platform.machine(), "baseline_ref": args.baseline_ref,
              "wav_identical": True, "wav_sha256": hashes["updated"], "seconds": timings,
              "median_seconds": medians, "speedup": medians["baseline"] / medians["updated"]}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
