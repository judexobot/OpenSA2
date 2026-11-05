"""Exercise the real audio_block with a deterministic SDL queue and no audio device."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_live_underruns_restore_headroom_without_changing_replays_or_wavs(tmp_path):
    source = (ROOT / "host/player/player.cpp").read_text()
    start = source.index("void Player::audio_block() {")
    end = source.index("\n}\n", start) + 3
    driver = tmp_path / "queue.cpp"
    driver.write_text(HARNESS.replace("// AUDIO_BLOCK", source[start:end]))
    exe = tmp_path / "queue"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(driver), "-o", str(exe)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


HARNESS = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
struct Sa2Mixer {
    static constexpr unsigned kRate = 48000, kChannels = 2;
    std::vector<float> samples;
    const std::vector<float>& block() const { return samples; }
    void clear_block() { samples.clear(); }
};
namespace sa2tw { void mono_audio(Sa2Mixer&) {} void swap_audio(Sa2Mixer&) {} }
constexpr int kAudioLeadMs = 60, kAudioQueueBound = 48000 * 8 / 2;
constexpr int lead_bytes = 48000 * 8 * kAudioLeadMs / 1000;
struct SDL_AudioStream { int queued = 0; bool fail = false; std::vector<float> played; };
int SDL_GetAudioStreamQueued(SDL_AudioStream* s) { return s->queued; }
bool SDL_PutAudioStreamData(SDL_AudioStream* s, const void* data, int bytes) {
    if (s->fail) return false;
    const auto* p = static_cast<const float*>(data);
    s->played.insert(s->played.end(), p, p + bytes / sizeof(float));
    s->queued += bytes;
    return true;
}
const char* SDL_GetError() { return "device error"; }
std::string strf(const char*, const char* error) { return error; }
struct Player {
    Sa2Mixer storage; Sa2Mixer* mixer_ = &storage;
    SDL_AudioStream device; SDL_AudioStream* audio_stream_ = &device;
    struct { bool is_live = true; bool live() const { return is_live; } } rd_;
    struct { std::string audio_wav; } o_;
    struct {
        std::vector<float> samples;
        bool write(const float* p, size_t frames) { samples.insert(samples.end(), p, p + frames * 2); return true; }
        void flush() {}
    } wav_;
    struct { std::atomic<bool> feeding{false}; std::atomic<uint64_t> shortfalls{0}; } audio_tap_;
    bool wav_open_ = true, audio_lead_queued_ = false;
    uint64_t frame_ = 1, audio_pre_first_samples_ = 0, audio_depth_n_ = 0, audio_depth_sum_ = 0;
    uint64_t audio_dropped_ = 0, audio_blocks_ = 0, audio_bytes_queued_ = 0;
    uint64_t audio_shortfalls_seen_ = 0, audio_recoveries_ = 0, audio_recovery_bytes_ = 0;
    int audio_depth_min_ = -1, audio_depth_max_ = 0;
    std::string audio_failure_;
    void io_failure(const std::string&) { assert(false); }
    void audio_block();
    void frame(size_t samples = 800) { storage.samples.assign(samples * 2, 0.25f); audio_block(); }
};
// AUDIO_BLOCK
int main() {
    Player p;
    p.frame();
    assert(p.device.queued == lead_bytes && p.audio_tap_.feeding);
    assert(p.audio_pre_first_samples_ == 800 && p.audio_blocks_ == 0);
    p.device.queued = 0; p.device.played.clear(); p.audio_tap_.shortfalls = 4;
    p.frame();
    assert(p.device.queued == lead_bytes && p.audio_recoveries_ == 1);
    assert(p.audio_recovery_bytes_ == lead_bytes - 800 * 8);
    assert(p.audio_bytes_queued_ == 800 * 8);
    assert(std::all_of(p.device.played.begin(), p.device.played.end() - 1600, [](float x) { return x == 0; }));
    assert(std::all_of(p.device.played.end() - 1600, p.device.played.end(), [](float x) { return x == 0.25f; }));
    assert(p.wav_.samples.size() == 3200);
    assert(std::all_of(p.wav_.samples.begin(), p.wav_.samples.end(), [](float x) { return x == 0.25f; }));
    p.device.queued = 0;
    p.frame(); // No new callback shortfall: do not repeatedly insert the same lead.
    assert(p.audio_recoveries_ == 1 && p.device.queued == 800 * 8);
    p.audio_tap_.shortfalls++; p.device.queued = 0;
    p.frame(4000); // A large block already provides enough headroom.
    assert(p.audio_recoveries_ == 1 && p.device.queued == 4000 * 8);
    p.device.queued = kAudioQueueBound + 8;
    p.frame();
    assert(p.audio_dropped_ == 1 && p.device.queued == kAudioQueueBound + 8);
    p.device.queued = 800; p.audio_tap_.shortfalls++;
    p.frame();
    assert(p.device.queued == lead_bytes && p.audio_recoveries_ == 2);

    Player replay; replay.rd_.is_live = false;
    replay.frame(); replay.device.queued = 0; replay.audio_tap_.shortfalls = 10;
    replay.frame();
    assert(replay.device.queued == 800 * 8 && replay.audio_recoveries_ == 0);
    Player failed; failed.frame(); failed.device.queued = 0; failed.device.fail = true;
    failed.audio_tap_.shortfalls++;
    failed.frame();
    assert(failed.audio_recoveries_ == 0 && failed.audio_blocks_ == 0 && !failed.audio_failure_.empty());
    Player no_device; no_device.audio_stream_ = nullptr;
    no_device.frame();
    assert(no_device.wav_.samples.size() == 1600 && no_device.storage.samples.empty());
}
'''
