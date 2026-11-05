// mixer.cpp: see mixer.h.
#include "mixer.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstring>

namespace {

// Time in units of 1/6 microsecond: output sample n is at 6 * origin_us + 125 n.
constexpr uint64_t kUnitsPerUs = 6, kUnitsPerSample = 125, kUnitsPerSec = 6000000;

// The limiter. The game's own XAudio2 level matrix folds its 5.1 voice to stereo as
// FL + 0.707 FC + 0.707 BL, which passes full scale in loud stretches (the user's VM run
// of 2026-10-12: XAudio2 peak +6.85 dBFS, 14,118 samples clamped), and a clamp squares
// those peaks off. In live play the gain drops at once to hold the mix under kLimitCeil
// and comes back with an 80 ms time constant, so a loud stretch is quieter, not distorted.
constexpr float kLimitCeil = 0.98f, kLimitRelease = 1.0f / (0.080f * 48000.0f), kLimitUnity = 0.9999f;
using u128 = unsigned __int128;

// Breaks for the tests only (Documentation/scripts/tests/test_s5_audio.py compiles the
// renderer with one -D at a time to show the check that names it fails): the
// position clock ignores SetFrequency (CLOCK), a looping buffer does not wrap
// (NOWRAP), no source is interpolated (NOLERP), volume and pan are ignored (GAIN),
// the level matrix is ignored (MATRIX), a store is not held to its store_hash
// (NOHASH). No build of the player defines one.
#ifdef MIXER_BREAK_CLOCK
constexpr bool kBreakClock = true;
#else
constexpr bool kBreakClock = false;
#endif
#ifdef MIXER_BREAK_NOWRAP
constexpr bool kBreakNoWrap = true;
#else
constexpr bool kBreakNoWrap = false;
#endif
#ifdef MIXER_BREAK_NOLERP
constexpr bool kBreakNoLerp = true;
#else
constexpr bool kBreakNoLerp = false;
#endif
#ifdef MIXER_BREAK_GAIN
constexpr bool kBreakGain = true;
#else
constexpr bool kBreakGain = false;
#endif
#ifdef MIXER_BREAK_MATRIX
constexpr bool kBreakMatrix = true;
#else
constexpr bool kBreakMatrix = false;
#endif
#ifdef MIXER_BREAK_NOHASH
constexpr bool kBreakNoHash = true;
#else
constexpr bool kBreakNoHash = false;
#endif

uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
int16_t rd16s(const uint8_t* p) { int16_t v; std::memcpy(&v, p, 2); return v; }
float fbits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint64_t u64of(uint32_t lo, uint32_t hi) { return uint64_t(lo) | uint64_t(hi) << 32; }

std::string strf(const char* fmt, ...) {
    va_list ap, aq;
    va_start(ap, fmt);
    va_copy(aq, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string s(n > 0 ? size_t(n) : 0, '\0');
    if (n > 0) std::vsnprintf(s.data(), size_t(n) + 1, fmt, aq);
    va_end(aq);
    return s;
}
std::string jesc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
        else if (c < 0x20) o += strf("\\u%04x", c);
        else o += char(c);
    }
    return o;
}
std::string num(double v) { return std::isfinite(v) ? strf("%.9g", v) : std::string("null"); }
std::string dbfs(double v) { return v > 0 ? num(20.0 * std::log10(v)) : std::string("null"); }

// DirectSound's hundredths of a decibel as a linear gain; DSBVOLUME_MIN is silence.
double db_gain(int32_t hundredths) {
    if (hundredths <= -10000) return 0.0;
    if (hundredths >= 0) return 1.0;
    return std::pow(10.0, double(hundredths) / 2000.0);
}

uint32_t align_down(uint32_t pos, uint32_t block) { return block < 2u ? pos : pos - pos % block; }

// host/sa2_stream.h store_hash: the 64-bit FNV-1a of a store, one byte at a time, as the
// recorder hashes the guest's (host/sa2_present_rec.c, guest_fnv).
uint64_t fnv1a64(const std::vector<uint8_t>& s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : s) { h ^= b; h *= 0x100000001b3ull; }
    return h;
}

// The one clock of a DirectSound buffer, which the renderer plays by and the cursor
// checks hold to the device's answers: its byte position at time t (1/6 us units) as
// an exact count of 1/6,000,000 bytes, looped or halted by runtime/sa2_dsound.c's
// buf_play_cursor. At a whole microsecond u, total / 6e6 is anchor_pos + floor((u -
// anchor_us) * avg_bytes_per_sec / 1e6), the device's own integers.
struct DsPos { u128 total; bool halted; };
DsPos ds_position(const sa2s_ds_state& s, uint64_t t) {
    const uint64_t anchor = u64of(s.anchor_us_lo, s.anchor_us_hi) * kUnitsPerUs;
    const uint64_t elapsed = t > anchor ? t - anchor : 0;
    const uint64_t rate = kBreakClock ? uint64_t(s.samples_per_sec) * s.block_align : s.avg_bytes_per_sec;
    const u128 total = u128(s.anchor_pos) * kUnitsPerSec + u128(elapsed) * rate;
    const u128 bytes_units = u128(s.bytes) * kUnitsPerSec;
    if (s.looping && !kBreakNoWrap) return {total % bytes_units, false};
    return {total, total >= bytes_units};
}

// The play cursor buf_play_cursor gives at virtual time `now` (integer microseconds)
// from the state `p`, and whether the buffer is still playing after it settled a halt.
uint32_t ds_cursor(const sa2s_ds_state& p, uint64_t now, bool* playing) {
    *playing = p.playing != 0;
    if (!p.bytes) return 0;
    const uint32_t block = p.block_align ? p.block_align : 1u;
    if (!p.playing || !p.avg_bytes_per_sec) return align_down(p.anchor_pos % p.bytes, block);
    const DsPos q = ds_position(p, now * kUnitsPerUs);
    if (q.halted) {
        *playing = false;
        return align_down(p.bytes > block ? p.bytes - block : 0u, block);
    }
    return align_down(uint32_t(q.total / kUnitsPerSec), block);
}

// runtime/sa2_xaudio2.c frames_after: elapsed (1/6 us units) times samples_per_sec
// times the ratio, whose float32 bits are mant * 2^-shift, as an exact integer part
// and a remainder over the returned denominator. The renderer and the GetState check
// both call it.
struct Frames { uint64_t whole; uint64_t rem; u128 den; };
Frames frames_after(uint64_t elapsed, uint32_t sps, uint32_t ratio_bits) {
    const uint32_t e = (ratio_bits >> 23) & 0xFFu;
    const uint64_t mant = (ratio_bits & 0x7FFFFFu) | 0x800000u;
    uint32_t shift = 150u - e;
    if (shift > 60u) shift = 60u;                 // a ratio the runtime clamps away; never 0 frames per second
    const u128 n = u128(elapsed) * sps * mant;
    const u128 den = u128(kUnitsPerSec) << shift;
    return {uint64_t(n / den), uint64_t(n % den), den};
}

}  // namespace

// ------------------------------------------------------------------ levels

void Sa2Level::add(float l, float r) {
    const double a = std::fabs(double(l)), b = std::fabs(double(r));
    peak = std::max(peak, std::max(a, b));
    sumsq += double(l) * l + double(r) * r;
    samples += 2;
}
double Sa2Level::rms() const { return samples ? std::sqrt(sumsq / double(samples)) : 0.0; }
std::string Sa2Level::json() const {
    return "{\"peak\": " + num(peak) + ", \"rms\": " + num(rms()) + ", \"peak_dbfs\": " + dbfs(peak) +
           ", \"rms_dbfs\": " + dbfs(rms()) + "}";
}

// ------------------------------------------------------------------ the mixer

void Sa2Mixer::problem(const std::string& what) {
    problems_++;
    if (first_problems_.size() < 40) {
        first_problems_.push_back(what);
        std::fprintf(stderr, "audio: %s\n", what.c_str());
    }
}

std::string Sa2Mixer::where(const Sa2Record& r) const {
    return strf("seq %u audio op %u", r.seq, r.op);
}

void Sa2Mixer::start(uint64_t virtual_us) {
    started_ = true;
    origin_us_ = last_us_ = virtual_us;
    n_ = 0;
}

void Sa2Mixer::on_frame(uint64_t virtual_us) { render_to(virtual_us); }

// Every output sample whose time is before `virtual_us`: a record at that time takes
// effect from the sample at its time on.
void Sa2Mixer::render_to(uint64_t virtual_us) {
    if (!started_) start(virtual_us);
    if (virtual_us < last_us_) {
        late_records_++;
        problem(strf("a record at virtual %" PRIu64 " us after one at %" PRIu64 " us: the mix cannot go back",
                     virtual_us, last_us_));
        return;
    }
    last_us_ = virtual_us;
    const uint64_t target = ((virtual_us - origin_us_) * kUnitsPerUs + kUnitsPerSample - 1) / kUnitsPerSample;
    if (target <= n_) return;
    const uint64_t count = target - n_;
    ds_acc_.assign(size_t(count) * 2, 0.0f);
    xa_acc_.assign(size_t(count) * 2, 0.0f);
    for (auto& [addr, b] : ds_) render_ds(b, n_, target, ds_acc_);
    for (auto& [addr, v] : xa_) render_xa(v, n_, target, xa_acc_);
    // Keep geometric growth across the many AUDIO records between two FRAMEs.
    // Reserving the exact size here reallocates and copies the accumulated mix
    // for almost every record during the first (potentially long) frame.
    for (uint64_t i = 0; i < count; i++) {
        const float dl = ds_acc_[2 * i], dr = ds_acc_[2 * i + 1];
        const float xl = xa_acc_[2 * i], xr = xa_acc_[2 * i + 1];
        float l = dl + xl, r = dr + xr;
        ds_level_.add(dl, dr);
        xa_level_.add(xl, xr);
        mix_level_.add(l, r);
        if (limiter_) {
            lim_gain_ += (1.0f - lim_gain_) * kLimitRelease;
            if (lim_gain_ > kLimitUnity) lim_gain_ = 1.0f;   // float's step near 1 stops the approach short of it
            const float pk = std::max(std::fabs(l), std::fabs(r));
            if (pk * lim_gain_ > kLimitCeil) lim_gain_ = kLimitCeil / pk;
            if (lim_gain_ < 1.0f) {
                limited_++;
                lim_least_ = std::min(lim_least_, lim_gain_);
                l *= lim_gain_;
                r *= lim_gain_;
            }
        }
        if (l > 1.0f || l < -1.0f) { clipped_++; l = std::clamp(l, -1.0f, 1.0f); }
        if (r > 1.0f || r < -1.0f) { clipped_++; r = std::clamp(r, -1.0f, 1.0f); }
        block_.push_back(l);
        block_.push_back(r);
    }
    n_ = target;
}

// One frame of a buffer's store as two float samples; false when the store does not
// hold it (silence).
bool Sa2Mixer::ds_frame(const DsBuf& b, const std::vector<uint8_t>& store, uint64_t frame, float* l, float* r) {
    *l = *r = 0.0f;
    const sa2s_ds_state& s = b.st;
    const uint64_t off = frame * s.block_align;
    if (off + s.block_align > store.size()) return false;
    const uint8_t* p = store.data() + off;
    if (s.format_tag == 1 && s.bits_per_sample == 16) {
        *l = float(rd16s(p)) / 32768.0f;
        *r = s.channels >= 2 ? float(rd16s(p + 2)) / 32768.0f : *l;
    } else if (s.format_tag == 1 && s.bits_per_sample == 8) {
        *l = float(int(p[0]) - 128) / 128.0f;
        *r = s.channels >= 2 ? float(int(p[1]) - 128) / 128.0f : *l;
    } else {                                     // 32-bit float (ds_format_ok)
        *l = fbits(rd32(p));
        *r = s.channels >= 2 ? fbits(rd32(p + 4)) : *l;
    }
    return true;
}

static bool ds_format_ok(const sa2s_ds_state& s) {
    const bool pcm = s.format_tag == 1 && (s.bits_per_sample == 16 || s.bits_per_sample == 8);
    const bool flt = s.format_tag == 3 && s.bits_per_sample == 32;
    return (pcm || flt) && s.channels && s.block_align == s.channels * (s.bits_per_sample / 8u);
}

void Sa2Mixer::render_ds(DsBuf& b, uint64_t n0, uint64_t n1, std::vector<float>& acc) {
    const sa2s_ds_state& s = b.st;
    if (!s.playing || s.released || s.primary || !s.bytes || !s.block_align || !s.avg_bytes_per_sec) return;
    if (!ds_format_ok(s)) {
        if (!ds_unsupported_samples_)
            problem(strf("buffer 0x%08x plays format tag %u, %u bits, %u channels, block %u, which the mixer "
                         "does not decode", s.buffer, s.format_tag, s.bits_per_sample, s.channels, s.block_align));
        ds_unsupported_samples_ += n1 - n0;
        return;
    }
    const uint64_t frame_units = uint64_t(s.block_align) * kUnitsPerSec;
    const uint64_t bytes_units = uint64_t(s.bytes) * kUnitsPerSec;
    const uint64_t anchor = u64of(s.anchor_us_lo, s.anchor_us_hi) * kUnitsPerUs;
    const uint64_t rate = kBreakClock ? uint64_t(s.samples_per_sec) * s.block_align : s.avg_bytes_per_sec;
    const uint64_t first_time = origin_us_ * kUnitsPerUs + kUnitsPerSample * n0;
    const DsPos first = ds_position(s, first_time);
    if (first.halted) { b.halted_seen = true; return; }
    // ds_position reduces a loop, or proves a non-looping position is in range.
    // Both fit in 64 bits: even UINT32_MAX bytes * 6,000,000 is below 2^55.
    // Advance that exact integer clock instead of doing 128-bit division and
    // modulo for every sample (software division on Android arm64).
    uint64_t total = uint64_t(first.total);
    const uint64_t nframes = s.bytes / s.block_align;
    const bool exact = uint64_t(s.avg_bytes_per_sec) == uint64_t(kRate) * s.block_align;
    const auto it = s.mem ? stores_.find(s.mem) : stores_.end();
    const bool have_store = it != stores_.end();
    const double g = kBreakGain ? 1.0 : db_gain(b.volume);
    const float gl = kBreakGain ? 1.0f : float(g * (b.pan > 0 ? db_gain(-b.pan) : 1.0));
    const float gr = kBreakGain ? 1.0f : float(g * (b.pan < 0 ? db_gain(b.pan) : 1.0));
    for (uint64_t n = n0, t = first_time; n < n1; n++, t += kUnitsPerSample) {
        if (n != n0 && t > anchor)
            total += std::min(kUnitsPerSample, t - anchor) * rate;
        if (s.looping && !kBreakNoWrap && total >= bytes_units) total %= bytes_units;
        if (total >= bytes_units) {
            b.halted_seen = true;
            break;                               // the buffer ran out: silent from here on
        }
        if (!have_store) { ds_no_store_samples_++; continue; }
        const uint64_t fi = total / frame_units;
        float l, r;
        ds_frame(b, it->second, fi, &l, &r);
        if (!exact && !kBreakNoLerp) {
            ds_resampled_samples_++;
            const double frac = double(total % frame_units) / double(frame_units);
            uint64_t f2 = fi + 1;
            if (f2 >= nframes) f2 = s.looping ? 0 : fi;
            float l2, r2;
            ds_frame(b, it->second, f2, &l2, &r2);
            l = float(l + (l2 - l) * frac);
            r = float(r + (r2 - r) * frac);
        }
        acc[2 * (n - n0)] += l * gl;
        acc[2 * (n - n0) + 1] += r * gr;
    }
}

void Sa2Mixer::render_xa(XaVoice& v, uint64_t n0, uint64_t n1, std::vector<float>& acc) {
    const sa2s_xa_state& s = v.st;
    if (s.kind != 2 || s.destroyed || !s.running || !s.channels) return;
    const uint64_t played = u64of(s.played_lo, s.played_hi), qend = u64of(s.queued_end_lo, s.queued_end_hi);
    if (!s.queued || played >= qend) { xa_starved_samples_ += n1 - n0; return; }
    const uint64_t anchor = u64of(s.anchor_us_lo, s.anchor_us_hi) * kUnitsPerUs;
    const uint64_t anchor_frame = u64of(s.anchor_frame_lo, s.anchor_frame_hi);
    const bool exact = s.samples_per_sec == kRate && s.ratio_bits == 0x3F800000u;
    const uint32_t ch = s.channels;
    const bool matrix = s.matrix_src == ch && s.matrix_dst == 2u && ch * 2u <= 12u;
    float m[12] = {};
    for (uint32_t k = 0; k < 12u; k++) m[k] = fbits(s.matrix[k]);
    const float vol = fbits(v.volume_bits);
    auto frame_at = [&](uint64_t pos, float* out) -> bool {
        for (const XaBuf& q : v.queue)
            if (q.start <= pos && pos < q.end) {
                std::memcpy(out, q.data.data() + size_t(pos - q.start) * ch, sizeof(float) * ch);
                return true;
            }
        return false;
    };
    std::vector<float> x(ch), y(ch);
    for (uint64_t n = n0; n < n1; n++) {
        const uint64_t t = origin_us_ * kUnitsPerUs + kUnitsPerSample * n;
        const uint64_t elapsed = t > anchor ? t - anchor : 0;
        const Frames f = frames_after(elapsed, s.samples_per_sec, s.ratio_bits);
        const uint64_t pos = anchor_frame + f.whole;
        if (pos >= qend) { xa_starved_samples_ += n1 - n; break; }
        // A voice's position never goes back (an anchor's frame is where it had played
        // to), so a buffer that ends at or before it is consumed for good.
        while (!v.queue.empty() && v.queue.front().end <= pos) v.queue.pop_front();
        if (!frame_at(pos, x.data())) {
            problem(strf("voice 0x%08x: stream frame %" PRIu64 " is queued, and no submit the stream holds "
                         "covers it", s.voice, pos));
            break;
        }
        if (!exact && !kBreakNoLerp) {
            xa_resampled_samples_++;
            const double frac = double(f.rem) / double(f.den);
            if (pos + 1 < qend && frame_at(pos + 1, y.data()))
                for (uint32_t c = 0; c < ch; c++) x[c] = float(x[c] + (y[c] - x[c]) * frac);
        }
        float l = 0.0f, r = 0.0f;
        if (matrix && !kBreakMatrix) {
            for (uint32_t c = 0; c < ch; c++) { l += m[c] * x[c]; r += m[ch + c] * x[c]; }
        } else if (ch == 1) {
            l = r = x[0];
        } else if (ch == 2) {
            l = x[0];
            r = x[1];
        } else if (ch == 6) {                    // 5.1: FL FR FC LFE BL BR
            xa_default_matrix_samples_++;
            const float k = 0.70710678f;
            l = x[0] + k * x[2] + k * x[4];
            r = x[1] + k * x[2] + k * x[5];
        } else {                                 // no layout the mixer knows: the first two
            xa_default_matrix_samples_++;
            l = x[0];
            r = x[1];
        }
        acc[2 * (n - n0)] += l * vol;
        acc[2 * (n - n0) + 1] += r * vol;
    }
}

bool Sa2Mixer::ds_state(const Sa2Record& r, int arg, sa2s_ds_state* out) {
    const Sa2Blob* b = r.blob(SA2S_BLOB_STATE, arg);
    if (!b || b->len != sizeof(sa2s_ds_state)) {
        problem(where(r) + ": no " + std::to_string(sizeof(sa2s_ds_state)) + "-byte buffer STATE blob");
        return false;
    }
    std::memcpy(out, b->data, sizeof *out);
    return true;
}
bool Sa2Mixer::xa_state(const Sa2Record& r, int arg, sa2s_xa_state* out) {
    const Sa2Blob* b = r.blob(SA2S_BLOB_STATE, arg);
    if (!b || b->len != sizeof(sa2s_xa_state)) {
        problem(where(r) + ": no " + std::to_string(sizeof(sa2s_xa_state)) + "-byte voice STATE blob");
        return false;
    }
    std::memcpy(out, b->data, sizeof *out);
    return true;
}

void Sa2Mixer::on_audio(const Sa2Record& r, const Sa2StreamReader& rd) {
    render_to(r.virtual_us);
    records_++;
    const Sa2AudioRow* row = rd.audio_row(r.op);
    if (!row) { problem(where(r) + ": an op the header's audio table does not name"); return; }
    by_op_[row->name]++;
    if (r.api == SA2S_AU_DSOUND) apply_ds(r, row);
    else if (r.api == SA2S_AU_XAUDIO2) apply_xa(r, row);
    else problem(where(r) + strf(": api %u", r.api));
}

void Sa2Mixer::apply_ds(const Sa2Record& r, const Sa2AudioRow* row) {
    const uint32_t op = r.op;
    if (op == SA2S_AU_DS_COOPERATIVE) return;
    if (op == SA2S_AU_DS_CREATE_BUFFER || op == SA2S_AU_DS_DUPLICATE) {
        sa2s_ds_state stv;
        if (!ds_state(r, 1, &stv)) return;
        const sa2s_ds_state* st = &stv;
        const Sa2Blob* out = r.blob(SA2S_BLOB_OUT, 1);
        if (!out || out->len < 4 || rd32(out->data) != st->buffer) {
            problem(where(r) + strf(": its STATE describes 0x%08x and the call wrote another buffer", st->buffer));
            return;
        }
        DsBuf b;
        b.st = *st;
        ds_[st->buffer] = b;                     // a new buffer's volume and pan are 0, a duplicate's too
        if (st->primary) ds_primary_++;
        store_of(r, *st);
        return;
    }
    auto it = ds_.find(r.self);
    if (it == ds_.end()) { problem(where(r) + strf(" (%s): buffer 0x%08x, which no record made", row->name.c_str(), r.self)); return; }
    DsBuf& b = it->second;
    sa2s_ds_state stv;
    if (!ds_state(r, -1, &stv)) return;
    const sa2s_ds_state* st = &stv;
    if (st->buffer != r.self) { problem(where(r) + strf(": its STATE describes 0x%08x, not 0x%08x", st->buffer, r.self)); return; }
    const uint64_t now = r.virtual_us;
    if (op == SA2S_AU_DS_GET_POSITION) {
        bool playing;
        const uint32_t play = ds_cursor(b.st, now, &playing);
        uint32_t write = play;
        if (b.st.bytes && playing) {
            uint64_t w = uint64_t(play) + b.st.write_lead;
            if (w >= b.st.bytes) w -= b.st.bytes;
            write = align_down(uint32_t(w), b.st.block_align ? b.st.block_align : 1u);
        }
        for (int k = 0; k < 2; k++) {
            const Sa2Blob* o = r.blob(SA2S_BLOB_OUT, k);
            if (!o || o->len < 4) continue;
            ds_cursor_checks_++;
            const uint32_t want = k ? write : play, got = rd32(o->data);
            if (got != want) {
                ds_cursor_mismatches_++;
                problem(where(r) + strf(": buffer 0x%08x's %s cursor was %u to the guest; the mixer's clock gives %u",
                                         r.self, k ? "write" : "play", got, want));
            }
        }
    } else if (op == SA2S_AU_DS_GET_STATUS) {
        bool playing;
        ds_cursor(b.st, now, &playing);
        const uint32_t want = (playing ? 1u : 0u) | (playing && b.st.looping ? 4u : 0u);
        if (const Sa2Blob* o = r.blob(SA2S_BLOB_OUT, 0); o && o->len >= 4) {
            ds_status_checks_++;
            if (rd32(o->data) != want) {
                ds_status_mismatches_++;
                problem(where(r) + strf(": buffer 0x%08x's status was 0x%x to the guest; the mixer's clock gives 0x%x",
                                         r.self, rd32(o->data), want));
            }
        }
    } else if (op == SA2S_AU_DS_SET_VOLUME) {
        b.volume = int32_t(r.arg(0));
    } else if (op == SA2S_AU_DS_SET_PAN) {
        b.pan = int32_t(r.arg(0));
    }
    b.st = *st;
    b.halted_seen = false;
    std::vector<uint8_t>* store = store_of(r, *st);
    if (op == SA2S_AU_DS_LOCK && st->mem) store_open_[st->mem]++;
    if (op == SA2S_AU_DS_UNLOCK) {
        for (int k : {0, 2}) {
            const uint32_t ptr = r.arg(size_t(k)), len = r.arg(size_t(k) + 1);
            if (!ptr || !len) continue;
            const Sa2Blob* pcm = r.blob(SA2S_BLOB_PCM, k);
            if (!pcm || pcm->len != len) {
                problem(where(r) + strf(": no %u-byte PCM blob for the segment at 0x%08x", len, ptr));
                continue;
            }
            if (st->primary) problem(where(r) + ": the guest wrote the primary buffer, which the mixer does not play");
            if (!store || ptr < st->mem || uint64_t(ptr - st->mem) + len > store->size()) {
                problem(where(r) + strf(": %u bytes at 0x%08x are outside buffer 0x%08x's store 0x%08x+%zu", len, ptr,
                                         r.self, st->mem, store ? store->size() : size_t(0)));
                continue;
            }
            std::memcpy(store->data() + (ptr - st->mem), pcm->data, len);
            ds_patches_++;
            ds_patch_bytes_ += len;
        }
        // The Unlock that closes its store's last Lock carries the recorder's hash of the
        // guest's whole store (host/sa2_stream.h, store_hash): this copy must hash the same.
        // A live session's (store_check_every_ > 1, mixer.h "The stores") holds the first such
        // Unlock with a hash and then one with a hash at least store_check_every_ closes after
        // the last it held, and counts the others, a close without a hash among them, as not
        // hashed: its recorder carries the hash on one close in 64 (SA2_REC_STORE_HASH_EVERY).
        // A HASH blob that is not 8 bytes is a problem in every mode.
        if (st->mem && !kBreakNoHash) {
            uint32_t& open = store_open_[st->mem];
            if (open) open--;
            if (!open) {
                const Sa2Blob* h = r.blob(SA2S_BLOB_HASH);
                const uint64_t close = ds_store_closes_++;
                const bool live = store_check_every_ > 1, malformed = h && h->len != 8;
                const bool due = !ds_store_checks_ || close - ds_store_checked_at_ >= store_check_every_;
                if (live && !malformed && (!h || !due)) {
                    ds_store_unhashed_++;
                } else {
                    ds_store_checks_++;
                    ds_store_checked_at_ = close;
                    if (!h || h->len != 8) {
                        ds_store_mismatches_++;
                        problem(where(r) + strf(": buffer 0x%08x's Unlock closes its store's last Lock and carries no "
                                                 "8-byte store_hash", r.self));
                    } else if (!store || fnv1a64(*store) != rd64(h->data)) {
                        ds_store_mismatches_++;
                        problem(where(r) + strf(": buffer 0x%08x's store 0x%08x, rebuilt from the stream's PCM, does "
                                                 "not hash to the guest's store after the Unlock (0x%016" PRIx64 "): "
                                                 "the stream does not hold what the guest wrote",
                                                 r.self, st->mem, rd64(h->data)));
                    }
                }
            }
        }
    }
}

// The store a buffer's STATE names, made the first time zero-filled, as the runtime's
// Lock makes it (runtime/sa2_dsound.c buf_alloc); nullptr for none, and nullptr and a
// problem (once) for a STATE whose mem_len is not its buffer's length: the runtime
// allocates each store at its buffer's length, so a stream that says otherwise is not
// one to size an allocation by.
std::vector<uint8_t>* Sa2Mixer::store_of(const Sa2Record& r, const sa2s_ds_state& st) {
    if (!st.mem) return nullptr;
    const auto it = stores_.find(st.mem);
    if (it != stores_.end()) return &it->second;
    if (st.mem_len != st.bytes) {
        if (bad_stores_.insert(st.mem).second)
            problem(where(r) + strf(": buffer 0x%08x's store 0x%08x is %u bytes and the buffer %u: the runtime makes "
                                     "a store its buffer's length", st.buffer, st.mem, st.mem_len, st.bytes));
        return nullptr;
    }
    std::vector<uint8_t>& s = stores_[st.mem];
    s.assign(st.mem_len, 0);
    return &s;
}

void Sa2Mixer::apply_xa(const Sa2Record& r, const Sa2AudioRow* row) {
    const uint32_t op = r.op;
    if (op == SA2S_AU_XA_CREATE_MASTER || op == SA2S_AU_XA_CREATE_SOURCE) {
        sa2s_xa_state stv;
        if (!xa_state(r, 0, &stv)) return;
        const sa2s_xa_state* st = &stv;
        const Sa2Blob* out = r.blob(SA2S_BLOB_OUT, 0);
        if (!out || out->len < 4 || rd32(out->data) != st->voice) {
            problem(where(r) + strf(": its STATE describes 0x%08x and the call wrote another voice", st->voice));
            return;
        }
        XaVoice v;
        v.st = *st;
        xa_[st->voice] = v;
        return;
    }
    auto it = xa_.find(r.self);
    if (it == xa_.end()) { problem(where(r) + strf(" (%s): voice 0x%08x, which no record made", row->name.c_str(), r.self)); return; }
    XaVoice& v = it->second;
    sa2s_xa_state stv;
    if (!xa_state(r, -1, &stv)) return;
    const sa2s_xa_state* st = &stv;
    if (st->voice != r.self) { problem(where(r) + strf(": its STATE describes 0x%08x, not 0x%08x", st->voice, r.self)); return; }
    const uint64_t qend_before = u64of(v.st.queued_end_lo, v.st.queued_end_hi);
    if (op == SA2S_AU_XA_GET_STATE) {
        // runtime/sa2_xaudio2.c voice_position and settle, from the last record.
        const sa2s_xa_state& p = v.st;
        uint64_t pos = u64of(p.played_lo, p.played_hi);
        if (p.running && p.queued) {
            const uint64_t anchor = u64of(p.anchor_us_lo, p.anchor_us_hi);
            const uint64_t el = r.virtual_us > anchor ? r.virtual_us - anchor : 0;
            pos = std::min(u64of(p.anchor_frame_lo, p.anchor_frame_hi) +
                               frames_after(el * kUnitsPerUs, p.samples_per_sec, p.ratio_bits).whole,
                           qend_before);
        }
        uint32_t queued = 0;
        for (const XaBuf& q : v.queue) queued += q.end > pos ? 1u : 0u;
        if (const Sa2Blob* o = r.blob(SA2S_BLOB_OUT, 0); o && o->len >= 16) {
            xa_state_checks_++;
            const uint32_t got_q = rd32(o->data + 4);
            const uint64_t got_p = u64of(rd32(o->data + 8), rd32(o->data + 12));
            if (got_q != queued || got_p != pos) {
                xa_state_mismatches_++;
                problem(where(r) + strf(": voice 0x%08x answered %u queued, %" PRIu64 " played; the mixer's clock gives "
                                         "%u, %" PRIu64, r.self, got_q, got_p, queued, pos));
            }
        }
    } else if (op == SA2S_AU_XA_SET_VOLUME) {
        v.volume_bits = r.arg(0);
    }
    v.st = *st;
    const uint64_t qend = u64of(st->queued_end_lo, st->queued_end_hi);
    if (qend < qend_before) {                    // a flush: the buffers past the new end are gone
        xa_flushed_frames_ += qend_before - qend;
        while (!v.queue.empty() && v.queue.back().start >= qend) v.queue.pop_back();
        if (!v.queue.empty() && v.queue.back().end > qend) {
            v.queue.back().data.resize(size_t(qend - v.queue.back().start) * st->channels);
            v.queue.back().end = qend;
        }
    }
    if (op == SA2S_AU_XA_SUBMIT) {
        const Sa2Blob* xb = r.blob(SA2S_BLOB_IN, 0);
        if (!xb || xb->len != 36) { problem(where(r) + ": no 36-byte XAUDIO2_BUFFER blob"); return; }
        const uint32_t bytes = rd32(xb->data + 4);
        const Sa2Blob* pcm = r.blob(SA2S_BLOB_PCM, 0);
        if (!pcm || pcm->len != bytes) { problem(where(r) + strf(": no %u-byte PCM blob", bytes)); return; }
        const uint32_t ba = st->block_align, ch = st->channels;
        const bool f32 = st->bits_per_sample == 32 && (st->format_tag == 3 || st->format_tag == 0xFFFEu);
        const bool s16 = st->bits_per_sample == 16 && (st->format_tag == 1 || st->format_tag == 0xFFFEu);
        if (!ba || !ch || bytes % ba || ba != ch * (st->bits_per_sample / 8u) || !(f32 || s16)) {
            problem(where(r) + strf(": %u bytes of format tag 0x%x, %u bits, %u channels, block %u: not whole frames the "
                                     "mixer decodes", bytes, st->format_tag, st->bits_per_sample, ch, ba));
            return;
        }
        const uint64_t frames = bytes / ba;
        if (frames > qend) { problem(where(r) + ": a submit past the voice's queue end"); return; }
        XaBuf q;
        q.start = qend - frames;
        q.end = qend;
        q.data.resize(size_t(frames) * ch);
        for (size_t i = 0; i < q.data.size(); i++)
            q.data[i] = f32 ? fbits(rd32(pcm->data + 4 * i)) : float(rd16s(pcm->data + 2 * i)) / 32768.0f;
        while (!v.queue.empty() && v.queue.back().end > q.start) {   // never, unless a flush went unrecorded
            problem(where(r) + ": a submit that overlaps the frames queued before it");
            v.queue.pop_back();
        }
        v.queue.push_back(std::move(q));
        xa_submits_++;
        xa_submit_bytes_ += bytes;
    }
}

std::string Sa2Mixer::json_fields() const {
    std::string ops;
    for (const auto& [name, n] : by_op_) ops += strf("%s\"%s\": %" PRIu64, ops.empty() ? "" : ", ", jesc(name).c_str(), n);
    std::string first;
    for (const auto& p : first_problems_) first += (first.empty() ? "\"" : ", \"") + jesc(p) + "\"";
    uint32_t ds_playing = 0, xa_running = 0;
    for (const auto& [a, b] : ds_) ds_playing += b.st.playing && !b.st.released && !b.halted_seen;
    for (const auto& [a, v] : xa_) xa_running += v.st.running && !v.st.destroyed;
    return strf("\"records\": %" PRIu64 ", \"by_op\": {%s}, \"sample_rate\": %u, \"channels\": %u, "
                "\"origin_us\": %" PRIu64 ", \"rendered_until_us\": %" PRIu64 ", \"samples\": %" PRIu64 ", "
                "\"late_records\": %" PRIu64 ", ",
                records_, ops.c_str(), kRate, kChannels, origin_us_, last_us_, n_, late_records_) +
           strf("\"ds\": {\"buffers\": %zu, \"primary_buffers\": %" PRIu64 ", \"stores\": %zu, \"playing_at_end\": %u, "
                "\"patches\": %" PRIu64 ", \"patch_bytes\": %" PRIu64 ", \"store_checks\": %" PRIu64
                ", \"store_mismatches\": %" PRIu64 ", \"store_check_every\": %u, \"stores_not_hashed\": %" PRIu64
                ", \"cursor_checks\": %" PRIu64
                ", \"cursor_mismatches\": %" PRIu64 ", \"status_checks\": %" PRIu64 ", \"status_mismatches\": %" PRIu64
                ", \"samples_without_store\": %" PRIu64 ", \"unsupported_format_samples\": %" PRIu64
                ", \"resampled_samples\": %" PRIu64 ", \"level\": ",
                ds_.size(), ds_primary_, stores_.size(), ds_playing, ds_patches_, ds_patch_bytes_, ds_store_checks_,
                ds_store_mismatches_, store_check_every_, ds_store_unhashed_, ds_cursor_checks_,
                ds_cursor_mismatches_, ds_status_checks_, ds_status_mismatches_, ds_no_store_samples_,
                ds_unsupported_samples_, ds_resampled_samples_) + ds_level_.json() + "}, " +
           strf("\"xa\": {\"voices\": %zu, \"running_at_end\": %u, \"submits\": %" PRIu64 ", \"submit_bytes\": %" PRIu64
                ", \"flushed_frames\": %" PRIu64 ", \"state_checks\": %" PRIu64 ", \"state_mismatches\": %" PRIu64
                ", \"starved_samples\": %" PRIu64 ", \"default_matrix_samples\": %" PRIu64
                ", \"resampled_samples\": %" PRIu64 ", \"level\": ",
                xa_.size(), xa_running, xa_submits_, xa_submit_bytes_, xa_flushed_frames_, xa_state_checks_,
                xa_state_mismatches_, xa_starved_samples_, xa_default_matrix_samples_, xa_resampled_samples_) +
           xa_level_.json() + "}, \"mix\": {\"level\": " + mix_level_.json() +
           strf(", \"clipped_samples\": %" PRIu64 ", \"limiter\": {\"on\": %s, \"limited_frames\": %" PRIu64
                ", \"least_gain\": ", clipped_, limiter_ ? "true" : "false", limited_) +
           num(lim_least_) + "}}, " +
           strf("\"problems\": {\"count\": %" PRIu64 ", \"first\": [%s]}", problems_, first.c_str());
}

// ------------------------------------------------------------------ WAV

namespace {
void le32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
void le16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }
// RIFF, a WAVE_FORMAT_IEEE_FLOAT fmt chunk of 18 bytes, a fact chunk, the data chunk:
// 58 bytes before the samples.
void wav_header(uint8_t h[58], uint64_t frames) {
    const uint64_t data = frames * 8u;
    const uint32_t d32 = data > 0xFFFFFFFFull - 58u ? 0xFFFFFFFFu - 58u : uint32_t(data);
    std::memcpy(h, "RIFF", 4);
    le32(h + 4, 50u + d32);
    std::memcpy(h + 8, "WAVEfmt ", 8);
    le32(h + 16, 18u);
    le16(h + 20, 3u);                    // WAVE_FORMAT_IEEE_FLOAT
    le16(h + 22, 2u);
    le32(h + 24, Sa2Mixer::kRate);
    le32(h + 28, Sa2Mixer::kRate * 8u);
    le16(h + 32, 8u);
    le16(h + 34, 32u);
    le16(h + 36, 0u);                    // cbSize
    std::memcpy(h + 38, "fact", 4);
    le32(h + 42, 4u);
    le32(h + 46, frames > 0xFFFFFFFFull ? 0xFFFFFFFFu : uint32_t(frames));
    std::memcpy(h + 50, "data", 4);
    le32(h + 54, d32);
}
}  // namespace

bool Sa2Wav::open(const std::string& path, std::string* err) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) { *err = "cannot write " + path; failed_ = true; return false; }
    uint8_t h[58];
    wav_header(h, 0);
    if (std::fwrite(h, 1, sizeof h, f_) != sizeof h) { *err = "cannot write " + path; failed_ = true; return false; }
    return true;
}

bool Sa2Wav::write(const float* interleaved, size_t frames) {
    if (!f_ || failed_) return false;
    if (frames && std::fwrite(interleaved, sizeof(float) * 2, frames, f_) != frames) { failed_ = true; return false; }
    frames_ += frames;
    return true;
}

bool Sa2Wav::flush() {
    if (!f_ || failed_) return false;
    uint8_t h[58];
    wav_header(h, frames_);
    const long at = std::ftell(f_);
    if (at < 0 || std::fseek(f_, 0, SEEK_SET) != 0 || std::fwrite(h, 1, sizeof h, f_) != sizeof h ||
        std::fseek(f_, at, SEEK_SET) != 0 || std::fflush(f_) != 0)
        failed_ = true;
    return !failed_;
}

bool Sa2Wav::close() {
    if (!f_) return !failed_;
    flush();
    if (std::fclose(f_) != 0) failed_ = true;
    f_ = nullptr;
    return !failed_;
}

// ------------------------------------------------------------------ the renderer

#ifdef SA2_MIXER_MAIN
// sa2_audio_render STREAM --wav OUT.wav [--json OUT.json] [--limit] [--store-check-every N]: a
// stream's sound with no device, no SDL and no window. --limit and --store-check-every N are a
// live session's player's settings (player.cpp: the limiter, kLiveStoreCheckEvery), for the
// tests; without them every closing Unlock is held to its store_hash and one without is a
// problem. Every AUDIO record goes to the mixer and every FRAME closes a block, as
// host/player/player.cpp does at each Present; the WAV holds the blocks, and last what the
// records after the last FRAME rendered, so it holds every sample the mixer counts. The JSON holds the mixer's counters, the first FRAME's
// virtual time and the WAV's own numbers. Exit 0: no audio problem; 1: an audio problem (or a version 1
// stream, which holds no audio); 2: usage or a file that could not be written; 3: the
// stream refused at open or malformed.
#include <cstdlib>

int main(int argc, char** argv) {
    std::string stream, wav_path, json_path;
    bool limit = false;
    uint32_t check_every = 1;
    const char* usage = "usage: sa2_audio_render STREAM --wav OUT.wav [--json OUT.json] [--limit] "
                        "[--store-check-every N]\n";
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--wav" && i + 1 < argc) wav_path = argv[++i];
        else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (a == "--limit") limit = true;
        else if (a == "--store-check-every" && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long n = std::strtoul(argv[++i], &end, 10);
            if (!*argv[i] || *end || n < 1 || n > 65536) { std::fprintf(stderr, "%s", usage); return 2; }
            check_every = uint32_t(n);
        }
        else if (!a.empty() && a[0] != '-') stream = a;
        else { std::fprintf(stderr, "%s", usage); return 2; }
    }
    if (stream.empty() || wav_path.empty()) {
        std::fprintf(stderr, "%s", usage);
        return 2;
    }
    Sa2StreamReader rd;
    std::string err;
    const int op = rd.open(stream, &err);
    if (op != Sa2StreamReader::OPEN_OK) { std::fprintf(stderr, "%s\n", err.c_str()); return op == Sa2StreamReader::OPEN_IO ? 2 : 3; }
    Sa2Mixer mx;
    mx.set_limiter(limit);
    mx.set_store_check_every(check_every);
    Sa2Wav wav;
    if (!wav.open(wav_path, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    if (rd.version() != SA2S_VERSION_AUDIO) mx.problem("the stream is version 1: it holds no audio");
    Sa2Record r;
    uint64_t frames = 0, first_frame_us = 0, audio = 0;
    int got;
    while ((got = rd.next(&r, &err)) > 0) {
        if (r.kind == SA2S_AUDIO && rd.version() == SA2S_VERSION_AUDIO) {
            audio++;
            mx.on_audio(r, rd);
        } else if (r.kind == SA2S_FRAME) {
            if (!frames++) first_frame_us = r.virtual_us;
            mx.on_frame(r.virtual_us);
            wav.write(mx.block().data(), mx.block().size() / 2);
            mx.clear_block();
        }
    }
    if (got < 0) { std::fprintf(stderr, "malformed: %s\n", err.c_str()); return 3; }
    wav.write(mx.block().data(), mx.block().size() / 2);       // after the last FRAME
    mx.clear_block();
    const bool wav_ok = wav.close();
    if (!json_path.empty()) {
        FILE* f = std::fopen(json_path.c_str(), "w");
        if (!f) return 2;
        std::fprintf(f, "{\"tool\": \"host/player sa2_audio_render\", \"stream\": \"%s\", \"version\": %u, "
                        "\"frames\": %" PRIu64 ", \"first_frame_us\": %" PRIu64 ", \"audio_records\": %" PRIu64
                        ", \"wav\": {\"path\": \"%s\", \"frames\": %" PRIu64 ", \"bytes\": %" PRIu64 ", \"ok\": %s}, "
                        "\"audio\": {%s}}\n",
                     jesc(stream).c_str(), rd.version(), frames, first_frame_us, audio, jesc(wav_path).c_str(),
                     wav.frames(), wav.bytes(), wav_ok ? "true" : "false", mx.json_fields().c_str());
        std::fclose(f);
    }
    std::printf("audio: %" PRIu64 " records, %" PRIu64 " frames, %" PRIu64 " samples from %" PRIu64 " us; %" PRIu64
                " problem(s)\n", audio, frames, mx.samples(), mx.origin_us(), mx.problems());
    if (!wav_ok) return 2;
    return mx.problems() ? 1 : 0;
}
#endif
