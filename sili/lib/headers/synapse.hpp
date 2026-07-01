/**
 * @file synapse.hpp
 * @brief FP4 quantization, ULEB128 delta encoding, and the SynapseRecord
 *        fixed-width storage format for SILi's flat-buffer delta-CSR weights.
 *
 *  ▗▄▄▖▗▖  ▗▖ ▗▄▖ ▗▄▄▖  ▗▄▄▖▗▄▄▄▖
 * ▐▌   ▝▚▞▘▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌      █
 *  ▝▀▚▖ ▐▌ ▐▛▀▜▌▐▛▀▘  ▝▀▚▖ █
 * ▗▄▄▞▘ ▐▌ ▐▌ ▐▌▐▌   ▗▄▄▞▘ █
 *
 * Radiation model
 * ───────────────
 * In a 30 GB continuously-saved model, "radiation" means a random bit flip
 * anywhere in the flat weight buffer.  Two safety properties we want:
 *
 *   (a) A corrupted *value* (FP4 packed byte) only affects one synapse's
 *       weight and importance — it cannot corrupt column indices.
 *   (b) A corrupted *index delta* (ULEB128 bytes) only affects that one
 *       synapse's column — it cannot misalign the rest of the row because
 *       records are fixed-width; the next record always starts at a known
 *       byte offset.
 *
 * Both properties are guaranteed by the SynapseRecord layout.
 *
 * Layout per synapse (COL_TYPE = uint32_t, kDeltaBytes = 5):
 * ┌─ byte 0 ──────┬─ byte 1 ──────┬ ··· ┬─ byte 4 ──────┬─ byte 5 ────┐
 * │ ULEB128[0]    │ ULEB128[1]    │     │ ULEB128[4]    │ hi=w lo=imp │
 * └───────────────┴───────────────┴ ··· ┴───────────────┴─────────────┘
 *  ◄─────────── kDeltaBytes = uleb128_max_bytes<COL_TYPE>() ──────────►  ← 1 byte
 *  ◄──────────────────────── kSize = kDeltaBytes + 1 ─────────────────────────►
 *
 * The ULEB128 uses only as many bytes as needed; remaining delta bytes are
 * zeroed.  The decoder stops at the first byte with bit-7 clear, so trailing
 * zeros (0x00 = "stop, value contribution 0") are harmless.
 */

#ifndef SILI_SYNAPSE_HPP
#define SILI_SYNAPSE_HPP

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// ── FP4 lookup table ─────────────────────────────────────────────────────────
// Index layout: 0=zero, 1-7=positive, 8=NaN (treated as 0), 9-15=negative.
// Two values pack into one byte (high nibble / low nibble) with no spanning.

static constexpr float FP4_TABLE[16] = {
     0.0f,    // 0000
     0.5f,    // 0001
     1.0f,    // 0010
     1.5f,    // 0011
     2.0f,    // 0100
     3.0f,    // 0101
     4.0f,    // 0110
     6.0f,    // 0111
     0.0f,    // 1000  NaN slot — treated as zero on read, never written
    -0.5f,    // 1001
    -1.0f,    // 1010
    -1.5f,    // 1011
    -2.0f,    // 1100
    -3.0f,    // 1101
    -4.0f,    // 1110
    -6.0f,    // 1111
};

/// Nearest-neighbour quantize @p v to a 4-bit FP4 index. Skips NaN slot (8).
inline uint8_t fp4_quantize(float v) {
    uint8_t best     = 0;
    float   best_err = std::abs(v - FP4_TABLE[0]);
    for (uint8_t i = 1; i < 16; ++i) {
        if (i == 8) continue;                        // skip NaN slot
        const float err = std::abs(v - FP4_TABLE[i]);
        if (err < best_err) { best_err = err; best = i; }
    }
    return best;
}

// ── ULEB128 ───────────────────────────────────────────────────────────────────

/// Maximum bytes to encode T as ULEB128.
/// For uint32_t: ceil(32/7) = 5.  For uint16_t: 3.
template <typename T = uint32_t>
constexpr std::size_t uleb128_max_bytes() {
    return (sizeof(T) * 8u + 6u) / 7u;
}

/// Encode @p value into @p buf as ULEB128.  Returns bytes written.
/// Caller must ensure buf has at least uleb128_max_bytes<T>() bytes.
template <typename T = uint32_t>
inline std::size_t uleb128_encode(T value, uint8_t* buf) {
    std::size_t n = 0;
    do {
        uint8_t byte = static_cast<uint8_t>(value & 0x7Fu);
        value >>= 7;
        if (value) byte |= 0x80u;
        buf[n++] = byte;
    } while (value);
    return n;
}

/// Decode one ULEB128 value from @p buf starting at byte offset *pos.
/// Advances *pos past the last byte consumed.
template <typename T = uint32_t>
inline T uleb128_decode(const uint8_t* buf, std::size_t& pos) {
    T result = 0;
    int shift = 0;
    uint8_t byte;
    do {
        byte    = buf[pos++];
        result |= static_cast<T>(byte & 0x7Fu) << shift;
        shift  += 7;
    } while (byte & 0x80u);
    return result;
}

// ── SynapseRecord ─────────────────────────────────────────────────────────────

template <typename T = uint32_t>
struct Synapse{
    struct ElemRef {
        uint8_t* _byte_ptr;
        bool     _hi;

        operator float() const {
            return FP4_TABLE[_hi ? (*_byte_ptr >> 4) : (*_byte_ptr & 0xF)];
        }
        ElemRef& operator=(float v) {
            const uint8_t idx = fp4_quantize(v);
            if (_hi) *_byte_ptr = (*_byte_ptr & 0x0Fu) | static_cast<uint8_t>(idx << 4u);
            else     *_byte_ptr = (*_byte_ptr & 0xF0u) | idx;
            return *this;
        }
        ElemRef& operator=(const ElemRef& other) { return *this = float(other); }
        ElemRef& operator+=(float v) { return *this = float(*this) + v; }
        ElemRef& operator-=(float v) { return *this = float(*this) - v; }
        ElemRef& operator*=(float v) { return *this = float(*this) * v; }
        ElemRef& operator/=(float v) { return *this = float(*this) / v; }
    };

    struct IndexRef {
        uint8_t* _buf;
        std::size_t _pos;
        std::size_t num_bytes=0;
        Synapse<T>& parent;

        inline std::size_t current_bytes() const {
            if (num_bytes > 0) return num_bytes;
            std::size_t temp_pos = _pos;
            uleb128_decode<T>(_buf, temp_pos);
            return temp_pos - _pos;
        }

        operator const T() {
            std::size_t temp_pos = _pos;
            return uleb128_decode<T>(_buf, temp_pos);
        }
        IndexRef& operator=(T value) {
            uint8_t tmp[uleb128_max_bytes<T>()];
            std::size_t new_len = uleb128_encode<T>(value, tmp);
            std::size_t old_len = current_bytes();

            if (new_len == old_len) {
                // Simple path: size hasn't changed
                std::memcpy(_buf + _pos, tmp, new_len);
            } else {
                // Complex path: shift required
                if (new_len > old_len && parent._growth_length < (new_len - old_len)) {
                    throw std::runtime_error("Cannot grow synapse record: no growth length remaining");
                }

                std::size_t shift_start = _pos + old_len;
                std::size_t bytes_to_shift = parent._row_end - shift_start;

                std::memmove(_buf + _pos + new_len, _buf + shift_start, bytes_to_shift);
                std::memcpy(_buf + _pos, tmp, new_len);

                // IMPORTANT: Update parent offsets because the buffer just shifted!
                std::ptrdiff_t diff = new_len - old_len;
                parent._val_offset   += diff;
                parent.next_synapse  += diff;
                parent._row_end      += diff;
                parent._growth_length -= diff; 
                num_bytes = new_len; // update cached length
            }
            return *this; // Fixed: return statement added
        }
    };

    uint8_t* record;
    std::size_t _ind_offset; // Actual offset where the ULEB128 index starts
    std::size_t _val_offset; // Replaces uint8_t& _val
    std::size_t next_synapse;
    std::size_t _row_start;
    std::size_t _row_end;
    std::size_t _growth_length;

    static Synapse<T> unpack(uint8_t* record, std::size_t& index, std::size_t row_start, std::size_t row_end, std::size_t growth_length) {
        Synapse<T> syn;
        syn.record = record;
        
        syn._ind_offset = index; // Store byte offset of the index
        uleb128_decode(record, index);
        
        syn._val_offset = index; // Store byte offset of the weights
        index++;                 // Move past the weights byte
        
        syn.next_synapse = index; 
        syn._row_start = row_start;
        syn._row_end = row_end;
        syn._growth_length = growth_length;
        
        return syn;
    }

    ElemRef weight() { return ElemRef{&record[_val_offset], true}; }
    ElemRef importance() { return ElemRef{&record[_val_offset], false}; }
    IndexRef delta_index() { return IndexRef{record, _ind_offset, 0, *this}; }
};


/**
 * @brief Fixed-width POD record for one synapse in the flat weight buffer.
 *
 * The record is always kSize bytes.  The first kDeltaBytes bytes hold the
 * ULEB128-encoded column *delta* from the previous synapse in the same row
 * (zero-padded).  The last byte is the FP4BiPacked value:
 *   high nibble = weight FP4 index, low nibble = importance FP4 index.
 *
 * Being fixed-width means:
 *  - Record k in row r lives at buf[(neuron.record_start + k) * kSize].
 *  - Weight/importance are always at offset kDeltaBytes within the record.
 *    → random-access reads/writes require no column scan.
 *  - A corrupt delta byte cannot shift subsequent records.
 *
 * @tparam COL_TYPE  Unsigned integer type for column indices (default uint32_t).
 */
template <typename COL_TYPE = uint32_t>
struct SynapseRecord {
    static constexpr std::size_t kDeltaBytes = uleb128_max_bytes<COL_TYPE>();
    static constexpr std::size_t kSize       = kDeltaBytes + 1u;

    uint8_t delta[kDeltaBytes];  ///< ULEB128 column delta, zero-padded to kDeltaBytes
    uint8_t packed;              ///< high nibble = weight FP4 idx, low nibble = importance FP4 idx

    // ── value accessors ───────────────────────────────────────────────────────

    float weight()     const { return FP4_TABLE[(packed >> 4) & 0xFu]; }
    float importance() const { return FP4_TABLE[ packed        & 0xFu]; }

    void set_weight(float w) {
        packed = static_cast<uint8_t>((fp4_quantize(w) << 4u) | (packed & 0x0Fu));
    }
    void set_importance(float imp) {
        packed = static_cast<uint8_t>((packed & 0xF0u) | fp4_quantize(imp));
    }
    void set(float w, float imp) {
        packed = static_cast<uint8_t>((fp4_quantize(w) << 4u) | fp4_quantize(imp));
    }

    // ── delta accessors ───────────────────────────────────────────────────────

    /// Encode @p delta_val into the delta field.  Zeros remainder bytes so
    /// the decoder cannot read stale data.
    void encode_delta(COL_TYPE delta_val) {
        std::memset(delta, 0, kDeltaBytes);
        uleb128_encode<COL_TYPE>(delta_val, delta);
    }

    /// Decode the column delta from this record.
    COL_TYPE decode_delta() const {
        std::size_t pos = 0;
        return uleb128_decode<COL_TYPE>(delta, pos);
    }
};

// Statically verify no compiler-inserted padding.
static_assert(sizeof(SynapseRecord<uint32_t>) == 6,  "SynapseRecord<uint32_t> must be 6 bytes");
static_assert(sizeof(SynapseRecord<uint16_t>) == 4,  "SynapseRecord<uint16_t> must be 4 bytes");
static_assert(alignof(SynapseRecord<uint32_t>) == 1, "SynapseRecord must have alignment 1");

// ── FP4BiPacked ───────────────────────────────────────────────────────────────
// Legacy two-lane packed storage used by CSRSynapses (old shared_ptr-CSR path).
// New code should store values inline in SynapseRecord instead.
//
// Two FP4 values per byte: high nibble = values[0] (weight),
//                           low nibble = values[1] (importance/strength).

struct FP4BiPacked {
    using storage_type = std::vector<uint8_t>;
    using size_type    = std::size_t;

    struct ElemRef {
        uint8_t& _byte;
        bool     _hi;

        operator float() const {
            return FP4_TABLE[_hi ? (_byte >> 4) : (_byte & 0xF)];
        }
        ElemRef& operator=(float v) {
            const uint8_t idx = fp4_quantize(v);
            if (_hi) _byte = (_byte & 0x0Fu) | static_cast<uint8_t>(idx << 4u);
            else     _byte = (_byte & 0xF0u) | idx;
            return *this;
        }
        ElemRef& operator=(const ElemRef& other) { return *this = float(other); }
        ElemRef& operator+=(float v) { return *this = float(*this) + v; }
        ElemRef& operator-=(float v) { return *this = float(*this) - v; }
        ElemRef& operator*=(float v) { return *this = float(*this) * v; }
        ElemRef& operator/=(float v) { return *this = float(*this) / v; }
    };

    struct Lane {
        std::shared_ptr<std::vector<uint8_t>>* _dp;
        bool _hi;

        // ── shared_ptr-like interface ─────────────────────────────────────────
        explicit operator bool() const { return bool(*_dp); }

        Lane&       operator*()        { return *this; }
        const Lane& operator*()  const { return *this; }
        Lane*       operator->()       { return this; }
        const Lane* operator->() const { return this; }

        /// Assigning make_shared<vector<T>> ensures the backing store exists.
        /// Both lanes always share one array.
        template <class T>
        Lane& operator=(std::shared_ptr<std::vector<T>>) {
            if (!*_dp)
                *_dp = std::make_shared<std::vector<uint8_t>>();
            return *this;
        }

        // ── vector-like interface ─────────────────────────────────────────────
        ElemRef operator[](std::size_t i) { return ElemRef{(**_dp)[i], _hi}; }
        float   operator[](std::size_t i) const {
            const uint8_t b = (**_dp)[i];
            return FP4_TABLE[_hi ? (b >> 4) : (b & 0xF)];
        }

        std::size_t size()     const { return *_dp ? (*_dp)->size()     : 0; }
        std::size_t capacity() const { return *_dp ? (*_dp)->capacity() : 0; }
        bool        empty()    const { return !*_dp || (*_dp)->empty(); }

        void reserve(std::size_t n) {
            if (!*_dp) *_dp = std::make_shared<std::vector<uint8_t>>();
            (*_dp)->reserve(n);
        }
        void resize(std::size_t n, float v = 0.0f) {
            if (!*_dp) *_dp = std::make_shared<std::vector<uint8_t>>();
            const std::size_t old = (*_dp)->size();
            (*_dp)->resize(n, uint8_t(0));
            if (v != 0.0f) {
                const uint8_t idx = fp4_quantize(v);
                for (std::size_t i = old; i < n; ++i) {
                    auto& b = (**_dp)[i];
                    if (_hi) b = (b & 0x0Fu) | static_cast<uint8_t>(idx << 4u);
                    else     b = (b & 0xF0u) | idx;
                }
            }
        }
        /// push_back allocates one new byte (both nibbles share it).
        /// Call on one lane per element; set the other nibble via operator[].
        void push_back(float v) {
            if (!*_dp) *_dp = std::make_shared<std::vector<uint8_t>>();
            const uint8_t idx  = fp4_quantize(v);
            const uint8_t byte = _hi ? static_cast<uint8_t>(idx << 4u) : idx;
            (*_dp)->push_back(byte);
        }
        void clear() { if (*_dp) (*_dp)->clear(); }
    };

    // ── state ─────────────────────────────────────────────────────────────────
    // _data declared before _lanes so &_data is valid in initialiser lists.

    std::shared_ptr<std::vector<uint8_t>> _data;
    Lane _lanes[2];

    // ── constructors ──────────────────────────────────────────────────────────
    FP4BiPacked()
        : _data(), _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    explicit FP4BiPacked(std::vector<uint8_t>&& data)
        : _data(std::make_shared<std::vector<uint8_t>>(std::move(data)))
        , _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    FP4BiPacked(const FP4BiPacked& o)
        : _data(o._data), _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    FP4BiPacked(FP4BiPacked&& o) noexcept
        : _data(std::move(o._data)), _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    FP4BiPacked& operator=(const FP4BiPacked& o) {
        if (this != &o) _data = o._data;
        return *this;
    }
    FP4BiPacked& operator=(FP4BiPacked&& o) noexcept {
        if (this != &o) _data = std::move(o._data);
        return *this;
    }

    // ── class-level vector interface (touches both nibbles in one store) ──────
    void reserve(std::size_t n) {
        if (!_data) _data = std::make_shared<std::vector<uint8_t>>();
        _data->reserve(n);
    }
    void resize(std::size_t n, float weight = 0.0f, float importance = 0.0f) {
        if (!_data) _data = std::make_shared<std::vector<uint8_t>>();
        const std::size_t old = _data->size();
        _data->resize(n, uint8_t(0));
        if (weight != 0.0f || importance != 0.0f) {
            const uint8_t fill = static_cast<uint8_t>(
                (fp4_quantize(weight) << 4u) | fp4_quantize(importance));
            for (std::size_t j = old; j < n; ++j) (*_data)[j] = fill;
        }
    }
    void push_back(float weight, float importance) {
        if (!_data) _data = std::make_shared<std::vector<uint8_t>>();
        _data->push_back(static_cast<uint8_t>(
            (fp4_quantize(weight) << 4u) | fp4_quantize(importance)));
    }
    static void move(FP4BiPacked& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0 || !v._data) return;
        std::memmove(v._data->data() + dest, v._data->data() + src, count);
    }
    void clear() { if (_data) _data->clear(); }

    std::vector<uint8_t> serialize() const {
        return _data ? *_data : std::vector<uint8_t>{};
    }
    static FP4BiPacked deserialize(std::vector<uint8_t>&& buffer) {
        return FP4BiPacked(std::move(buffer));
    }

    Lane&       operator[](std::size_t i)       { return _lanes[i]; }
    const Lane& operator[](std::size_t i) const { return _lanes[i]; }

    std::size_t size()  const { return _data ? _data->size()  : 0; }
    bool        empty() const { return !_data || _data->empty(); }
};

// tuple_size specialisation so sparse_struct::n_value_arrays constexpr compiles.
namespace std {
    template <> struct tuple_size<FP4BiPacked>
        : integral_constant<size_t, 2> {};
}

#endif // SILI_SYNAPSE_HPP