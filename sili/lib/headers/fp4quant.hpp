#ifndef __FP4_HPP_
#define __FP4_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// ── FP4 lookup table — "FP4 All the Way" ─────────────────────────────────────
// Index layout: 0=zero, 1-7=positive, 8=NaN (treated as 0), 9-15=negative.
// Fits in 4 bits; two values pack exactly into one byte with no spanning.

static constexpr float FP4_TABLE[16] = {
     0.0f,    // 0000
     0.5f,    // 0001
     1.0f,    // 0010
     1.5f,    // 0011
     2.0f,    // 0100
     3.0f,    // 0101
     4.0f,    // 0110
     6.0f,    // 0111
     NAN,     // 1000
    -0.5f,    // 1001
    -1.0f,   // 1010
    -1.5f,   // 1011
    -2.0f,   // 1100
    -3.0f,   // 1101
    -4.0f,   // 1110
    -6.0f,   // 1111
};

///Nearest-neighbour quantize @p v to a 4-bit FP4 index. NaN slot (8) is skipped.
inline uint8_t fp4_quantize(float v) {
    uint8_t best     = 0;
    float   best_err = std::abs(v - FP4_TABLE[0]);
    for (uint8_t i = 1; i < 16; ++i) {
        if (i == 8) continue;
        const float err = std::abs(v - FP4_TABLE[i]);
        if (err < best_err) { best_err = err; best = i; }
    }
    return best;
}

// ── FP4BiPacked ───────────────────────────────────────────────────────────────
// Two FP4 values per byte: high nibble = values[0] (weight),
//                           low nibble = values[1] (importance/strength).
// One byte per connection — no bit-spanning, no cross-byte logic, no masking beyond 0xF.

struct FP4BiPacked {
    using storage_type = std::vector<uint8_t>;
    using size_type = std::size_t;

    struct ElemRef {
        uint8_t& _byte;
        bool     _hi;
 
        operator float() const {
            return FP4_TABLE[_hi ? (_byte >> 4) : (_byte & 0xF)];
        }
        ElemRef& operator=(float v) {
            const uint8_t idx = fp4_quantize(v);
            if (_hi) _byte = (_byte & 0x0F) | (idx << 4);
            else     _byte = (_byte & 0xF0) |  idx;
            return *this;
        }
        ElemRef& operator=(const ElemRef& other) { return *this = float(other); }
        ElemRef& operator+=(float v) { return *this = (float)*this + v; }
        ElemRef& operator-=(float v) { return *this = (float)*this - v; }
        ElemRef& operator*=(float v) { return *this = (float)*this * v; }
        ElemRef& operator/=(float v) { return *this = (float)*this / v; }
        
    };

    struct Lane {
        std::shared_ptr<std::vector<uint8_t>>* _dp;
        bool _hi;
 
        // ── shared_ptr-like interface ─────────────────────────────────────────
 
        explicit operator bool() const { return bool(*_dp); }
 
        ///operator* and operator-> return *this — Lane is both ptr and vector.
        Lane&       operator*()        { return *this; }
        const Lane& operator*()  const { return *this; }
        Lane*       operator->()       { return this; }
        const Lane* operator->() const { return this; }
 
        ///Assignment from make_shared<vector<T>> ensures the backing store exists.
        ///Both lanes always share one array; the assigned ptr's contents are ignored.
        template <class T>
        Lane& operator=(std::shared_ptr<std::vector<T>>) {
            if (!*_dp)
                *_dp = std::make_shared<std::vector<uint8_t>>();
            return *this;
        }
 
        // ── vector-like interface ─────────────────────────────────────────────
 
        ElemRef operator[](std::size_t i) {
            return ElemRef{(**_dp)[i], _hi};
        }
        float operator[](std::size_t i) const {
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
                    if (_hi) b = (b & 0x0F) | (idx << 4);
                    else     b = (b & 0xF0) |  idx;
                }
            }
        }
        ///push_back allocates one new byte (both nibbles share it).
        ///Call on one lane per element; set the other nibble via operator[].
        void push_back(float v) {
            if (!*_dp) *_dp = std::make_shared<std::vector<uint8_t>>();
            const uint8_t idx  = fp4_quantize(v);
            const uint8_t byte = _hi ? uint8_t(idx << 4) : idx;
            (*_dp)->push_back(byte);
        }
        void clear() { if (*_dp) (*_dp)->clear(); }
    };
 
    // ── state ─────────────────────────────────────────────────────────────────
    // _data declared before _lanes so &_data is valid in initializer lists.
 
    std::shared_ptr<std::vector<uint8_t>> _data;
    Lane _lanes[2];
 
    // ── constructors ──────────────────────────────────────────────────────────
    // Each ctor rebuilds _lanes pointing to this->_data.
    // Copy/move assignment only touches _data; _lanes already point to it.
 
    FP4BiPacked()
        : _data()
        , _lanes{Lane{&_data, true}, Lane{&_data, false}}
    {}

    FP4BiPacked(std::vector<uint8_t>&& data)
        : _data(std::make_shared<std::vector<uint8_t>>(std::move(data)))
        , _lanes{Lane{&_data, true}, Lane{&_data, false}}
    {}
 
    FP4BiPacked(const FP4BiPacked& o)
        : _data(o._data)
        , _lanes{Lane{&_data, true}, Lane{&_data, false}}
    {}
 
    FP4BiPacked(FP4BiPacked&& o) noexcept
        : _data(std::move(o._data))
        , _lanes{Lane{&_data, true}, Lane{&_data, false}}
    {}
 
    FP4BiPacked& operator=(const FP4BiPacked& o) {
        if (this != &o) _data = o._data;       // _lanes._dp = &_data already correct
        return *this;
    }
    FP4BiPacked& operator=(FP4BiPacked&& o) noexcept {
        if (this != &o) _data = std::move(o._data);
        return *this;
    }

    // ── class-level vector interface ─────────────────────────────────────────────
    // Mirrors Lane's interface but operates on both nibbles simultaneously.
    // Prefer these over lane[0]/lane[1] calls when weight and importance
    // are known together — avoids touching the backing store twice.

    void reserve(std::size_t n) {
        if (!_data) _data = std::make_shared<std::vector<uint8_t>>();
        _data->reserve(n);
    }

    void resize(std::size_t n, float weight = 0.0f, float importance = 0.0f) {
        if (!_data) _data = std::make_shared<std::vector<uint8_t>>();
        const std::size_t old = _data->size();
        _data->resize(n, uint8_t(0));
        if (weight != 0.0f || importance != 0.0f) {
            const uint8_t fill = uint8_t((fp4_quantize(weight) << 4)
                                    |  fp4_quantize(importance));
            for (std::size_t j = old; j < n; ++j)
                (*_data)[j] = fill;
        }
    }

    void push_back(float weight, float importance) {
        if (!_data) _data = std::make_shared<std::vector<uint8_t>>();
        _data->push_back(uint8_t((fp4_quantize(weight) << 4)
                                |  fp4_quantize(importance)));
    }

    

    void clear() { if (_data) _data->clear(); }

    /// Serializes the internal data into a standard byte vector.
    std::vector<uint8_t> serialize() const {
        if (_data) {
            // Returns a copy of the underlying shared storage
            return *_data;
        }
        return std::vector<uint8_t>{};
    }

    /// Deserializes from a standard byte vector (moves the buffer to avoid a copy).
    static FP4BiPacked deserialize(std::vector<uint8_t>&& buffer) {
        return FP4BiPacked(std::move(buffer));
    }
 
    Lane&       operator[](std::size_t i)       { return _lanes[i]; }
    const Lane& operator[](std::size_t i) const { return _lanes[i]; }
 
    std::size_t size()  const { return _data ? _data->size()  : 0; }
    bool        empty() const { return !_data || _data->empty(); }
};

// tuple_size specialization so sparse_struct::n_value_arrays constexpr compiles.
namespace std {
    template <> struct tuple_size<FP4BiPacked>
        : integral_constant<size_t, 2> {};
}

#endif
