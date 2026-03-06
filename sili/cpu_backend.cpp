#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lib/sparse_linear.cpp"
#include "lib/sparse_conv2d.cpp"
#include "lib/sparse_conv2d_over_on.cpp"
#include "lib/utils.cpp"

namespace py = pybind11;


/*
#include <sstream>
#include <string>
#include <array>
#include <cstdint>
#include <bit>
#include <stdfloat>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Endianness Helpers and Serialization Functions (same as before)
// -----------------------------------------------------------------------------

constexpr bool is_host_little_endian() {
    return std::endian::native == std::endian::little;
}

template <typename T>
T swap_endian(T val) {
    static_assert(std::is_arithmetic_v<T>, "swap_endian only supports arithmetic types.");
    union {
        T val;
        unsigned char bytes[sizeof(T)];
    } src, dest;
    src.val = val;
    for (std::size_t i = 0; i < sizeof(T); i++) {
        dest.bytes[i] = src.bytes[sizeof(T) - 1 - i];
    }
    return dest.val;
}

template <typename T>
void write_value(std::ostream& os, const T& val) {
    T tmp = val;
    if (!is_host_little_endian()) {
        tmp = swap_endian(tmp);
    }
    os.write(reinterpret_cast<const char*>(&tmp), sizeof(T));
}

template <typename T>
void read_value(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
    if (!is_host_little_endian()) {
        val = swap_endian(val);
    }
}

// Serialization for a single sparse_struct into an existing stream.
template <typename SIZE_TYPE, typename PTRS, typename INDICES, typename VALUES>
void serialize_sparse_struct(std::ostream& os, const sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES>& mat) {
    // Write dimensions and reserved space.
    write_value(os, mat.rows);
    write_value(os, mat.cols);
    write_value(os, mat._reserved_space);
    // Write nnz.
    SIZE_TYPE nnz = mat.nnz();
    write_value(os, nnz);
    // Write pointer arrays.
    if constexpr(is_std_array_v<decltype(mat.ptrs)>) {
        for (const auto& ptr : mat.ptrs) {
            for (std::size_t i = 0; i < static_cast<std::size_t>(mat.rows) + 1; ++i) {
                write_value(os, ptr.get()[i]);
            }
        }
    } else {
        write_value(os, mat.ptrs);
    }
    // Write indices arrays.
    for (const auto& idx_ptr : mat.indices) {
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            write_value(os, idx_ptr.get()[i]);
        }
    }
    // Write value arrays.
    using value_t = std::remove_pointer_t<decltype(mat.values[0].get())>;
    for (const auto& val_ptr : mat.values) {
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            write_value(os, val_ptr.get()[i]);
        }
    }
}

// Deserialization from an existing stream.
template <typename SIZE_TYPE, typename PTRS, typename INDICES, typename VALUES>
void deserialize_sparse_struct(std::istream& is, sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES>& mat) {
    read_value(is, mat.rows);
    read_value(is, mat.cols);
    read_value(is, mat._reserved_space);
    SIZE_TYPE nnz = 0;
    read_value(is, nnz);
    // Read pointer arrays.
    if constexpr(is_std_array_v<decltype(mat.ptrs)>) {
        for (auto& ptr : mat.ptrs) {
            ptr.reset(new SIZE_TYPE[mat.rows + 1]);
            for (std::size_t i = 0; i < static_cast<std::size_t>(mat.rows) + 1; ++i) {
                read_value(is, ptr.get()[i]);
            }
        }
    } else {
        read_value(is, mat.ptrs);
        nnz = mat.ptrs;
    }
    // Read indices arrays.
    for (auto& idx_ptr : mat.indices) {
        idx_ptr.reset(new SIZE_TYPE[nnz]);
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            read_value(is, idx_ptr.get()[i]);
        }
    }
    // Read value arrays.
    using value_t = std::remove_pointer_t<decltype(mat.values[0].get())>;
    for (auto& val_ptr : mat.values) {
        val_ptr.reset(new value_t[nnz]);
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            read_value(is, val_ptr.get()[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Pickle Support Functions for CSRInput
// -----------------------------------------------------------------------------

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes csr_input_getstate(const CSRInput<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline CSRInput<SIZE_TYPE, VALUE_TYPE> csr_input_setstate(py::bytes state) {
    std::string buffer = state;  // Implicit conversion to std::string.
    std::istringstream iss(buffer, std::ios::binary);
    CSRInput<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj);
    return obj;
}

// -----------------------------------------------------------------------------
// Pickle Support Functions for CSRSynapses
// -----------------------------------------------------------------------------

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes csrsynapses_getstate(const CSRSynapses<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline CSRSynapses<SIZE_TYPE, VALUE_TYPE> csrsynapses_setstate(py::bytes state) {
    std::string buffer = state;
    std::istringstream iss(buffer, std::ios::binary);
    CSRSynapses<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj);
    return obj;
}


//COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes coosynaptogenesis_getstate(const COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> coosynaptogenesis_setstate(py::bytes state) {
    std::string buffer = state;
    std::istringstream iss(buffer, std::ios::binary);
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj);
    return obj;
}

//SparseLinearWeights

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes SparseLinearWeights_getstate(const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self.connections);
    serialize_sparse_struct(oss, self.probes);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> SparseLinearWeights_setstate(py::bytes state) {
    std::string buffer = state;
    std::istringstream iss(buffer, std::ios::binary);
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj.connections);
    deserialize_sparse_struct(iss, obj.probes);
    return obj;
}

// -----------------------------------------------------------------------------
// Pybind11 Module Registration
// -----------------------------------------------------------------------------

template <typename SIZE_TYPE, typename VALUE_TYPE>
void declare_CSRInput(py::module &m, const std::string &size_typestr, const std::string &value_typestr) {
    // Create a unique class name, e.g., "CSRInput_uint32_double"
    std::string pyclass_name = "CSRInput_" + size_typestr + "_" + value_typestr;

    // Alias for the specific instantiation
    using CSRInput_t = CSRInput<SIZE_TYPE, VALUE_TYPE>;

    // Bind the class to Python
    py::class_<CSRInput_t>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def("nnz", &CSRInput_t::nnz)
        .def(py::pickle(
            // Serialization (getstate)
            [](const CSRInput_t &self) {
                std::ostringstream oss(std::ios::binary);
                serialize_sparse_struct(oss, self);
                return py::bytes(oss.str());
            },
            // Deserialization (setstate)
            [](py::bytes state) {
                std::string buffer = state;
                std::istringstream iss(buffer, std::ios::binary);
                CSRInput_t obj;
                deserialize_sparse_struct(iss, obj);
                return obj;
            }
        ));
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void declare_SparseLinearWeights(py::module &m, const std::string &size_typestr, const std::string &value_typestr) {
    std::string pyclass_name = "SparseLinearWeights_" + size_typestr + "_" + value_typestr;
    using SparseLinearWeights_t = SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>;

    py::class_<SparseLinearWeights_t>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def_readonly("connections", &SparseLinearWeights_t::connections)
        .def_readonly("probes", &SparseLinearWeights_t::probes)
        .def(py::pickle(
            [](const SparseLinearWeights_t &self) {
                std::ostringstream oss(std::ios::binary);
                serialize_sparse_struct(oss, self.connections);
                serialize_sparse_struct(oss, self.probes);
                return py::bytes(oss.str());
            },
            [](py::bytes state) {
                std::string buffer = state;
                std::istringstream iss(buffer, std::ios::binary);
                SparseLinearWeights_t obj;
                deserialize_sparse_struct(iss, obj.connections);
                deserialize_sparse_struct(iss, obj.probes);
                return obj;
            }
        ));
}

PYBIND11_MODULE(sparse_bindings, m) {
    m.doc() = "Bindings for sparse matrix structures with pickle support.";

    declare_CSRInput<uint64_t, float>(m, "u64", "f32");
    declare_CSRInput<uint32_t, float>(m, "u32", "f32");
    declare_CSRInput<uint16_t, float>(m, "u16", "f32");
    declare_CSRInput<uint8_t, float>(m, "u8", "f32");

    declare_SparseLinearWeights<uint64_t, float>(m, "u64", "f32");
    declare_SparseLinearWeights<uint32_t, float>(m, "u32", "f32");
    declare_SparseLinearWeights<uint16_t, float>(m, "u16", "f32");
    declare_SparseLinearWeights<uint8_t, float>(m, "u8", "f32");
}*/

class SparseTensorCPU {
public:
    using S = size_t;
    using V = float;

    SparseLinearWeights<S, V> weights;
    py::array_t<V>            _grad;   // Python-visible grad buffer (nnz,)
    std::pair<S, S>           _shape;

    SparseTensorCPU(SparseLinearWeights<S, V> w, std::pair<S, S> shape)
        : weights(std::move(w)), _shape(shape) {}

    // ── forward ──────────────────────────────────────────────────────────────

    py::array_t<V> mm(py::array_t<V> x) {
        auto xbuf   = x.request();
        S    batch  = (xbuf.ndim == 2) ? xbuf.shape[0] : 1;
        S    out_sz = batch * weights.connections.cols;

        py::array_t<V> output({(py::ssize_t)batch,
                                (py::ssize_t)weights.connections.cols});
        auto obuf = output.request();
        std::fill((V*)obuf.ptr, (V*)obuf.ptr + out_sz, V(0));

        // Build a CSRInput view over x's numpy buffer
        CSRInput<S, V> input_view = make_csr_input_view(x, batch);

        sparse_linear_csr_csc_forward(
            input_view,
            weights,
            (V*)obuf.ptr,
            /*train=*/false
        );
        return output;
    }

    // ── backward: gradient w.r.t. input x ────────────────────────────────────

    py::array_t<V> backward_x(py::array_t<V> x, py::array_t<V> dy) {
        auto xbuf  = x.request();
        S    batch = (xbuf.ndim == 2) ? xbuf.shape[0] : 1;
        S    in_sz = batch * _shape.second;

        py::array_t<V> dx({(py::ssize_t)batch,
                            (py::ssize_t)_shape.second});
        auto dxbuf = dx.request();
        std::fill((V*)dxbuf.ptr, (V*)dxbuf.ptr + in_sz, V(0));

        CSRInput<S, V> input_view = make_csr_input_view(x, batch);

        sparse_linear_csr_csc_backward(
            input_view,
            weights,
            (V*)dy.request().ptr,
            (V*)dxbuf.ptr
        );
        return dx;
    }

    // ── backward: gradient w.r.t. nonzero values ─────────────────────────────

    py::array_t<V> backward_vals(py::array_t<V> x, py::array_t<V> dy) {
        // The weight grad is accumulated into weights.connections.values[1]
        // during backward_x (same kernel call).  We expose it here as a
        // fresh numpy array so the Python _acc_sparse helper can accumulate it.
        S nnz = weights.connections.nnz();
        py::array_t<V> dvals(nnz);
        auto buf = dvals.request();
        std::copy(
            weights.connections.values[1].get(),
            weights.connections.values[1].get() + nnz,
            (V*)buf.ptr
        );
        // Reset the C++-side accumulator so it doesn't double-count.
        std::fill(
            weights.connections.values[1].get(),
            weights.connections.values[1].get() + nnz,
            V(0)
        );
        return dvals;
    }

    // ── vals property (nonzero weights, shape (nnz,)) ────────────────────────

    py::array_t<V> get_vals() {
        S nnz = weights.connections.nnz();
        // Zero-copy view into the underlying buffer.
        return py::array_t<V>(
            {(py::ssize_t)nnz},
            {sizeof(V)},
            weights.connections.values[0].get(),
            py::cast(this)   // keep 'this' alive as long as the array lives
        );
    }

    void set_vals(py::array_t<V> v) {
        auto buf = v.request();
        S    nnz = weights.connections.nnz();
        if ((S)buf.size != nnz)
            throw std::runtime_error("vals size mismatch");
        std::copy((V*)buf.ptr, (V*)buf.ptr + nnz,
                  weights.connections.values[0].get());
    }

    // ── grad property ─────────────────────────────────────────────────────────

    py::object get_grad() {
        if (_grad.size() == 0)
            return py::none();
        return _grad;
    }

    void set_grad(py::object g) {
        if (g.is_none()) {
            _grad = py::array_t<V>();
        } else {
            _grad = g.cast<py::array_t<V>>();
        }
    }

    // ── shape ─────────────────────────────────────────────────────────────────

    std::pair<S, S> shape() const { return _shape; }

    // ── device movement ───────────────────────────────────────────────────────

    // to(device_str): CPU → CPU is always a no-op.
    // Cross-device moves are handled by the Python backend calling to_host()
    // on this object, then the target backend's from_host().
    SparseTensorCPU& to(const std::string& device) {
        if (device != "cpu" && device.substr(0, 3) != "cpu")
            throw std::runtime_error(
                "SparseTensorCPU.to(): use backend.move() for cross-device transfers.");
        return *this;
    }

    // Returns a plain dict of numpy arrays that the Python layer can hand to
    // another backend's from_host() without knowing the internal layout.
    py::dict to_host() {
        S nnz  = weights.connections.nnz();
        S rows = _shape.first;

        auto vals = py::array_t<V>(nnz, weights.connections.values[0].get());
        auto grad = py::array_t<V>(nnz, weights.connections.values[1].get());
        auto crow = py::array_t<S>(rows + 1, weights.connections.ptrs[0].get());
        auto col  = py::array_t<S>(nnz, weights.connections.indices[0].get());

        py::dict d;
        d["vals"]  = vals;
        d["grad"]  = grad;
        d["crow"]  = crow;
        d["col"]   = col;
        d["shape"] = py::make_tuple(_shape.first, _shape.second);
        return d;
    }

private:
    // Build a CSRInput that borrows from a dense numpy array.
    // This lets us reuse the sparse kernel even with a dense input
    // (the CSRInput view treats every element as a nonzero).
    CSRInput<S, V> make_csr_input_view(py::array_t<V>& x, S batch) {
        // Delegate to whatever helper is in utils.cpp / csr.hpp.
        // Adjust to match your actual CSRInput construction API.
        return dense_to_csr_input_view<S, V>(
            (V*)x.request().ptr,
            batch,
            _shape.second
        );
    }
};


// ─────────────────────────────────────────────────────────────────────────────
//  Module
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_cpu, m)
{
    // ── SparseTensorCPU class ─────────────────────────────────────────────────
    //
    // This is what SparseLinear (and _is_sparse()) work with on the Python side.
    // All other C++ linear variants (sidlso_olist, sidlso_scipy, …) are exposed
    // as module-level functions below for direct use when needed.

    py::class_<SparseTensorCPU>(m, "SparseTensorCPU")
        .def("mm",            &SparseTensorCPU::mm)
        .def("backward_x",    &SparseTensorCPU::backward_x)
        .def("backward_vals", &SparseTensorCPU::backward_vals)
        .def("to",            &SparseTensorCPU::to,  py::return_value_policy::reference)
        .def("to_host",       &SparseTensorCPU::to_host)
        .def_property("vals", &SparseTensorCPU::get_vals, &SparseTensorCPU::set_vals)
        .def_property("grad", &SparseTensorCPU::get_grad, &SparseTensorCPU::set_grad)
        .def_property_readonly("shape", &SparseTensorCPU::shape)
        .def("zero_grad", [](SparseTensorCPU& self){
            self._grad = py::array_t<float>();
        });

    // ── Factory: make_sparse_tensor(dense_numpy_array) → SparseTensorCPU ─────
    //
    // Called from SparseMLP's weight_factory and from_dense() equivalent.
    // Converts a dense (rows, cols) numpy array to a SparseTensorCPU.

    m.def("make_sparse_tensor", [](py::array_t<float> dense) {
        auto buf  = dense.request();
        size_t rows = buf.shape[0], cols = buf.shape[1];
        auto weights = dense_to_sparse_linear_weights<size_t, float>(
            (float*)buf.ptr, rows, cols
        );
        return SparseTensorCPU(std::move(weights), {rows, cols});
    });

    // ── Low-level module functions (existing API, kept for direct use) ────────

    // sidlso = sparse input, dense layer, sparse output
    m.def("linear_sidlso_olist",          &sparse_linear_vectorized_forward_wrapper);
    m.def("linear_sidlso_backward_olist", &sparse_linear_vectorized_backward_wrapper);
    m.def("linear_sidlso_scipy",          &sparse_linear_vectorized_forward_wrapper);
    m.def("linear_sidlso_backward_scipy", &sparse_linear_vectorized_backward_wrapper);
    m.def("linear_sidlso",                &sparse_linear_vectorized_forward_wrapper);
    m.def("linear_sidlso_backward",       &sparse_linear_vectorized_backward_wrapper);
}