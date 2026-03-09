#pragma once
#include <cstddef>
#include <vector>
#include <cmath>

// ── MSE loss and gradient ─────────────────────────────────────────────────────
// loss = (1/n) * sum((desired - output)^2)
// grad = -2 * (desired - output) / n

// ── Serial ────────────────────────────────────────────────────────────────────

template <typename VALUE_TYPE>
VALUE_TYPE mse_loss(
    const VALUE_TYPE* output,
    const VALUE_TYPE* desired,
    const size_t n)
{
    VALUE_TYPE acc = VALUE_TYPE(0);
    for (size_t i = 0; i < n; ++i) {
        const VALUE_TYPE d = desired[i] - output[i];
        acc += d * d;
    }
    return acc / VALUE_TYPE(n);
}

template <typename VALUE_TYPE>
void mse_grad(
    const VALUE_TYPE* output,
    const VALUE_TYPE* desired,
    VALUE_TYPE*       grad_out,
    const size_t      n)
{
    const VALUE_TYPE scale = VALUE_TYPE(-2) / VALUE_TYPE(n);
    for (size_t i = 0; i < n; ++i)
        grad_out[i] = scale * (desired[i] - output[i]);
}

// ── Vector overloads (convenience, allocates) ─────────────────────────────────

template <typename VALUE_TYPE>
VALUE_TYPE mse_loss(
    const std::vector<VALUE_TYPE>& output,
    const std::vector<VALUE_TYPE>& desired)
{
    return mse_loss(output.data(), desired.data(), output.size());
}

template <typename VALUE_TYPE>
std::vector<VALUE_TYPE> mse_grad(
    const std::vector<VALUE_TYPE>& output,
    const std::vector<VALUE_TYPE>& desired)
{
    std::vector<VALUE_TYPE> g(output.size());
    mse_grad(output.data(), desired.data(), g.data(), output.size());
    return g;
}

// ── Parallel variants ─────────────────────────────────────────────────────────

template <typename VALUE_TYPE>
VALUE_TYPE mse_loss_parallel(
    const VALUE_TYPE* output,
    const VALUE_TYPE* desired,
    const size_t      n,
    const int         num_cpus)
{
    VALUE_TYPE acc = VALUE_TYPE(0);
    #pragma omp parallel for num_threads(num_cpus) schedule(static) reduction(+:acc)
    for (size_t i = 0; i < n; ++i) {
        const VALUE_TYPE d = desired[i] - output[i];
        acc += d * d;
    }
    return acc / VALUE_TYPE(n);
}

template <typename VALUE_TYPE>
void mse_grad_parallel(
    const VALUE_TYPE* output,
    const VALUE_TYPE* desired,
    VALUE_TYPE*       grad_out,
    const size_t      n,
    const int         num_cpus)
{
    const VALUE_TYPE scale = VALUE_TYPE(-2) / VALUE_TYPE(n);
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (size_t i = 0; i < n; ++i)
        grad_out[i] = scale * (desired[i] - output[i]);
}