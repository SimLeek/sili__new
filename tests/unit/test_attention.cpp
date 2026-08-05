#include "../../sili/lib/headers/attention.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>
#include <numeric>
#include <random>

// ── Helper: naive dense attention (reference) ─────────────────────────────────

static std::vector<float> naive_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::size_t T, std::size_t d)
{
    const float scale = 1.0f / std::sqrt(float(d));
    std::vector<float> out(T * d, 0.0f);
    for (std::size_t q = 0; q < T; ++q) {
        // dot products
        std::vector<float> s(T);
        float max_s = -1e38f;
        for (std::size_t k = 0; k < T; ++k) {
            float dot = 0.0f;
            for (std::size_t i = 0; i < d; ++i) dot += Q[q*d+i] * K[k*d+i];
            s[k] = dot * scale;
            if (s[k] > max_s) max_s = s[k];
        }
        float sum_e = 0.0f;
        for (float& v : s) { v = std::exp(v - max_s); sum_e += v; }
        for (float& v : s) v /= sum_e;
        for (std::size_t k = 0; k < T; ++k)
            for (std::size_t i = 0; i < d; ++i)
                out[q*d+i] += s[k] * V[k*d+i];
    }
    return out;
}

// ── Helper: finite-difference gradient check ─────────────────────────────────
// Perturbs each element of x by +/-eps and computes the numerical dL/dx[i]
// via (f(x+eps) - f(x-eps)) / (2*eps), where f sums the output * dO.
// Returns max absolute difference between numerical and analytical gradient.

static float fd_grad_check(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    const std::vector<float>& dO,   // upstream gradient
    std::size_t T, std::size_t d,
    // analytical grads
    const std::vector<float>& dQ_anal,
    const std::vector<float>& dK_anal,
    const std::vector<float>& dV_anal,
    float eps = 1e-3f)
{
    float max_err = 0.0f;
    // Check only a few elements per matrix to keep the test fast.
    auto check = [&](const std::vector<float>& X, const std::vector<float>& dX_anal,
                     std::size_t rows, auto fwd_fn) {
        for (std::size_t r = 0; r < rows; r += 2) {
            for (std::size_t i = 0; i < d; i += 3) {
                auto Xp = X, Xm = X;
                Xp[r*d+i] += eps; Xm[r*d+i] -= eps;
                auto op = fwd_fn(Xp), om = fwd_fn(Xm);
                float num = 0.0f;
                for (std::size_t j = 0; j < T*d; ++j) num += (op[j] - om[j]) * dO[j];
                num /= 2.0f * eps;
                float err = std::abs(num - dX_anal[r*d+i]);
                if (err > max_err) max_err = err;
            }
        }
    };
    // dQ
    check(Q, dQ_anal, T, [&](const std::vector<float>& Qp) {
        std::vector<float> o(T*d, 0.0f);
        banded_attention_forward(Qp.data(), K.data(), V.data(), o.data(), T, T, d, T, 1);
        return o;
    });
    // dK
    check(K, dK_anal, T, [&](const std::vector<float>& Kp) {
        std::vector<float> o(T*d, 0.0f);
        banded_attention_forward(Q.data(), Kp.data(), V.data(), o.data(), T, T, d, T, 1);
        return o;
    });
    // dV
    check(V, dV_anal, T, [&](const std::vector<float>& Vp) {
        std::vector<float> o(T*d, 0.0f);
        banded_attention_forward(Q.data(), K.data(), Vp.data(), o.data(), T, T, d, T, 1);
        return o;
    });
    return max_err;
}

// ── Forward tests (ported from pre-merge test_sili.py) ───────────────────────

TEST_CASE("banded_attention: full bandwidth matches naive dense attention",
         "[attention][forward]") {
    const std::size_t T = 6, d = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out(T*d, 0.0f);
    banded_attention_forward(Q.data(), K.data(), V.data(), out.data(), T, T, d, T, 1);
    auto ref = naive_attention(Q, K, V, T, d);

    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out[i] == Catch::Approx(ref[i]).margin(1e-4f));
}

TEST_CASE("banded_attention: weights sum to one (all-ones V gives all-ones output)",
         "[attention][forward]") {
    const std::size_t T = 8, d = 6;
    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d, 2.5f);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);

    std::vector<float> out(T*d, 0.0f);
    banded_attention_forward(Q.data(), K.data(), V.data(), out.data(), T, T, d, 2, 1);
    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out[i] == Catch::Approx(2.5f).margin(1e-5f));
}

TEST_CASE("sparse_banded_attention: inner_k=0 matches banded_attention exactly",
         "[attention][forward]") {
    const std::size_t T = 8, d = 12;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out_banded(T*d, 0.0f), out_sparse(T*d, 0.0f);
    banded_attention_forward(Q.data(), K.data(), V.data(), out_banded.data(), T, T, d, 3, 1);
    sparse_banded_attention_forward(Q.data(), K.data(), V.data(), out_sparse.data(), T, T, d, 3, 0, 1);

    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out_banded[i] == Catch::Approx(out_sparse[i]).margin(1e-5f));
}

TEST_CASE("sparse_attention: full top-k (k=T) matches naive dense attention",
         "[attention][forward]") {
    const std::size_t T = 4, d = 8;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out(T*d, 0.0f);
    sparse_attention_forward(Q.data(), K.data(), V.data(), out.data(), T, d, T, 1);
    auto ref = naive_attention(Q, K, V, T, d);

    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out[i] == Catch::Approx(ref[i]).margin(1e-4f));
}

// ── Backward tests ────────────────────────────────────────────────────────────
//
// Core property: attention is NOT self-adjoint, so dQ/dK/dV are all
// non-trivially different, and there were no backward ops in the pre-merge
// codebase -- written from scratch. Verified via finite-difference gradient
// checks (numerical vs analytical dL/dX for each of Q, K, V).

TEST_CASE("banded_attention_backward: dV is correct (finite-difference check)",
         "[attention][backward][regression]") {
    // dL/dV[k] = sum_q(attn[q,k] * dO[q]) -- the simplest of the three,
    // no softmax Jacobian. Tested separately to catch it specifically.
    const std::size_t T = 4, d = 4;
    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d), dO(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> dQ(T*d,0), dK(T*d,0), dV(T*d,0);
    banded_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                              dQ.data(), dK.data(), dV.data(), T, T, d, T, 1);

    // Numerical dL/dV[k, i] via f(V+eps) - f(V-eps)) / 2eps, where f = sum(out * dO).
    const float eps = 1e-3f;
    for (std::size_t k = 0; k < T; k += 1) {
        for (std::size_t i = 0; i < d; i += 1) {
            auto Vp = V, Vm = V;
            Vp[k*d+i] += eps; Vm[k*d+i] -= eps;
            std::vector<float> op(T*d,0), om(T*d,0);
            banded_attention_forward(Q.data(), K.data(), Vp.data(), op.data(), T, T, d, T, 1);
            banded_attention_forward(Q.data(), K.data(), Vm.data(), om.data(), T, T, d, T, 1);
            float num = 0.0f;
            for (std::size_t j = 0; j < T*d; ++j) num += (op[j]-om[j]) * dO[j];
            num /= 2.0f * eps;
            CHECK(dV[k*d+i] == Catch::Approx(num).margin(5e-3f));
        }
    }
}

TEST_CASE("banded_attention_backward: dQ and dK match finite differences",
         "[attention][backward][regression]") {
    // Uses full bandwidth (same as dense attention) so the gradient is
    // analytically well-understood and easy to verify numerically.
    const std::size_t T = 4, d = 4;
    std::mt19937 rng(2);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d), dO(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> dQ(T*d,0), dK(T*d,0), dV(T*d,0);
    banded_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                              dQ.data(), dK.data(), dV.data(), T, T, d, T, 1);

    const float eps = 1e-3f;
    const float tol = 5e-3f;
    for (std::size_t r = 0; r < T; ++r) {
        for (std::size_t i = 0; i < d; ++i) {
            // dQ check
            auto Qp = Q, Qm = Q;
            Qp[r*d+i] += eps; Qm[r*d+i] -= eps;
            std::vector<float> op(T*d,0), om(T*d,0);
            banded_attention_forward(Qp.data(), K.data(), V.data(), op.data(), T, T, d, T, 1);
            banded_attention_forward(Qm.data(), K.data(), V.data(), om.data(), T, T, d, T, 1);
            float numQ = 0.0f;
            for (std::size_t j = 0; j < T*d; ++j) numQ += (op[j]-om[j]) * dO[j];
            numQ /= 2.0f * eps;
            CHECK(dQ[r*d+i] == Catch::Approx(numQ).margin(tol));
            // dK check
            auto Kp = K, Km = K;
            Kp[r*d+i] += eps; Km[r*d+i] -= eps;
            std::fill(op.begin(), op.end(), 0.0f);
            std::fill(om.begin(), om.end(), 0.0f);
            banded_attention_forward(Q.data(), Kp.data(), V.data(), op.data(), T, T, d, T, 1);
            banded_attention_forward(Q.data(), Km.data(), V.data(), om.data(), T, T, d, T, 1);
            float numK = 0.0f;
            for (std::size_t j = 0; j < T*d; ++j) numK += (op[j]-om[j]) * dO[j];
            numK /= 2.0f * eps;
            CHECK(dK[r*d+i] == Catch::Approx(numK).margin(tol));
        }
    }
}

TEST_CASE("sparse_attention_backward: finite-difference check (full k=T case)",
         "[attention][backward][regression]") {
    // With k=T, sparse_attention_forward is equivalent to dense attention,
    // so the backward gradient should also be exact.
    const std::size_t T = 4, d = 4;
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d), dO(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> dQ(T*d,0), dK(T*d,0), dV(T*d,0);
    sparse_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                              dQ.data(), dK.data(), dV.data(), T, d, T, 1);

    const float eps = 1e-3f, tol = 5e-3f;
    for (std::size_t r = 0; r < T; ++r) {
        for (std::size_t i = 0; i < d; ++i) {
            auto Qp = Q, Qm = Q;
            Qp[r*d+i] += eps; Qm[r*d+i] -= eps;
            std::vector<float> op(T*d,0), om(T*d,0);
            sparse_attention_forward(Qp.data(), K.data(), V.data(), op.data(), T, d, T, 1);
            sparse_attention_forward(Qm.data(), K.data(), V.data(), om.data(), T, d, T, 1);
            float num = 0.0f;
            for (std::size_t j = 0; j < T*d; ++j) num += (op[j]-om[j]) * dO[j];
            CHECK(dQ[r*d+i] == Catch::Approx(num / (2.0f*eps)).margin(tol));
        }
    }
}

TEST_CASE("sparse_banded_attention_backward: inner_k=0 backward matches banded_attention_backward",
         "[attention][backward]") {
    // When inner_k=0 (all band keys), the backward should match banded_backward exactly.
    const std::size_t T = 6, d = 6;
    std::mt19937 rng(9);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d), dO(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> dQ_b(T*d,0), dK_b(T*d,0), dV_b(T*d,0);
    std::vector<float> dQ_s(T*d,0), dK_s(T*d,0), dV_s(T*d,0);
    const std::size_t bw = 2;
    banded_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                              dQ_b.data(), dK_b.data(), dV_b.data(), T, T, d, bw, 1);
    sparse_banded_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                                     dQ_s.data(), dK_s.data(), dV_s.data(), T, T, d, bw, 0, 1);

    for (std::size_t i = 0; i < T*d; ++i) {
        CHECK(dQ_b[i] == Catch::Approx(dQ_s[i]).margin(1e-4f));
        CHECK(dK_b[i] == Catch::Approx(dK_s[i]).margin(1e-4f));
        CHECK(dV_b[i] == Catch::Approx(dV_s[i]).margin(1e-4f));
    }
}

// ── Causal masking ──────────────────────────────────────────────────────────
//
// None of the three variants mask the future by default -- band/top-k
// selection is purely geometric/L2-norm-driven and can pick keys on either
// side of a query. causal=true restricts each query to keys at or before
// its own sequence position (self-attention only, T == K).

static std::vector<float> naive_causal_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::size_t T, std::size_t d)
{
    const float scale = 1.0f / std::sqrt(float(d));
    std::vector<float> out(T * d, 0.0f);
    for (std::size_t q = 0; q < T; ++q) {
        std::vector<float> s(q + 1);
        float max_s = -1e38f;
        for (std::size_t k = 0; k <= q; ++k) {
            float dot = 0.0f;
            for (std::size_t i = 0; i < d; ++i) dot += Q[q*d+i] * K[k*d+i];
            s[k] = dot * scale;
            if (s[k] > max_s) max_s = s[k];
        }
        float sum_e = 0.0f;
        for (float& v : s) { v = std::exp(v - max_s); sum_e += v; }
        for (float& v : s) v /= sum_e;
        for (std::size_t k = 0; k <= q; ++k)
            for (std::size_t i = 0; i < d; ++i)
                out[q*d+i] += s[k] * V[k*d+i];
    }
    return out;
}

TEST_CASE("banded_attention: causal with a full-width band matches naive causal attention",
         "[attention][forward][causal]") {
    const std::size_t T = 6, d = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out(T*d, 0.0f);
    banded_attention_forward(Q.data(), K.data(), V.data(), out.data(), T, T, d, T, 1, true);
    auto ref = naive_causal_attention(Q, K, V, T, d);

    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out[i] == Catch::Approx(ref[i]).margin(1e-4f));
}

TEST_CASE("banded_attention: causal output at t does not depend on keys/values after t",
         "[attention][forward][causal][regression]") {
    // Direct leakage check, independent of the naive-reference comparison
    // above: perturbing K/V at a future position must not change an
    // earlier query's output at all.
    const std::size_t T = 6, d = 8;
    std::mt19937 rng(19);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    const std::size_t t = 2;   // query position under test
    std::vector<float> out_before(T*d, 0.0f);
    banded_attention_forward(Q.data(), K.data(), V.data(), out_before.data(), T, T, d, T, 1, true);

    auto K2 = K, V2 = V;
    for (std::size_t k = t + 1; k < T; ++k)
        for (std::size_t i = 0; i < d; ++i) { K2[k*d+i] += 5.0f; V2[k*d+i] += 5.0f; }

    std::vector<float> out_after(T*d, 0.0f);
    banded_attention_forward(Q.data(), K2.data(), V2.data(), out_after.data(), T, T, d, T, 1, true);

    for (std::size_t i = 0; i < d; ++i)
        CHECK(out_before[t*d+i] == Catch::Approx(out_after[t*d+i]).margin(1e-5f));
}

TEST_CASE("sparse_banded_attention: causal inner_k=0 matches banded_attention causal",
         "[attention][forward][causal]") {
    const std::size_t T = 8, d = 12;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out_banded(T*d, 0.0f), out_sparse(T*d, 0.0f);
    banded_attention_forward(Q.data(), K.data(), V.data(), out_banded.data(), T, T, d, 3, 1, true);
    sparse_banded_attention_forward(Q.data(), K.data(), V.data(), out_sparse.data(), T, T, d, 3, 0, 1, true);

    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out_banded[i] == Catch::Approx(out_sparse[i]).margin(1e-5f));
}

TEST_CASE("sparse_attention: causal full top-k (k=T) matches naive causal attention",
         "[attention][forward][causal]") {
    const std::size_t T = 6, d = 8;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out(T*d, 0.0f);
    sparse_attention_forward(Q.data(), K.data(), V.data(), out.data(), T, d, T, 1, true);
    auto ref = naive_causal_attention(Q, K, V, T, d);

    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out[i] == Catch::Approx(ref[i]).margin(1e-4f));
}

TEST_CASE("sparse_attention: causal output at t does not depend on keys/values after t",
         "[attention][forward][causal][regression]") {
    // sparse_attention selects queries/keys globally by L2 norm, so a
    // "selected" key can land on either side of any given query -- this is
    // the case most likely to leak the future if causal masking is wrong.
    const std::size_t T = 6, d = 8;
    std::mt19937 rng(23);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    const std::size_t t = 1;
    std::vector<float> out_before(T*d, 0.0f);
    sparse_attention_forward(Q.data(), K.data(), V.data(), out_before.data(), T, d, T, 1, true);

    auto K2 = K, V2 = V;
    for (std::size_t k = t + 1; k < T; ++k)
        for (std::size_t i = 0; i < d; ++i) { K2[k*d+i] += 5.0f; V2[k*d+i] += 5.0f; }

    std::vector<float> out_after(T*d, 0.0f);
    sparse_attention_forward(Q.data(), K2.data(), V2.data(), out_after.data(), T, d, T, 1, true);

    for (std::size_t i = 0; i < d; ++i)
        CHECK(out_before[t*d+i] == Catch::Approx(out_after[t*d+i]).margin(1e-5f));
}

TEST_CASE("banded_attention_backward: causal gradients match finite differences",
         "[attention][backward][causal]") {
    const std::size_t T = 6, d = 6;
    std::mt19937 rng(31);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d), dO(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> dQ(T*d,0), dK(T*d,0), dV(T*d,0);
    banded_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                              dQ.data(), dK.data(), dV.data(), T, T, d, T, 1, true);

    float max_err = 0.0f;
    const float eps = 1e-3f;
    auto fwd = [&](const std::vector<float>& Qp, const std::vector<float>& Kp,
                  const std::vector<float>& Vp) {
        std::vector<float> o(T*d, 0.0f);
        banded_attention_forward(Qp.data(), Kp.data(), Vp.data(), o.data(), T, T, d, T, 1, true);
        return o;
    };
    for (std::size_t r = 0; r < T; r += 2) {
        for (std::size_t i = 0; i < d; i += 2) {
            auto Qp = Q, Qm = Q; Qp[r*d+i] += eps; Qm[r*d+i] -= eps;
            auto op = fwd(Qp, K, V), om = fwd(Qm, K, V);
            float num = 0.0f;
            for (std::size_t j = 0; j < T*d; ++j) num += (op[j]-om[j]) * dO[j];
            num /= 2.0f*eps;
            max_err = std::max(max_err, std::abs(num - dQ[r*d+i]));
        }
    }
    CHECK(max_err < 1e-2f);
}

// ── gaussian_attention: full attention with a learnable per-query Gaussian bias

TEST_CASE("gaussian_attention: sigma->0 with center on an integer key concentrates output on that key's V",
         "[attention][gaussian]") {
    const std::size_t K = 5, d = 3;
    std::mt19937 rng(40);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(d), Kmat(K*d), V(K*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : Kmat) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    const std::size_t j0 = 2;
    std::vector<float> centers = {float(j0)};
    std::vector<float> sigmas  = {1e-3f};
    std::vector<float> out(d, 0.0f);
    gaussian_attention_forward(Q.data(), Kmat.data(), V.data(), out.data(),
                               1, K, d, centers.data(), sigmas.data(), 1, false);

    for (std::size_t i = 0; i < d; ++i)
        CHECK(out[i] == Catch::Approx(V[j0*d+i]).margin(1e-4));
}

TEST_CASE("gaussian_attention: with sigma very large, reduces to plain dense softmax attention (Gaussian bias vanishes)",
         "[attention][gaussian]") {
    const std::size_t T = 4, d = 4;
    std::mt19937 rng(41);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Q(T*d), K(T*d), V(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> centers(T, 0.0f);
    std::vector<float> sigmas(T, 1e6f);   // bias term ~0 for any finite (j-center)
    std::vector<float> out(T*d, 0.0f);
    gaussian_attention_forward(Q.data(), K.data(), V.data(), out.data(),
                               T, T, d, centers.data(), sigmas.data(), 1, false);

    auto naive = naive_attention(Q, K, V, T, d);
    for (std::size_t i = 0; i < T*d; ++i)
        CHECK(out[i] == Catch::Approx(naive[i]).margin(1e-3));
}

TEST_CASE("gaussian_attention_backward: dCenters and dSigmas match finite differences",
         "[attention][gaussian][backward]") {
    const std::size_t T = 5, K = 6, d = 4;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> center_dist(0.0f, float(K));
    std::uniform_real_distribution<float> sigma_dist(0.5f, 2.0f);

    std::vector<float> Q(T*d), Kmat(K*d), V(K*d), dO(T*d);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : Kmat) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    for (auto& v : dO) v = dist(rng);
    std::vector<float> centers(T), sigmas(T);
    for (auto& v : centers) v = center_dist(rng);
    for (auto& v : sigmas)  v = sigma_dist(rng);

    std::vector<float> dQ(T*d,0), dK(K*d,0), dV(K*d,0), dC(T,0), dS(T,0);
    gaussian_attention_backward(Q.data(), Kmat.data(), V.data(), dO.data(),
                                dQ.data(), dK.data(), dV.data(), dC.data(), dS.data(),
                                T, K, d, centers.data(), sigmas.data(), 1, false);

    auto fwd_loss = [&](const std::vector<float>& c, const std::vector<float>& s) {
        std::vector<float> o(T*d, 0.0f);
        gaussian_attention_forward(Q.data(), Kmat.data(), V.data(), o.data(),
                                   T, K, d, c.data(), s.data(), 1, false);
        float loss = 0.0f;
        for (std::size_t j = 0; j < T*d; ++j) loss += o[j] * dO[j];
        return loss;
    };

    const float eps = 1e-3f;
    float max_err = 0.0f;
    for (std::size_t t = 0; t < T; ++t) {
        auto cp = centers, cm = centers;
        cp[t] += eps; cm[t] -= eps;
        float num_dc = (fwd_loss(cp, sigmas) - fwd_loss(cm, sigmas)) / (2.0f * eps);
        max_err = std::max(max_err, std::abs(num_dc - dC[t]));

        auto sp = sigmas, sm = sigmas;
        sp[t] += eps; sm[t] -= eps;
        float num_ds = (fwd_loss(centers, sp) - fwd_loss(centers, sm)) / (2.0f * eps);
        max_err = std::max(max_err, std::abs(num_ds - dS[t]));
    }
    CHECK(max_err < 1e-2f);
}
