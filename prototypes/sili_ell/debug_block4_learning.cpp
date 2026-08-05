// Sanity check: does block4_weight_update actually reduce loss over many
// steps, the way disldo's own stochastic-rounding fix was validated
// earlier this session? Small regression task: fit one block4 layer to
// match a fixed target layer's output, watch loss trend down.
#include "../../sili/lib/headers/dense_block4.hpp"
#include <cstdio>
#include <random>

int main() {
    fp4_seed_stochastic_rng(7);
    const uint32_t n_out = 16, n_in = 16;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u01(0, 1), xv(-1, 1);
    std::uniform_int_distribution<int> magpick(1, 7);

    auto build_random = [&](std::mt19937& r) {
        std::vector<uint32_t> ptrs(n_out + 1, 0), idx;
        std::vector<float> w, imp;
        for (uint32_t row = 0; row < n_out; ++row) {
            idx.push_back(0); w.push_back(6.0f); imp.push_back(0.5f);   // guarantee row_scale=1
            for (uint32_t c = 1; c < n_in; ++c) {
                if (u01(r) < 0.5f) {
                    idx.push_back(c);
                    w.push_back(FP4_TABLE[magpick(r)] * (u01(r) < 0.5f ? -1.f : 1.f));
                    imp.push_back(0.5f);
                }
            }
            ptrs[row + 1] = uint32_t(idx.size());
        }
        return split_for_block4(n_out, n_in, ptrs, idx, w, imp, 0.10f);
    };

    std::mt19937 target_rng(2), student_rng(3);
    auto target = build_random(target_rng).block4;
    auto student = build_random(student_rng).block4;

    const int N = 800;
    const float lr = 0.05f;
    double loss_first10 = 0, loss_last10 = 0;
    std::vector<float> losses;
    for (int step = 0; step < N; ++step) {
        std::vector<float> x(n_in);
        for (auto& v : x) v = xv(rng);

        std::vector<float> y_target(n_out, 0.f), y_student(n_out, 0.f);
        block4_forward(target, x.data(), y_target.data(), 1);
        block4_forward(student, x.data(), y_student.data(), 1);

        float loss = 0;
        std::vector<float> dy(n_out);
        for (uint32_t i = 0; i < n_out; ++i) {
            float err = y_student[i] - y_target[i];
            loss += err * err;
            dy[i] = 2.0f * err / n_out;   // dL/dy_student
        }
        loss /= n_out;
        losses.push_back(loss);

        block4_weight_update(student, x.data(), dy.data(), lr, 1);

        if (step < 10) loss_first10 += loss;
        if (step >= N - 10) loss_last10 += loss;
    }
    loss_first10 /= 10; loss_last10 /= 10;
    printf("loss: first10_avg=%.4f  last10_avg=%.4f  ratio=%.4f (should be < 1.0 if learning works)\n",
           loss_first10, loss_last10, loss_last10 / loss_first10);
    printf("%s\n", (loss_last10 < loss_first10 * 0.5) ? "PASS (loss dropped by >2x)" : "FAIL or inconclusive");
    return 0;
}
