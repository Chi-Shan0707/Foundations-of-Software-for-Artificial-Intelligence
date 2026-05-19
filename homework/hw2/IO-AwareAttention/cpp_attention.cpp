// C++ Attention Benchmark: 仅测量 Attention 前向计算时间
// 编译: g++ -O2 -o attn_benchmark_cpp attn_benchmark_cpp.cpp
// 运行: ./attn_benchmark_cpp <N> <d>
//
// 复用 cpp_microgpt.cpp 的核心逻辑:
//   Value* 指针级运算 + softmax + 手动 dot product
//   测试目标: 移除计算图构建（跳过 children/local_grads），测定单纯量数值计算效率

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
using namespace std;

// ─── 从 cpp_microgpt.cpp 摘取的 softmax ────────────────────────────────
// 对应原版实现中的 softmax 函数
void softmax_inplace(double* x, int len) {
    double max_val = -1e9;
    for (int i = 0; i < len; ++i) if (x[i] > max_val) max_val = x[i];
    double sum_exp = 0.0;
    for (int i = 0; i < len; ++i) { x[i] = exp(x[i] - max_val); sum_exp += x[i]; }
    for (int i = 0; i < len; ++i) x[i] /= sum_exp;
}

/*
 * Attention 前向计算
 *
 * 提取自 cpp_microgpt.cpp AttentionBlock::forward 的前向计算逻辑:
 *
 *   // 原版 L128-134: 计算注意力分数
 *   double scale_factor = 1.0 / sqrt((double)head_dim);
 *   for(size_t t = 0; t < seq_len; ++t) {
 *       Value* dot = Value::make_new(0.0);
 *       for(size_t i = 0; i < q.data.size(); ++i)
 *           dot = Value::add(dot, Value::mul(q.data[i], keys[t].data[i]));
 *       scores.data[t] = Value::mul(dot, scale_node);
 *   }
 *
 *   // 原版 L135: softmax
 *   Vector weights = softmax(scores);
 *
 *   // 原版 L138-141: 加权求和
 *   for (size_t t = 0; t < seq_len; ++t) {
 *       Value* w = weights.data[t];
 *       for (size_t i = 0; i < dim_embd; ++i)
 *           output.data[i] = Value::add(output.data[i], Value::mul(w, values[t].data[i]));
 *   }
 *
 * 这里去掉了所有 Value* 计算图对象包装，替换为等价的底级 double 运算，
 * 用于评估剥离 Autograd 开销后的前向推理纯数学算力极值。
 */
void attention_forward(const double* Q, const double* K, const double* V,
                       double* O, int N, int d) {
    double scale_factor = 1.0 / sqrt((double)d); // 对应 cpp L129

    // scores[i][j] = dot(Q[i], K[j]) * scale_factor
    // 对应 cpp L128-134 的双重循环
    vector<double> scores(N);

    for (int i = 0; i < N; ++i) {
        // ── 1. 计算注意力分数: dot(q, keys[t]) * scale ──
        // 对应 cpp L130-133:
        //   Value* dot = Value::make_new(0.0);
        //   for(size_t i = 0; i < q.data.size(); ++i)
        //       dot = Value::add(dot, Value::mul(q.data[i], keys[t].data[i]));
        //   scores.data[t] = Value::mul(dot, scale_node);
        for (int t = 0; t < N; ++t) {
            double dot = 0.0;
            for (int ii = 0; ii < d; ++ii) {
                dot += Q[i * d + ii] * K[t * d + ii];
            }
            scores[t] = dot * scale_factor;
        }

        // ── 2. Softmax 归一化 ──
        // 对应 cpp L135: Vector weights = softmax(scores);
        softmax_inplace(scores.data(), N);

        // ── 3. 加权求和: output += weights[t] * values[t] ──
        // 对应 cpp L138-141:
        //   for (size_t t = 0; t < seq_len; ++t) {
        //       Value* w = weights.data[t];
        //       for (size_t i = 0; i < dim_embd; ++i)
        //           output.data[i] = Value::add(output.data[i], Value::mul(w, values[t].data[i]));
        //   }
        for (int ii = 0; ii < d; ++ii) {
            double sum = 0.0;
            for (int t = 0; t < N; ++t) {
                sum += scores[t] * V[t * d + ii];
            }
            O[i * d + ii] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16;
    int d = (argc > 2) ? atoi(argv[2]) : 16;
    int num_iters = (argc > 3) ? atoi(argv[3]) : 100;

    vector<double> Q(N * d), K(N * d), V(N * d), O(N * d);
    for (auto& x : Q) x = (double)rand() / RAND_MAX - 0.5;
    for (auto& x : K) x = (double)rand() / RAND_MAX - 0.5;
    for (auto& x : V) x = (double)rand() / RAND_MAX - 0.5;

    attention_forward(Q.data(), K.data(), V.data(), O.data(), N, d);

    auto start = chrono::high_resolution_clock::now();
    for (int it = 0; it < num_iters; ++it) {
        attention_forward(Q.data(), K.data(), V.data(), O.data(), N, d);
    }
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count() / num_iters;

    printf("%.6f", ms);
    return 0;
}
