/*
 * 比较4种方式计算Fibonacci的性能：
 * 1. Python 普通递归
 * 2. Python Lambda递归 (Church数)
 * 3. C++ 普通递归
 * 4. C++ Lambda递归
 */

#include <iostream>
#include <chrono>
#include <functional>

using namespace std;
using namespace std::chrono;

// ============================================================
// C++ 普通递归
// ============================================================

long long fib_normal(long long n) {
    if (n <= 1) return n;
    return fib_normal(n - 1) + fib_normal(n - 2);
}

// ============================================================
// C++ Lambda递归 (模仿Church数)
// ============================================================

// Church数类型: 接受一个函数f，返回一个函数
using Church = function<function<long long(long long)>(function<long long(long long)>)>;

// 转换Church数到整数
long long to_int(Church n) {
    return n([](long long k) { return k + 1; })(0);
}

// 整数转Church数
Church to_church(long long n) {
    if (n == 0) {
        return [](function<long long(long long)> f) { return [f](long long x) { return x; }; };
    }
    Church prev = to_church(n - 1);
    return [prev](function<long long(long long)> f) {
        return [prev, f](long long x) { return f(prev(f)(x)); };
    };
}

// 后继 SUCC(n) = n + 1
Church succ(Church n) {
    return [n](function<long long(long long)> f) {
        return [n, f](long long x) { return f(n(f)(x)); };
    };
}

// 加法 ADD(m)(n) = m + n
Church add(Church m, Church n) {
    return [m, n](function<long long(long long)> f) {
        return [m, n, f](long long x) { return m(f)(n(f)(x)); };
    };
}

// Lambda递归求Fibonacci
Church fib_lambda(long long n) {
    if (n <= 1) return to_church(n);
    return add(fib_lambda(n - 1), fib_lambda(n - 2));
}

// ============================================================
// 性能测试
// ============================================================

void compare(int n) {
    cout << "\n计算 fib(" << n << "):" << endl;
    cout << string(30, '-') << endl;

    // C++ 普通递归
    auto start = high_resolution_clock::now();
    long long result1 = fib_normal(n);
    auto end = high_resolution_clock::now();
    double time1 = duration_cast<microseconds>(end - start).count() / 1000000.0;
    cout << "C++  普通递归: " << result1 << ", 耗时 " << time1 << " 秒" << endl;

    // C++ Lambda递归
    start = high_resolution_clock::now();
    Church church_result = fib_lambda(n);
    long long result2 = to_int(church_result);
    end = high_resolution_clock::now();
    double time2 = duration_cast<microseconds>(end - start).count() / 1000000.0;
    cout << "C++  Lambda递归: " << result2 << ", 耗时 " << time2 << " 秒" << endl;

    cout << "\nC++ Lambda递归慢了 " << (time2 / time1) << " 倍" << endl;
}

int main() {
    cout << string(40, '=') << endl;
    cout << "Fibonacci性能对比 (C++)" << endl;
    cout << string(40, '=') << endl;
    cout << "\n提示: Python版本结果请运行 compare.py" << endl;
    cout << "      可以将两边的输出对照比较" << endl;

    for (int n : {10, 12, 14, 16, 18}) {
        compare(n);
    }

    cout << "\n" << string(40, '=') << endl;
    cout << "结论: C++ Lambda递归同样比普通递归慢得多" << endl;
    cout << "      但C++整体比Python快很多" << endl;

    return 0;
}
