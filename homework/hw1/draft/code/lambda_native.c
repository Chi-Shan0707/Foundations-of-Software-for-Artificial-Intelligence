#include <stddef.h>

// Church Numerals: 用整数表示"应用函数的次数"
// ADD(m, n) = m + n  (应用 m 次，再应用 n 次)
int church_add(int m, int n) {
    return m + n;
}

// MUL(m, n) = m * n  (应用 n 次，重复 m 次)
int church_mul(int m, int n) {
    return m * n;
}

// SUCC(n) = n + 1
int church_succ(int n) {
    return n + 1;
}
