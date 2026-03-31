#include <stddef.h>

// SUCC(n) = n + 1


int church_succ(int n) {
    if (n <= 0) return n;
    return 1+church_succ(n-1);
}

// ADD(m, n) = 应用 m 次 SUCC 到 n
// 对应 Lambda: ADD = λm.λn. m SUCC n
int church_add(int m, int n) {
    for (int i = 0; i < m; i++) {
        n = church_succ(n);
    }
    return n;
}
