#include <iostream>
#include <vector>
using namespace std;

// 简单的矩阵乘法实现
// C = A * B
void matrixMultiply(const vector<vector<double>>& A,
                   const vector<vector<double>>& B,
                   vector<vector<double>>& C) {
    int m = A.size();      // A的行数
    int n = A[0].size();   // A的列数 = B的行数
    int p = B[0].size();   // B的列数

    // 初始化结果矩阵C为全0
    C.assign(m, vector<double>(p, 0.0));

    // 三层循环实现矩阵乘法
    // C[i][j] = sum(A[i][k] * B[k][j])
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 打印矩阵
void printMatrix(const vector<vector<double>>& M) {
    for (const auto& row : M) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

int main() {
    // 示例：A是2x3矩阵，B是3x2矩阵
    vector<vector<double>> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    vector<vector<double>> B = {
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}
    };

    vector<vector<double>> C;

    cout << "矩阵 A:" << endl;
    printMatrix(A);

    cout << "\n矩阵 B:" << endl;
    printMatrix(B);

    matrixMultiply(A, B, C);

    cout << "\n结果 C = A * B:" << endl;
    printMatrix(C);

    return 0;
}
