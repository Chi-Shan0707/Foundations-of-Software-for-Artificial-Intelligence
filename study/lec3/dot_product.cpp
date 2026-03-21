/*
 * ============================================================================
 * C++ 命令行操作指南
 * ============================================================================
 *
 * 1. 基础编译:
 *    g++ dot_product.cpp -o dot_product
 *    g++ dot_product.cpp                    # 默认输出 a.out
 *
 * 2. 带优化等级编译:
 *    g++ -O0 dot_product.cpp -o dot_product  # 无优化（默认）
 *    g++ -O1 dot_product.cpp -o dot_product  # 基础优化
 *    g++ -O2 dot_product.cpp -o dot_product  # 常用优化
 *    g++ -O3 dot_product.cpp -o dot_product  # 最高优化
 *    g++ -Os dot_product.cpp -o dot_product  # 优化体积
 *
 * 3. 调试相关:
 *    g++ -g dot_product.cpp -o dot_product   # 生成调试信息
 *    g++ -g -O0 dot_product.cpp -o dot_product # 调试版本（无优化）
 *
 * 4. 警告选项:
 *    g++ -Wall dot_product.cpp -o dot_product   # 开启所有常用警告
 *    g++ -Wextra dot_product.cpp -o dot_product # 额外警告
 *    g++ -Werror dot_product.cpp -o dot_product # 将警告视为错误
 *
 * 5. 标准版本:
 *    g++ -std=c++11 dot_product.cpp -o dot_product
 *    g++ -std=c++17 dot_product.cpp -o dot_product
 *    g++ -std=c++20 dot_product.cpp -o dot_product
 *
 * 6. 运行程序:
 *    ./dot_product
 *    ./dot_product arg1 arg2              # 传递命令行参数
 *
 * 7. GDB 调试命令:
 *    gdb ./dot_product                    # 启动 GDB
 *
 *    GDB 内部常用命令:
 *    (gdb) run (r)                        # 运行程序
 *    (gdb) break (b) main                 # 在 main 设置断点
 *    (gdb) break dot_product              # 在函数设置断点
 *    (gdb) break 15                       # 在第15行设置断点
 *    (gdb) continue (c)                   # 继续执行
 *    (gdb) next (n)                       # 单步执行（不进入函数）
 *    (gdb) step (s)                       # 单步执行（进入函数）
 *    (gdb) print (p) result               # 打印变量值
 *    (gdb) display result                 # 每次停止时显示变量
 *    (gdb) info breakpoints               # 查看所有断点
 *    (gdb) delete 1                       # 删除断点1
 *    (gdb) backtrace (bt)                 # 查看调用栈
 *    (gdb) frame (f) 0                    # 切换栈帧
 *    (gdb) info locals                    # 查看局部变量
 *    (gdb) info registers                 # 查看寄存器
 *    (gdb) disas                          # 反汇编当前函数
 *    (gdb) quit (q)                       # 退出 GDB
 *
 *    GDB 直接运行并调试:
 *    gdb -ex "break main" -ex "run" -ex "quit" ./dot_product
 *
 * 8. 反汇编/逆向工程:
 *    objdump -d dot_product               # 反汇编所有section
 *    objdump -d -M intel dot_product      # 使用 Intel 语法反汇编
 *    objdump -S dot_product               # 反汇编并显示源代码（需用 -g 编译）
 *    objdump -t dot_product               # 显示符号表
 *    objdump -T dot_product               # 显示动态符号表
 *
 *    nm dot_product                       # 显示符号表
 *    nm -C dot_product                    # 解码 C++ 符号
 *    nm -D dot_product                    # 显示动态符号
 *
 *    readelf -h dot_product               # 读取 ELF 文件头
 *    readelf -S dot_product               # 读取 section 头
 *    readelf -s dot_product               # 读取符号表
 *    readelf -l dot_product               # 读取程序头
 *
 * 9. 查看动态库依赖:
 *    ldd dot_product                      # 显示程序依赖的共享库
 *
 * 10. 汇编级查看:
 *     g++ -S dot_product.cpp -o dot_product.s  # 生成汇编文件
 *     g++ -S -fverbose-asm dot_product.cpp     # 生成带注释的汇编
 *
 * 11. 预处理:
 *     g++ -E dot_product.cpp              # 只进行预处理，输出到 stdout
 *     g++ -E dot_product.cpp -o dot_product.i # 输出到文件
 *
 * 12. 链接相关:
 *     g++ -c dot_product.cpp              # 只编译不链接，生成 .o
 *     g++ dot_product.o -o dot_product    # 单独链接
 *
 * 13. 查看依赖:
 *     g++ -M dot_product.cpp              # 生成 make 规则
 *     g++ -MM dot_product.cpp             # 生成依赖（不含系统头文件）
 *
 * 14. 性能分析:
 *     g++ -pg dot_product.cpp -o dot_product  # 启用 gprof 分析
 *     ./dot_product                       # 运行生成 gmon.out
 *     gprof dot_product gmon.out          # 查看分析结果
 *
 * 15. 内存检查 (Valgrind):
 *     g++ -g dot_product.cpp -o dot_product
 *     valgrind --leak-check=full ./dot_product
 *     valgrind --tool=memcheck ./dot_product
 *
 * 16. 查看二进制内容:
 *     hexdump -C dot_product | head       # 查看二进制文件的十六进制
 *     strings dot_product                 # 查看可打印字符串
 *
 * ============================================================================
 */

#include<iostream>
#include<vector>
int dot_product(const std::vector<int>& a, const std::vector<int>& b) {
    int result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
int main()
{
    std::cout << "Dot product: ";
    std::vector<int> A = {1, 2, 3, 4, 5};
    std::vector<int> B = {6, 7, 8, 9, 10};
    std::cout << dot_product(A, B) << std::endl;
    return 0;
}