"""
数据竞态演示

演示：
- 两个线程同时修改共享变量
- 期望结果 vs 实际结果
- sleep() 如何放大竞态窗口
"""

import threading
import time


# 全局共享变量
counter = 0
N = 10  # 每个线程增加 10 次，期望总计 20


def task(name: str):
    """线程任务：读取-修改-写入"""
    global counter
    for i in range(N):
        temp = counter  # ① 读
        # time.sleep(0.00001)  # 取消注释可以放大竞态窗口
        counter = temp + 1  # ② 写


def main():
    global counter

    print("=== 数据竞态演示 ===")
    print(f"期望结果: {2 * N} (两个线程各增加 {N} 次)")

    # 重置计数器
    counter = 0

    # 创建两个线程
    t1 = threading.Thread(target=task, args=("t1",))
    t2 = threading.Thread(target=task, args=("t2",))

    # 启动线程
    t1.start()
    t2.start()

    # 等待线程结束
    t1.join()
    t2.join()

    print(f"实际结果: {counter}")

    if counter == 2 * N:
        print("✓ 结果正确（这次没有发生竞态，但不是保证的！）")
    else:
        print(f"✗ 结果错误！丢失了 {2 * N - counter} 次更新")
        print("  原因：两个线程同时读取到相同的旧值，覆盖了对方的更新")


if __name__ == "__main__":
    # 多次运行，观察结果的不确定性
    for i in range(5):
        print(f"\n--- 第 {i+1} 次运行 ---")
        main()
