"""
简单线程示例

演示：
- 如何创建和启动线程
- join() 的作用
- 线程执行的不确定性
"""

import threading
import time


def worker(name: str, duration: float):
    """线程工作函数"""
    print(f"[{name}] 开始执行")
    time.sleep(duration)
    print(f"[{name}] 执行完毕")


def main():
    print("=== 创建两个线程 ===")

    # 创建线程
    t1 = threading.Thread(target=worker, args=("线程1", 1.0))
    t2 = threading.Thread(target=worker, args=("线程2", 0.5))

    # 启动线程
    t1.start()
    t2.start()

    print("主线程继续执行...")

    # 等待线程结束
    t1.join()
    t2.join()

    print("所有线程都执行完毕")


if __name__ == "__main__":
    main()
