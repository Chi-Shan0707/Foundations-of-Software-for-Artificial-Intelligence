"""
使用锁解决数据竞态

演示：
- 如何使用 Lock 保护临界区
- with 语句自动获取和释放锁
- 共享同一把锁的重要性
"""

import threading
import time


# 全局变量
counter = 0
N = 10


def task_without_lock(name: str):
    """没有锁保护的线程"""
    global counter
    for _ in range(N):
        temp = counter
        time.sleep(0.00001)  # 放大竞态窗口
        counter = temp + 1


def task_with_lock(name: str, lock: threading.Lock):
    """有锁保护的线程"""
    global counter
    for _ in range(N):
        with lock:  # 自动获取锁
            temp = counter
            time.sleep(0.00001)  # 即使有 sleep，其他线程也无法进入
            counter = temp + 1
        # 离开 with 块，锁自动释放


def main():
    global counter

    print("=== 锁的作用演示 ===\n")

    # ===== 情况1: 没有锁 =====
    print("【情况1】两个线程，没有锁")
    counter = 0

    t1 = threading.Thread(target=task_without_lock, args=("t1",))
    t2 = threading.Thread(target=task_without_lock, args=("t2",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}\n")

    # ===== 情况2: 各自用不同的锁（错误！）=====
    print("【情况2】两个线程，各自使用不同的锁（错误示例）")
    counter = 0

    lock1 = threading.Lock()
    lock2 = threading.Lock()

    t3 = threading.Thread(target=task_with_lock, args=("t3", lock1))
    t4 = threading.Thread(target=task_with_lock, args=("t4", lock2))

    t3.start()
    t4.start()
    t3.join()
    t4.join()

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}")
    print("原因：使用的是不同的锁对象，不会互斥！\n")

    # ===== 情况3: 共享同一把锁（正确！）=====
    print("【情况3】两个线程，共享同一把锁（正确示例）")
    counter = 0

    shared_lock = threading.Lock()

    t5 = threading.Thread(target=task_with_lock, args=("t5", shared_lock))
    t6 = threading.Thread(target=task_with_lock, args=("t6", shared_lock))

    t5.start()
    t6.start()
    t5.join()
    t6.join()

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}")
    print("原因：两个线程竞争同一把锁，临界区互斥执行！")


if __name__ == "__main__":
    main()
