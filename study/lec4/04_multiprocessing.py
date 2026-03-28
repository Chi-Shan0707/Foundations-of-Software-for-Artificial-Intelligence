"""
进程演示：内存隔离

演示：
- 进程间的内存隔离
- 子进程修改不会影响父进程
- 如何通过 Queue 进行进程间通信
"""

import multiprocessing


# 全局变量
counter = 0


def worker_without_ipc(name: str):
    """工作进程：修改自己的 counter 副本"""
    global counter
    print(f"[{name}] start, local counter = {counter}")
    counter += 1
    print(f"[{name}] end, local counter = {counter}")


def worker_with_queue(queue: multiprocessing.Queue, name: str):
    """工作进程：通过 Queue 发送消息"""
    queue.put(f"来自 {name} 的消息")
    queue.put(f"{name} 完成工作")


def main():
    global counter

    print("=== 进程内存隔离演示 ===\n")

    # ===== 情况1: 无 IPC，内存隔离 =====
    print("【情况1】子进程修改不会影响父进程")
    counter = 0

    p1 = multiprocessing.Process(target=worker_without_ipc, args=("p1",))
    p2 = multiprocessing.Process(target=worker_without_ipc, args=("p2",))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(f"[父进程] counter = {counter}")
    print("观察：父进程的 counter 没有被子进程修改\n")

    # ===== 情况2: 使用 Queue 进行 IPC =====
    print("【情况2】使用 Queue 进行进程间通信")

    queue = multiprocessing.Queue()

    p3 = multiprocessing.Process(target=worker_with_queue, args=(queue, "p3"))
    p4 = multiprocessing.Process(target=worker_with_queue, args=(queue, "p4"))

    p3.start()
    p4.start()

    p3.join()
    p4.join()

    print("从 Queue 接收消息：")
    while not queue.empty():
        print(f"  - {queue.get()}")


if __name__ == "__main__":
    # Windows 下必须放在 if __name__ == "__main__" 下
    main()
