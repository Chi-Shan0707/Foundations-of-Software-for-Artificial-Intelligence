"""
协程竞态演示

演示：
- 协程也会出现数据竞态
- await 点是切换时机
- 使用 asyncio.Lock 解决竞态
- 轮询 vs 事件驱动
"""

import asyncio


# 全局变量
counter = 0
safe_counter = 0
N = 5


async def unsafe_worker(name: str):
    """没有锁保护的协程"""
    global counter
    for _ in range(N):
        old = counter  # 读
        await asyncio.sleep(0)  # 让出执行权，其他协程可能修改 counter
        counter = old + 1  # 写
    print(f"[{name}] counter_snapshot = {counter}")


async def safe_worker(name: str, lock: asyncio.Lock):
    """有锁保护的协程"""
    global safe_counter
    for _ in range(N):
        async with lock:  # 获取锁
            old = safe_counter
            await asyncio.sleep(0)
            safe_counter = old + 1
    print(f"[{name}] safe_counter_snapshot = {safe_counter}")


async def polling_demo():
    """轮询：反复检查任务是否完成"""
    print("\n=== 轮询示例 ===")

    task = asyncio.create_task(asyncio.sleep(0.2))

    while not task.done():  # 反复检查
        print("[polling] 任务未完成...")
        await asyncio.sleep(0.05)

    print("[polling] 任务完成！")


async def event_driven_demo():
    """事件驱动：直接等待任务完成"""
    print("\n=== 事件驱动示例 ===")

    print("等待任务完成...")
    await asyncio.sleep(0.2)
    print("任务完成！")


async def main():
    print("=== 协程竞态演示 ===\n")

    # ===== 情况1: 没有锁 =====
    print("【情况1】协程没有锁保护")
    counter = 0

    t1 = asyncio.create_task(unsafe_worker("c1"))
    t2 = asyncio.create_task(unsafe_worker("c2"))

    await asyncio.gather(t1, t2)

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}\n")

    # ===== 情况2: 有锁 =====
    print("【情况2】协程有 asyncio.Lock 保护")
    safe_counter = 0
    lock = asyncio.Lock()

    s1 = asyncio.create_task(safe_worker("s1", lock))
    s2 = asyncio.create_task(safe_worker("s2", lock))

    await asyncio.gather(s1, s2)

    print(f"期望: {2 * N}, 实际: {safe_counter}, 结果: {'✓' if safe_counter == 2 * N else '✗'}\n")

    # ===== 情况3: 轮询 vs 事件驱动 =====
    await polling_demo()
    await event_driven_demo()


if __name__ == "__main__":
    asyncio.run(main())
