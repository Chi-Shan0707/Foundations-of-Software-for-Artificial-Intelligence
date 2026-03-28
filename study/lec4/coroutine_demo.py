import asyncio


counter = 0
safe_counter = 0
N = 5


async def worker(name: str) -> None:
    global counter
    for _ in range(N):
        # 协程在单线程里运行，但在 await 处会主动让出执行权。
        # 这里故意把 "读-改-写" 拆开，演示协程之间也会出现逻辑竞态。
        old = counter
        await asyncio.sleep(0)
        counter = old + 1
    print(f"[{name}] counter_snapshot = {counter}")


async def safe_worker(name: str, lock: asyncio.Lock) -> None:
    global safe_counter
    for _ in range(N):
        # 使用同一把 asyncio.Lock 后，多个协程在临界区内互斥。
        async with lock:
            old = safe_counter
            await asyncio.sleep(0)
            safe_counter = old + 1
    print(f"[{name}] safe_counter_snapshot = {safe_counter}")


async def polling_demo() -> None:
    # 轮询: 反复检查任务是否完成。
    # 这种方式可读性一般，且会产生额外检查开销。
    task = asyncio.create_task(asyncio.sleep(0.2))
    while not task.done():
        print("[polling] task not done yet")
        await asyncio.sleep(0.05)
    print("[polling] task done")


async def main() -> None:
    print("== 协程竞态示例(无锁) ==")
    t1 = asyncio.create_task(worker("c1"))
    t2 = asyncio.create_task(worker("c2"))
    await asyncio.gather(t1, t2)
    print(f"expected = {2 * N}, actual = {counter}")

    print("\n== 协程加锁示例(共享同一把 asyncio.Lock) ==")
    lock = asyncio.Lock()
    s1 = asyncio.create_task(safe_worker("s1", lock))
    s2 = asyncio.create_task(safe_worker("s2", lock))
    await asyncio.gather(s1, s2)
    print(f"expected = {2 * N}, actual = {safe_counter}")

    print("\n== 轮询示例 ==")
    await polling_demo()


if __name__ == "__main__":
    # 协程函数不会自动执行，需要交给事件循环运行。
    asyncio.run(main())
