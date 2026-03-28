"""
协程基础演示

演示：
- async/await 语法
- 如何创建和运行协程
- 协程的并发执行
"""

import asyncio


async def worker(name: str, duration: float):
    """协程工作函数"""
    print(f"[{name}] 开始")
    await asyncio.sleep(duration)  # 模拟 I/O 操作，主动让出执行权
    print(f"[{name}] 结束")


async def sequential():
    """顺序执行：一个接一个"""
    print("\n=== 顺序执行 ===")
    await worker("任务1", 0.5)
    await worker("任务2", 0.3)
    print("总耗时约: 0.8 秒")


async def concurrent():
    """并发执行：同时进行"""
    print("\n=== 并发执行 ===")
    # 创建并发任务
    t1 = asyncio.create_task(worker("任务1", 0.5))
    t2 = asyncio.create_task(worker("任务2", 0.3))

    # 等待所有任务完成
    await asyncio.gather(t1, t2)
    print("总耗时约: 0.5 秒（并行执行）")


async def main():
    print("=== 协程基础演示 ===")

    # 顺序执行
    await sequential()

    # 并发执行
    await concurrent()


if __name__ == "__main__":
    # 运行协程
    asyncio.run(main())
