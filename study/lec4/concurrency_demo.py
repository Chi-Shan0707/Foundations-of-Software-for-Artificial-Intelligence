import threading

import time



cnt  = 0
N = 10

temp = 0

def task():
    global cnt
    global temp
    for _ in range(N):
        temp = cnt
        # time.sleep(0.00001)
        cnt = temp + 1

t1 = threading.Thread(target=task)
t2 = threading.Thread(target=task)

t1.start()

t2.start()

# 等待线程 t1 执行结束；主线程会在这里阻塞
t1.join()

# 等待线程 t2 执行结束；确保两个线程都完成后再继续
t2.join()

# 只有在两个 join 都返回后，才会打印最终计数结果
# 情况1: t1/t2 没有加锁，cnt 的读改写可能发生竞态
print(cnt)



tot = 0

M = 20
def task_with_lock(lock):
    global tot
    for _ in range(M):
        with lock:
            temp = tot
            time.sleep(0.00001)
            tot = temp + 1

# 情况2: t3/t4 各自拿不同的锁对象，线程之间仍然不会互斥
t3 = threading.Thread(target=task_with_lock, args=(threading.Lock(),))
t4 = threading.Thread(target=task_with_lock, args=(threading.Lock(),))

t3.start()
t4.start()

t3.join()
t4.join()

# 因为不是同一把锁，结果通常会小于 2 * M
print(tot)


safe_tot = 0

def task_with_shared_lock(lock):
    global safe_tot
    for _ in range(M):
        with lock:
            temp = safe_tot
            time.sleep(0.00001)
            safe_tot = temp + 1

# 情况3: t5/t6 共享同一把锁，临界区真正互斥
shared_lock = threading.Lock()
t5 = threading.Thread(target=task_with_shared_lock, args=(shared_lock,))
t6 = threading.Thread(target=task_with_shared_lock, args=(shared_lock,))

t5.start()
t6.start()

t5.join()
t6.join()

# 两个线程各执行 M 次，且每次更新都受同一把锁保护，结果稳定为 2 * M
print(safe_tot)




