import multiprocessing


# 主进程里的全局变量。
# 注意: 在 multiprocessing 中，每个子进程都有自己的独立内存副本。
counter = 0


def worker(name: str) -> None:
	global counter
	print(f"[{name}] start, local counter = {counter}")

	# 这里只会修改当前子进程自己的 counter 副本，
	# 不会回写到主进程的 counter。
	counter += 1

	print(f"[{name}] end, local counter = {counter}")


if __name__ == "__main__":
	processes = []

	for i in range(2):
		p = multiprocessing.Process(target=worker, args=(f"p{i}",))
		processes.append(p)
		p.start()

	for p in processes:
		p.join()

	# 主进程的 counter 从未被子进程真正修改，所以仍然是 0。
	print(f"[main] counter = {counter}")
