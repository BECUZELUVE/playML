import time

def sum_n_2(n):
    start = time.time()
    the_sum = (n*(n+1))/2
    end = time.time()
    return (end-start)

print("计算前10000项之和需要", sum_n_2(10000), "毫秒")
print("计算前100000项之和需要", sum_n_2(100000), "毫秒")
print("计算前1000000项之和需要", sum_n_2(1000000), "毫秒")
