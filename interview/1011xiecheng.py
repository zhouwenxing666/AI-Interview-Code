
# import sys
# input = sys.stdin.readline


### 题目1 好的字符串与不错的字符串
n = int(input())

good = ok = 0

def is_good(s):
    return all((s[i].isupper() ^ s[i+1].isupper()) for i in range(len(s) - 1))

for _ in range(n):
    input() # m 不用实际参数逻辑
    s = input().strip()
    if is_good(s):
        good += 1
        ok += 1
    else:
        found = False
        for i in range(len(s)):
            t = s[:i] + (s[i].swapcase()) + s[i+1:]
            if is_good(t):
                ok += 1
                found = True
                break
        if not found:
            pass
print(good, ok)


### 题2 淘汰赛中胜场数
T = int(input())
for _ in range(T):
    n = int(input())
    s = list(map(int, list(input().strip())))
    if n == 1:
        print(0)
        continue
    idx = 0 # 队伍1位置
    win = 0
    while n > 1:
        pair = idx ^ 1 # 同组对手位置（idx奇偶翻转）
        if s[idx] < s[pair]:
            break
        win += 1
        idx //= 2
        nxt = []
        for i in range(0, n, 2):
            nxt.append(max(s[i], s[i+1]))
            s = nxt
            n //= 2
    print(win)


### 题3 寻找最小完美数
# 解法1 暴力解 通过20% 剩下超时
import sys
input = sys.stdin.readline
t = int(input())
for _ in range(t):
    x = int(input())
    ans = x
    while True:
        b = bin(ans)[2:]
        if b.count('1') == b.count('0'):
            print(ans)
            break
        ans += 1

#解法2
# 该问题关键在于构造。对于给定的x，目标y要么和x有相同的二进制长度，要么更长。
# 1 先计算出比x更长的，最小的完美数 candidate_longer
# 2 寻找和x等长的，且大于等于x的最小完美数 candidate_same_len
# 3 如果x的二进制长度数奇数，或找不到满足条件的candidate_same_len，则答案是candidate_longer
# 4 否则答案就是candidate_same_len
import sys
input = sys.stdin.readline

test_cases = int(input())
for _ in range(test_cases):
    num_val = int(input())
    bin_str = bin(num_val)[2:]
    bit_len = len(bin_str)

    # 计算比num_val更长的最小完美数 candidate_longer
    target_len = bit_len + 1 if bit_len % 2 != 0 else bit_len + 2
    half_len = target_len // 2
    longer_s = '1' + '0' * half_len + '1' * (half_len - 1)
    candidate_longer = int(longer_s, 2)

    if bit_len % 2 != 0:
        print(candidate_longer)
        continue

    half_len = bit_len // 2
    ones_count = bin_str.count('1')

    if ones_count == half_len:
        print(num_val)
        continue

    # 从右往左遍历 寻找第一个可以从0变1的位置
    found_same_len = False
    for i in range(bit_len - 1, -1, -1):
        if bin_str[i] == '0':
            prefix = bin_str[:i] + '1'
            ones_in_prefix = prefix.count('1')
            ones_for_suffix = half_len - ones_in_prefix
            suffix_len = bit_len - len(prefix)

            if 0 <= ones_for_suffix <=. suffix_len:
                zeros_for_suffix = suffix_len - ones_for_suffix
                suffix = '0' * zeros_for_suffix + '1' * ones_for_suffix

                candidate_same_len_s = prefix + suffix
                print(int(candidate_same_len_s, 2))
                found_same_len = True
                break
    
    if not found_same_len:
        print(candidate_longer)

### 题4 

# 使用树状数组（fenwick tree/bit）和坐标压缩（离散化）
import sys

# 树状数组更新操作
def updata_bit(bit_arr, max_idx, idx, val):
    while idx <= max_idx:
        bit_arr[idx] += val
        idx += idx & -idx

# 树状数组查询操作
def query_bit(bit_arr, idx):
    total = 0
    while idx > 0:
        total += bit_arr[idx]
        idx -= idx & -idx
    return total

# 读取测试用例
input = sys.stdin.readline
num_test_cases = int(input())

for _ in range(num_test_cases):
    n = int(input())
    a = list(map(int, input().split()))

    # 由于a[i]的值域很大，需要进行坐标压缩
    distinct_vals = sorted(list(set(a)))
    rank_map = {val: i + 1 for i, val in enumerate(distinct_vals)}
    m = len(distinct_vals)

    # 计算每个位置左侧值小于等于a[i]的元素数量
    left_le_counts = [0] * n
    bit1 = [0] * (m + 1)
    for i in range(n):
        rank = rank_map[a[i]]
        left_le_counts[i] = query_bit(bit1, rank)
        updata_bit(bit1, m, rank, 1)
    
    # 计算每个位置右侧值小于等于a[i]的元素数量
    right_le_counts = [0] * n
    bit2 = [0] * (m + 1)
    for i in range(n-1, -1, -1):
        rank = rank_map[a[i]]
        right_le_counts[i] = query_bit(bit2, rank)
        updata_bit(bit2, m, rank, 1)
    
    # 累加每个元素的总贡献
    total_contribution = 0
    for i in range(n):
        # 包含a[i]的总区间数
        total_intervals_for_i = i * (n - 1 - i)
        # a[i]不产生贡献的去见数
        non_contributing_intervals = left_le_counts[i] * right_le_counts[i]

        total_contribution += total_intervals_for_i - non_contributing_intervals
    
    print(total_contribution)



import sys
