### 8.24 小红书笔试

import sys
import math

## 题1
# def main():
#     n = int(sys.stdin.readline().strip())
#     a = list(map(int, sys.stdin.readline().split()))
#     ans = 0
#     for d in range(2, 101):
#         i = 0
#         cnt = 0
#         while i < n:
#             if a[i] % d == 0:
#                 j = i
#                 while j < n and a[j] % d == 0:
#                     j += 1
#                 L = j - i
#                 cnt += (L + 1) // 2
#                 i = j
#             else:
#                 i += 1
#         if cnt > ans:
#             ans = cnt
    
#     print(ans)

## 题2
# def main():
    # try:
    #     input = sys.stdin.readline
    #     # 读取用户数n 和查询数m
    #     n, m = map(int, input().split())

    #     # 读取n个用户的分数
    #     scores = list(map(int,input().split()))

    #     max_val = 500001
    #     # 1 预处理 
    #     counts = [0] * max_val
    #     for score in scores:
    #         if score < max_val:
    #             counts[score] += 1
    #     # 2 预处理 找出每个数字的倍数和约数在输入数据中的总数
    #     multiples_count = [0] * max_val
    #     divisor_sum = [0] * max_val

    #     for i in range(1, max_val):
    #         for j in range(i, max_val, i):
    #             if counts[j] > 0:
    #                 multiples_count[i] += counts[j]
    #             if counts[i] > 0:
    #                 divisor_sum[j] += counts[i]
    #     # 3 处理m次查询
    #     output = []
    #     for _ in range(m):
    #         x = int(input())

    #         result = multiples_count[x] + divisor_sum[x] - counts[x]

    #         output.append(str(result))
    #     sys.stdout.write('\n'.join(output)+'\n')
    # except(IOError, IndexError) as e:
    #     pass

#题3
# def main():
#     try:
#         n_str = sys.stdin.readline()
#         if not n_str:
#             return
#         n = int(n_str)
#         s = sys.stdin.readline().strip()
#     except(IOError, ValueError):
#         return
    
#     # step1 确定各字符的目标数量
#     orig_counts = [0] * 26
#     indices=[[]for _ in range(26)]
#     for i,char in enumerate(s):
#         code = ord(char) - ord('a')
#         orig_counts[code]+=1
#         indices[code].append(i)

#     # step2 确定最优的目标字符数量
#     target_counts=[0]*26
#     target_counts[25]=orig_counts[25]
#     for i in range(24,-1,-1):
#         target_counts[i]=min(orig_counts[i],target_counts[i+1])

#     # step3 贪心构造
#     result_chars=[]
#     s_cursor=0
#     needed_counts=list(target_counts)
#     total_len=sum(needed_counts)
    
#     indices_ptr=[0]*26
#     for _ in range(total_len):
#         limit_idx = n
#         for char_code in range(26):
#             if needed_counts[char_code] > 0:
#                 k = needed_counts[char_code]
#                 last_possible_pos=indices[char_code][-k]
#                 limit_idx=min(limit_idx,last_possible_pos)

#         for char_code in range(26):
#             if needed_counts[char_code] > 0:
#                 ptr=indices_ptr[char_code]
#                 pos=indices[char_code][ptr]
                
#                 if pos <= limit_idx:
#                     result_chars.append(chr(ord('a') + char_code))
#                     s_cursor=pos+1
#                     needed_counts[char_code]-=1
#                     for code in range(26):
#                         while(indices_ptr[code]<orig_counts[code] and indices[code][indices_ptr[code]]<s_cursor):
#                             indices_ptr[code]+=1
#                     break
#     print("".join(result_chars))
# def main():
#     try:
#         n_str = sys.stdin.readline()
#         if not n_str:
#             return
#         n = int(n_str)
#         s = sys.stdin.readline().strip()

#     except(IOError, ValueError):
#         return
    
#     # step1 确定各字符的目标数量
#     orig_counts = [0] * 26
#     indices=[[]for _ in range(26)]
#     for i,char in enumerate(s):
#         code = ord(char) - ord('a')
#         orig_counts[code]+=1
#         indices[code].append(i)
#     # for char in s:
#     #     orig_counts[ord(char)-ord('a')] += 1

    
#     target_counts=[0]*26
#     target_counts[25]=orig_counts[25]
#     for i in range(24,-1,-1):
#         target_counts[i]=min(orig_counts[i],target_counts[i+1])
    
#     # step 2 构造字典序最小字符串
#     suffix_counts=[[0]*26 for _ in range(n + 1)]
#     for i in range(n-1, -1, -1):
#         for j in range(26):
#             suffix_counts[i][j] = suffix_counts[i+1][j]
#             suffix_counts[i][ord(s[i])-ord('a')] += 1
        
#     result = []
#     current_s_pos = 0
#     needed_counts = list(target_counts)
#     total_len=sum(needed_counts)

#     for _ in range(total_len):
#         for char_code in range(26):
#             # 如果当前字符还需要
#             if needed_counts[char_code] > 0:
#                 # 在s中从current_s_pos开始寻找该字符
#                 try:
#                     char_pos = s.index(chr(ord('a') + char_code), current_s_pos)
#                 except ValueError:
#                     continue

#                 needed_counts[char_code]-=1
#                 is_possible=True
#                 for j in range(26):
#                     if needed_counts[j] > suffix_counts[char_pos + 1][j]:
#                         is_possible = False
#                         break
#                 if is_possible:
#                     result.append(chr(ord('a')+char_code))
#                     current_s_pos=char_pos+1
#                     break
#                 else:
#                     needed_counts[char_code]+=1
            
#     print("".join(result))




## 给一个数组 算出最大连续子数组的和
# [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# [-2, 1, -3, 4, -1]
# [-3 -2 -1 1 4]
def count_max_numpy_sum(nums: list):
    
    # nums.sort()
    print(nums)
    counts = []
    sum = nums[0]
    for i,num in enumerate(nums):
        if i == 0:
            continue
        # lianxu
        if nums[i] - nums[i-1] == 1 or nums[i] == nums[i-1]:
            sum += nums[i]
        else:
            counts.append(sum)
            sum = 0
        
    counts.sort()
    print(counts[-1])
    return counts[-1]

def count_max_numpy_sum_1(nums: list):
# [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# n. (n-1) / 2 

    left = 0
    right = 0

    max_sum = 0
    win_sum = nums[0]
    for right, num in enumerate(nums):
        if right == 0:
            continue

        if nums[right] > 0:
            win_sum += nums[right]
            max_sum = max(max_sum,win_sum)
        
        # 左移条件
        while win_sum < 0 and left <= right:
            left += 1
            win_sum -= nums[left]
            max_sum = max(max_sum,win_sum)
            if win_sum >= 0:
                break
        
    return max_sum
        

result = count_max_numpy_sum_1([-2, 1, -3, 4, -1, 2, 1, -5, 4])
print(result)

# if __name__ == "__main__":
    
#     main()

