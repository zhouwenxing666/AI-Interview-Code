from typing import List

class Solution:
    # lc134加油站  贪心
    def canCompleteCircuit(self , gas: List[int], cost: List[int]) -> int:
        # write code here
        n = len(gas)
        cur_sum = 0
        min_sum = float('inf')

        for i in range(n):
            cur_sum += gas[i] - cost[i]
            min_sum = min(min_sum, cur_sum)
        
        if cur_sum < 0: return -1
        if min_sum >= 0: return 0

        for j in range(n-1, 0, -1):
            min_sum += gas[j] - cost[j]
            if min_sum >= 0:
                return j
            
        return -1


    # lcr82 组合总和2  回溯
    def combinationSum2(self , nums: List[int], target: int) -> List[List[int]]:
        # write code here

        # 先对nums排序
        nums.sort()


        result = []
        path = []
        cur_sum = 0
        start_idx = 0 # 控制nums的起点

        self.backtracking(nums, target, path, start_idx, result, cur_sum)

        return result
    
    def backtracking(self, nums, target, path, start_idx, result, cur_sum):
        if cur_sum > target:
            return
        if cur_sum == target:
            result.append(path[:])
            return
        
        
        for i in range(start_idx, len(nums)):
            path.append(nums[i])

            self.backtracking(nums, target, path, i, result, cur_sum + nums[i])

            path.pop()

    #lc539，lcr035 最小时差
    def findMinDifference(self , timePoints: List[str]) -> int:
        # write code here
        # 因为时间点最多只有24 * 60个 长度超过次 说明有重复的时间点 返回0
        if len(timePoints) > 24 * 60:
            return 0

        # 遍历时间列表 将其转换为分钟列表mins 如对于时间13:14转成 13 * 60 + 14
        mins = sorted(int(t[:2] * 60 + int(t[3:]) for t in timePoints))
        mins.append(mins[0] + 24 * 60)
        return min(mins[i] - mins[i-1] for i in range(1, len(mins)))
