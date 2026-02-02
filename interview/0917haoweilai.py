import os 
from typing import List
#题1 环路上有n个加油站 油箱容量无限的汽车 找到出发时加油站的编号
def main1():
    gas = list(map(int, input().split()))
    cost = list(map(int, input().split()))
    n = len(gas)

    total_tank = 0
    current_tank = 0
    start_station = 0

    for i in range(n):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        # 如果从当前起点无法到达下一站
        if current_tank < 0:
            # 则将下一站设为新起点 并重置当前邮箱
            start_station = i + 1
            current_tank = 0
        
    
    # 如果总油量小于总消耗 则误解
    # 否则 start_station为答案
    if total_tank < 0:
        return -1
    else:
        return start_station


# 题2 给定一个候选人集合 nums 和 target 找出nums中数字和为target的组合 升序排序输出
def combinationSum2(self , nums: List[int], target: int) -> List[List[int]]:
    # write code here
    nums.sort()

    res = []

    def dfs(start, path, total):
        if total == target:
            res.append(path[:])
            return
        for i in range(start, len(nunms)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            if total + nums[i] > target:
                break
        
            path.append(nums[i])
        
            dfs(i + 1, path=path, total = total + nums[i])
            path.pop()

    dfs(0,[],0)
    return res

#题3 最小时差
def findMinDifference(self , timePoints: List[str]) -> int:
    # write code here
    mins = sorted(int(t[:2]) * 60 + int(t[3:]) for t in timePoints)
    mins.append(mins[0] + 1440)
    return min(b - a for a,b in zip(mins, mins[1:]))



if __name__ == "__main__":
    # main()
