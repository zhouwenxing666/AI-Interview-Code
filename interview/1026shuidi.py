

import sys
from collections import deque
input = sys.stdin.readline

## 牛客：世界树上找米库
def main():
    sys.setrecursionlimit(1<<25)
    t = int(input())

    for _ in range(t):
        n = int(input())
        graph = [[] for _ in range(n+1)]
        degree = [0] * (n + 1)

        for _ in range(n-1):
            u, v = map(int, input().split())
            graph[u].append(v)
            graph[v].append(u)
            degree[u] += 1
            degree[v] += 1

            #找出所有叶子节点
            sekai = [i for i in range(1, n + 1) if degree[i] == 1]

            #多源距离bfs 记录每个节点到最近sekai点点距离
            dist = [-1] * (n + 1)
            q = deque()

            for s in sekai:
                dist[s] = 0
                q.append(s)
            
            while q:
                u = q.popleft()
                for v in graph[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)

            # 在非叶子节点中 找出距离值最大的点
            max_dist = -1
            miku = []
            for i in range(1, n + 1):
                if degree[i] > 1:
                    if dist[i] > max_dist:
                        max_dist = dist[i]
                        miku = [i]
                    elif dist[i] == max_dist:
                        miku.append(i)
            
            # 输出结果
            print(len(miku))
            print("".join(map(str, sorted(miku))))

main()



## 牛客 能量项链
n = int(input())
a = list(map(int, input().split()))
a *= 2 # 环 -> 链

# dp[i][j] 表示合并i～j的最大能力
dp = [[0] * (2 * n) for _ in range(2 * n)]

for l in range(2, n+1): # 区间长度
    for i in range(2*n -l): # 区间起点
        j = i + l - 1
        for k in range(i+1, j):
            dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + a[i]*a[k]*a[j+1])

ans = 0
for i in range(n): #枚举每个起点 取最大值
    ans = max(ans, dp[i][i+n-1])

print(ans)
