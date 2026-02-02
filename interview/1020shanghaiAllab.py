import sys

input = sys.stdin.readline

n = int(input())

for _ in range(n):
    len = int(input())
    s = input().strip()
    stk = []

    for c in s:
        if stk and stk[-1] == c:
            stk.pop()
        else:
            stk.append(c)
    print(len(stk))