
# 一组人 在工厂n 人和人 尊重 [1,2,3]
# 老板
# 

def ack(a, b) -> bool:
    pass
# 
# 调用ask 找出老板 最少调用ack
def find_boss(nums):
    
    n = len(nums)
    boss = -111111111

    #1 暴力 双重for循环
    # for i in range(n-1):
    #     for j in range(i,n):
    #         if ask(nums[i], nums[j]):
    #             boss = nums[i]
    
    #2
    # left = 0
    # for right in range(1,n):
    #     left_num = nums[left]
    #     right_num = nums[right]
    #     if ask(left_num, right_num):
    #             boss = left_num
        
    #3 o(n)
    # 1 left, right
    # 2 pre 1表示尊重后面一位 0不尊重后面一位 , behind 1尊重前面一位，0不尊重前面一位 尊重数组[0,1]
    # [1,0,,xxx,0]
    # [0,xxxx,0,1]
    pre = [0] * n
    behind = [0] * n
    for i in range(0,n-1): 
        if ack(nums[i], nums[i+1]):
            pre[i] = 1
        else:
            pre[i] = 0


    for i in range(n-1, -1, -1):
        if ack(nums[i],nums[i-1]):
            pre[i] = 1
        else:
            pre[i] = 0

    boss = -111111
    left = 0
    # 先找到第一个都为0都下标 赋给left
    for i in range(n):
        if pre[i] and behind[i] == 0:
            left = i
    # 在开始逐步去更新  
    for right in range(left + 1, n):
        if pre[right] and behind[right] == 0:
            if ack(nums[left], nums[right]):
                boss = nums[left]
            else:
                boss = nums[right]
                left = right
    


    return boss
