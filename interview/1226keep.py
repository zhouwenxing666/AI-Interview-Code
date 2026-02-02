
# [3, 1, 2, 5, 4]

# [1 2 3 5, 4]



def quick_sort(nums):
    # sort_quickly(nums, 0, len(nums)-1)
    quick_sort_0121(nums, 0, len(nums)-1)
    return nums

def sort_quickly(nums, start, end):
    left = start
    right = end
    if left >= right:
        return
    pivot = nums[left]
    
    while left < right:
        while left < right and nums[right] >= pivot:
            right -= 1
        # 找到第一个小于基准值的的right位置 把它复制给left 第一次的时候left元素已经存储在pivot变量中了
        nums[left] = nums[right]
        while left < right and nums[left] <= pivot:
            left += 1
        nums[right] = nums[left]
    nums[left] = pivot

    sort_quickly(nums, start, left-1)
    sort_quickly(nums, left + 1, end)

def quick_sort_0121(nums, start, end):
    # 递归终止条件
    if start >= end:
        return
    
    left = start
    right = end 
    pivot = nums[left]

    while left < right:
        while left < right and nums[right] >= pivot:
            right -= 1
        nums[left] = nums[right]
        while left < right and nums[left] <= pivot:
            left += 1
        nums[right] = nums[left]
    nums[left] = pivot
    quick_sort_0121(nums, start, left - 1)
    quick_sort_0121(nums, left + 1, end)

    return nums


nums = [3, 1, 2, 5, 4]
# new_nums = quick_sort(nums)
new_nums = quick_sort_0121(nums,0,len(nums)-1)

print(f"new_nums is:{new_nums}")


# X的平方根等于多少，保留3位小数。比如25 = 5
def sqrt_zwx(num):
    left = 0
    right = num
    mid = 0
    #[0, num]
    while left <= right:
        mid = left + (right - left) // 2
        if abs(mid * mid - num) < 0.001:
            return round(mid, 3)
        elif mid * mid < num:
            left = mid + 0.001
        else:
            right = mid - 0.001
    return round(right, 3)

# result = sqrt_zwx(30.0)
# print(f"result is:{result}")


# maxlen =win_len 2
# win c
# s = "abdcd" b = "abcddsa"




