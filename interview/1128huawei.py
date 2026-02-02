"""
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

 

示例 1：

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
示例 2：

输入：nums = [2,0,1]
输出：[0,1,2]
"""
from collections import defaultdict

# ### 我的解法
# def sort_colors_2(nums):
#     n = len(nums)
#     left = 0
#     right = n - 1
#     cur = 0
#     while cur <= right:
#         # 若当前cur对应的数字为0则 将其与left指向的元素交换 然后left，cur同时右移动
#         # 若当前cur对应的数字为2则 将其与right指向的元素交换 然后cur右移动，right左移
#         cur_num = nums[cur]
#         if cur_num == 0:
#             nums[left], nums[cur] = nums[cur], nums[left]
#             left += 1
#             cur += 1
#         elif cur_num == 2:
#             nums[right], nums[cur] = nums[cur], nums[right]
#             right -= 1
#         else:
#             cur += 1
        
#     return nums

# # 示例 1
# nums1 = [2, 0, 2, 1, 1, 0]
# result = sort_colors_2(nums1)
# print(f"示例 1 输出: {result}")  # 输出: [0, 0, 1, 1, 2, 2]

# nums2 = [2, 0, 1]
# result2 = sort_colors_2(nums2)
# print(f"示例 2 输出: {result2}")  

# nums2 = [2, 2, 0, 1]
# result3 = sort_colors_2(nums2)
# print(f"示例 3 输出: {result3}")  

# def sort_colors_1(nums):
#     n = len(nums)
#     cnt = [0] * 3 
    
#     for num in nums:
#         cnt[num] += 1
#     idx = 0
#     for i in range(3):
#         for _ in range(cnt[i]):
#             nums[idx] = i
#             idx += 1

#     return nums

# ####  最优解 双指针 指针1指向0 位置 指针2指向2位置 中间位置为1 初始时curr指针从左到右遍历
# def sortColors(nums):
#     """
#     使用双指针法对颜色进行原地排序
#     时间复杂度: O(n)
#     空间复杂度: O(1)
#     """
#     p0 = 0
#     curr = 0
#     p2 = len(nums) - 1

#     while curr <= p2:
#         if nums[curr] == 0:
#             # 如果当前元素是0，与p0指向的元素交换
#             nums[p0], nums[curr] = nums[curr], nums[p0]
#             # p0和curr都向右移动
#             p0 += 1
#             curr += 1
#         elif nums[curr] == 2:
#             # 如果当前元素是2，与p2指向的元素交换
#             nums[p2], nums[curr] = nums[curr], nums[p2]
#             # p2向左移动，curr不动，因为交换过来的元素需要检查
#             p2 -= 1
#         else: # nums[curr] == 1
#             # 如果当前元素是1，它就在正确的位置，curr向右移动
#             curr += 1

# # 示例 1
# nums1 = [2, 0, 2, 1, 1, 0]
# sortColors(nums1)
# print(f"示例 1 输出: {nums1}")  # 输出: [0, 0, 1, 1, 2, 2]

# # 示例 2
# nums2 = [2, 0, 1]
# sortColors(nums2)
# print(f"示例 2 输出: {nums2}")  # 输出: [0, 1, 2]


# def sort_colors(nums):
#     n = len(nums)

#     for i in range(n):
#         for j in range(0, n-i-1):
#             if nums[j] > nums[j+1]:
#                 nums[j], nums[j+1] = nums[j+1], nums[j]
    
#     return nums

# nums = [2,0,2,1,1,0]
# result = sort_colors_1(nums)
# print(result)
# def sort_colors(nums):
#     dic = defaultdict(int)

#     result = []
#     # 先记录每个色出现的次数
#     for num in nums:
#         dic[num] +=1
    
#     # 按照0，1， 2的顺序天下
#     for i in range(3):
#         result.extend([i] * dic[i])
#     return result


##
"""
某种脚本语言中，一个形如 x+(a+pi-xn)+eps 的运算表达式由以下元素组成：
•	操作数：所有操作数都是变量，变量名为全小写字母且长度不超过10个字符；
•	操作符：只有+、-两种双目运算符；
•	括号：用于改变运算的优先级。当运算的优先级相同时，表达式从左向右进行计算。
给出一个运算表达式 calExpression后，为方便计算，将其转换为二叉树的表达形式，树中每个子树的计算结果会用于更上一层的子树的计算，
这样更清晰的展示了运算次序。最后以先序遍历方式输出树的节点序列。

比如，样例1中的运算表达式转换为以下的运算树：
输入
运算表达式 calExpression，仅包含小写字母和+、-、(、)字符，字符串长度不超过1000。
用例保证输入的字符串是一个合法的运算表达式。
输出
一个字符串数组，每个元素的值为节点中的操作数或者运算符。

样例1
复制输入：
"x+(a+pi-xn)+eps"
复制输出：
["+", "+", "x", "-", "+", "a", "pi", "xn", "eps"]

解释：
见题干中的图

样例2
复制输入：
"length+length"
复制输出：
["+", "length", "length"]
解释：
不同位置的操作数可以用同一个变量

样例3
复制输入：
"((x))"
复制输出：
["x"]
解释：
一个单独的操作数也是合法的表达式；括号内的内容也可以是不含操作符的操作数。
提示
【先序遍历】 以深度优先方式不重复地访问树的所有节点的遍历方式：先访问根节点，然后访问左子树， 最后访问右子树。
"""
from collections import defaultdict
from collections import deque

# class TreeNode:
#     def __init__(self, val):
#         self.val = val
#         self.left = None
#         self.right = None

def build_expression_tree(expression: str):
    n = len(expression)
    # step1: 将中缀表达式expression转为后缀表达式 用栈作为中间结构
    output = []
    stack = []
    for i in range(n):
        char = expression[i]
        if char.isalpha():
            # 读取完整的变量名
            var_start = i
            while i + 1 < n and expression[i + 1].isalpha():
                i += 1
            var_name = expression[var_start:i + 1]
            output.append(var_name) 
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
    print("output after reading char:", output)

expression = "x+(a+pi-xn)+eps"
result = build_expression_tree(expression)
print(f"result is: {result}")


    # # 仅包含小写字母和+、-、(、)字符

    # def precedence(op):
    #     if op == '+' or op == '-':
    #         return 1
    #     return 0
