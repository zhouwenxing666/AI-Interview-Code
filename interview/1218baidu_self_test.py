class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
class Solution:
    def searchRange(self, head: ListNode, target: int):
         # 返回 [start_index, end_index]，不存在返回 [-1, -1]
         
        idx = 0
        left = -1
        right = -1
        cur = head

        # 一直右移 直到碰到第一个值等于target的节点
        while cur and cur.val < target:
            idx += 1
            cur = cur.next
        # 判空和判错
        if not cur or cur.val != target:
            return [-1, -1]
        left = idx # 此时left 为当前idx
        
        # 一直右移 直到碰到第一个值大于target的节点
        while cur and cur.val == target:
            idx += 1
            cur = cur.next
        right = idx - 1# 此时right为当前idx-1

        return [left, right]
        




