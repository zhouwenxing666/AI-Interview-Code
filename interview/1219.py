# 实现反转链表

class ListNode:
    def __init__(self, val, next: None):
        self.val = val
        self.next = next
    

def reverseListNode(head):
    if not head:
        return None

    pre = ListNode(-1, head)
    cur = head

    while cur.next:
        next = cur.next

        cur.next = next.next
        next.next = pre.next
        pre.next = next
    
    return pre.next

node4 = ListNode(4,None)
node3 = ListNode(3,node4)
node2 = ListNode(2,node3)
node1 = ListNode(1,node2)

print(f"head is:{node1.val}")


now_head = reverseListNode(node1)

cur = now_head
while cur:
    print(cur.val)
    cur = cur.next




import torch
import torch.nn as nn
import math

class RMS_NORM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.afs = nn.Parameter(dim=hidden_size)
        # self.afa = afa
    
    def forward(self, x):
        #return (x / rms)  * afa
        #均方根
        rms = torch.sqrt((x**2).mean(dim=-1))

        return (x / (rms + 'inf'))  * self.afs
