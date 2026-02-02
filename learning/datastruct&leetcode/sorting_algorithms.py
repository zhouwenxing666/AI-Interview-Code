# -*- coding: utf-8 -*-

"""
常见排序算法汇总
包含: 冒泡排序, 选择排序, 插入排序, 希尔排序, 归并排序, 快速排序, 堆排序, 桶排序

"""

import random
from typing import List

def bubble_sort(arr: List[int]) -> List[int]:
    """
    冒泡排序 (Bubble Sort)
    
    原理:
    比较相邻的元素。如果第一个比第二个大，就交换他们两个。
    对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
    针对所有的元素重复以上的步骤，除了最后一个。
    持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
    
    时间复杂度: 平均 O(n^2), 最坏 O(n^2), 最好 O(n)
    空间复杂度: O(1)
    稳定性: 稳定
    """
    n = len(arr)
    if n <= 1:
        return arr
    
    # 也就是需要遍历 n-1 次
    for i in range(n - 1):
        # 设定一个标记，若为False，则表示此次循环没有进行交换，也就是待排序列已经有序，排序已经完成。
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def selection_sort(arr: List[int]) -> List[int]:
    """
    选择排序 (Selection Sort)
    
    原理:
    首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。
    再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
    重复第二步，直到所有元素均排序完毕。
    
    时间复杂度: O(n^2)
    空间复杂度: O(1)
    稳定性: 不稳定
    """
    n = len(arr)
    if n <= 1:
        return arr
        
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def insertion_sort(arr: List[int]) -> List[int]:
    """
    插入排序 (Insertion Sort)
    
    原理:
    通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
    
    时间复杂度: 平均 O(n^2), 最坏 O(n^2), 最好 O(n)
    空间复杂度: O(1)
    稳定性: 稳定
    """
    n = len(arr)
    if n <= 1:
        return arr
        
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        # 将选出的元素 key 与前面的元素进行比较，如果前面的元素大于 key，则将前面的元素后移
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def shell_sort(arr: List[int]) -> List[int]:
    """
    希尔排序 (Shell Sort)
    
    原理:
    是插入排序的一种更高效的改进版本。也称为缩小增量排序。
    先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序。
    
    时间复杂度: 取决于增量序列，平均 O(n log n) ~ O(n^2)
    空间复杂度: O(1)
    稳定性: 不稳定
    """
    n = len(arr)
    if n <= 1:
        return arr
        
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

def merge_sort(arr: List[int]) -> List[int]:
    """
    归并排序 (Merge Sort)
    
    原理:
    采用分治法（Divide and Conquer）的一个非常典型的应用。
    将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。
    
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    稳定性: 稳定
    """
    if len(arr) <= 1:
        return arr
        
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr: List[int], left: int = None, right: int = None) -> List[int]:
    """
    快速排序 (Quick Sort)
    
    原理:
    通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，
    然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。
    
    时间复杂度: 平均 O(n log n), 最坏 O(n^2), 最好 O(n log n)
    空间复杂度: O(log n)
    稳定性: 不稳定
    """
    # 为了方便调用，处理默认参数
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
        
    if left < right:
        partition_index = partition(arr, left, right)
        quick_sort(arr, left, partition_index - 1)
        quick_sort(arr, partition_index + 1, right)
    return arr

def partition(arr: List[int], left: int, right: int) -> int:
    pivot = left
    index = pivot + 1
    i = index
    while i <= right:
        if arr[i] < arr[pivot]:
            arr[i], arr[index] = arr[index], arr[i]
            index += 1
        i += 1
    arr[pivot], arr[index - 1] = arr[index - 1], arr[pivot]
    return index - 1


def heap_sort(arr: List[int]) -> List[int]:
    """
    堆排序 (Heap Sort)
    
    原理:
    是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：
    即子结点的键值或索引总是小于（或者大于）它的父节点。
    
    时间复杂度: O(n log n)
    空间复杂度: O(1)
    稳定性: 不稳定
    """
    n = len(arr)
    
    # 构建大顶堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
        
    # 一个个交换元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]   # 交换
        heapify(arr, i, 0)
        
    return arr

def heapify(arr: List[int], n: int, i: int):
    largest = i
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
    
    if l < n and arr[i] < arr[l]:
        largest = l
        
    if r < n and arr[largest] < arr[r]:
        largest = r
        
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换
        heapify(arr, n, largest)

def bucket_sort(arr: List[int], bucket_size: int = 5) -> List[int]:
    """
    桶排序 (Bucket Sort)
    
    原理:
    桶排序是计数排序的升级版。它利用了函数的映射关系，高效与否的关键就在于这个映射函数的确定。
    工作原理：假设输入数据服从均匀分布，将数据分到有限数量的桶里，每个桶再分别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排）。
    
    时间复杂度: 平均 O(n + k), 最坏 O(n^2), 最好 O(n)
    空间复杂度: O(n + k)
    稳定性: 稳定 (取决于桶内排序算法)
    """
    if len(arr) == 0:
        return arr

    min_val, max_val = min(arr), max(arr)
    
    # 桶的数量
    bucket_count = (max_val - min_val) // bucket_size + 1
    buckets = [[] for _ in range(bucket_count)]
    
    # 利用映射函数将数据分配到各个桶中
    for i in range(len(arr)):
        buckets[(arr[i] - min_val) // bucket_size].append(arr[i])
        
    arr.clear()
    for bucket in buckets:
        # 这里使用插入排序，因为桶内数据量较小，插入排序效率较高
        insertion_sort(bucket)
        arr.extend(bucket)
        
    return arr

if __name__ == "__main__":
    # 测试代码
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("原始数组:", test_arr)
    
    print("\n冒泡排序:", bubble_sort(test_arr.copy()))
    print("选择排序:", selection_sort(test_arr.copy()))
    print("插入排序:", insertion_sort(test_arr.copy()))
    print("希尔排序:", shell_sort(test_arr.copy()))
    print("归并排序:", merge_sort(test_arr.copy()))
    
    qs_arr = test_arr.copy()
    quick_sort(qs_arr)
    print("快速排序:", qs_arr)
    
    hs_arr = test_arr.copy()
    heap_sort(hs_arr)
    print("堆排序:  ", hs_arr)

    bs_arr = test_arr.copy()
    print("桶排序:  ", bucket_sort(bs_arr))
