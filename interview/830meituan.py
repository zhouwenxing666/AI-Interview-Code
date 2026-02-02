### 8.30 美团笔试
import sys
import json
import numpy as np

# 题1

def main1():
    try:
        # 读取一行输入
        line = sys.stdin.readline().strip()
        if not line:
            return
        n, k = map(int, line.split())

        # visited字典用于存储遇到的数字以及在哪一步遇到，格式为{数字，步数}
        # path列表用于按顺序记录演变路径上的每一个数组
        visited = {n: 0}
        path = [n]

        step = 0
        while step < k:
            if n % 2 == 0:
                n //=2 
            else:
                n = 3 * n + 1
            
            step += 1

            # 检查当前数字是否之前遇到过，如果遇到过，说明找到了循环
            if n in visited:
                # 循环开始的步数
                prev_step = visited[n]
                cycle_len = step - prev_step # 循环的长度
                remaining_k = k - step # 剩下还需要走的步数
                offset = remaining_k % cycle_len # 在循环内的偏移量
                # 最终结果是循环开始点之后偏移offset的那个数
                final_n = path[prev_step + offset]
                print(final_n)
                return

            # 如果是新数字 记录下来
            visited[n] = step
            path.append(n)
        
        # 如果k比较小，在找当前循环前就结束了，直接输出当前的n
        print(n)
    except(IOError,ValueError):
        return

def main2():
    try:
        # 读取一行输入
        line = sys.stdin.readline().strip()
        # 如何将line转换为json对象        
        data = json.loads(line)
    except(json.JSONDecodeError,IndexError):
        return

    # 将列表转换为numpy数组以便进行向量化计算
    # c0 c1 分别代表类别0和类别1的训练数据
    C0 = np.array(data['train'][0])
    C1 = np.array(data['train'][1])
    test_samples = np.array(data['test'])

    # 2.参数估计
    # 计算每个类别的均值向量
    mu0 = np.mean(C0[:, :-1], axis=0)
    mu1 = np.mean(C1[:, :-1], axis=0)

    # 获取特征纬度
    m = test_samples.shape[1]

    # 计算类内散度矩阵sw
    # （c - mu).t @ (c - mu)是计算sum( (x -mu)(x-mu)^t)的一种高效方法
    S0 = (C0[:, : -1] - mu0).T @ (C0[:, : -1] - mu0)
    S1 = (C1[:, : -1] - mu1).T @ (C1[:, : -1] - mu1)
    Sw = S0 + S1

    # 添加一个小的单位矩阵（正则化）以确保矩阵可逆
    Sw_regularized = Sw + 1e-6 * np.identity(m)

    # 3.计算最佳投影方向w
    # w = inv(sw)*(mu1-mu0)
    Sw_inv = np.linalg.inv(Sw_regularized)
    w = Sw_inv @ (mu1 - mu0)

    # 4.分类准则
    # 计算两个类别均值在w上的投影
    m0_proj = w.T @ mu0
    m1_proj = w.T @ mu1

    predictions = []
    # 遍历每个测试样本进行分类
    # nums = [1,2 ,3,4,5,6,7,8,9,10]
    # for i in enumerate(nums,0,len(nums)-2):
    for x in test_samples:
        # 将测试样本投影到w上
        y_proj = w.T @ x

        # 判断投影点离哪个类的投影中心更近
        if abs(y_proj - m0_proj) < abs(y_proj - m1_proj):
            predictions.append(0)
        else:
            predictions.append(1)
    
    # 5 打印要求的json格式输出结果
    print(json.dumps(predictions))

def main2_1():
    data = json.loads(sys.stdin.readline().strip())
    train = np.array(data['train'], dtype=float)
    test = np.array(data['test'], dtype=float)

    X, y = train[:, :-1], train[:, -1].astype(int)
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)

    Sw = np.zeros((X.shape[1], X.shape[1]))

    for c, mu in [(0, mu0), (1, mu1)]:
        diff = X[y == c] - mu
        Sw += diff.T @ diff
    Sw += 1e-6 * np.eye(X.shape[1])

    w = np.linalg.inv(Sw) @ (mu1 - mu0)
    m0, m1 = w @ mu0, w @ mu1
    t = (m0 + m1) / 2

    preds = [int(abs(w @ x - m1) < abs(w @ x - m0)) for x in test]
    print(json.dumps(preds))

# 题3 暴力解 只过0.2
def main3():
    n = int(sys.stdin.readline().strip())
    arr = list(map(int, sys.stdin.readline().split()))
    
    total = 0
    for i in range(n):
        mn =arr[i]
        mx = arr[i]
        seen = set()
        for j in range(i, n):
            if arr[j] in seen:
                break
            seen.add(arr[j])
            mn = min(mn, arr[j])
            mx = max(mx, arr[j])
            total += (mx - mn + 1) - (j - i + 1)
    print(total)

# 题3最优解
def main3_1():
    n = int(sys.stdin.readline().strip())
    a = list(map(int, sys.stdin.readline().split()))
    
    # 子数组总和与所有子数组长度之和
    total_sub = n * (n + 1) // 2
    total_len = n * (n + 1) * (n + 2) // 6

    # 所有子数组最大值之和
    left = [0]*n
    right = [0]*n
    st = []
    for i in range(n):
        while st and a[st[-1]] <= a[i]:
            st.pop()
        left[i] = i - st[-1] if st else i + 1
        st.append(i)
    st.clear()

    for i in range(n-1, -1, -1):
        while st and a[st[-1]] < a[i]:
            st.pop()
        right[i] = st[-1] - i if st else n - i
        st.append(i)

    sum_max = 0
    for i in range(n):
        sum_max += a[i] * left[i] * right[i]

    # 所有子数组最小值之和
    left = [0]*n
    right = [0]*n
    st.clear()
    for i in range(n):
        while st and a[st[-1]] >= a[i]:
            st.pop()
        left[i] = i - st[-1] if st else i + 1
        st.append(i)

    st.clear()
    for i in range(n-1, -1, -1):
        while st and a[st[-1]] > a[i]:
            st.pop()
        right[i] = st[-1] - i if st else n - i
        st.append(i)

    sum_min = 0
    for i in range(n):
        sum_min += a[i] * left[i] * right[i]

    ans = (sum_max - sum_min + total_sub) - total_len
    print(ans)

if __name__ == "__main__":
    main2_1()


