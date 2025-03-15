from typing import List
import random

def phaseFraction_generator(num_phases: int,
                            phaseFraction_min: int = 8,
                            phaseFraction_step: int = 4) -> List[List[int]]:
    """
    Function 枚举所有满足以下条件的相分数组合:
      1. 相数 = num_phases = 1/2/3/4
      2. 各相分数皆 >= phaseFraction_min = 8， 比 8 小， XRD就看不到了
      3. 各相分数之和 = 100
      4. 每个相分数的步进为 phaseFraction_step (可模拟笔记中 8,12,16 等递增方式)

    参数:
    num_phases : int
        相的个数(例如 2 表示 2 相, 3 表示 3 相, 4 表示 4 相, 等等)
    phaseFraction_min : int, 默认 8
        每个相所占的最小比例(如笔记中每个相至少 8%)
    total : int, 默认 100
        总和(如笔记中相分数总和 100%)
    phaseFraction_step : int, 默认 4
        步长(如笔记中常见的 8,12,16,20... 这类 4 的倍数)

    返回:
    ----------
    List[List[int]], 所有符合条件的相分数列表, 每个列表的长度等于 num_phases
    =[
        num_phases=1 [
            [100],
        ]
        num_phases=2 [
            [x, 100-x], x_min = 8, x step size = 4
        ]
        num_phases=3 [
            [x, y, 100-x-y], x_min = 8, y_min = 8, x step size = 4, y step size = 4
        ]
        num_phases=4 [
            [x, y, z, 100-x-y-z], x_min = 8, y_min = 8, z_min = 8, x step size = 4, y step size = 4, z step size = 4
        ]
    ]

    refer
    """
    TOTAL = 100
    results = []

    # 1) 相数 = 1
    if num_phases == 1:
        # 只有 [100] 一种可能
        if 100 >= phaseFraction_min and (100 - phaseFraction_min) % phaseFraction_step == 0:
            results.append([100])
        return results

    # 2) 相数 = 2
    elif num_phases == 2:
        # 遍历第一个相 x, 第二个相 = 100 - x
        for x in range(phaseFraction_min, TOTAL + 1, phaseFraction_step):
            y = TOTAL - x
            if y >= phaseFraction_min and (y - phaseFraction_min) % phaseFraction_step == 0:
                results.append([x, y])
        return results

    # 3) 相数 = 3
    elif num_phases == 3:
        # x + y + z = 100
        # x, y, z 均 >= phaseFraction_min，且步进 = phaseFraction_step
        for x in range(phaseFraction_min, TOTAL + 1, phaseFraction_step):
            for y in range(phaseFraction_min, TOTAL - x + 1, phaseFraction_step):
                z = TOTAL - x - y
                if z >= phaseFraction_min and (z - phaseFraction_min) % phaseFraction_step == 0:
                    results.append([x, y, z])
        return results

    # 4) 相数 = 4
    elif num_phases == 4:
        # x + y + z + w = 100
        for x in range(phaseFraction_min, TOTAL + 1, phaseFraction_step):
            for y in range(phaseFraction_min, TOTAL - x + 1, phaseFraction_step):
                for z in range(phaseFraction_min, TOTAL - x - y + 1, phaseFraction_step):
                    w = TOTAL - x - y - z
                    if w >= phaseFraction_min and (w - phaseFraction_min) % phaseFraction_step == 0:
                        results.append([x, y, z, w])
        return results

    # 如果 num_phases 不在 1~4 范围内, 返回空列表或抛出异常
    else:
        # 也可改为 raise ValueError("num_phases must be 1, 2, 3, or 4.")
        return results
# 演示如何使用 phaseFraction_generator 函数
# 示例: 2 相组合
# combos_2 = phaseFraction_generator(num_phases=2, phaseFraction_min=8, phaseFraction_step=4)
# print("2 相组合示例(仅显示前 10 个):", combos_2[:10], "... 共", len(combos_2), "个")

# 示例: 3 相组合
# combos_3 = phaseFraction_generator(num_phases=3, phaseFraction_min=8, phaseFraction_step=4)
# print("3 相组合示例(仅显示前 10 个):", combos_3[:10], "... 共", len(combos_3), "个")

# 示例: 4 相组合
# combos_4 = phaseFraction_generator(num_phases=4, phaseFraction_min=8, phaseFraction_step=4)
# print("4 相组合示例(仅显示前 10 个):", combos_4[:10], "... 共", len(combos_4), "个")

# 示例: 1 相组合
# combos_1 = phaseFraction_generator(num_phases=1, phaseFraction_min=8, phaseFraction_step=4)
# print("1 相组合示例:", combos_1)