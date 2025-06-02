# main.py

import numpy as np
import matplotlib.pyplot as plt
import json
from config import param_bounds
# np.random.seed(42)
# 适应度函数 ------------核心！！！
def evaluate_fitness(ind):
    pitch, skew, rake, thick, c_root, c_tip = ind

    # 升阻比估计  弦长总和(c_root + c_tip)/(后掠角abs(skew) + 前掠角abs(rake) + 厚度thick*10)
    lift_drag = (c_root + c_tip) / (0.01 + abs(skew) + abs(rake) + thick * 10)

    # 叶片能效估计 升阻比加权lift_drag * 增益因子(1 + 0.5 * (pitch - 1)) - 厚度thick*10
    efficiency = lift_drag * (1 + 0.5 * (pitch - 1)) - thick * 10

    # 启动性能估计： 螺距对初始推力的贡献 pitch*0.5 - 掠角 abs(skew)*0.01 + 厚度thick*0.2
    startup = pitch * 0.5 - abs(skew) * 0.01 + thick * 0.2

    # 综合适应度（加权）
    return 0.5 * efficiency + 0.3 * startup + 0.2 * lift_drag


# 初始化种群 随机数
def init_population(size):
    # 返回种群数组
    return np.array([
        [np.random.uniform(*param_bounds[k]) for k in param_bounds]
        for _ in range(size)
    ])


# 选择优秀个体作为父代
def select_parents(pop, fitnesses, num):
    # 对适应度从小到大排序，返回索引列表
    idx = np.argsort(fitnesses)[-num:]
    # 返回这些最优个体
    return pop[idx]


# 交叉操作
def crossover(parents, n_offspring):
    #  初始化空列表
    offspring = []
    # 交叉生成子代
    for _ in range(n_offspring):
        # 从父代中随机选两个不同个体
        p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
        # 混合两个个体的基因
        alpha = np.random.rand()
        child = alpha * p1 + (1 - alpha) * p2
        # 添加子代
        offspring.append(child)
    return np.array(offspring)


# 变异操作
# 用于引入新的基因变异，以保持种群多样性、避免陷入局部最优。
def mutate(pop, rate=0.1):
    # 遍历每个个体
    for ind in pop:
        # 以一定概率对个体的某个参数进行变异
        if np.random.rand() < rate:
            # 随机选择变异的位置
            idx = np.random.randint(len(ind))
            key = list(param_bounds.keys())[idx]
            # 执行变异
            ind[idx] = np.random.uniform(*param_bounds[key])
    return pop


# 遗传算法主程序
def run_genetic_algorithm(generations=500, pop_size=300, n_parents=100):
    # 初始化种群
    pop = init_population(pop_size)
    best_history = []
    # 进化过程
    for gen in range(generations):
        # 适应度评估
        fitnesses = np.array([evaluate_fitness(ind) for ind in pop])
        best_fit = np.max(fitnesses)
        best_history.append(best_fit)

        print(f"Generation {gen + 1}: Best Fitness = {best_fit:.4f}")
        parents = select_parents(pop, fitnesses, n_parents)
        offspring = crossover(parents, pop_size - n_parents)
        offspring = mutate(offspring, rate=0.2)
        pop = np.vstack((parents, offspring))

    # 最优个体输出
    best_idx = np.argmax([evaluate_fitness(ind) for ind in pop])
    best_ind = pop[best_idx]
    best_params = {name: float(value) for name, value in zip(param_bounds.keys(), best_ind)}

    print("\nBest Design Parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.4f}")

    # 保存为 JSON 文件
    with open("result.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\nResult saved to result.json.")

    # 绘图展示优化过程
    plt.plot(best_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Optimization Progress")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_genetic_algorithm()
