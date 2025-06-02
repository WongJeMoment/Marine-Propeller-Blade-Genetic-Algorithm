# main.py

import numpy as np
import matplotlib.pyplot as plt
import json
from config import param_bounds

# 适应度函数
def evaluate_fitness(ind):
    pitch, skew, rake, thick, c_root, c_tip = ind

    # 升阻比估计：厚度、掠角小，弦长大 → 升力好
    lift_drag = (c_root + c_tip) / (0.01 + abs(skew) + abs(rake) + thick * 10)

    # 叶片能效估计：结合升阻比与螺距比
    efficiency = lift_drag * (1 + 0.5 * (pitch - 1)) - thick * 10

    # 启动性能估计：低速时的扭矩响应
    startup = pitch * 0.5 - abs(skew) * 0.01 + thick * 0.2

    # 综合适应度（加权）
    return 0.5 * efficiency + 0.3 * startup + 0.2 * lift_drag


# 初始化种群
def init_population(size):
    return np.array([
        [np.random.uniform(*param_bounds[k]) for k in param_bounds]
        for _ in range(size)
    ])


# 选择优秀个体作为父代
def select_parents(pop, fitnesses, num):
    idx = np.argsort(fitnesses)[-num:]
    return pop[idx]


# 交叉操作
def crossover(parents, n_offspring):
    offspring = []
    for _ in range(n_offspring):
        p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
        alpha = np.random.rand()
        child = alpha * p1 + (1 - alpha) * p2
        offspring.append(child)
    return np.array(offspring)


# 变异操作
def mutate(pop, rate=0.1):
    for ind in pop:
        if np.random.rand() < rate:
            idx = np.random.randint(len(ind))
            key = list(param_bounds.keys())[idx]
            ind[idx] = np.random.uniform(*param_bounds[key])
    return pop


# 遗传算法主程序
def run_genetic_algorithm(generations=50, pop_size=30, n_parents=10):
    pop = init_population(pop_size)
    best_history = []

    for gen in range(generations):
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
