def compute_cost(x, y, w, b):

    m = x.shape[0]
    total_cost = 0

    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b

        cost = (f_wb - y[i]) ** 2

        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost
