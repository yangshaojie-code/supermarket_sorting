def step_func(current, target, step):
    if current < target - step:
        return current + step
    if current > target + step:
        return current - step
    return target
