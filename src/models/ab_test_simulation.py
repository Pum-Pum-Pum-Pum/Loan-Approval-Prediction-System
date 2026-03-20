import numpy as np
from scipy import stats


RANDOM_STATE = 42
N_USERS_PER_GROUP = 5000
BASE_CONVERSION_A = 0.12
BASE_CONVERSION_B = 0.135


def simulate_ab_test(n_a: int, n_b: int, p_a: float, p_b: float, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    group_a = rng.binomial(1, p_a, size=n_a)
    group_b = rng.binomial(1, p_b, size=n_b)
    return group_a, group_b


def two_proportion_z_test(group_a: np.ndarray, group_b: np.ndarray):
    x1, n1 = group_a.sum(), len(group_a)
    x2, n2 = group_b.sum(), len(group_b)

    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        'conversion_a': p1,
        'conversion_b': p2,
        'uplift_abs': p2 - p1,
        'uplift_pct': (p2 - p1) / p1,
        'z_stat': z,
        'p_value': p_value,
    }


def main() -> None:
    group_a, group_b = simulate_ab_test(
        n_a=N_USERS_PER_GROUP,
        n_b=N_USERS_PER_GROUP,
        p_a=BASE_CONVERSION_A,
        p_b=BASE_CONVERSION_B,
        random_state=RANDOM_STATE,
    )

    results = two_proportion_z_test(group_a, group_b)

    print('A/B TEST SIMULATION')
    print('-------------------')
    print(f"Group A users        : {len(group_a)}")
    print(f"Group B users        : {len(group_b)}")
    print(f"Conversion A         : {results['conversion_a']:.4f}")
    print(f"Conversion B         : {results['conversion_b']:.4f}")
    print(f"Absolute uplift      : {results['uplift_abs']:.4f}")
    print(f"Relative uplift      : {results['uplift_pct']:.2%}")
    print(f"Z-statistic          : {results['z_stat']:.4f}")
    print(f"P-value              : {results['p_value']:.6f}")
    print(f"Statistically significant at 0.05? : {results['p_value'] < 0.05}")


if __name__ == '__main__':
    main()
