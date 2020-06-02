import math
from matplotlib import pyplot as plt
from tqdm import tqdm


def dxdt(x, k, phi):
    a = 1e-1
    b = 1e-2
    c = 1e-4
    u0 = 0.0016
    v0 = 0.1472
    q = 1  # 0.629
    return 2 * c * math.sqrt(b / a) * (k - x) ** 2 * math.sqrt(x) + phi * (u0 + v0 - q * x)


def runge_kutta(h, x0, t0, k, phi):
    x = x0
    xs = [x0]
    times = [0]
    for i in tqdm(range(50000000)):
        k1 = dxdt(x, k, phi)
        k2 = dxdt(x + k1, k, phi)
        k3 = dxdt(x + k2 * (h / 2), k, phi)
        k4 = dxdt(x + k3 * h, k, phi)
        x_next = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x_next
        xs.append(x)
        t0 += h
        times.append(t0)
    return xs, times


def main():
    for k in [0.3, 0.5, 0.8]:
        for phi in [1e-7, 5e-5, 5e-3]:
            print(k, phi)
            xs, times = runge_kutta(h=0.1, x0=0.25, t0=0, k=k, phi=phi)
            plt.clf()
            plt.plot(times, xs)
            plt.xlabel('Time')
            plt.ylabel('$x$')
            plt.ylim((0, 1))
            plt.title(f'RK plot for $x$ over time with k={k}, $\\varphi$={phi}')
            plt.savefig(f'rk_plots/{k}_{phi}.png')


if __name__ == '__main__':
    main()
