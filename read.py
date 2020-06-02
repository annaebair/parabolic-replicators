import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os


def average_trajectories(params_string):
    """
    Averages the trajectories for the given params_string on two different seeds
    """
    trajectory_pairs = {}
    path = 'sim_data/'
    if os.path.exists(path):
        items = os.listdir(path)
        for item in items:
            params = item[:-10]
            # Line below restricts to only the given param string
            # TODO: clean this up to have an option later
            if params == params_string:
                img_name = item[:-4]
                x_counts = []
                times = []
                with open('sim_data/' + item) as f:
                    for row in f:
                        if len(row) > 25:
                            seed, a_rate, b_rate, c_rate, frac_occupied, phi, diffusion_param, frac_x, grid_size, sim_time = row.split()
                        elif len(row) > 0:
                            t, x = row.split()
                            t, x = float(t), float(x)
                            times.append(t)
                            x_counts.append(x)
                    if params in trajectory_pairs:
                        trajectory_pairs[params].append((x_counts, times))
                    else:
                        trajectory_pairs[params] = [(x_counts, times)]

    # TODO: Assumes only two trajectories; make more flexible for multiple trajectories in the future.
    for k, v in trajectory_pairs.items():
        pair1, pair2 = v
        x1, t11 = pair1
        x2, t21 = pair2

    x1_idx = 0
    x2_idx = 0
    current_1 = x1[0]
    current_2 = x2[0]
    x_vals = [current_1, current_2]
    t1 = deque(t11)
    t2 = deque(t21)
    current_time_1 = t1.popleft()
    current_time_2 = t2.popleft()
    time_options = [current_time_1, current_time_2]
    averages = []
    times = []
    while True:
        if t1 and t2:
            next_checkpoint = min(time_options)
            traj = np.argmin(time_options)
            averages.append(np.mean(x_vals))
            times.append(next_checkpoint)
            # print(time_options)
            if traj == 0:
                time_options[0] = t1.popleft()
                x1_idx += 1
                x_vals[0] = x1[x1_idx]
            else:
                time_options[1] = t2.popleft()
                x2_idx += 1
                x_vals[1] = x2[x2_idx]
        else:
            break

    plt.plot(t11, x1)
    plt.plot(t21, x2)
    plt.plot(times, averages)
    plt.show()


def detect_stationarity(num_intervals):
    path = 'sim_data/'
    if os.path.exists(path):
        items = os.listdir(path)
        for item in items:
            param = item[:-4]
            x_counts = []
            times = []
            with open(os.path.join(path, item)) as f:
                # TODO reformulate so that the first line is read in as the parameters and all subsequent lines are data
                for row in f:
                    if len(row) > 25:
                        seed, a_rate, b_rate, c_rate, frac_occupied, phi, diffusion_param, frac_x, grid_size, sim_time = row.split()
                    elif len(row) > 0:
                        t, x = row.split()
                        t, x = float(t), float(x)
                        times.append(t)
                        x_counts.append(x)

            interval = float(sim_time) / num_intervals
            checkpoint = interval
            to_average = []
            idx = 0
            prev_avg = 0
            tau_s = None
            for t in times[:-1]:
                if t > checkpoint:
                    checkpoint += interval
                    avg = np.mean(to_average)
                    stdev = np.std(to_average)
                    if abs(avg-prev_avg) < stdev:
                        tau_s = t
                        break
                    else:
                        to_average = [x_counts[idx]]
                        prev_avg = avg
                else:
                    to_average.append(x_counts[idx])
                idx += 1
            plt.clf()
            plt.plot(times, [i / 1250 for i in x_counts])
            if tau_s is not None:
                plt.axvline(x=tau_s, color='r')
            plt.title(f'$\\varphi$:{phi}, diff:{diffusion_param}, $\\tau_s$: {tau_s}')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(f'sim_plots/D{num_intervals}', param + ".png"))


def main():
    detect_stationarity(num_intervals=25)


if __name__ == '__main__':
    main()
