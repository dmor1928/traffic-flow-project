import datetime  # Saving filenames at unique time
from numba import njit, config  # Faster compiler
import numpy as np
from math import log10, floor  # Rounding in round_sig function
from scipy import signal, fftpack

# Rounds x to sig = significant figures, used when displaying avg_v in visualisation
def round_sig(x, sig=2):
    if x == 0.0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

@njit()
def correlation_lags(in1_len, in2_len, mode=0):
    if mode == 0:  # 0 = "full"
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == 1:  # 1 = "same"
        lags = np.arange(-in2_len + 1, in1_len)
        mid = lags.size // 2
        lag_bound = in1_len // 2
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == 2:  # 2 = "valid"
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags


# Freq_finder takes in oscillating data signal and desired cut_off
# Returns
    # modified copy of sample,
    # 2D array with frequencies and respective amplitudes, as two rows,
    # The freq and amplitude with the greatest amplitude,
    # Signal filtered to the primary frequencies.
def freq_finder(data, cut_off):
    # print(f"freq finder cut_off: {cut_off}")
    sample = data[cut_off:].copy()
    sample -= np.mean(sample)  # Remove DC component
    range = np.ptp(sample)  # Currently unused but might be interesting
    sample *= signal.windows.hann(num_ticks - cut_off)

    fft = fftpack.fft(sample)
    amp = np.abs(fft)
    freq = fftpack.fftfreq(sample.size, d=1)
    freq_amp = np.array([freq, amp])
    amp_pos = freq_amp[1, :].argmax()
    peak_freq = freq_amp[0, amp_pos]
    peak_freq_amp = [peak_freq, amp_pos]

    high_freq_fft = fft.copy()
    high_freq_fft[np.abs(freq) > peak_freq] = 0
    filtered_signal = fftpack.ifft(high_freq_fft)
    # print(f"number of high_freq_fft: {np.count_nonzero(high_freq_fft)}")

    return sample, freq_amp, peak_freq_amp, filtered_signal


# Cross correlation of car pairs to find total time lag and thus soliton speed
# Returns
    # The x oscillation separation between cars
    # The time lag behind the leading car using cross correlation
def cross_correlate_pairs(cars):  # Finds the time lag and avg x separation from the car in front, for each car.
    time_lags = np.empty(number_of_cars)
    x_osc_separation = np.empty(number_of_cars)

    for n, car in enumerate(cars):
        # leading_x = car.leading_car.X[cut_off:]
        car_x = car[0, cut_off:]
        leading_car = cars[int(car[2, 0])]
        leading_x = leading_car[0, cut_off:]
        if n + 1 == number_of_cars:
            leading_x += width

        car_avg_x = np.mean(car_x)
        leading_avg_x = np.mean(leading_x)  # Average

        x_osc_separation[n] = (
                    leading_avg_x - car_avg_x)  # Finds the x separation between car oscillation waves
        trans_zero = car_x[0]  # C entering data at zero gives more accurate correlation results
        only_pos = [num_ticks - cut_off,
                    num_ticks - cut_off + width]  # Only returns the +ve displacement and obviously < width
        correlation = signal.correlate(car_x - trans_zero, leading_x - trans_zero, method='fft')[
                      only_pos[0]:only_pos[1]]
        lags = correlation_lags(num_ticks - cut_off, num_ticks - cut_off)[
               only_pos[0]:only_pos[1]]  # Array of the lags that signal.correlate calculates at
        lag = lags[np.argmax(correlation)]
        time_lags[n] = lag  # The lag value where there is maximum correlation coefficient is assigned for car_n
    return x_osc_separation, time_lags


def soliton_final_state_detection(cars):
    decimal = 8
    earliest_tick_per_car = np.empty(number_of_cars)
    earliest_x_per_car = np.empty(number_of_cars)

    for i in range(number_of_cars):
        single_car_peak_ticks, _ = signal.find_peaks(cars[i, 0])
        num_peaks = len(single_car_peak_ticks)
        unique, counts = np.unique(cars[i, 0][single_car_peak_ticks].round(decimals=decimal), return_counts=True)
        single_car_peaks = np.dstack((unique, counts))
        single_car_peaks = single_car_peaks[np.all(single_car_peaks > 2, axis=2)]
        if single_car_peaks.size == 0:
            return 0, 0, 0, 0
        rows = np.shape(single_car_peaks)[0]

        peak_ticks = np.empty(rows, dtype=int)
        peak_x_positions = np.empty(rows)
        for j in range(rows):  # For each soliton...
            # print(np.where(np.round(car.X, decimals=decimal) == single_car_peaks[j, 0])[0])
            peak_ticks[j] = np.where(np.round(cars[i, 0], decimals=decimal) == single_car_peaks[j, 0])[0][0]
            peak_x_positions[j] = cars[i, 0, peak_ticks[j]]
            # print(f"wave {j} first tick: {peak_ticks[j]}")

        # print(f"peak_ticks: {peak_ticks}")
        earliest_tick_per_car[i] = np.amin(peak_ticks)
        k = np.argmin(peak_ticks)
        # print(f"k: {k}")
        # print(f"car.X[k] = {car.X[peak_ticks[k]]}")
        earliest_x_per_car[i] = cars[i, 0, peak_ticks[k]]
    number_of_solitons = rows

    earliest_tick = np.sort(earliest_tick_per_car)[0]
    earliest_x_position = earliest_x_per_car[np.argsort(earliest_tick_per_car)[0]]
    soliton_period_est = np.mean(
        np.diff(np.where(np.round(np.array(cars[number_of_cars-1, 0]), decimals=decimal) == single_car_peaks[j, 0])))
    return int(earliest_tick), earliest_x_position, int(soliton_period_est), number_of_solitons


@njit()
def numpy_move(x, v, dec, width, min_delta):
    # Boundary loop
    x[x >= width] -= width
    
    # Calculates delta_x
    delta_x = np.empty(number_of_cars)
    delta_x[0:-1] = x[1:] % width - x[0:-1] % width
    delta_x[-1] = x[0] % width - x[-1] % width
    delta_x %= width  # rather than delta_x[x<=0] += width faster?
    # print(delta_x)
    
    # Decelerating
    leading_car_v = np.empty(number_of_cars)
    leading_car_v[-1:] = v[0]
    leading_car_v[:-1] = v[1:]
    v[delta_x <= 10] = leading_car_v[delta_x <= 10] - dec
    
    # Accelerating
    # np.add(v, acc, out = v, where = (delta_x > 10))
    v[delta_x > 10] += acc
    
    # No reversing and sets max speed
    v[v < 0] = 0
    v[v > max_v] = max_v
    
    return x + v, v


@njit()
def numpy_move_avg(x, v, dec, width, min_delta):
    # Returns x_pos shifted by avg_v
    # Also needs to account for reverse boundary loop
    
    # Boundary loop
    # x[x >= width] -= width
    # x[x <= 0] = width - x[x <= 0]

    # Algorithm as usual
    # Calculates delta_x
    delta_x = np.empty(number_of_cars)
    delta_x[0:-1] = x[1:] % width - x[0:-1] % width
    delta_x[-1] = x[0] % width - x[-1] % width
    delta_x %= width
    # print(delta_x)
    
    # Decelerating
    leading_car_v = np.empty(number_of_cars)
    leading_car_v[-1:] = v[0]
    leading_car_v[:-1] = v[1:]
    v[delta_x <= min_delta] = leading_car_v[delta_x <= min_delta] - dec
    
    # Accelerating
    # np.add(v, acc, out = v, where = (delta_x > 10))
    v[delta_x > min_delta] += acc
    
    # No reversing and sets max speed
    # if v < -avg_v then it's moving v < 0 in fixed frame
    v[v < 0] = 0.0
    v[v > max_v] = max_v

    return x + v - np.mean(v), v


# Initial Vehicle Spawning

# Generates cars with random starting x_pos no clipping
# Generates random velocity, normal distribution (around 0.5), standard dev. (sd)
def random_car_setup(number_of_cars, sd):
    initial_v = np.random.normal(loc=0.5, scale=sd, size=number_of_cars)
    initial_x = np.zeros(number_of_cars)

    for i in range(1, number_of_cars):
        new_pos = np.random.randint(0, high=width - 10)
        while not (np.abs(new_pos - initial_x[:i]) >= 10).all():
            new_pos = np.random.randint(0, high=width - 10)
        initial_x[i] = new_pos
    initial_x = np.sort(initial_x)
    return initial_x, initial_v


# Generates cars with equally-spaced starting positions from [0, width - 10] (10 is car width)
# Generates random velocities, normal distribution (around 0.5), standard dev. (sd) for each car
def equal_car_setup(number_of_cars, sd):
    initial_v = np.random.normal(loc=0.5, scale=sd, size=number_of_cars)
    initial_x = np.linspace(0, width - 10, num=number_of_cars)
    return initial_x, initial_v


# Runs simulation (if num_ticks = 0 it never stops)
def run_sim(num_ticks, sample_modulus, initial_x, initial_v, dec, width, min_delta):
    tick = 1

    x = np.copy(initial_x)
    v = np.copy(initial_v)

    all_pos_data = np.empty((int(num_ticks / sample_modulus), number_of_cars))
    average_v = np.empty(int(num_ticks / sample_modulus))
    average_v[0] = np.mean(v)

    extra_time = 0  # For final state detection
    num_ticks_new = num_ticks_limit + 1
    earliest_tick, soliton_init_x, soliton_period_est, soliton_count = 0, 0, 0, 0
    running = True
    while running:

        if tick % sample_modulus == 0:
            sample_index = int(tick/sample_modulus) - 1
            all_pos_data[sample_index] = x
            average_v[sample_index] = np.mean(v)

        x, v = numpy_move_avg(x, v, dec, width, min_delta)

        # FINAL STATE DETECTIONS #

        # Soliton final state detection
        if tick == 5_000 + extra_time * 5_000:  #Checks for soliton peaks, every 5_000 ticks
            # print(f"current tick: {tick}")
            cars_final_state_det = translate_to_numpy(all_pos_data[0:sample_index], tick-1)
            earliest_tick, soliton_init_x, soliton_period_est, soliton_count = soliton_final_state_detection(cars_final_state_det)
            if (earliest_tick, soliton_init_x, soliton_period_est, soliton_count) == (0, 0, 0, 0):
                # print(f"extra time")
                final_state_reached = False
                extra_time += 1
            else:
                num_ticks_new = earliest_tick + int(soliton_period_est * 11)
                all_pos_data = all_pos_data[0:int(num_ticks_new / sample_modulus), :]  # Cut all_pos_data to new tick shape
                average_v = average_v[0:int(num_ticks_new / sample_modulus)]  # Cut average_v to new tick shape
                final_state_reached = True
                # print(f"Soliton final state detected. Running up to tick {num_ticks_new}... ")
                # print(f"earliest_tick: {earliest_tick}")
                # print(f"earliest_x_position: {soliton_init_x}")
                # print(f"soliton_period_est: {soliton_period_est}")

        # If soliton and ticks required < num_ticks, end early
        if tick >= num_ticks_new:
            running = False
            # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
            #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
            #                             ff_jump? (1/0), ff_jump_tick (o def)]
            all_pos_data = all_pos_data[0:int(num_ticks_new / sample_modulus), :]
            final_state_information = np.array([1, earliest_tick, 1, soliton_count, soliton_init_x, soliton_period_est,
                                                0, 0], dtype=int)
            return all_pos_data, average_v, num_ticks_new, final_state_information  #, earliest_x_position, soliton_period_est

        # Hard tick limit reached
        if tick == num_ticks:  # Reached end of simulation
            if final_state_reached is True:
                # print("Final state detected, but reached tick limit before 11 periods recorded.")
                # print(f"extra_time: {extra_time}")
                running = False
                # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
                #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
                #                             ff_jump? (1/0), ff_jump_tick (0 def)]
                final_state_information = np.array([1, earliest_tick, 1, soliton_count, soliton_init_x, soliton_period_est,
                                                    0, 0], dtype=int)
                return all_pos_data, average_v, num_ticks, final_state_information
            else:
                # print(f"extra_time: {extra_time}")
                # print("Final state was never detected. Region likely unstable.")
                running = False
                # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
                #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
                #                             ff_jump? (1/0), ff_jump_tick (0 def)]
                final_state_information = np.array([0, num_ticks, 2, 0, 0, 0, 0, 0], dtype=int)
                return all_pos_data, average_v, num_ticks, final_state_information  # Returns cut_off value = 0.

        # Free-flow final state detection
        if np.all(v == 1.):  # Optimisation makes free-flow runs end early.
            remaining_pos_data = np.tile(x, (int(num_ticks / sample_modulus) - sample_index, 1))
            remaining_pos_data = np.reshape(remaining_pos_data, (int(num_ticks / sample_modulus) - sample_index, number_of_cars))
            all_pos_data = np.concatenate((all_pos_data[:sample_index], remaining_pos_data))
            # all_pos_data[sample_index:] = remaining_pos_data
            average_v = np.concatenate((average_v[:sample_index], np.repeat(1., int(num_ticks / sample_modulus) - sample_index)), axis=None)

            # average_v[sample_index:] = np.repeat(1., int(num_ticks / sample_modulus) - sample_index)

            # Check if they JUMPED to free flow and if so what time
            # ff_jump_detection()
            # return ff_jump?, jump_tick

            # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
            #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
            #                             ff_jump? (1/0), ff_jump_tick]
            final_state_information = np.array([1, tick, 0, soliton_count, soliton_init_x, soliton_period_est, 0, 0], dtype=int)

            return all_pos_data, average_v, num_ticks, final_state_information  # Returns cut_off value of where free flow was reached.

        tick += 1


# Run analysis function.
# Inputs:
#   final_state_information, specifically (final state type and cut_off value for when final state is reached)
#   cars array (all_pos_data reformatted)
# Returns
#   Frequencies of system oscillations from FFT / power spectra frequency finder analysis
#   Soliton speed measurements from cross correlation method
def run_analysis(final_state_information):
    # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
    #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
    #                             ff_jump? (1/0), ff_jump_tick]
    final_state_reached = final_state_information[0]
    cut_off = final_state_information[1]  # = time to reach final state
    final_state_type = int(final_state_information[2])

    # SOLITON FINAL STATE ANALYSIS #
    if final_state_type == 1:

        # Cross correlation of car pairs to find time lags --> soliton speed
        x_osc_separation, time_lags = cross_correlate_pairs(cars)
        # inst_soliton_speed = x_osc_separation / time_lags  # Attempt at instantaneous speed calculation, DOESN'T WORK
        overall_soliton_speed = width / np.sum(time_lags)  # More consistent and makes much more sense
        # print(f"time_lags: {time_lags}")
        # print(f"cross_correlation result: {overall_soliton_speed}")
        # print(f"soliton_speed: {soliton_speed}")
        # print(f"overall_soliton_speed: {overall_soliton_speed}")

        # Avg speed frequency finder
        avg_v_sample, avg_v_freq_amp, avg_v_peak_freq_amp, avg_v_filtered_sig = freq_finder(avg_v, cut_off)
        # print(f"avg. v peak_freq:  {avg_v_peak_freq_amp[0]}")

        # X position frequency finder (for car_0)
        x_sample, x_freq_amp, x_peak_freq_amp, x_filtered_sig = freq_finder(cars[0, 0], cut_off)
        # print(f"x peak_freq:       {x_peak_freq_amp[0]}")
        # fft_speed_result = width * x_peak_freq_amp[0]
        # print(f"sol. speed from x position FFT power spectrum result: {fft_speed_result}")

        # Variance frequency finder
        velocities = np.stack([car[1] for car in cars])
        velocities += avg_v
        v_variance = np.var(velocities, axis=0)
        v_var_sample, v_var_freq_amp, v_var_peak_freq_amp, v_var_filtered_sig = freq_finder(v_variance, cut_off)
        # print(f"var_peak_freq_amp: {v_var_peak_freq_amp[0]}")

        # Peak detection from moving box, counting peaks   WORKING, BUT NOT AS ACCURATE?
        # final_state_tick = num_ticks - 10_000
        # soliton_period = round(width / overall_soliton_speed)
        # # print(f"soliton_period: {soliton_period}")
        # num_boxes = (num_ticks - final_state_tick) // soliton_period
        # box_ticks = np.split(np.arange(final_state_tick, final_state_tick + round(num_boxes * soliton_period)), num_boxes)
        # peaks = np.empty(num_boxes)
        # # print(f"num_boxes: {num_boxes}")
        # for n in range(num_boxes):
        #     box_n = cars[0, 0, int(final_state_tick + soliton_period * n): int(final_state_tick + soliton_period * (n+1))]
        #     # print(signal.find_peaks((box_n)))
        #     peaks_n, _ = signal.find_peaks(box_n)
        #     peaks[n] = len(peaks_n)
        # print(f"peaks: {peaks}")

        final_avg_v = np.mean(avg_v[cut_off:])
        final_var_v = np.mean(v_variance[cut_off:])

        final_state_analysis = np.asarray([
            final_avg_v, final_var_v,
            overall_soliton_speed,  # num_solitons,
            x_peak_freq_amp[0], x_peak_freq_amp[1],
            avg_v_peak_freq_amp[0], avg_v_peak_freq_amp[1],
            v_var_peak_freq_amp[0], v_var_peak_freq_amp[1],
            cut_off, num_ticks
        ])
        return final_state_analysis

    # END OF SOLITON FINAL STATE ANALYSIS

    # FREE FLOW FINAL STATE ANALYSIS #
    if final_state_type == 0:
        # Free-flow jumping detection (bouncing not yet working)
        sampling = 2_000
        jump_tick = 0
        mean_avg_v_boxes = np.empty((num_ticks // sampling + 1))
        x = np.empty_like(mean_avg_v_boxes)
        for i in range(num_ticks // sampling + 1):
            if i == num_ticks // sampling:
                mean_avg_v_boxes[i] = np.mean(avg_v[sampling * i:num_ticks - 1])
                x[i] = 0.5 * (sampling * i + num_ticks - 1)
            else:
                mean_avg_v_boxes[i] = np.mean(avg_v[sampling * i:sampling * (i + 1)])
                x[i] = 0.5 * (sampling * i + sampling * (i + 1))

        mean_avg_v_diffs = np.diff(mean_avg_v_boxes)
        jump_up_err = 0.05
        for i in range(1, num_ticks // sampling):
            if (mean_avg_v_diffs[i] > jump_up_err): # Gradient dramatically increased (shot to FF) (blue)
                jump_tick = i
                # print(f"jumped to free flow at ~{i}")
                break

        if jump_tick != 0:
            final_state_information[6] = 1
            final_state_information[7] = jump_tick

        final_avg_v = 1.0
        final_var_v = 0

        final_state_analysis = np.asarray([
            final_avg_v, final_var_v,
            0,  # num_solitons,
            0, 0,
            0, 0,
            0, 0,
            cut_off, num_ticks
        ])
        return final_state_analysis

    # END OF FREE FINAL STATE ANALYSIS

    # Didn't reach final state
    if final_state_type == 2:
        velocities = np.stack([car[1] for car in cars])
        velocities += avg_v
        v_variance = np.var(velocities, axis=0)
        # print(avg_v)
        final_avg_v = np.mean(avg_v[cut_off:])
        # print(final_avg_v)
        final_var_v = np.mean(v_variance[cut_off:])

        final_state_analysis = np.asarray([
            final_avg_v, final_var_v,
            0,  # num_solitons,
            0, 0,
            0, 0,
            0, 0,
            cut_off, num_ticks
        ])
        return final_state_analysis

    # END OF FREE FLOW FINAL STATE ANALYSIS

    #final_state_analysis = np.asarray([
    #   overall mean avg_v and v_var, (determines free flow or jam)   +2
    #   soliton effect speed, number of solitons,                     +2
    #   Oscillation data:
    #       x_position frequencies and amplitude of oscillation,      +2
    #       avg_v frequencies and amplitude of oscillation,           +2
    #       v_var frequencies and amplitude of oscillation,           +2

    #   The ticks data is recorded at (e.g. 10_000 to 20_000 ticks
    #   ])


# Complete removal of Cars class to enable numba implementation and parallel processing
def translate_to_numpy(all_pos_data, num_ticks):
    cars_np = np.empty((number_of_cars, 3, num_ticks))
    for n, car_np in enumerate(cars_np):
        car_np[0] = all_pos_data[:, n]
        car_np[1] = np.append(init_v[n], all_pos_data[:, n][1:] - all_pos_data[:, n][:-1])
        car_np[1][car_np[1] <= -width + max_v] += width
        if n == number_of_cars - 1:
            car_np[2, 0] = 0
        else:
            car_np[2, 0] = n + 1
    return cars_np


# PROGRAM RUNNING #

# Running and sample parameters
num_ticks_limit = 100_000  # 1_000_000
cut_off = 2_000  # How long system takes to settle. Larger cut_off ==> longer run time
sample_mod = 1  # How often x_pos_data is collected in run_sim
num_runs = 1  # The number of runs per dec/acc value

# Initial value parameters
init_v_sd = 0.15
np.random.seed(42)

# System parameters
max_v = 1.0  # Always fixed
acc = 0.0005  # Always fixed
number_of_cars = 100  # Might be changing variable in future?

# Changing variables
vertical_pixels = 201
horizontal_pixels = 201
width_N_vals = np.linspace(10., 35., num=vertical_pixels, endpoint=True)  # width / number_of_cars dimensionless
print(width_N_vals)
dec_acc_vals = np.linspace(0, 50., num=horizontal_pixels, endpoint=True)   # dec / acc dimensionless
print(dec_acc_vals)
#width_vals = np.linspace(300., 3300., num=100, endpoint=False)  # array of width values being run over = width_N_vals * N

min_delta = 10  # Potentially change in future?

print(datetime.datetime.now().strftime("%H%M%S"))

# RUNNING NEW RANDOM SIMULATIONS iterating over range of dec/acc
initial_parameters = np.empty((vertical_pixels, horizontal_pixels, num_runs, 2, number_of_cars))
system_parameters = np.empty((vertical_pixels, horizontal_pixels, 6))
final_state_analysis = np.empty((vertical_pixels, horizontal_pixels, num_runs, 11))
final_state_information = np.empty((vertical_pixels, horizontal_pixels, num_runs, 8))

width = int(number_of_cars * min_delta)
init_x, fixed_v = equal_car_setup(number_of_cars, init_v_sd)  # Generates the fixed_v values, for when fixing initial v.
print(fixed_v)
init_v = fixed_v

for i, width_N in enumerate(width_N_vals):
    width = int(width_N * number_of_cars)
    print(f"width: {width}")
    for j, dec_acc in enumerate(dec_acc_vals):
        dec = dec_acc * acc
        system_parameters[i, j] = np.asarray([number_of_cars, width, acc, dec, min_delta, num_ticks_limit])
        # print(f"dec/acc = {dec_acc}: ")
        num_ticks = num_ticks_limit
        for run in range(num_runs):
            # print(f"run {run+1} / {num_runs}")

            # For randomising initial speed and spacing#
            # if width_N <= 15:  # random_cat_setup is inefficient at high densities, and is essentially equal spacing.
            #     init_x, init_v = equal_car_setup(number_of_cars, init_v_sd)
            # else:
            #     init_x, init_v = random_car_setup(number_of_cars, init_v_sd)

            # When inital speed is fixed but initial position spacing changes as width changes.
            init_x, _ = equal_car_setup(number_of_cars, init_v_sd)

            # Run sim function
            # Returns
            #   x position data for all cars at every tick
            #   average speed at every tick (since if avg_v frame is taken, information is lost)
            #   num_ticks_new bad variable name, but returns the actual number of ticks the simulation ran for.
            #       This is because some arrays end early (e.g. 11 soliton oscillations in 15K ticks,
            #       Others run to tick limit, and although ff systems don't run to limit, they *are* calculated to limit.
            #       Although not specified in its input, it is crucial for run_analysis.
            #       (a better variable name would be good)
            #   The final state properties =
            #       [ final_state_reached?, time_to_reach, state_type, soliton_count,
            #         soliton_init_x , soliton_period_est, ff_jump?, ff_jump_tick ]
            # (Note: The accuracy of ff_jump and ff_jump_tick has not yet been tested)
            all_pos_data, avg_v, num_ticks_new, final_state_run_info = run_sim(num_ticks, sample_mod, init_x, init_v, dec, width, min_delta)
            num_ticks = num_ticks_new
            # print(final_state_run_info)
            # print(f"Ticks ran:{num_ticks}")

            # Translates all_pos_data information into a cars numpy array
            #   If not making plots, could probably just be implemented at the end of run_sim tbh, but it doesn't make a difference.
            cars = translate_to_numpy(all_pos_data, num_ticks)

            final_state_analysis[i, j, run] = run_analysis(final_state_run_info)  # Saves final_state_properties from run_analysis
            initial_parameters[i, j, run] = np.asarray([init_x, init_v])  # Saves initial positions and speeds
            final_state_information[i, j, run] = final_state_run_info  # Saves final_state_info from run_sim

# print(final_state_information)
# file_ID = datetime.datetime.now().strftime("%H%M%S%f-%d-%m-" + str(num_ticks))  # Generates unique file name
np.savez("databank/05-02-200x200_50000ticks_eqspacing_100N_0to50da_seed42_fixed",
         init_param=initial_parameters,
         final_param=final_state_analysis,
         sys_param=system_parameters,
         final_info=final_state_information)

print(datetime.datetime.now().strftime("%H%M%S"))  # Prints end time
