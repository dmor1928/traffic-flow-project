import datetime  # Saving filenames at unique time
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from numba import njit, config  # Faster compiler
import numba_scipy
import numpy as np
import pygame
import random
from math import log10, floor  # Rounding in round_sig function
from os import listdir, getcwd  # For reading filenames from /databank
from os.path import isfile, join  # For reading filenames from /databank
from scipy import signal, fftpack
from time import sleep
import cProfile


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'CMU Serif Roman'
plt.rcParams.update({'font.size': 12})

(width, height) = (600, 200)
background = (255, 255, 255)
road_height = height * 0.5

screen = pygame.display.set_mode((width, height))

pygame.font.init()
my_font = pygame.font.SysFont('Arial', 16)

pygame.display.set_caption('Traffic Flow') 
pygame.event.set_allowed(pygame.QUIT)
screen.fill(background)


# Rounds x to sig = significant figures, used when displaying avg_v in visualisation
def round_sig(x, sig=2):
    if x == 0.0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)


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
        lags = signal.correlation_lags(num_ticks - cut_off, num_ticks - cut_off)[
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
        single_car_peaks = single_car_peaks[np.all(single_car_peaks > 1, axis=2)]
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

def read_data(filename):
    global acc, dec, dec_acc
    file = np.load(filename)
    x_init = file['x_init']
    v_init = file['v_init']
    param = file['param']
    # x_data = file['x_data']

    cars = []
    number_of_cars, width, acc, dec = param
    dec_acc = dec / acc
    number_of_cars = int(number_of_cars)
    
    for n in range(number_of_cars):
        cars.append(Car(x_init[n], road_height, v_init[n]))

    return cars


class Car: 
    def __init__(self, x, y, v): 
        
        self.x = x
        self.y = y
        self.v = v
        self.colour = (0, 0, 255)
        self.X = None
        self.T = None
        self.V = None
        self.leading_car = None
        self.laps = 0

        #(SCRAPPED) variables for first density thing
        #self.L = 200
        #self.density = 0
        #self.Density = []
    
    # Find the leading car and assign it to self.leading_car
    def assign_leading_car(self):
        if cars.index(self) == number_of_cars - 1:
            self.leading_car = cars[0]
        else:
            self.leading_car = cars[cars.index(self) + 1]
    
    def assign_X_V(self, tick):
        #What if we only saved every nth X / V value?
        self.X[tick-1] = self.x 
        self.V[tick-1] = self.v
    
    def display(self):
        #Each vehicle represented by 10-pixel diameter ball
        pygame.draw.circle(screen, self.colour, (self.x, self.y), 5)
    
    def move(self):
        #Boundary loop
        if self.x > width:
            self.x = self.x - width
        
        #Distance from leading car (accounting for if leading car looped)
        if self.leading_car.x < self.x:
            delta_x = (width - self.x) + self.leading_car.x
        else:
            delta_x = self.leading_car.x - self.x

        #Rough implementation of Netlogo patches
        # Instead of looking at squares in fixed static grid,
        # Always looks 10 pixels ahead
        
        min_delta = 10
        if delta_x <= min_delta:
            self.v = self.leading_car.v - dec
            #Future SUVAT implementation?
            #self.a = ( front_car_v**2 - self.v**2 ) / (2 * delta_x)   
        else:
            self.v += acc
        
        #Sets max_v
        if self.v > max_v:
            self.v = max_v
            
        #Sets min_v = 0
        if self.v <= 0:
            self.v = 0
        
        #Movement
        self.x += self.v

    def calc_density(self):
        
        if self.L > width:
            print("Error: L > width")
            return 0
        
        num_cars = 1 #counts itself
        l_0 = self.x - self.L/2
        l_1 = self.x + self.L/2
        

        #Looks a distance L/2 infront + behind, 
        # counts number of cars within distance
        if l_0 < 0:
            
            # (0, self.x)
            for car in cars:
                if car.x < self.x:
                    num_cars += 1
                if car.x == self.x:
                    break
            # (width - L/2, width)
            for car in reversed(cars):
                if car.x > width - self.L/2:
                    num_cars += 1
                else: 
                    break
            
        if l_1 > width:
            
            # (self.x, width)
            for car in reversed(cars):
                if car.x > self.x:
                    num_cars += 1
                else:
                    break
            # (0, l_1 - width)
            for car in cars:
                if car.x < l_1 - width:
                    num_cars += 1
                else: 
                    break
        
        self.Density.append(num_cars / self.L)


# Calculates traffic density across road at sample points.
def system_density(sample_spacing, L):  # Needs to include t in parameters
    sample_points = np.arange(0, width, sample_spacing)
    sample_density = np.empty(sample_spacing)
    density_normaliser = 0.1 * (sample_spacing - (sample_spacing % 10)) + 1

    for p in sample_points:
        l_0 = p - L / 2
        l_1 = p + L / 2
        car_count = 0

        # Looks a distance L/2 infront + behind, counts number of cars within distance
        if l_0 < 0:

            # (0, p)
            for car in cars:
                if car.x < p:
                    car_count += 1
                    if car.v > 0.4:
                        car_count -= 1
                if car.x >= p:
                    break
            # (width + (p-L/2), width)
            for car in reversed(cars):
                if car.x > width + (p - 20):
                    car_count += 1
                    if car.v > 0.4:
                        car_count -= 1
                else:
                    break

        elif l_1 > width:

            # (p, width)
            for car in reversed(cars):
                if car.x >= p:
                    car_count += 1
                    if car.v > 0.4:
                        car_count -= 1
                else:
                    break
            # (0, L/2 - (width-p))
            for car in cars:
                if car.x <= L / 2 - width + p:
                    car_count += 1
                    if car.v > 0.4:
                        car_count -= 1
                else:
                    break

        else:
            for car in cars:
                if (l_0 <= car.x) and (car.x <= l_1):
                    car_count += 1
                    if car.v > 0.4:
                        car_count -= 1
                if car.x > l_1:
                    break
        sample_density = np.append(sample_density, car_count)
    sample_density *= density_normaliser
    return sample_points, sample_density


# Draws the black dots representing density in display
def draw_density(sample_points, sample_density):
    if np.any(sample_points):
        for i in range(len(sample_points)):
            if sample_density[i] > 0:
                x = sample_points[i]
                y = road_height - sample_density[i] * (height - road_height) * 0.8
                pygame.draw.circle(screen, (0, 0, 0), (x, y), 2)


def drawRoad():
    pygame.draw.line(screen, (0,0,0), (0, road_height), (width, road_height))


def display_cars(x):
    # Each vehicle represented by 10-pixel diameter ball (radius = 5)
    for x_pos in x:
        pygame.draw.circle(screen, cars[int(np.asarray(x == x_pos).nonzero()[0])].colour, (x_pos % width, road_height), 5)


#@njit()
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


#@njit()
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

    # cars[5].colour = (255, 0, 0)  #See oscillation easier in PyGame sim

    #soliton_points_x = []
    #soliton_points_t = []

    x = np.copy(initial_x)
    v = np.copy(initial_v)

    all_pos_data = np.empty((int(num_ticks / sample_modulus), number_of_cars))
    average_v = np.empty(int(num_ticks / sample_modulus))
    average_v[0] = np.mean(v)

    extra_time = 0  # For final state detection
    num_ticks_new = num_ticks + 1
    running = True
    while running:

        if tick % sample_modulus == 0:
            sample_index = int(tick/sample_modulus) - 1
            all_pos_data[sample_index] = x
            average_v[sample_index] = np.mean(v)

        x, v = numpy_move_avg(x, v, dec, width, min_delta)

        # REMOVE COMMENT (AND DISABLE @NJIT()) TO DISPLAY \/
        #sleep(0.00002)
        screen.fill(background)
        drawRoad()
        display_cars(x)
        avg_v_display = my_font.render(f"avg. speed: {round_sig(np.mean(v), 4)}", False, (0, 0, 0))
        screen.blit(avg_v_display, (0, height - 20))
        tick_display = my_font.render(f"tick: {tick}", False, (0, 0, 0))
        screen.blit(tick_display, (0, height - 40))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                running = False

        """ Drawing density (old and probably broken)
        Calculates density on road from samples every 50 ticks
        if ticks % 50 == 0:
            sample_points, sample_density = system_density(10, 40)
            #Collecting soliton points when above certain value
            for p in range(len(sample_points)):
                if sample_density[p] >= 0.8:
                    soliton_points_x = np.append(soliton_points_x, sample_points[p])
                    soliton_points_t = np.append(soliton_points_t, ticks)
        if ticks >= 100:
            draw_density(sample_points, sample_density)
        """

        # FINAL STATE DETECTIONS #

        # Soliton final state detection
        if tick == 5_000 + extra_time * 5_000:  #Checks for soliton peaks, every 5_000 ticks
            # print(f"current tick: {tick}")
            cars_final_state_det = translate_to_numpy(all_pos_data[0:sample_index], tick-1)
            earliest_tick, soliton_init_x, soliton_period_est, soliton_count = soliton_final_state_detection(cars_final_state_det)
            if (earliest_tick, soliton_init_x, soliton_period_est) == (0, 0, 0):
                # print(f"extra time")
                extra_time += 1
            else:
                num_ticks_new = earliest_tick + int(soliton_period_est * 11)
                all_pos_data = all_pos_data[0:int(num_ticks_new / sample_modulus), :]  # Cut all_pos_data to new tick shape
                average_v = average_v[0:int(num_ticks_new / sample_modulus)]  # Cut average_v to new tick shape
                print(f"Soliton final state detected. Running up to tick {num_ticks_new}... ")
                # print(f"earliest_tick: {earliest_tick}")
                # print(f"earliest_x_position: {earliest_x_position}")
                # print(f"soliton_period_est: {soliton_period_est}")

        # If soliton and ticks required < num_ticks, end early
        if tick == num_ticks_new:
            running = False
            # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
            #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
            #                             ff_jump? (1/0), ff_jump_tick (o def)]

            final_state_information = np.array([1, earliest_tick, 1, soliton_count, soliton_init_x, soliton_period_est,
                                                0, 0], dtype=int)
            return all_pos_data, average_v, num_ticks_new, final_state_information  #, earliest_x_position, soliton_period_est

        # Hard tick limit reached
        if tick == num_ticks:  # Reached end of simulation
            if extra_time == (num_ticks // 5_000) - 1:
                print("Final state was never detected. Region likely unstable.")
                running = False
                # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
                #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
                #                             ff_jump? (1/0), ff_jump_tick (0 def)]
                final_state_information = np.array([0, num_ticks, 2, 0, 0, 0, 0, 0], dtype=int)
                return all_pos_data, average_v, num_ticks, final_state_information  # Returns cut_off value = num_ticks.
            else:
                print("Final state detected, but reached tick limit before 11 periods recorded.")
                running = False
                # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
                #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
                #                             ff_jump? (1/0), ff_jump_tick (0 def)]
                final_state_information = np.array([1, earliest_tick, 1, soliton_count, soliton_init_x, soliton_period_est,
                                                    0, 0], dtype=int)
                return all_pos_data, average_v, num_ticks, final_state_information

        # Free-flow final state detection
        if np.all(v == 1.):  # Optimisation makes free-flow runs end early.
            remaining_pos_data = np.tile(x, (int(num_ticks / sample_modulus) - sample_index, 1))
            remaining_pos_data = np.reshape(remaining_pos_data, (int(num_ticks / sample_modulus) - sample_index, number_of_cars))
            all_pos_data[sample_index:] = remaining_pos_data
            average_v[sample_index:] = np.repeat(1., int(num_ticks / sample_modulus) - sample_index)

            # Check if they JUMPED to free flow and if so what time
            # ff_jump_detection()
            # return ff_jump?, jump_tick

            # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
            #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
            #                             ff_jump? (1/0), ff_jump_tick]
            final_state_information = np.array([1, tick, 0, 0, 0, 0, 0, 0], dtype=int)

            return all_pos_data, average_v, num_ticks, final_state_information  # Returns cut_off value of where free flow was reached.

        tick += 1


# GRAPHS AND PLOTS #

def run_plots(cut_off):
    
    #(scrapped) Car Density around car_0
    """
    plt.plot(cars[0].Density, linewidth = 1)
    plt.savefig('Density.png', dpi=300)
    plt.show()
    """

    # (likely scrapped) Soliton points from system_density
    """ 
    plot = False
    if plot is True:
        fig2, ax2 = plt.subplots()
        ax2.scatter(soliton_points_x, soliton_points_t, marker="+", linewidths = 1)
        ax2.set_title("Soliton points for dec_acc= "+str(dec_acc)+", run= "+str(run+1))
        ax2.set_xlabel("Road coordinate x")
        ax2.set_ylabel("Time (ticks)")
        plt.show() 
    """

    # Car path plot (ticks on y-axis, x on x-axis)
    # Inputs:
    #   ax:                 Axis to be plotted on
    #   cars_array:         Array of cars to be plotted (so can just be a pair or entire collection)
    #   frame:              Changes the title of plot accordingly
    #   tick_range [a, b]:  Plots x positions of each car in cars_array to tick_range
    # Returns
    #   Nothing, but plots the formatted data on the inputted axis ax
    def plot_car_paths(ax, fig, cars_array, frame, tick_range):
        if cars_array == cars:
            ax.set_title(
                f"Car paths in {frame} frame of reference in final oscillating state \n"
                fr" $\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'acc'}}}$ = {dec_acc}, N = {number_of_cars}")
        else:
            ax.set_title(
                fr"Path of every 5$^\mathsf{{{'th'}}}$ vehicle in {frame} frame of reference" f"\n"
                fr"$\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'acc'}}}$ = {dec_acc}, N = {number_of_cars}, L = {width}")
        ax.set_xlabel(fr"$x$-position on road")
        ax.set_ylabel(fr"ticks")
        colour_map = plt.get_cmap('viridis')
        plot_colours = [colour_map(i) for i in np.linspace(0, 1, len(cars_array))]

        border_cars = []
        not_border_cars = []
        for car in cars_array:
            for t in range(tick_range[0], tick_range[1]):  # range(1, ticks - 1):
                if (car.X[t - 1] - car.X[t] >= width - 1) or (car.X[t - 1] - car.X[t] <= -width + 1):
                    border_cars.append(car)
                    break
                elif t == tick_range[1] - 1:
                    not_border_cars.append(car)
        for car in border_cars:
            last_split = tick_range[0]
            plot_colour = plot_colours[cars_array.index(car)]
            for t in range(tick_range[0], tick_range[1]):
                if (car.X[t - 1] - car.X[t] >= width - 1) or (car.X[t - 1] - car.X[t] <= -width + 1):
                    ax.plot(car.X[last_split:(t - 1)], np.arange(last_split, t - 1, 1), color=plot_colour)
                    last_split = t
                if t == tick_range[1]:
                    ax.plot(car.X[last_split:(t - 1)], np.arange(last_split, t - 1, 1), color=plot_colour)
        for car in not_border_cars:
            plot_colour = plot_colours[cars_array.index(car)]
            ax.plot(car.X[tick_range[0]: tick_range[1]], np.arange(tick_range[0], tick_range[1], 1), color=plot_colour)
        #plt.savefig('Car paths, speed var = default, random spacing.png', dpi=300)

    # fig3, ax3 = plt.subplots()
    # car_arr = [cars[0], cars[5], cars[10], cars[15], cars[20], cars[25]]
    # plot_car_paths(ax3, fig3, car_arr, "average speed", [cut_off, num_ticks - 1])
    # plt.savefig("varying_oscillations.png", dpi=300)
    # plt.savefig("oscillations.png", dpi=300)
    
    # Plot sample of x position for car pairs (plot_car_paths can be used instead?)
    plot = False
    if plot is True:
        car_path_plots = []
        for n, car in enumerate(cars):
            fig4, ax4 = plt.subplots(3)
            fig4.suptitle(f"Oscillation of all cars in avg_v frame \n "
                          fr"$\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'dec'}}}$ = {dec_acc}, N = {number_of_cars}")

            ax4[0].plot(np.arange(num_ticks - 10_000, num_ticks), car.X[-10_000:])
            dxdt = np.diff(car.X[-10_000:])
            ax4[1].plot(np.arange(num_ticks - 10_000, num_ticks-1), dxdt)
            d2xdt2 = np.diff(dxdt)
            ax4[2].plot(np.arange(num_ticks - 10_000, num_ticks-2), d2xdt2)
            # ax4.set_title(f"Oscillation of car {n + 1} and {1 if n == number_of_cars - 1 else n + 2} in avg_v frame \n "
                          # fr"$\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'dec'}}}$ = {dec_acc}, N = {number_of_cars}")
            # ax4.plot(np.arange(num_ticks-10_000, num_ticks), car.leading_car.X[-10_000:])
            # plt.savefig(f"car_pair_oscillations/{n}", dpi=300)
            car_path_plots.append((fig4, ax4))
            plt.close(fig4)

    # Cross correlation

    # FOR DEBUGGING CROSS CORRELATION BUG
    # if time_lags[time_lags == 1.].any:
    #     print(f"time_lags: {time_lags}")
    #     fig5, ax5 = plt.subplots()
    #     ax5.set_xlabel("time translation")
    #     ax5.set_ylabel("correlation value")
    #     ax5.set_title(f"{n}")
    #     ax5.plot(correlation)
    #     ax5.vlines(np.argmax(correlation), ymin=np.min(correlation), ymax=np.max(correlation), colors=["red"])
    #     plt.show()

    # plot = False
    # if plot is True:
    #     plt.plot(soliton_speed)
    #     plt.show()

    # Oscillations #

    def frequency_plot(
            suptitle, fig, ax,
            sample, sample_label, cut_off, filtered, ylabel0, mean_label0,
            freq_amp, peak_freq_amp
    ):
        fig.suptitle(f"{suptitle}")
        ax[0].set_xlabel("Ticks")
        ax[0].set_ylabel(ylabel0)
        ax[0].plot(np.arange(cut_off, num_ticks, 1), sample, label=sample_label)
        ax[0].plot(np.arange(cut_off, num_ticks, 1), filtered, label="Filtered")
        ax[0].legend(loc="upper right", framealpha=0.6)

        # ax[1].set_xlabel("Frequency (ticks" + r"$^{-1}$" + ")")
        # ax[1].set_ylabel("Amplitude")
        # ax[1].plot(freq_amp[0], freq_amp[1])
        # ax[1].vlines(peak_freq_amp[0], ymin=0, ymax=freq_amp[1, peak_freq_amp[1]], colors=["orange"])
        # ax[1].locator_params(axis='y', nbins=2)

        freq_amp_zoom = freq_amp.T[((freq_amp.T[:, 0] > 0) & (freq_amp.T[:, 0] < 0.05))]
        freq_zoom = freq_amp_zoom[:, 0]
        amp_zoom = freq_amp_zoom[:, 1]
        ax[1].set_xlabel("Frequency (ticks" + r"$^{-1}$" + ")")
        ax[1].set_ylabel("Amplitude")
        ax[1].plot(freq_zoom, amp_zoom)
        ax[1].vlines(peak_freq_amp[0], ymin=0, ymax=freq_amp[1, peak_freq_amp[1]], colors=["orange"])
        ax[1].annotate(f"Peak frequency = {round_sig(peak_freq_amp[0], 5)}", (0.1, 0.9), xycoords='axes fraction',
                       va='top')
        ax[1].locator_params(axis='y', nbins=2)

        samplemax = max(sample.min(), sample.max(), key=abs)
        ax[0].set_ylim(-abs(samplemax), abs(samplemax))
        func = lambda x, pos: mean_label0 if np.isclose(x, 0) else "{:.2e}".format(x)
        ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(func))
        # ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        fig.align_ylabels(ax)

    # Avg. speed oscillation frequency finder
    avg_v_sample, avg_v_freq_amp, avg_v_peak_freq_amp, avg_v_filtered_sig = freq_finder(avg_v, cut_off)
    # print(f"avg. v peak_freq:  {avg_v_peak_freq_amp[0]}")

    # avg_v_range = np.ptp(avg_v_sample)
    # avg_v_period = 1 / avg_v_peak_freq_amp[0]
    # print(f"avg_v osc. period: {avg_v_period}")

    # Avg. v power spectrum

    fig6, ax6 = plt.subplots(2, layout="tight")
    frequency_plot(
        f"Frequency for oscillation of instantaneous average system speed\n run = {run}", fig6, ax6,
        avg_v_sample, "Avg. speed sample", cut_off, avg_v_filtered_sig, "Avg. system speed", r"$\overline{v}$",
        avg_v_freq_amp, avg_v_peak_freq_amp
    )
    plt.close(fig6)

    # Finding frequency for car_0 x-position oscillation
    x_sample, x_freq_amp, x_peak_freq_amp, x_filtered_sig = freq_finder(cars[0].X, cut_off)
    # print(f"x peak_freq:       {x_peak_freq_amp[0]}")

    # Power Spectrum of car_0 X position
    fig7, ax7 = plt.subplots(2, layout="tight")
    frequency_plot(
        f"Frequency for oscillation of car 0\n run = {run}", fig7, ax7,
        x_sample, "X-coordinate sample", cut_off, x_filtered_sig, "X-coordinate", r"$\overline{x}$",
        x_freq_amp, x_peak_freq_amp
    )
    plt.close(fig7)

    # Variance compared to instantaneous mean speed
    velocities = np.stack([car.V for car in cars])
    velocities += avg_v
    variance = np.var(velocities, axis=0)

    # Variance frequency finder
    var_sample, var_freq_amp, var_peak_freq_amp, var_filtered_sig = freq_finder(variance, cut_off)
    # print(f"var_peak_freq_amp: {var_peak_freq_amp[0]}")

    # Variance power spectrum plot

    fig8, ax8 = plt.subplots(2, layout="tight")
    frequency_plot(
        f"Frequency for oscillation of speed variance\n run = {run}", fig8, ax8,
        var_sample, "Variance sample", cut_off, var_filtered_sig, "Variance", r"$\overline{\sigma^2}$",
        var_freq_amp, var_peak_freq_amp
    )
    plt.close(fig8)

    # Final state detection graph
    fig9, ax9 = plt.subplots()
    averaged_avg_v = np.empty(num_ticks)
    for i in range(num_ticks):
        averaged_avg_v[i] = np.mean(avg_v[0:i])

    avged_avg_v_variance = np.empty(num_ticks)
    for i in range(2, num_ticks):
        avged_avg_v_variance[i] = np.var(averaged_avg_v[0:i])
    ax9.plot(avg_v)
    ax9.plot(averaged_avg_v)
    #ax9[1].plot(avged_avg_v_variance)
    plt.close(fig9)

    decimal = 8
    earliest_tick_per_car = np.empty(number_of_cars)
    earliest_x_per_car = np.empty(number_of_cars)

    # for i, car in enumerate(cars):
    #     single_car_peak_ticks, _ = signal.find_peaks(car.X)
    #     num_peaks = len(single_car_peak_ticks)
    #     unique, counts = np.unique(cars[i].X[single_car_peak_ticks].round(decimals=decimal), return_counts=True)
    #     single_car_peaks = np.dstack((unique, counts))
    #     single_car_peaks = single_car_peaks[np.all(single_car_peaks > 1, axis=2)]
    #     print(f"car {i}: {single_car_peaks}")
    #     rows = np.shape(single_car_peaks)[0]
    #
    #     peak_ticks = np.empty(rows, dtype=int)
    #     peak_x_positions = np.empty(rows)
    #     for j in range(rows):  # For each soliton...
    #         # print(np.where(np.round(car.X, decimals=decimal) == single_car_peaks[j, 0])[0])
    #         peak_ticks[j] = np.where(np.round(car.X, decimals=decimal) == single_car_peaks[j, 0])[0][0]
    #         peak_x_positions[j] = car.X[peak_ticks[j]]
    #         # print(f"wave {j} first tick: {peak_ticks[j]}")
    #
    #     # print(f"peak_ticks: {peak_ticks}")
    #     earliest_tick_per_car[i] = np.amin(peak_ticks)
    #     k = np.argmin(peak_ticks)
    #     # print(f"k: {k}")
    #     # print(f"car.X[k] = {car.X[peak_ticks[k]]}")
    #     earliest_x_per_car[i] = car.X[peak_ticks[k]]
    #
    # earliest_tick = np.sort(earliest_tick_per_car)[0]
    # earliest_x_position = earliest_x_per_car[np.argsort(earliest_tick_per_car)[0]]
    # soliton_period_est = np.mean(
    #     np.diff(np.where(np.round(np.array(cars[0].X), decimals=decimal) == single_car_peaks[0, 0])))

    # Phase portrait

    car_index = 4
    cars[car_index].V[1:] += avg_v[1:]
    #cars[0].X -= np.mean(cars[0].X)

    print(final_state_run_info)
    final_state_tick = int(final_state_run_info[1])
    final_state_type = final_state_run_info[2]
    soliton_period_est = final_state_run_info[5]

    #if final_state_type == 1.0:
        #ax10.plot(cars[car_index].X[final_state_tick:final_state_tick + soliton_period_est], cars[car_index].V[final_state_tick:final_state_tick+soliton_period_est], c='red', linewidth=2, zorder=0, linestyle='--', dashes=(5, 5))
    #if final_state_type == 0.0:
        #ax10.scatter(x=cars[car_index].X[:final_state_tick][-1], y=cars[car_index].V[:final_state_tick][-1], linewidths=3, c='red')

    norm = plt.Normalize(1, final_state_tick)

    # Colorbar
    fig11, ax11 = plt.subplots(layout="tight")

    # louis stuff
    fig10, ax10 = plt.subplots(layout="tight")
    #ax10.set_axis_off()

    for t in range(1, 10):

        carpos_av = []
        carpos_osc = []
        for n in range(number_of_cars):
            cacheval = np.average(cars[n].X[cut_off:])

            carpos_av.append(cacheval)
            carpos_osc.append(cars[n].X[10000+(t*300)] - cacheval)

        peakpos = carpos_av[np.argmax(carpos_osc)]
        if t == 1:
            superpeakpos = peakpos
            posdiff = 0
        else:
            posdiff = superpeakpos - peakpos
        carpos_final = np.add(carpos_av, posdiff)


        ax10.plot(carpos_final, carpos_osc, color=[0, 0.5 + t / 20, 0])


    # Plot lables
    label_title = ['free flow', 'soliton-oscillation', 'unknown final'][final_state_type]
    fig10.suptitle(f"Profiles of multiple traffic waves  \n overlaid on top of each other  \n"
                   fr"$\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'acc'}}}$ = {dec_acc}, "
                   fr"N = {number_of_cars}, L = {width}")
    ax10.set_xlabel(fr"Position along soliton $x$")
    ax10.set_ylabel(fr"Oscillation amplitude $X$")
    plt.xlim(0, 2000)

    # Color bar labels
    ax10.grid()
    plt.savefig("hellogamers.png", dpi=300)


    plt.close()

    # for (fig, ax) in car_path_plots[0]:
    #     show_figure(fig)
    #     plt.show()

    show_figure(fig10)
    plt.show()


def run_analysis(final_state_information):
    # final_state_information = [ final_state_reached? (1/0), time_to_reach, ff_or_soliton_or_none (0/1/2),
    #                             soliton_count (0 def), soliton_init_x (0 def), soliton_period_est (0 def),
    #                             ff_jump? (1/0), ff_jump_tick]
    final_state_reached = final_state_information[0]
    cut_off = final_state_information[1]  # = time to reach final state
    final_state_type = final_state_information[2]

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

    # END OF SOLITON FINAL STATE ANALYSIS


    # FREE FLOW FINAL STATE ANALYSIS #
    if final_state_type == 1:

        # Free-flow jumping detection (bouncing not yet working)
        sampling = 5_000
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

    # END OF FREE FLOW FINAL STATE ANALYSIS

    final_state_analysis = np.asarray([
        np.mean(avg_v[cut_off:]), np.mean(v_variance[cut_off:]),
        overall_soliton_speed,  # num_solitons,
        x_peak_freq_amp[0], x_peak_freq_amp[1],
        avg_v_peak_freq_amp[0], avg_v_peak_freq_amp[1],
        v_var_peak_freq_amp[0], v_var_peak_freq_amp[1],
        cut_off, num_ticks
    ])

    return final_state_analysis

    #final_state_analysis = np.asarray([
    #   overall mean avg_v and v_var, (determines free flow or jam)   +2
    #   soliton effect speed, number of solitons,                     +2
    #   Oscillation data:
    #       x_position frequencies and amplitude of oscillation,      +2
    #       avg_v frequencies and amplitude of oscillation,           +2
    #       v_var frequencies and amplitude of oscillation,           +2

    #   The ticks data is recorded at (e.g. 10_000 to 20_000 ticks
    #   ])


def translate_to_class():
    cars = []
    for n in range(number_of_cars):
        cars.append(Car(init_x[n], road_height, init_v[n]))

    for n, car in enumerate(cars):
        car.X = all_pos_data[:, n]

        car.V = np.empty_like(car.X)
        car.V[0] = init_v[n]
        car.V[1:] = car.X[1:] - car.X[:-1]
        car.V[car.V <= -width + max_v] += width

        car.T = np.arange(1, len(car.X) + 1)

        if n == number_of_cars - 1:
            car.leading_car = cars[0]
        else:
            car.leading_car = cars[n + 1]
    return cars


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


def all_runs_mean_speed_plot(full_axis=False):

    final_mean_avg_vs = np.mean(avg_vs[:, -10_000:], axis=1)
    overall_mean_avg_vs = np.mean(avg_vs, axis=1)
    sort_index = overall_mean_avg_vs.argsort()
    avg_vs_sorted = avg_vs[sort_index]

    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.1]}, sharey=True, layout="tight")
    fig.suptitle(f"Mean speed of system over time ({num_runs} runs) \n"
                 fr" $\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'acc'}}}$ = {dec_acc}, N = {number_of_cars}, L = {width}")
    # fig.suptitle(fr" $\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'acc'}}}$ = {dec_acc}")
    ax[0].set_xlabel("Ticks")
    ax[0].set_ylabel("Avg. speed")
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])

    if full_axis is True:
        ax[0].set_ylim(0, 1.1)
        ax[1].set_ylim(0, 1.1)

    colour_map = plt.get_cmap('viridis')
    plot_colours = [colour_map(i) for i in np.linspace(0, 1, num_runs)]
    for n, avg_v in enumerate(avg_vs_sorted):
        plot_colour = plot_colours[n]
        ax[0].plot(avg_v, color=plot_colour, alpha=0.6)
        ax[1].scatter(y=final_mean_avg_vs[n], x=0, s=4.5, color=plot_colour)
        ax[1].hlines(y=final_mean_avg_vs[n], xmin=-1, xmax=1, linewidth=1, linestyles='dashed', colors='black')

    for n, avg_v in enumerate(avg_vs_sorted):
        # Bouncing and free-flow jumping detection
        sampling = 10_000
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
        err = 0 + 0.005
        bounce_up_err = 0.05
        jump_up_err = 0.05
        for i in range(1, num_ticks // sampling):

            # Gradient dramatically increased (shot to FF) (blue)
            if (mean_avg_v_diffs[i] > jump_up_err):
                ax[0].scatter(y=mean_avg_v_boxes[i], x=x[i], color='blue', marker='o', linewidths=0.5, zorder=2.5)


            # # Bounced up (red)
            # elif (mean_avg_v_diffs[i] - mean_avg_v_diffs[i - 1] > err):  # ~2nd deriv. minimum
            #     ax[0].scatter(y=mean_avg_v_boxes[i], x=x[i], color='r', marker='o', linewidths=0.5, zorder=2.5)
            #
            # # Bounced down (green)
            # elif (mean_avg_v_diffs[i] - mean_avg_v_diffs[i - 1] < -err):  # ~2nd deriv. maximum
            #     if mean_avg_v_boxes[i + 1] != 1:
            #         ax[0].scatter(y=mean_avg_v_boxes[i], x=x[i], color='g', marker='o', linewidths=0.5, zorder=2.5)

            else:
                ax[0].scatter(y=mean_avg_v_boxes[i], x=x[i], color='black', marker='x', linewidths=0.5, alpha=1.0)

    return fig


def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


# PROGRAM RUNNING #

# Running and sample parameters
num_ticks = 20_000  # 1_000_000
cut_off = 2_000  # How long system takes to settle. Larger cut_off ==> longer run time
sample_mod = 1  # How often x_pos_data is collected in run_sim
num_runs = 5  # The number of runs per dec/acc value

# Initial value parameters
init_v_sd = 0.15
# np.random.seed(42)

# System parameters
max_v = 1.0  # Always fixed
acc = 0.0005  # Always fixed
number_of_cars = 100  # Might be changing variable in future?

# Changing variables
vertical_pixels = 200
horizontal_pixels = 200
width_N_vals = np.linspace(20., 110., num=vertical_pixels, endpoint=False)  # width / number_of_cars dimensionless
dec_acc_vals = np.linspace(0., 200., num=horizontal_pixels, endpoint=False)   # dec / acc dimensionless
#width_vals = np.linspace(300., 3300., num=100, endpoint=False)  # array of width values being run over = width_N_vals * N
min_delta = 10  # Potentially change in future?

width_N_vals = np.linspace(20., 21., num=1, endpoint=False)  # width / number_of_cars dimensionless
dec_acc_vals = np.linspace(60, 61, num=1, endpoint=True)   # dec / acc dimensionless
#dec_acc_vals = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
# dec_acc_vals += 5

print(datetime.datetime.now().strftime("%H%M%S"))

# RUNNING NEW RANDOM SIMULATIONS iterating over range of dec/acc
initial_parameters = np.empty((200, 200, num_runs, 2, number_of_cars))
system_parameters = np.empty((200, 200, 5))
final_state_analysis = np.empty((200, 200, num_runs, 11))
final_state_information = np.empty((200, 200, num_runs, 8))

for i, width_N in enumerate(width_N_vals):
    width = int(width_N * number_of_cars)
    print(f"width: {width}")
    for j, dec_acc in enumerate(dec_acc_vals):
        dec = dec_acc * acc
        print(f"dec/acc = {dec_acc}: ")
        system_parameters[i, j] = np.asarray([number_of_cars, width, acc, dec, min_delta])

        avg_vs = np.zeros((num_runs, num_ticks))

        for run in range(num_runs):
            if width_N <= 15:
                init_x, init_v = equal_car_setup(number_of_cars, init_v_sd)
            else:
                init_x, init_v = random_car_setup(number_of_cars, init_v_sd)

            # Only needed for displaying
            cars = []
            for n in range(number_of_cars):
                cars.append(Car(init_x[n], road_height, init_v[n]))

            import sys
            np.set_printoptions(threshold=sys.maxsize)

            all_pos_data, avg_v, num_ticks_new, final_state_run_info = run_sim(num_ticks, sample_mod, init_x, init_v, dec, width, min_delta)
            num_ticks = num_ticks_new


            print(f"run {run+1} / {num_runs}")
            print(f"Ticks ran:{num_ticks}")
            # cars = translate_to_numpy(all_pos_data, num_ticks)

            avg_vs[run, 0:len(avg_v)] = avg_v  # (also disabled when running for saving, used only for fig1 plot)

            initial_parameters[i, j, run] = np.asarray([init_x, init_v])  # or maybe just save standard deviations?
            # final_state_analysis[i, j, run] = run_analysis(final_state_run_info)
            final_state_information[i,j, run] = final_state_run_info

            cars = translate_to_class()  # (temporary fix, will be changed for efficiency for 1000 x 1000)
            run_plots(cut_off)  # (disabled when running for saving)

        fig1 = all_runs_mean_speed_plot(full_axis=True)
        show_figure(fig1)
        # plt.savefig(f"dodgy_region_limit/N = {number_of_cars}.png", dpi=300)
        plt.show()

# file_ID = datetime.datetime.now().strftime("%H%M%S%f-%d-%m-" + str(num_ticks))  # Generates unique file name
# np.savez("databank/sim_run_data-" + file_ID,
#          init_param=initial_parameters,
#          final_param=final_state_analysis,
#          sys_param=system_parameters,
#          final_info=final_state_information)


# FOR RUNNING FROM SAVED DATA IN /databank

# Saves init to cars, runs sim then makes plots,
# Faster than reading saved data.
# mypath = getcwd() + "/databank"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# for f in onlyfiles:
#     print(f)
#     cars = read_data("databank/"+f)
#     all_pos_data, avg_v = run_sim(num_ticks, 1)
#     make_plots()

pygame.quit()
print(datetime.datetime.now().strftime("%H%M%S"))  # Prints end time
