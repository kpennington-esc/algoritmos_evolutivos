import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from math import cos, sin, atan2, pi, sqrt
%matplotlib widget

# ========== CONSTANTS ==========
W, H = 200, 200
MAX_SENSOR_RANGE = 80.0 # how far the sensors can see
SENSOR_ANGLES = [-0.35, 0.0, 0.35] # radians relative to car heading
DT = 1.0 # time step in seconds
MAX_SPEED = 320.0 # units per second
MAX_STEER_RATE = 0.20 

# GA parameters
POP_SIZE = 40
GENERATIONS = 3
TOURNAMENT = 4 
CROSSOVER_RATE = 0.7
MUTATION_STD = 0.2
MUTATION_STEP = 0.1
MAX_MUTATION_STD = 1.0
STAGNATION_LIMIT = 10
rng = np.random.RandomState(24)

# Track parameters
cx, cy = W // 2, H // 2
outer_rx, outer_ry = 70, 90 # outer ellipse radii
inner_rx, inner_ry = 40, 60 # inner ellipse radii

# Network architecture
INPUT_DIM = 6
HIDDEN = 8
OUTPUT_DIM = 3

# Genome size!!
NUM_WEIGHTS = INPUT_DIM * HIDDEN + HIDDEN + HIDDEN * OUTPUT_DIM + OUTPUT_DIM


# ========== TRACK CREATION ==========
def make_ring_track(W, H, cx, cy, outer_rx, outer_ry, inner_rx, inner_ry):
    """
    Create a ring-shaped track mask. The track is defined by an outer ellipse
    and an inner ellipse (the hole). The area between the two ellipses is the
    track where the car can drive (value 1), and the rest is off-track (value 0).
    0 = off-track, 1 = on-track
    """
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    bbox_outer = [cx - outer_rx, cy - outer_ry, cx + outer_rx, cy + outer_ry]
    draw.ellipse(bbox_outer, fill=1)
    bbox_inner = [cx - inner_rx, cy - inner_ry, cx + inner_rx, cy + inner_ry]
    draw.ellipse(bbox_inner, fill=0)
    mask = np.array(img, dtype=np.uint8)
    return mask


track_mask = make_ring_track(W, H, cx, cy, outer_rx, outer_ry, inner_rx,
                             inner_ry)
center_rx = (outer_rx + inner_rx) / 2.0
center_ry = (outer_ry + inner_ry) / 2.0
center_R = (center_rx + center_ry) / 2.0


def in_track(x, y):
    """
    This is a naive check, but works fine for our simple track.
    """
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= W or yi >= H:
        return False
    return track_mask[yi, xi] == 1


def sensor_distances_fast(x, y, heading, step=4.0):
    """
    Calculate distances to track edges for each sensor angle.
    Uses a step size to incrementally check along the ray until hitting
    the edge of the track or going out of bounds.
    A higher step size is faster but less accurate.
    """
    dists = []
    maxsteps = int(MAX_SENSOR_RANGE / step) + 1
    for a in SENSOR_ANGLES:
        ray_ang = heading + a
        hit_dist = MAX_SENSOR_RANGE
        for i in range(maxsteps):
            dist = i * step
            sx = x + dist * cos(ray_ang)
            sy = y + dist * sin(ray_ang)
            if not (0 <= sx < W and 0 <= sy < H) or not in_track(sx, sy):
                hit_dist = dist
                break
        dists.append(min(hit_dist, MAX_SENSOR_RANGE))
    return np.array(dists, dtype=float) / MAX_SENSOR_RANGE


# ========== PROGRESS CALCULATION ==========
def progress_along_track(x, y):
    """
    This function calculates the angle of the point (x, y) relative to the
    center of the track (cx, cy). The angle is in radians and normalized to
    the range [0, 2*pi). This allows us to determine how far along the
    circular track the point is.

    We ll need a more complex function for more complex tracks.
    """
    ang = atan2(y - cy, x - cx)
    return ang if ang >= 0 else ang + 2 * pi


# TODO: check & verify this function
def calculate_net_progress(trace):
    """
    Calculate *cumulative* angular progress along the track given a trace of (x, y) points.
    Returns: (progress_percent, laps_completed, direction, total_distance)

    - Uses signed smallest-angle differences between consecutive points around track center
      to accumulate total angular change (so multiple laps and back-and-forth are counted).
    - direction: +1 for CCW, -1 for CW (based on sign of cumulative angle).
    """
    if not trace or len(trace) < 2:
        return 0.0, 0, 1, 0.0

    # total distance along path
    total_distance = 0.0
    for i in range(1, len(trace)):
        x1, y1 = trace[i - 1]
        x2, y2 = trace[i]
        total_distance += sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # accumulate signed angular deltas about the track center
    def ang_at(p):
        return atan2(p[1] - cy, p[0] - cx)

    cum_angle = 0.0
    prev_ang = ang_at(trace[0])
    for i in range(1, len(trace)):
        a = ang_at(trace[i])
        # smallest signed delta in (-pi, pi]
        delta = (a - prev_ang + pi) % (2 * pi) - pi
        cum_angle += delta
        prev_ang = a

    # determine direction and laps
    direction = 1 if cum_angle >= 0 else -1
    abs_cum = abs(cum_angle)
    laps_completed = int(np.floor(abs_cum / (2 * pi)))
    progress_percent = (abs_cum % (2 * pi)) / (2 * pi) * 100.0

    # if more than full laps, add 100% for each full lap
    progress_percent += laps_completed * 100.0

    return progress_percent, laps_completed, direction, total_distance
# --------------------------------------------------------------------



# ========== NEURAL NETWORK ==========
def genome_to_network(genome):
    """ Convert a flat genome array into network weights and biases.
    """
    idx = 0
    w1 = genome[idx: idx + INPUT_DIM * HIDDEN].reshape((INPUT_DIM, HIDDEN))
    idx += INPUT_DIM * HIDDEN
    b1 = genome[idx: idx + HIDDEN]
    idx += HIDDEN
    w2 = genome[idx: idx + HIDDEN * OUTPUT_DIM].reshape((HIDDEN, OUTPUT_DIM))
    idx += HIDDEN * OUTPUT_DIM
    b2 = genome[idx: idx + OUTPUT_DIM]
    idx += OUTPUT_DIM
    return w1, b1, w2, b2


def forward_network(genome, inputs):
    w1, b1, w2, b2 = genome_to_network(genome)
    # forward pass
    h = np.tanh(np.dot(inputs, w1) + b1)
    out = np.dot(h, w2) + b2

    # output layer activation
    steer = np.tanh(out[0])  # * 0.8 # tanh -1 to 1 scaled to -0.8 to 0.8
    accel = 1.0 / (1.0 + np.exp(-out[1]))  # sigmoid 0-1
    brake = 1.0 / (1.0 + np.exp(-out[2]))  # sigmoid 0-1

    if accel > 0.7 and brake > 0.7:
        brake = 0.3

    return steer, accel, brake


# ========== SIMULATION ==========
def simulate_one(genome, max_steps=1000, sensor_step=4.0, return_trace=False,
                 return_controls=False):
    """
    Simulate one car controlled by the given genome, from start until max_steps
    or crash.
    Returns final position, trace, crash status, and controls data.
    """
    theta = -pi # start angled left
    r = (center_rx + center_ry) / 2.0 # start radius
    x = cx + r * cos(theta)
    y = cy + r * sin(theta)
    heading = theta + pi / 2 # tangent to circle
    speed = 0.0
    trace = [(x, y)]
    controls_data = []

    # race!
    for step in range(max_steps):
        sensor = sensor_distances_fast(x, y, heading, step=sensor_step)
        tangent = atan2(y - cy, x - cx) + pi / 2
        rel_angle = ((heading - tangent) + pi) % (2 * pi) - pi
        speed_norm = speed / MAX_SPEED
        next_turn_curvature = 1.0 / max(1.0, center_R)

        inputs = np.array(
            [sensor[0], sensor[1], sensor[2], speed_norm, rel_angle / pi,
             next_turn_curvature], dtype=float)
        steer, accel, brake = forward_network(genome, inputs)

        if return_controls:
            controls_data.append(
                {'step': step, 
                 'steer': steer,
                 'accel': accel, 
                 'brake': brake,
                'speed': speed,
                'sensor_left': sensor[0],
                'sensor_center': sensor[1],
                'sensor_right': sensor[2]
                 })

        # physics
        steer_rate = steer * MAX_STEER_RATE
        heading += steer_rate * DT
        accel_force = accel * 2.2
        brake_force = brake * 2.8
        speed += (accel_force - brake_force) * DT
        speed *= 0.988
        speed = max(0.0, min(MAX_SPEED, speed))

        # update position
        prev_x, prev_y = x, y
        x += speed * cos(heading) * DT
        y += speed * sin(heading) * DT

        if return_trace:
            trace.append((x, y))

        # crash check
        if not (0 <= x < W and 0 <= y < H) or not in_track(x, y):
            return None, (
                trace if return_trace else None), True, None, controls_data

    return None, (trace if return_trace else None), False, None, controls_data


# ========== FITNESS FUNCTION ==========
def evaluate_pop_fitness(pop):
    """
    Evaluate fitness of the entire population. Returns fitness array and progress data for monitoring.
    1. simulate each individual
    2. calculate progress, direction, distance
    3. compute fitness with penalties and bonuses
    4. return fitness array and progress data
    """
    fits = np.zeros(len(pop), dtype=float)
    progress_data = []  

    for i, ind in enumerate(pop):
        _, trace, crashed, _, _ = simulate_one(ind, max_steps=1000,
                                               return_trace=True)

        if not trace or len(trace) < 10:
            fits[i] = 0.1
            progress_data.append((0.0, 0, 1, 0.0, True))
            continue

        progress_percent, laps_completed, direction, total_distance = calculate_net_progress(
            trace)

        progress_data.append(
            (progress_percent, laps_completed, direction, total_distance, crashed))

        # simple fitness: progress percentage
        fitness = progress_percent * 100 

        # penalties# for going backwards and crashing
        if direction < 0:  # backwards
            fitness *= 0.1  # 90% penalty
        if crashed:
            fitness *= 0.5  # 50% penalty 

        # efficiency bonus for progress with less distance
        if progress_percent > 10 and total_distance > 0:
            efficiency = progress_percent / (total_distance / 100)
            fitness += min(50, efficiency * 5)

        fits[i] = max(1.0, fitness)

    return fits, progress_data


# ========== GENETIC ALGORITHM ==========
def create_population(pop_size):
    return rng.normal(scale=0.3, size=(pop_size, NUM_WEIGHTS)).astype(
        np.float32)


def tournament_select(pop, fits, k=TOURNAMENT):
    """
    tournament selection: pick k random individuals and return the best one.
    1. randomly select k indices from the population
    2. find the index of the best fitness among them
    3. return a copy of the best individual
    4. ensures diversity and selection pressure

    best inddividual in the current population is not guaranteed to be selected
    """
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = idxs[np.argmax(fits[idxs])]
    return pop[best].copy()


def crossover(p1, p2):
    """
    uniform crossover: for each gene, randomly choose from one of the parents.

    """
    if rng.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    alpha = rng.rand(NUM_WEIGHTS)
    return alpha * p1 + (1 - alpha) * p2, (1 - alpha) * p1 + alpha * p2


def mutate(g, sigma=MUTATION_STD):
    return g + rng.normal(scale=sigma, size=g.shape)


def plot_controls(controls_data, title="Control Inputs Over Time"):
    if not controls_data:
        print("No control data to plot")
        return
        
    steps = [c['step'] for c in controls_data]
    steer = [c['steer'] for c in controls_data]
    accel = [c['accel'] for c in controls_data]
    brake = [c['brake'] for c in controls_data]
    speed = [c['speed'] for c in controls_data]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(steps, steer, 'b-', linewidth=2)
    ax1.set_ylabel('Steering (-1 to 1)')
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True)
    ax1.set_title('Steering Input')
    
    ax2.plot(steps, accel, 'g-', label='Throttle', linewidth=2)
    ax2.plot(steps, brake, 'r-', label='Brake', linewidth=2)
    ax2.set_ylabel('Throttle/Brake (0 to 1)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Throttle and Brake')
    
    ax3.plot(steps, speed, 'purple', linewidth=2)
    ax3.set_ylabel('Speed')
    ax3.set_xlabel('Simulation Step')
    ax3.grid(True)
    ax3.set_title('Speed Over Time')
    
    sensor_left = [c['sensor_left'] for c in controls_data]
    sensor_center = [c['sensor_center'] for c in controls_data] 
    sensor_right = [c['sensor_right'] for c in controls_data]
    
    ax4.plot(steps, sensor_left, 'orange', label='Left Sensor', alpha=0.7)
    ax4.plot(steps, sensor_center, 'brown', label='Center Sensor', alpha=0.7)
    ax4.plot(steps, sensor_right, 'pink', label='Right Sensor', alpha=0.7)
    ax4.set_ylabel('Sensor Distance (0-1)')
    ax4.set_xlabel('Simulation Step')
    ax4.legend()
    ax4.grid(True)
    ax4.set_title('Sensor Readings')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()


# ========== MAIN EVOLUTION ==========
population = create_population(POP_SIZE)
best_history, mean_history = [], []
best_trace_overall = None
best_controls_overall = None
best_fitness_overall = -1.0
best_genome_overall = None
stagnation_counter = 0
current_mutation_std = MUTATION_STD

print("Starting evolution...")
print("Gen | Best Fitness | Progress% | Laps | Dir | Distance | Status | MutRate")
print("----|--------------|-----------|------|-----|----------|--------|--------")


for gen in range(GENERATIONS):
    fits, progress_data = evaluate_pop_fitness(population)
    mean_history.append(float(np.mean(fits)))
    best_idx = int(np.argmax(fits))
    best_val = float(fits[best_idx])
    best_history.append(best_val)

    # best individual progress info
    best_progress, best_laps, best_direction, best_distance, best_crashed = progress_data[
        best_idx]
    direction_symbol = "→" if best_direction > 0 else "←"
    status = "CRASH" if best_crashed else "OK"

    # Check if new best
    is_new_best = best_val > best_fitness_overall
    if is_new_best:
        best_fitness_overall = best_val
        best_genome_overall = population[best_idx].copy()
        _, best_trace_overall, _, _, best_controls_overall = simulate_one(best_genome_overall,
                                                      max_steps=1000,
                                                      return_trace=True, return_controls=True)
        new_best_marker = "NEW BEST!"
        stagnation_counter = 0
        current_mutation_std = MUTATION_STD
    else:
        stagnation_counter += 1
        new_best_marker = ""

    # adaptive mutation adjustment
    if stagnation_counter > 0 and stagnation_counter % STAGNATION_LIMIT == 0:
        current_mutation_std = min(MAX_MUTATION_STD, current_mutation_std + MUTATION_STEP)
        print(f" Stagnation detected ({stagnation_counter} gens) → Mutation std increased to {current_mutation_std:.2f}")

    print(
        f"{gen + 1:3d} | {best_val:12.0f} | {best_progress:9.1f}% | {best_laps:5d} | {direction_symbol}  | {best_distance:9.0f} | {status:5} | {current_mutation_std:.2f} {new_best_marker}"
    )


    new_pop = []

    #  best individual
    elite_idx = np.argsort(fits)[-1:]
    new_pop.append(population[elite_idx[0]].copy())

    # add random individuals
    for _ in range(3):
        new_pop.append(create_population(1)[0])

    # fill rest with selection/crossover/mutation
    while len(new_pop) < POP_SIZE:
        p1 = tournament_select(population, fits)
        p2 = tournament_select(population, fits)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_pop.append(c1)
        if len(new_pop) < POP_SIZE:
            new_pop.append(c2)

    population = np.vstack(new_pop)[:POP_SIZE]

# ========== RESULTS ==========
print(f"\nEvolution completed! Final best: {best_fitness_overall:.0f} fitness")

if best_trace_overall:
    progress_percent, laps_completed, direction, total_distance = calculate_net_progress(
        best_trace_overall)
    direction_text = "Counter-Clockwise" if direction > 0 else "Clockwise"
    print(
        f"Best driver: {progress_percent:.1f}% progress, {direction_text}, {total_distance:.0f} units")

# ========== PLOTS ==========
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(best_history, label='Best', linewidth=2)
plt.plot(mean_history, label='Mean', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Evolution Progress')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.imshow(track_mask, origin='lower')
if best_trace_overall:
    tx = [p[0] for p in best_trace_overall]
    ty = [p[1] for p in best_trace_overall]
    plt.plot(tx, ty, 'r-', linewidth=2, label='Best Driver')

plt.scatter([cx], [cy], marker='x', color='cyan', s=100)
plt.scatter(tx[-1], ty[-1], marker='o', color='green', s=50, label='End Position')
plt.axis('off')
plt.title('Best Trajectory')
plt.legend()

plt.tight_layout()
plt.show()
if best_controls_overall:
    plot_controls(best_controls_overall, title="Best Driver Control Inputs Over Time")

