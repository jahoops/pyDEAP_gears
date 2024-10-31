from functools import partial 
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import random
import debugpy
from deap import base, creator, tools, algorithms
import os
import logging
from multiprocessing import Pool

# Check if debugging is enabled
if os.getenv("DEBUG") == "1":
    # Set up debugpy for remote debugging
    debugpy.listen(("0.0.0.0", 5680))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached.")

# Initialize Pygame and Pymunk
pygame.init()
# Define screen dimensions at the global scope
width, height = 800, 600  # Set these to your desired window size
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
running = True

space = pymunk.Space()
space.gravity = (0, 980)

# Create walls of the rectangle
def create_wall(space, start, end):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, start, end, 5)
    shape.friction = 1.0
    space.add(body, shape)
    return shape

# Left and right walls for upward travel
left_wall = create_wall(space, (200, 0), (200, 600))  # Left wall
right_wall = create_wall(space, (600, 0), (600, 600))  # Right wall

# Bottom solid triangular guide
def create_bottom_triangle(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Poly(body, [
        (200, 600), (400, 400), (600, 600)
    ])
    shape.friction = 1.0
    space.add(body, shape)

create_bottom_triangle(space)

# Bottom solid triangular guide
def create_top(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Poly(body, [
        (200, 5), (600, 5), (600, 15), (200, 15)
    ])
    shape.friction = 1.0
    space.add(body, shape)

# Define collision types
BALL_COLLISION_TYPE = 1
RAMP_COLLISION_TYPE = 2
POLYGON_COLLISION_TYPE = 3

create_top(space)

# Create ramps on both sides
def create_ramp(space, start, end):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, start, end, 5)
    shape.friction = 1.0
    shape.collision_type = RAMP_COLLISION_TYPE
    space.add(body, shape)
    return shape

# Adjusted ramps
create_ramp(space, (235, 100), (375, 150))  # Left ramp
create_ramp(space, (565, 100), (425, 150))  # Right ramp

# Create solid horizontal lift
def create_lift(space, position, width, height):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = position
    shape = pymunk.Poly.create_box(body, (width, height))
    shape.friction = 1.0
    space.add(body, shape)
    return body, shape

# Create balls
def create_ball(space, position, name):
    body = pymunk.Body(mass=1, moment=10)
    body.position = position
    shape = pymunk.Circle(body, 10)
    shape.elasticity = 0.5
    shape.friction = 0.9
    shape.collision_type = BALL_COLLISION_TYPE
    body.rewarded = False
    body.name = name
    space.add(body, shape)
    return body

# Update lift positions
def update_lifts(lift, lift_shape, direction):
    if direction == "going down":
        lift.velocity = Vec2d(0, 200)  # Move downwards
        lift_shape.sensor = True  # Disable collision (folded)
        if lift.position.y >= 600:
            direction = "going up"
            lift_shape.sensor = False  # Enable collision (rigid)
    else:
        lift.velocity = Vec2d(0, -200)  # Move upwards
        if lift.position.y <= 30:
            direction = "going down"
            lift_shape.sensor = True  # Disable collision (folded)
    return direction

# Define DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Define the number of points for each shape (5 points per shape, total 10 points)
POINTS_PER_SHAPE = 5
TOTAL_POINTS = POINTS_PER_SHAPE * 2  # For two shapes

# Define bounds for x and y coordinates
X_MIN_SHAPE_1, X_MAX_SHAPE_1 = 200, 240
Y_MIN_SHAPE_1, Y_MAX_SHAPE_1 = 20, 80
X_MIN_SHAPE_2, X_MAX_SHAPE_2 = 560, 600
Y_MIN_SHAPE_2, Y_MAX_SHAPE_2 = 20, 80

# Attribute generator for shape 1 x-coordinates
toolbox.register("attr_shape1_x", random.uniform, X_MIN_SHAPE_1, X_MAX_SHAPE_1)
# Attribute generator for shape 1 y-coordinates
toolbox.register("attr_shape1_y", random.uniform, Y_MIN_SHAPE_1, Y_MAX_SHAPE_1)

# Attribute generator for shape 2 x-coordinates
toolbox.register("attr_shape2_x", random.uniform, X_MIN_SHAPE_2, X_MAX_SHAPE_2)
# Attribute generator for shape 2 y-coordinates
toolbox.register("attr_shape2_y", random.uniform, Y_MIN_SHAPE_2, Y_MAX_SHAPE_2)

# Register the individual generator
def create_individual():
    individual = []
    # Generate points for shape 1
    for _ in range(POINTS_PER_SHAPE):
        individual.append(toolbox.attr_shape1_x())
        individual.append(toolbox.attr_shape1_y())
    # Generate points for shape 2
    for _ in range(POINTS_PER_SHAPE):
        individual.append(toolbox.attr_shape2_x())
        individual.append(toolbox.attr_shape2_y())
    return creator.Individual(individual)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def get_world_vertices(shape):
    verts = []
    for v in shape.get_vertices():
        # Transform to world coordinates
        v_world = shape.body.position + v.rotated(shape.body.angle)
        # Adjust for Pygame's coordinate system (flip y-axis)
        v_py = v_world.x, height - v_world.y  # Access 'height' from global scope
        verts.append(v_py)
    return verts

# Create a static brown polygon (e.g., ramp)
def create_brown_polygon(vertices, name="ramp"):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, 0)
    body.name = name
    shape = pymunk.Poly(body, vertices)
    shape.elasticity = 0.4
    shape.friction = 1.0
    shape.color = pygame.Color("brown")  # For visualization
    shape.collision_type = POLYGON_COLLISION_TYPE
    space.add(body, shape)
    return shape

def ramp_collision_handler(arbiter, space, data):
    # Retrieve the shapes involved in the collision
    shape_a, shape_b = arbiter.shapes

    # Identify which shape is the ball and which is the ramp
    if shape_a.collision_type == BALL_COLLISION_TYPE:
        ball_shape = shape_a
        ramp_shape = shape_b
    else:
        ball_shape = shape_b
        ramp_shape = shape_a

    # Access the ball's body
    ball_body = ball_shape.body

    # Get the ball's position at collision
    ball_x_at_collision = ball_body.position.x
    
    closer_distance = 0
    if ball_body.name == "right ball" and ball_x_at_collision < 550:
        closer_distance = abs(ball_x_at_collision - 600)
    if ball_body.name == "left ball" and ball_x_at_collision > 240:
        closer_distance = abs(ball_x_at_collision - 200)

    if closer_distance == 0:
        return False
    
    # Scale the reward (the closer to the wall, the higher the reward)
    scaling_factor = 100  # Adjust this value as needed
    reward = max(0, scaling_factor * (1 / (closer_distance + 1)))

    if reward > 0:
        ball_body.rewarded = True
        data['performance'] += reward

    return True

# Initialize collision handler before usage
handler = space.add_collision_handler(BALL_COLLISION_TYPE, RAMP_COLLISION_TYPE)
handler.begin = ramp_collision_handler
handler.data['performance'] = 0  # Initialize performance data
def check_ball_height(balls):
    for ball in balls:
        if ball.name == "right ball":
            if ball.position.y < 30:
                handler.data['performance'] -= 100
        elif ball.name == "left ball":
            if ball.position.y < 30:
                handler.data['performance'] -= 100

def eval_shapes(individual, show_visualization=False, space=space):
    balls = [
        create_ball(space, (215, 500), "left ball"),  
        create_ball(space, (590, 500), "right ball"),
        #create_ball(space, (250, 90))  
    ]
    # Reset performance
    handler.data['performance'] = 0
    # Create left and right lifts
    left_lift, left_lift_shape = create_lift(space, (200, height), 50, 10)
    right_lift, right_lift_shape = create_lift(space, (600, height), 50, 10) 

    shape_1_points = [(individual[i], individual[i+1]) for i in range(0, 10, 2)]
    shape_2_points = [(individual[i], individual[i+1]) for i in range(10, 20, 2)]

    shape_1 = create_brown_polygon(shape_1_points, "left bumper")
    shape_2 = create_brown_polygon(shape_2_points, "right bumper")

    # Variables to track performance
    performance = 0
    
    left_lift_direction = "going up"
    right_lift_direction = "going up"
    # Run simulation for enough time for balls to interact
    while left_lift_direction == "going up" or (left_lift_direction == "going down" and  left_lift.position.y < height / 2):
        left_lift_direction = update_lifts(left_lift, left_lift_shape, left_lift_direction)
        right_lift_direction = update_lifts(right_lift, right_lift_shape, right_lift_direction)

        # Define the total time step and number of sub-steps
        total_dt = 1/60.0  # Total time step per frame
        sub_steps = 4      # Number of sub-steps
        sub_dt = total_dt / sub_steps

        if show_visualization:
        # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            space.step(sub_dt)
            check_ball_height(balls)
            # Clear the screen with a white background
            screen.fill((255, 255, 255))
            # Draw the current state of the physics space
            space.debug_draw(pymunk.pygame_util.DrawOptions(screen))     
            # Update the full display Surface to the screen
            pygame.display.flip()
            # Control the frame rate
            clock.tick(240)
        else:
            # Update the physics simulation with smaller sub-steps
            for _ in range(sub_steps):
                space.step(sub_dt)
                check_ball_height(balls)


    # Remove shapes after evaluation
    space.remove(shape_1, shape_2)
    for ball in balls:
        space.remove(ball, list(ball.shapes)[0])
    space.remove(left_lift, left_lift_shape)
    space.remove(right_lift, right_lift_shape)
    # Retrieve performance
    performance = handler.data['performance']
    print(f"Performance: {performance}")
    return (performance,)

toolbox.register("evaluate", eval_shapes, space=space)
toolbox.register("evaluate_visual", partial(eval_shapes, show_visualization=True, space=space))
toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# DEAP Evolution Loop
population = toolbox.population(n=50)
ngen = 40

def check_bounds(min_val, max_val):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                # Constrain points for the first shape
                for i in range(0, 10, 2):
                    child[i] = min(max(child[i], 200), 240)  # x between 200 and 220
                    child[i+1] = min(max(child[i+1], 20), 80)  # y between 20 and 80
                # Constrain points for the second shape
                for i in range(10, 20, 2):
                    child[i] = min(max(child[i], 560), 600)  # x between 580 and 600
                    child[i+1] = min(max(child[i+1], 20), 80)  # y between 20 and 80
            return offspring
        return wrapper
    return decorator

VISUALIZE_BEST = False

for gen in range(ngen):
    print(f"Generation {gen}")
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    most_fit = tools.selBest(population, 1)[0]
    print(f"Best individual of generation {gen}: {most_fit.fitness.values}")

    # Optionally visualize the best individual after each generation
    if VISUALIZE_BEST:
        print("Visualizing the best individual...")
        fitness = toolbox.evaluate_visual(most_fit)
        print(f"Visual evaluation performance: {fitness[0]}")
        

print(f"Best overall individual: {most_fit}")
fitness = toolbox.evaluate_visual(most_fit)
# Disable all callbacks
handler.begin = None
handler.separate = None
handler.post_solve = None
handler.pre_solve = None
pygame.quit()
