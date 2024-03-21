# Loading packages
using Pkg
# pkg"add https://github.com/zsunberg/DMUStudent.jl"
# pkg"add QuickPOMDPs"
# pkg"add POMDPTools"
# pkg"add POMDPs"
# pkg"add IJulia"
# pkg"add ElectronDisplay "
# pkg"add DataFrames"
# pkg"add VegaLite"
# pkg"add DeepQLearning"
# pkg"add POMDPSimulators"
# pkg"add POMDPPolicies"
# pkg"add QMDP"
# pkg"add POMDPSolve"

using DMUStudent.HW5: HW5, mc

import POMDPs
using POMDPModels
using POMDPSolve
using POMDPSimulators
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using POMDPPolicies: EpsGreedyPolicy, LinearDecaySchedule, SoftmaxPolicy
using QMDP: QMDPSolver

using IJulia
using CommonRLInterface: actions, observe, act!, terminated, AbstractEnv
using CommonRLInterface.Wrappers: QuickWrapper
# using VegaLite
using ElectronDisplay
using DataFrames: DataFrame
using DeepQLearning

using Statistics: mean, std
using Random: rand, randperm
using Distributions: Categorical
using Plots: scatter, scatter!, plot, plot!, savefig, heatmap
using Flux: Chain, Dense, Optimise, params, setup, relu, mse, gradient, ADAM, σ, RMSProp
import Flux
using StaticArrays
using Base.Threads: @threads


############
# Question 1
############

#=
# Definition POMDP
monitor_treat = QuickPOMDP(
    states = [:healthy, :in_situ_cancer, :invasive_cancer, :death],
    actions = [:wait, :test, :treat],
    observations = [:positive, :negative],

    # Transition function
    transition = function (s, a)
        if s == :healthy
            outcomes = [:in_situ_cancer, :healthy]
            return outcomes[rand(Categorical([0.02, 0.98]))]
        elseif s == :in_situ_cancer && a == :treat
            outcomes = [:healthy, :in_situ_cancer]
            return outcomes[rand(Categorical([0.6, 0.4]))]
        elseif s == :in_situ_cancer && a != :treat
            outcomes = [:invasive_cancer, :in_situ_cancer]
            return outcomes[rand(Categorical([0.1, 0.9]))]
        elseif s == :invasive_cancer && a == :treat
            outcomes = [:healthy, :death, :invasive_cancer]
            return outcomes[rand(Categorical([0.2, 0.2, 0.6]))]
        elseif s == :invasive_cancer && a != :treat
            outcomes = [:death, :invasive_cancer]
            return outcomes[rand(Categorical([0.2, 0.8]))]
        else
            return s
        end
    end,

    # Observation function
    observation = function (s, a, sp)
        if a == :test
            if sp == :healthy
                outcomes = [:negative, :positive]
                return outcomes[rand(Categorical([0.95, 0.05]))]
            elseif sp == :in_situ_cancer
                outcomes = [:negative, :positive]
                return outcomes[rand(Categorical([0.2, 0.8]))]
            elseif sp == :invasive_cancer
                outcomes = [:negative, :positive]
                return outcomes[rand(Categorical([0.0, 1.0]))]
            else
                return :negative
            end
        elseif a == :treat
            if sp == :in_situ_cancer || sp == :invasive_cancer
                outcomes = [:negative, :positive]
                return outcomes[rand(Categorical([0.0, 1.0]))]
            else
                return :negative
            end
        else
            return :negative
        end
    end,

    # Reward function 
    reward = function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        else
            return 0.0
        end
    end,
    initialstate = Uniform([:healthy]),
    discount = 0.99
)

# Policy WAIT
function wait_policy(m, o)
    return :wait
end

# Roll-out generation function 
function rollout(m, policy, s0, depth=100)
    # Initialization
    r_total = 0.0
    t = 0
    o = :negative
    s = s0

    # Generate a trajetory 
    while t < depth
        a = policy(m, o)
        sp = POMDPs.transition(m, s, a)
        o = POMDPs.observations(m, s, a, sp)
        r = POMDPs.reward(m, s, a)
        r_total += POMDPs.discount(m)^t * r
        t += 1
        s = sp
    end

    return r_total
end

# Run Statistics
function compute_mean_and_sem(rewards)
    mean_reward = mean(rewards)
    sem = std(rewards) / sqrt(length(rewards))
    return mean_reward, sem
end

# Monte-Carlo Evaluate
results_wait = [rollout(monitor_treat, wait_policy, rand(POMDPs.initialstate(monitor_treat))) for _ in 1:10000]
@show compute_mean_and_sem(results_wait)


############
# Question 2
############

# Initialization data
n = 300
dx = rand(Float32, n)
dy = convert.(Float32, (1 .- dx) .* sin.(20 .* log.(dx .+ 0.2)))

# Initialize NN
model = Chain(Dense(1 => 20, σ), Dense(20 => 20, σ), Dense(20 => 20, σ), Dense(20 => 20, σ), Dense(20 => 1))

# Loss function
loss(x, y) = Flux.mse(model(x), y)

# Create training data (Flux wants tuples!)
data = [(SVector(dx[i]), SVector(dy[i])) for i in 1:length(dx)]

# Train the model 
training_steps = 100
num_threads = Threads.nthreads()
loss_values = Float64[]
@threads for i in 1:num_threads
    for j in 1:training_steps÷num_threads
        Flux.train!(loss, Flux.params(model), repeat(data, 100), ADAM())
        push!(loss_values, loss(reshape(dx, 1, :), reshape(dy, 1, :))) # TODO: This is not correct
    end
end

# Dataset for plotting
n = 100
dx = rand(Float32, n)
dy = convert.(Float32, (1 .- dx) .* sin.(20 .* log.(dx .+ 0.2)))

# Plot results
p1 = plot(sort(dx), x -> ((1 - x) * sin(20 * log(x + 0.2))), linewidth=1.5, xlabel="x [-]", ylabel="y [-]", label="Analytical")
plot!(p1, sort(dx), first.(model.(SVector.(sort(dx)))), linewidth=1.5, xlabel="x [-]", ylabel="y [-]", label="NN Approximation")
scatter!(p1, dx, dy, label="Data")
p2 = plot(loss_values, linewidth=1.5, xlabel="Training Step [-]", ylabel="Loss [-]", label="Loss")
plot(p1, p2, layout=(2, 1), grid=true, legend=true)
savefig("hw5/PlotNN.pdf")
=#

############
# Question 3
############

# Initialization environment (Discrete A, position and velocity observations )
env = QuickWrapper(HW5.mc,
    actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
    observe=mc -> observe(mc)[1:2]
)

# Define the DQN function
function dqn(env)
    # Initialization of the Q-network (IN: state, OUT: Q-values for each action)
    Q = Chain(Dense(2, 128, relu),
        Dense(128, length(actions(env))))

    # Experience buffer
    buffer = []

    # Training parameters
    γ = 0.99  # Discount factor
    α = 0.0005  # Learning rate
    batch_size = 10  # Batch size for training

    # Function to calculate Q-learning loss
    function loss(Q, s, a_ind, r, sp, done)
        Q_sp = Q(sp)
        target = r + (1 - done) * γ * maximum(Q_sp)  # Q-learning target
        return Flux.mse(Q(s)[a_ind], target)  # Mean squared error loss
    end

    # Training loop
    for episode in 1:1000  # Example: 1000 episodes
        # Reset environment and observe initial state
        s = observe(env)
        done = false

        while !done
            # Select action using ε-greedy policy
            if rand() < 0.1  # Exploration (ε = 0.1)
                a_ind = rand(1:length(actions(env)))
            else
                a_ind = argmax(Q(s))
            end

            # Take action and observe next state and reward
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)

            # Store experience in buffer
            push!(buffer, (s, a_ind, r, sp, done))

            # Sample a minibatch from the buffer
            minibatch = rand(buffer, min(length(buffer), batch_size))

            # Compute loss and perform gradient descent
            grads = Flux.gradient(() -> loss(Q, minibatch[1][1], minibatch[1][2], minibatch[1][3], minibatch[1][4], minibatch[1][5]), params(Q))
            Optimise.update!(ADAM(α), params(Q), grads)

            # Move to next state
            s = sp
        end

        # Evaluate, print, and plot (not shown here)
        # You may want to save your best policy
        println("Episode $episode completed")

    end

    return Q
end

# Define the DQN function
function dqn2(env)
    # Initialize the Q-network (input: state, output: Q-values for each action)
    Q = Chain(Dense(2, 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 128, relu),
        Dense(128, length(actions(env))))

    # Experience buffer
    buffer = []

    # Training parameters
    γ = 0.99  # Discount factor
    α = 0.00005  # Learning rate
    batch_size = 32 # Batch size for training
    training_episodes = 100000 # Number of training episodes
    target_update_freq = 1000 # Frequency to update target network

    # Target network
    Q_target = deepcopy(Q)
    update_count = 0  # Counter for target network update

    # Store loss for each episode
    training_losses = []

    # Function to compute Q-learning loss
    function loss(Q, Q_target, s, a_ind, r, sp, done)
        targets = [r[i] + (1 - done[i]) * γ * maximum(Q_target(sp[i])) for i in 1:length(s)]  # Q-learning target
        current = [Q(s[i])[a_ind[i]] for i in 1:length(s)] # Current Q-value
        return Flux.mse(current, targets)  # Mean squared error loss
    end

    # Training loop
    for episode in 1:training_episodes
        # Reset environment and observe initial state
        s = observe(env)
        done = false

        # Accumulate trajectories in the buffer
        trajectory = []

        while !done
            # Select action using ε-greedy policy
            if rand() < 0.1  # Exploration (ε = 0.1)
                a_ind = rand(1:length(actions(env)))
            else
                a_ind = argmax(Q(s))
            end

            # Take action and observe next state and reward
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)

            # Store experience in trajectory
            push!(trajectory, (s, a_ind, r, sp, done))

            # Move to next state
            s = sp
        end

        # Store trajectory in buffer
        for exp in trajectory
            push!(buffer, exp)
        end

        # Update target network if necessary
        if update_count % target_update_freq == 0
            Q_target = deepcopy(Q)
        end

        # Increment target update count
        update_count += 1

        # Training step after accumulating batch_size trajectories
        if episode % batch_size == 0
            # Sample a minibatch from the buffer
            minibatch = rand(buffer, min(length(buffer), batch_size))

            # Clear buffer
            empty!(buffer)

            # Accumulate losses for the minibatch
            minibatch_loss = 0.0
            all_s, all_a_ind, all_r, all_sp, all_done = zip(minibatch...)
            minibatch_loss = loss(Q, Q_target, all_s, all_a_ind, all_r, all_sp, all_done)

            # Store the loss for the current training step
            push!(training_losses, minibatch_loss)

            # Compute gradients and perform gradient descent
            grads = Flux.gradient(() -> loss(Q, Q_target, all_s, all_a_ind, all_r, all_sp, all_done), params(Q))
            Optimise.update!(ADAM(α), params(Q), grads)
            # opt_state = Flux.setup(ADAM(α), Q) 
            # grads = gradient(loss, Q, minibatch...)[1] # TODO: il problema sta qua!!
            # update!(opt_state, Q, grads)
        end

        # Evaluation, printing, and plotting
        println("Episode $episode completed")

    end

    return Q, training_losses
end

# Train the Q-network
Q, training_losses = dqn2(env)

# Plot training curve
p3 = plot(1:length(training_losses), training_losses,
    xlabel="Training Step [-]",
    ylabel="MSE Loss [-]",
    markershape=:circle,
    markersize=1,
    markerstrokecolor=:black,
    markerstrokewidth=1,
    grid=true,
)
savefig("hw5/PlotDQN.pdf")
# TODO: rileggi cosa vuole homeowkr testo!
# Evaluation and solution with POMDP packages
# HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "giovanni.fereoli@colorado.edu") 

#----------
# Rendering
#----------

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
# display(render(env))

# The following code allows you to render the value function
#xs = -3.0f0:0.1f0:3.0f0
#vs = -0.3f0:0.01f0:0.3f0
#heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")

#=
function render_value(value)
     xs = -3.0:0.1:3.0
     vs = -0.3:0.01:0.3

     data = DataFrame(
                      x = vec([x for x in xs, v in vs]),
                      v = vec([v for x in xs, v in vs]),
                      val = vec([value([x, v]) for x in xs, v in vs])
     )

     data |> @vlplot(:rect, "x:o", "v:o", color=:val, width="container", height="container")
end

display(render_value(s->maximum(Q(s))))

# Use of built-in solvers
model = Chain(Dense(2, 32), Dense(32, length(actions(env))))
exploration = EpsGreedyPolicy(env, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000 / 2))
solver = DeepQLearningSolver(qnetwork=model, max_steps=10000,
    exploration_policy=exploration,
    learning_rate=0.005, log_freq=500,
    recurrence=false, double_q=true, dueling=true, prioritized_replay=true)
solver = QMDPSolver()
policy_pomdp = solve(solver, env)
HW5.evaluate(policy_pomdp, "giovanni.fereoli@colorado.edu")

=#




