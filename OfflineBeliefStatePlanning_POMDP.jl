using POMDPs
using Pkg
# pkg"add https://github.com/zsunberg/DMUStudent.jl"
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP
using ParticleFilters: ParticleCollection
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions

##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

# Discrete state filter:
# Recursive Bayesian estimation to assign a probability mass to each state
function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    # Initalize
    states_m = ordered_states(up.m)
    probs = zeros(length(states_m))

    # Update
    for i in 1:length(states_m)
        z = observation(up.m, a, states_m[i])
        sp = states_m[i]
        probs[i] = pdf(z, o) * sum(s -> pdf(b, s) * pdf(transition(up.m, s, a), sp), states_m)
    end
    probs ./= sum(probs)

    return DiscreteBelief(up.m, probs)
end

# Note, you can access the transition and observation probabilities as follows:
# Z(o | a, s') can be programmed with Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)

# Initialize belief simplex
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}} #OSS: instead of conditional plans, policy as alpha(action)
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    bvec = beliefvec(b)
    num_vectors = length(p.alphas)
    best_idx = 1
    max_value = -Inf
    for i = 1:num_vectors
        temp_value = bvec .* p.alphas[i]
        if temp_value > max_value
            max_value = temp_value
            best_idx = i
        end
    end
    return p.alpha_actions[best_idx]
    # return first(actions(b.pomdp))
end

#------
# QMDP
#------


function qmdp_solve(m, discount=discount(m))
    # Initialize
    acts = actiontype(m)[act for act in actions(m)]
    Γ = Vector{Float64}[zeros(length(states(m))) for _ in actions(m)]

    # Alpha vector iteration
    k_max = 100
    for _ in 1:k_max
        Γ = [[reward(m, s, a) + discount * sum(pdf(transition(m, s, a), sp) * maximum(alpha[j] for alpha in Γ)
                                               for (j, sp) in enumerate(ordered_states(m))) for s in ordered_states(m)]
             for a in actions(m)]
    end

    return HW6AlphaVectorPolicy(Γ, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

#=
###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(

    # Fill in your actual code from last homework here

    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],
    transition = (s, a) -> Deterministic(s),
    observation = (a, sp) -> Deterministic(false),
    reward = (s, a) -> 0.0,
    discount = 0.99,
    initialstate = Deterministic(:death),
    isterminal = s->s==:death,
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)

                               # Fill in your heuristic policy here
                               # Use pdf(b, s) to get the probability of a state

                               return :wait
                           end
                          )

@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79

#####################
# Problem 3: LaserTag
#####################

m = LaserTagPOMDP()

qmdp_p = qmdp_solve(m)
up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# Use this version with only 100 episodes to check how well you are doing quickly
@show HW6.evaluate((qmdp_p, up), n_episodes=100)

# A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
using BasicPOMCP
function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
    solver = POMCPSolver(tree_queries=10,
                         c=1.0,
                         default_action=first(actions(m)),
                         estimate_value=FORollout(FunctionPolicy(s->rand(actions(m)))))
    return solve(solver, m)
end
pomcp_p = pomcp_solve(m)

@show HW6.evaluate((pomcp_p, up), n_episodes=100)

# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((qmdp_p, up), "REPLACE_WITH_YOUR_EMAIL@colorado.edu")

#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
using POMDPGifs
import Cairo, Fontconfig # needed to display properly

makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")

# You can render a single frame like this
using POMDPTools: stepthrough, render
using Compose: draw, PNG

history = []
for step in stepthrough(m, qmdp_p, up, max_steps=10)
    push!(history, step)
end
displayable_object = render(m, last(history))
# display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
draw(PNG("lasertag.png"), displayable_object)
=#
