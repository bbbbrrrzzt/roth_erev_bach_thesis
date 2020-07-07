#!/usr/bin/env julia
## Preamble
import Pkg;
Pkg.add("StatsBase");
Pkg.add("Plots");
Pkg.add("Statistics")
using Statistics
using StatsBase
using Plots
gr()
DIR = pwd()

#= First we define a simple agent based on a game that's solvable by hand
The sample game will be a multi-round game with 2 players who have perfect information
about the payoffs of all players.
=#

# First define initial distributions of choices

# Sets of probability distributions, movesets, weights, etc., simplify the 'subsetting'
# process... Otherwise we have to use string manipulation, which is not simple or reliable
# Each player is denoted by a number from 1 to N, where N is the total number of players
# in the game

# Parameters of the game world
#const
α = 10 # memory parameter, determines how many steps backwards the move/payoff history goes
# must be >=1 if we use any learning algorithm
#const
N = 3 # player number parameter
#const
#Φ = 1 / 10 # experimentation/generalisation parameter
Φ = 1 / 10
#const
ε = 1 / 5 # `forgetting' or recency parameter
#const
s_1 = 50.0
# T = 100 # Rounds the game is played for?

# 'Choice' parameter, describes how many moves each player has.
# Can be adjusted per player.
ω = 2
C = [ω for c = 1:N]

share = 1

# Movesets
moves = [collect(1:1:C[i]) for i = 1:N]
pd_moves = [collect(1:-1:0) for i = 1:N]

# Probability distributions
function seed_prob(c)
    r = [rand() for i = 1:c]
    r = r / sum(r)
    return r
end
prob = Float64[]
prob = [seed_prob(C[i]) for i = 1:N]
# Weights, required to use the sample function
weight = [Weights(prob[i]) for i = 1:N]

L = Float16[]
# When the strength parameter should define a uniformly strengthened distribution
#L = [[convert(Float16, s_1) for n = 1:N] for move = 1:length(moves[1])]
# When the distribution must be specified
#L_start = [[s_1 for i in 1:C[j]] for j in 1:N]
L_start = map(+,prob*s_1,[[s_1 for i in 1:C[j]] for j in 1:N])
L = deepcopy(L_start)

diff_prob11 = []
diff_prob21 = []

# Games in vector form
function generator_multi_game(N)
    a = [1 for i = 1:N]
    for i = 1:(N-1)
        a = [a for i = 1:N]
    end
    return a
end

game_Dict = Dict(
    "Game 1" => [[[2, 4], [6, 0]], [[3, 3], [1, 5]]],
    "Game 2" => [[[8, 0], [3, 8]], [[0, 5], [5, 3]]],
    "Game 3" => [[[3, 7], [8, 2]], [[4, 6], [1, 9]]],
    "Game 4" => [[[3, -3], [-1, 1]], [[-9, 9], [3, -3]]],
    "Game 5" => [[[9, 0], [0, 1]], [[0, 1], [1, 0]]],
    "Game 6" => [[[4, 0], [0, 1]], [[0, 1], [1, 0]]],
    "Game 7" => [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
    # Here are some prototypical games for testing purposes
    "Matching Pennies" => [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]],
    "Prisoners Dilemma" => [[[3, 3], [0, 5]], [[5, 0], [1, 1]]],
    "Stag Hunt" => [[[8, 8], [0, 1]], [[1, 0], [1, 1]]],
    #"One-Zero" => [[[1, 1], [0, 0]], [[0, 0], [0, 0]]],
    #"Multi-Game" => reshape(repeat(repeat(repeat([Tuple(1:3)], 5), 3), 3), 3, 5, 3),
    # Ultimatum" analogies
    "Ultimatum 1" =>
        [[[1 - share, share], [0, 0]], [[share, 1 - share], [0, 0]]],
    "Ultimatum 2" => [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
)

#prisoners_dilemma_3D => [[[3, 3, 3], [0, 5]], [[5, 0], [1, 1]]]
#
#prisoners_dilemma_4D => [[[3, 3], [0, 5]], [[5, 0], [1, 1]]]
#
#prisoners_dilemma_5D => [[[3, 3], [0, 5]], [[5, 0], [1, 1]]]

# Sequential games
function seq_game(game)
    return game[P(1)]
end

#= Propensities are entries in a list. Note that the initial values of the entries are
# determined by the variable strength s_n.
=#

#function Λ(n,j,t)
#	L = fill(1,N,length(moves[n]),α) # this assumes every player has the same amount of
#					 # moves. Might need to be fixed at some point.
#end

# Strength "parameter"
#function s(n, game)
#    return sum([game[j][k][n] for j = 1:N, k = 1:N])
#end

# For the feedback and updating functions we need to define a 'history' of previous moves
# and payoffs for the players.
# It will consist of lists of size N, where N is the number of players and have the
# following format:
#
# move_list = (move_player1, move_player2, ..., move_playerN);
# payoff_list = (payoff_player1, payoff_player2, ..., payoff_playerN)
#
# This way moves and payoffs are easy to find.
temp_move_history = []
temp_payoff_history = []
move_history = []
payoff_history = []


#function S(n,game)
#	for l in 1:length(L[n])
#		L[n][l] = s
#	end
#end
#
# History function
function H()
    if length(move_history) > α
        popfirst!(move_history)
    end
    if length(payoff_history) > α
        popfirst!(payoff_history)
    end
end

# Move playing function
function P(n=1; vector = 0)
    if Bool(vector)
        move_vector = [0 for i = 1:N]
        for i = 1:N
            move_vector[i] = sample(pd_moves[i], weight[i])
        end
        push!(move_history, move_vector)
        return move_vector
    else
        move = sample(moves[n], weight[n])
        push!(temp_move_history, move)
        return move
    end
end


# Define payoff function Π(n)
# Note that this function returns a tuple, containing the payoffs for both
# players where n is the move of row player, m is the move of column
# player, and k is an optional parameter that describes the
# player whose payoffs we are interested in
function Π(n, m, key, k = 0)
    payoffs = (game_Dict[key][n][m][1], game_Dict[key][n][m][2])
    push!(temp_payoff_history, payoffs)
    if k != 0
        return payoffs[k] # return payoff of player k, if specified
    else
        return payoffs
    end
end

# Define the feedback function R
function R(n, move, τ = 0)
    # where n is the number of the player and τ is the number of steps
    # the function goes back to (like a lag parameter), default
    # value=1
    if move == move_history[length(move_history)-τ][n]
        return payoff_history[length(payoff_history)-τ][n]
    else
        return 0
    end
end

# Because the similarity of strategies cannot be linearly ordered, we use the
# following function

function E(n, move, τ = 0)
    if length(move_history) - τ > 0
        if move == move_history[(length(move_history)-τ)][n]
            return R(n, move) * (1 - ε)
        else
            return (R(n, move) * ε) / (length(moves[n]) - 1)
        end
    else
        return 0
    end
end

function q(n)
    for move in moves[n]
        global L[n][move] = (1 - Φ) * L[n][move] + E(n, move)
        #return (1-Φ)*L[n][move] + E(n,move)
    end
    for move in moves[n]
        if L[n][move] < 0
            global L[n][move] = 0
        end
    end
end

# Define the probability adjusting function (p in the paper)
function U()
    for n = 1:N
        for j = 1:length(L[n])
            prob[n][j] = L[n][j] / (sum(L[n]))#+s(n,game))
            #return L[n][j]/(sum(L[n]))#+s(n,game))
        end
    end
    weight = [Weights(prob[i]) for i = 1:N]
    #return [Weights(prob[1]),Weights(prob[2])]
end

# function mean_payoff(game)
# 	return (prob[1][1]*prob[2][1]*game[1][1][1]+
# 		prob[1][2]*prob[2][1]*game[2][1][1]+
# 		prob[1][1]*prob[2][2]*game[1][2][1]+
# 		prob[1][2]*prob[2][2]*game[2][2][1],
#
# 		prob[1][1]*prob[2][1]*game[1][1][2]+
# 		prob[1][2]*prob[2][1]*game[2][1][2]+
# 		prob[1][1]*prob[2][2]*game[1][2][2]+
# 		prob[1][2]*prob[2][2]*game[2][2][2])
# end

## Testing the game
function G(key="None";multi=0)
    if Bool(multi)
        pd(P(vector=1))
        H()
        for n = 1:N
            q(n)
        end
        U()
    else
        global N=2
        Π(P(1), P(2), key)
        H()
        if length(move_history) > α
            popfirst!(move_history)
        end
        for n = 1:N
            q(n)
        end
        U()
        global share = prob[1][1]
        #println(L)
    end
end
## This section contains functions & declarations purely for plotting purposes
plot_prob11 = zeros(0)
plot_prob12 = zeros(0)
plot_prob21 = zeros(0)
plot_prob22 = zeros(0)
plot_payoff1 = zeros(0)
plot_payoff2 = zeros(0)
p = []

function mean_payoff(game)
    return (
        prob[1][1] * prob[2][1] * game[1][1][1] +
        prob[1][2] * prob[2][1] * game[2][1][1] +
        prob[1][1] * prob[2][2] * game[1][2][1] +
        prob[1][2] * prob[2][2] * game[2][2][1],
        prob[1][1] * prob[2][1] * game[1][1][2] +
        prob[1][2] * prob[2][1] * game[2][1][2] +
        prob[1][1] * prob[2][2] * game[1][2][2] +
        prob[1][2] * prob[2][2] * game[2][2][2],
    )
end


function game_reset()
    p_1 = rand()
    q_1 = rand()
    global moves = [collect(1:1:C[i]) for i = 1:N]
    global pd_moves = [collect(1:-1:0) for i = 1:N]
    global weight = [Weights(prob[i]) for i = 1:N]
    global prob = [seed_prob(C[i]) for i = 1:N]
    global L_start = map(+,prob*s_1,[[s_1 for i in 1:C[j]] for j in 1:N])
    global L = deepcopy(L_start)
    #[[convert(Float16, s_1) for n = 1:N] for move = 1:length(moves[1])]
    global temp_move_history = []
    global temp_payoff_history = []
    global move_history = []
    global payoff_history = []
    global weight = [Weights(prob[i]) for i = 1:N]
    global plot_prob11 = zeros(0)
    global plot_prob12 = zeros(0)
    global plot_prob21 = zeros(0)
    global plot_prob22 = zeros(0)
    global plot_payoff1 = zeros(0)
    global plot_payoff2 = zeros(0)
    global prob_agg_1 = [[] for i = 1:100]
    global prob_agg_2 = [[] for i = 1:100]
    global diff_prob11 = []
    global diff_prob21 = []
    global max_conv11 = 0
    global max_conv21 = 0
end

## Testing functions

function test(key, K; diff = 0, G = G, multi=0)
    game_reset()
    #println(prob)
    for i = 1:K
        G(key,multi)
        append!(plot_prob11, prob[1][1])
        append!(plot_prob21, prob[2][1])
        #push!(prob_agg_1[i],prob[1][1])
        #push!(prob_agg_2[i],prob[2][1])
    end
    for j = 2:K
        append!(diff_prob11, plot_prob11[j] - plot_prob11[j-1])
        append!(diff_prob21, plot_prob21[j] - plot_prob21[j-1])
    end
    println(prob)
    if Bool(diff)
        diff_prob = [diff_prob11, diff_prob21]
        return diff_prob
    else
        plot_prob = [plot_prob11, plot_prob21]
        return weight_hist'
    end
end

function mass_test(key, K, N; G = G)
    prob_agg_1 = [0 for i = 1:K]
    prob_agg_2 = [0 for i = 1:K]
    test(key, K)
    for n = 1:N-1
        test(key, K)
        prob_agg_1 = hcat(prob_agg_1, plot_prob11)
        prob_agg_2 = hcat(prob_agg_2, plot_prob21)
    end
    mean_prob_1 = []
    mean_prob_2 = []
    lump_1 = []
    lump_2 = []
    for j = 1:K
        lump_1 = [prob_agg_1[k] for k = j:K:K*(N-1)+1]
        lump_2 = [prob_agg_2[k] for k = j:K:K*(N-1)+1]
        push!(mean_prob_1, mean(lump_1))
        push!(mean_prob_2, mean(lump_2))
    end
    return [mean_prob_1, mean_prob_2]
end

## Plotting functions
function plot_conv_parameter(key, param)
    plot_rel_std_phi = [[], [], []]
    paramName = ""
    global Φ = 1 / 10
    global ε = 1 / 5 # `forgetting' or recency parameter
    global s_1 = 9
    if param == 2
        range = collect(0:1:100)
    else
        range = collect(0:0.01:1)
    end
    for f in range
        if param == 0
            global Φ = f
            paramName = "phi"
        elseif param == 1
            global ε = f
            paramName = "epsilon"
        elseif param == 2
            global s_1 = f
            paramName = "strength"
        else
            return 1
        end
        p = test(key, 200, 1)
        p_1 = copy(p)
        p_2 = copy(p)
        p_1 = [mean(p[1][i-10:1:i]) for i in 1:length(p[1]) if i > 10]
        p_2 = [mean(p[2][i-10:1:i]) for i in 1:length(p[2]) if i > 10]
        sd_1 = [std(p[1][i-10:1:i]) for i in 1:length(p[1]) if i > 10]
        sd_2 = [std(p[2][i-10:1:i]) for i in 1:length(p[2]) if i > 10]
        sd_1_long = std(p[1][49:1:199])
        sd_2_long = std(p[2][49:1:199])
        p = [p_1, p_2, sd_1, sd_2]
        push!(plot_rel_std_phi[1], f)
        push!(plot_rel_std_phi[2], sd_1_long)
        push!(plot_rel_std_phi[3], sd_2_long)
        #display(
        plot(
            p,
            title = "Convergence $key ($paramName = $f)",
            label = ["Mean A1" "Std.Dev A1" "Mean A2" "Std.Dev A2"],
            ylims = (-.3, 0.3),
            ylabel = "Change",
            xlabel = "Round",
        )
        print(length(p))
    end
    xs = plot_rel_std_phi[1]
    display(plot(
        xs,
        [plot_rel_std_phi[2] plot_rel_std_phi[3]],
        ylims = (0, 0.2),
        label = ["Std.Dev A1" "Std.Dev A2"],
        xlabel = "$paramName",
        ylabel = "Std.Dev of Change",
        title = "Relation between $paramName and std.dev of change ($key)",
        titlefontsize = 12,
    ))
    png("plot_rel_std_$(paramName)_$(key)")
end

plot_pay = [plot_payoff1, plot_payoff2]

function plot_prob(key)
    p = mass_test(key, 200, 10)
    display(plot(
        [
            p[1][1:convert(Int16, floor(length(p[1]) / 5)):length(p[1])],
            p[2][1:convert(Int16, floor(length(p[2]) / 5)):length(p[2])],
        ],
        title = "$key",
        label = ["A1" "A2"],
        ylims = (0, 1),
        ylabel = "Probability",
        xlabel = "Round",
        markershape = :circle,
        markersize = 4,
    ))
    #png("ultimatum_2")
end

function plot_conv(key)
    p = test(key, 200, 1)
    p_1 = copy(p)
    p_2 = copy(p)
    p_1 = [mean(p[1][i-10:1:i]) for i in 1:length(p[1]) if i > 10]
    p_2 = [mean(p[2][i-10:1:i]) for i in 1:length(p[2]) if i > 10]
    sd_1 = [std(p[1][i-10:1:i]) for i in 1:length(p[1]) if i > 10]
    sd_2 = [std(p[2][i-10:1:i]) for i in 1:length(p[2]) if i > 10]
    p = [p_1, p_2, sd_1, sd_2]
    display(plot(
        p,
        title = "Convergence $key",
        label = ["A1" "A2"],
        ylims = (-.3, 0.3),
        ylabel = "Change",
        xlabel = "Round",
    ))
    #png("convergence_$key")
end

function plot_conv_percentage(key)
    test(key, 200, 1)
    global plot_conv_perc = [
        plot_prob11 / plot_prob11[length(plot_prob11)],
        plot_prob21 / plot_prob21[length(plot_prob21)],
    ]
    display(plot(
        plot_conv_perc,
        ylims = (0, 2),
        title = "Percentage of total convergence after N rounds ($key)",
        titlefontsize = 12,
        label = ["A1" "A2"],
        ylabel = "Percentage",
        xlabel = "Round",
    ))
    png("percentage_convergence_$(key)")
end

function pd(move_vector)
    pd = [0 for i = 1:length(move_vector)]
    x0 = 3
    x1 = 0
    x2 = 1
    x3 = 5
    # Define the edge cases
    for j = 1:length(move_vector)
        # No players defect
        if (sum(move_vector) == 0)
            pd = [x0 for i = 1:length(move_vector)]
        # Exactly one player defects
        elseif (sum(move_vector) == 1)
            if Bool(move_vector[j])
                pd[j] = x3
            else
                pd[j] = x1
            end
        # More than one player defects
        else
            if Bool(move_vector[j])
                pd[j] = x2
            else
                pd[j] = x1
            end
        end
    end
    push!(payoff_history, pd)
    return pd # a vector of payoffs
end

#sum([sample(pd_moves[2],weight[2]) for i in 1:100])

function play_pd(K)
    game_reset()
    weight_hist = [weight[n][1] for n in 1:N]
    for i in 1:K
        G(multi=1)
        weight_hist = hcat(weight_hist, [weight[n][2] for n in 1:N])
        #print("$L\n")
    end
    display(plot(
    ylims = (0,1),
    label = ["C1" "C2"],
    title = "Prisoner's Dilemma with $N players",
    weight_hist'
    ))
    print(sum([sample(pd_moves[1],weight[1]) for i in 1:100]))
end

function play_pd_multi(N,K,L)
    game_reset()
    weight_hist = [weight[n][1] for n in 1:N]
    for i in 1:K
        G(multi=1)
        weight_hist = hcat(weight_hist, [weight[n][2] for n in 1:N])
        #print("$L\n")
    end
    display(plot(
    ylims = (0,1),
    label = ["C1" "C2"],
    title = "Prisoner's Dilemma with $N players",
    weight_hist'
    ))
    print(sum([sample(pd_moves[1],weight[1]) for i in 1:100]))
end

function play_pd_multi(K,L)
    prob_agg_1 = [0 for i = 1:K]
    prob_agg_2 = [0 for i = 1:K]
    play_pd(K)
    for n = 1:L-1
        play_pd(K)
        prob_agg_1 = hcat(prob_agg_1, weight[1][1])
        prob_agg_2 = hcat(prob_agg_2, weight[2][1])
    end
    mean_prob_1 = []
    mean_prob_2 = []
    lump_1 = []
    lump_2 = []
    for j = 1:K
        lump_1 = [prob_agg_1[k] for k = j:K:K*(N-1)+1]
        lump_2 = [prob_agg_2[k] for k = j:K:K*(N-1)+1]
        push!(mean_prob_1, mean(lump_1))
        push!(mean_prob_2, mean(lump_2))
    end
    return [mean_prob_1, mean_prob_2]
end
