#!/usr/bin/env julia

#Greek letters
#αβγδεζηθκλμνξΦΘΓΔΣφπΠτ

## Preamble
import Pkg; Pkg.add("StatsBase") ; Pkg.add("Plots")
using StatsBase
using Plots
gr()

#= First we define a simple agent based on a game that's solvable by hand
The sample game will be a multi-round game with 2 players who have perfect information about
the payoffs of all players.
=#

# First define initial distributions of choices

# Sets of probability distributions, movesets, weights, etc., simplify the 'subsetting' 
# process... Otherwise we have to use string manipulation, which is not simple or reliable
# Each player is denoted by a number from 1 to N, where N is the total number of players in
# the game

# Probability distributions
prob = [[1/2,1/2],[1/2,1/2]]

# Movesets
moves = [[1,2],[1,2]]

# Weights, required to use the sample function
weights = [Weights(prob[1]),Weights(prob[2])]

# Let's also define the game itself
# We need three indices, two for the moves of each player and one for which player's payoff
# we are interested in

game = [[[2,4],[6,0]],
	[[3,3],[1,5]]]

game_matching_pennies = [[[1,-1],[-1,1]],
			 [[-1,1],[1,-1]]]


#= Propensities are entries in a list. Note that the initial values of the entries are 
# determined by the variable strength s_n.
=#

L = [[1,1],[1,1]]

# Here we may define the strength s[n] for each agent at the start of the game
s = 1
function S(n)
	for l in 1:length(L[n])
		L[n][l] = s
	end
end

# For the feedback and updating functions we need to define a 'history' of previous moves 
# and payoffs for the players. 
# It will consist of lists of size N, where N is the number of players and have the 
# following format: 
#
# move_list = (move_player1, move_player2, ..., move_playerN);
# payoff_list = (payoff_player1, payoff_player2, ..., payoff_playerN)
#
# This way moves and payoffs are easy to find.
temp_move_history=[]
move_history=[]
payoff_history=[]
α = 4 # memory parameter, determines how many steps backwards the move/payoff history goes,
      # must be >=1 if we use any learning algorithm
N = 2 # player number parameter

# Move playing function
function P(n)
	move = sample(moves[n], weights[n])
	push!(temp_move_history,move)
	if length(temp_move_history)>N*α # To ensure that history says length α
		popfirst!(temp_move_history)
	end
	return move
end

# Define payoff function Π(n)
function Π(n,m,game,k=0) # where n is the move of row player, m is the move of column 
			 # player, and k is an optional parameter that describes the player
			 # whose payoffs we are interested in
	payoffs = (game[n][m][1],game[n][m][2])
	#push!(payoff_history, payoffs[2])
	push!(payoff_history, payoffs)
	if length(payoff_history)>α 		
		popfirst!(payoff_history)
	end
	if k!=0
		return payoffs[k] # return payoff of player k, if specified
	end
end

# Define the feedback function R
function R(n,τ=1) # where n is the number of the player and τ is the number of steps the 
		# function goes back to (like a lag parameter), default value=1
	if length(move_history)>1
		for move in moves[n]
			if move==move_history[(length(move_history)-τ)][n] # check if the move was played τ rounds ago
				L[n][move] = L[n][move] + payoff_history[τ][n]
				if L[n][move]<1
					L[n][move]=1
				end
			end
		end
	end
end

# Define the updating function
function U()
	for i in 1:length(L)
		for j in 1:length(L[i])
			prob[i][j] = L[i][j]/(sum(L[i]))#+s[i])
		end
	end
end

function mean_payoff(game)
	return (prob[1][1]*prob[2][1]*game[1][1][1]+
		prob[1][2]*prob[2][1]*game[2][1][1]+
		prob[1][1]*prob[2][2]*game[1][2][1]+
		prob[1][2]*prob[2][2]*game[2][2][1],

		prob[1][1]*prob[2][1]*game[1][1][2]+
		prob[1][2]*prob[2][1]*game[2][1][2]+
		prob[1][1]*prob[2][2]*game[1][2][2]+
		prob[1][2]*prob[2][2]*game[2][2][2])
end

# Testing the game
function G(game)
	Π(P(1),P(2),game)
	push!(move_history,(temp_move_history[1],temp_move_history[2]))
	if length(move_history)>α
		popfirst!(move_history)
	end
	for n in 1:N
		R(n,1)
	end
	U()
end

function game_reset()
	for n in 1:N
		S(n)
		prob[n] = [1/2,1/2]
		weights[n] = Weights(prob[n])
	end
global payoff_history=[]
global move_history=[]
global plot_prob11 = zeros(0)
global plot_prob12 = zeros(0)
global plot_prob21 = zeros(0)
global plot_prob22 = zeros(0)
global plot_payoff1 = zeros(0)
global plot_payoff2 = zeros(0)
end

plot_prob11 = zeros(0)
plot_prob12 = zeros(0)
plot_prob21 = zeros(0)
plot_prob22 = zeros(0)
plot_payoff1 = zeros(0)
plot_payoff2 = zeros(0)
p=[]

function test(game,K)
game_reset()
for i in 1:K
	G(game) 
	push!(plot_prob11,prob[1][1])
	push!(plot_prob12,prob[1][2])
	push!(plot_prob21,prob[2][1])
	push!(plot_prob22,prob[2][2])
	push!(plot_payoff1, mean_payoff(game)[1])
	push!(plot_payoff2, mean_payoff(game)[2])
end
plot_prob = [[plot_prob11,plot_prob12],
	     [plot_prob21,plot_prob22],
	     [plot_payoff1,plot_payoff2]]
push!(p,plot(1:length(plot_prob[1][1]),plot_prob,
	     label=["Prob11" "Prob12" "Prob21" "Prob22" "MeanPayoff1" "MeanPayoff2"]))
game_reset()
end

for i in 1:9 
	test(game_matching_pennies,250) 
end 
display(plot(p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9], 
	     layout = (3,3), legend = false))

#println("prob=$(prob)")
#println("propensities=$(L)")
#println("payoff $(history=payoff_history)")
#println("mean payoff=$(mean_payoff(game))")
