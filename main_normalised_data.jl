## ------------------------- ##
##      Pendulum Solver      ##
## ------------------------- ##
using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux
using Statistics
using LinearAlgebra

# Store iteration data
mutable struct Data
	y::Vector
	x::Vector
	Data() = new()
end

# Struct of hyperparameters
mutable struct Lambda
	res::Float64
	weights::Float64
end

show_plot = false
save_plot = false

## File Loaded and Case Selection
println("-- Pendulum Solver --")
case = "central_diff"
#case = "euler"


## Load Pendulum Object
include("Dynsys.jl")
intitial_phi = 1.0
intitial_vel = 0.0

l = 1.0 / 30.0
g = 10.0
m = 1.0
c = 5.0
k = 0.0
pendulum = Dynsys.Math_pendulum(l, g, m, c, k, intitial_phi, intitial_vel)


## Load Integrator Object
ts = 2000
dt = 1.0 / ts
Integ = Dynsys.Integrator(dt, ts)

data = Data()
data.y = zeros(ts)
data.x = zeros(ts)


## Setup Time Intetration
# initial setting
if show_plot == true
	fig = Dynsys.create_fig(pendulum)
	Dynsys.plot_state(pendulum)
	display(fig)
end

# compute phi_-1
if case == "central_diff"
	acceleration = -pendulum.g / pendulum.l * sin(pendulum.phi)
	pendulum.phi_prev = pendulum.phi + 0.5 * Integ.delta_t * Integ.delta_t * acceleration
else
	pendulum.phi_prev = 0
end



## ------------------------- ##
##      Data Deneration      ##
## ------------------------- ##
println("Using '" * case * "' Solver!\nGenerating Data")
for i in 1:Integ.timesteps
	# integration step
	Dynsys.run_step(Integ, case, pendulum)

	# plot the state
	if show_plot == true
		fig = Dynsys.create_fig(pendulum)
		Dynsys.plot_state(pendulum)
		display(fig)
	end

	# save the step
	data.y[i] = pendulum.phi
	data.x[i] = dt * i
end
println("Data Generation Complete")

if save_plot == true
    println("Saving Data Plot")
    plot(data.time,data.phi)
    savefig(case * ".png")
    println("Save Complete")
end

## ------------------------- ##
##       Neural Network      ##
## ------------------------- ##



## Network Creation

Activation_Function = "tanh"
neural_network = Chain(
    Dense(1, 20, tanh),
    Dense(20,40, tanh),
    Dense(40, 1)
)

if Activation_Function == "relu"                    #Normalisation of data for Relu Activation Function
  min1 = findmin(data.y)
  max1 = findmax(data.y)
  data.y = (data.y .- min1[1])./(max1[1] .- min1[1])
else
  mean1 = mean(data.y)                             #Normalisation of data for other Activation Functions
  std1 = std(data.y)
  data.y = (data.y .- mean1)./std1
end
##Network Data
training_ts = Int(floor(0.35 * ts))
training_data_t = data.x[1:2:training_ts]'	# Time data (training data set)
training_data_y = data.y[1:2:training_ts]'	# Training data (training data set)
validation_data_t = data.x[2:2:training_ts]'	# Time data (validation data set)
validation_data_y = data.y[2:2:training_ts]'	# Training data (validation data set)

optimizer = Adam(0.01)																				# Selected optimizer
parameters = Flux.params(neural_network)															# Get NN parameters
input_data = Flux.Data.DataLoader((training_data_t, training_data_y), batchsize=25, shuffle=true)	# Load data


## Loss Functions
# Mean Squared Error Loss
function mse_loss(x, y)
	return mse = Flux.mse(neural_network(x),y)	# Compute mse loss

end

# Weight Decay (using L1 and L2 Regularization)
#=function weight_loss()
	weights = hcat(reshape(parameters[1],1,20),reshape(parameters[3],1,800),reshape(parameters[5],1,40))
	return 1e-4 * sum(abs2.(weights)) + 1e-4 * sum(abs.(weights))
end=#

# Residual Loss
res_dt = 0.001
res_n_points = 75
res_data_center = collect(range(res_dt, 1-res_dt, res_n_points))'		# Training data set for residaul computation (center)
res_data_right = res_data_center .+ res_dt								# Training data set for residaul computation (right)
res_data_left = res_data_center .- res_dt								# Training data set for residaul computation (left)

function res_loss()


	vals_center = neural_network(res_data_center)	# Central values for CFD
	vals_right = neural_network(res_data_right)		# Right values for CFD
	vals_left = neural_network(res_data_left)       # Left values for CFD

    if Activation_Function == "relu"
       vals_center = vals_center.*(max1[1].-min1[1]) .+min1[1]
       vals_right = vals_right.*(max1[1].-min1[1]) .+min1[1]
	    vals_left = vals_left.*(max1[1].-min1[1]) .+min1[1]
   else
	  vals_center = (vals_center.*std1) .+ mean1
      vals_right = (vals_right.*std1) .+ mean1
	  vals_left = (vals_left.*std1) .+ mean1
   end

	# println(vals_left)
	# println(vals_right)
	# println(vals_center)

    v = (vals_right .- vals_left) / (2*res_dt)						# Compute phi_dot using central difference
    a = (vals_right .- 2*vals_center .+ vals_left) / (res_dt^2)		# Compute phi_ddot using central difference
    res = (m * a) .+ (c * v) .+ (g / l * vals_center)				# Compute the physics residual

	return mean(abs2.(res))
end

lambda = Lambda(1e-4, 1e-4)
function loss_function(x, y)
	mse = mse_loss(x, y)
	res = res_loss()
	#weight = weight_loss()

    return mse + lambda.res * res #+ weight  # Compute the total loss function (mse + res + weight decay)
end


## Main Training Loop
tolerance = 1e-6	# Early termination tolerance
max_itrs = 10000	# Maximum number of training iterations
n_itrs = max_itrs
convergence = Data()
convergence.x = zeros(max_itrs)
convergence.y = zeros(max_itrs)
println("Training Network")
for e in 1:max_itrs
	Flux.train!(loss_function, parameters, input_data, optimizer)   # Training call
	convergence.x[e] = e
	convergence.y[e] = loss_function(training_data_t, training_data_y)
	if e % 100 == 0
		loss = loss_function(training_data_t, training_data_y)
		@printf("Iterations: %d, Loss: %.4e\n", e, loss)
	end

	# Check loss function on validation dataset
	if mse_loss(validation_data_t, validation_data_y) < tolerance
		@printf("Training converged in %d iterations\n", e)
		global n_itrs = e
		break
	end

	# Update training rate
	if e == 750
		optimizer.eta = 0.001
	elseif e == 1250
		optimizer.eta = 0.0005
	elseif e == 5000
		optimizer.eta = 0.0001
	elseif e == 10000
		optimizer.eta = 0.00005
	elseif e == 15000
		optimizer.eta = 0.00001
	end
end
println("Training Complete")


## Plot Trained Network Output and Compare
trained_y = neural_network(data.x')
if Activation_Function == "relu"
	data.y = data.y .* (max1[1] .- min1[1]) .+ min1[1]
	trained_y = trained_y .* (max1[1] .- min1[1]) .+ min1[1]
else
	data.y = (data.y .*std1) .+ mean1
	trained_y = (trained_y.*std1) .+ mean1
end
plot(data.x, [trained_y', data.y])
println("Saving Fitted Plot")
#savefig("FittedData.png")
println("Save Complete")

## Plot convergence
plot_convergence = true
if plot_convergence == true
	plot(convergence.x[1:n_itrs], convergence.y[1:n_itrs], yaxis=:log)
	println("Saving Convergence Plot")
	#savefig("Convergence.png")
	println("Save Complete")
end
