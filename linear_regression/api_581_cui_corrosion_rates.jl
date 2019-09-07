# API 581 Corrosion and regression
# Linear Interpolation

using CSV
using Statistics
using Missings
using Plots
using StatsBase

y1 = 0.005
x1 = 225
y2 = 0.01
x2 = 160
x = 200
y = y1 + ((x - x1) * ((y2-y1) / (x2 - x1)))
y = 0.0041 ipy # 240 operating temperature

operating_temperature = [10,18,43,90,160,225,275,325,350]
marine_corrosion_rate = [0,1,5,5,10,5,2,1,0]
marine_corrosion_rate = marine_corrosion_rate ./ 1000
plot(operating_temperature,marine_corrosion_rate, seriestype=:scatter,title="API 581 Table 16.2 - Corrosion Rates (ipy) - CUI", label ="Estimated Corrosion Rate API 581")

# First set of data
y = marine_corrosion_rate[1:5]
x = operating_temperature[1:5]
# Regression coefficients (y = b0 .+ (b1 .* x))
b1 = cov(x,y) / var(x) # b (slope)
b0 = mean(y) - b1 * mean(x) # a (intercept)

# Apply the regression formula to the training data
y_hat_train = fill(0.0,size(y,1)) # initialize array
y_hat_train = b0 .+ (b1 .* x)
training_set_residuals = y .- y_hat_train
r_2 = cor(x,y) ^2
n = size(train_set,1)
standard_error = sqrt((sum(y_hat_train .- y)^2) / (n-2))

plot!(x,y_hat_train, smooth=true, label = "Least Squares Regression")
plot_annotation = "y = ",b0," + (",b1," * x)"

#annotate!(30, 0.0018, text(plot_annotation, :red, :left, 10))

# Second set of data
y = marine_corrosion_rate[5:9]
x = operating_temperature[5:9]
# Regression coefficients (y = b0 .+ (b1 .* x))
b1 = cov(x,y) / var(x) # b (slope)
b0 = mean(y) - b1 * mean(x) # a (intercept)

# Apply the regression formula to the training data
y_hat_train = fill(0.0,size(y,1)) # initialize array
y_hat_train = b0 .+ (b1 .* x)
training_set_residuals = y .- y_hat_train
r_2 = cor(x,y) ^2
plot!(x,y_hat_train, smooth=true, label = "Least Squares Regression")

# margin of error
upper_bound = y + (.5 * y)
lower_bound = y - (.5 * y)
range = collect(lower_bound:.001:upper_bound)
difference_to_tmin = .280 - .120
time_to_governing_tmin = zeros(size(range,1))
for i in 1:length(range)
    time_to_governing_tmin[i] = difference_to_tmin / range[i]
end

plot(range,time_to_governing_tmin, seriestype=:scatter,title="Estimating +/-50% incorrect corrosion rate-Interpolated = 0.0041ipy", label ="Time To Governing Tmin")
plot!([y], seriestype="vline", label = "Interpolated Rate")
plot!([23.17], seriestype="hline", label = "Years MARS in-service")


#  Calculate B31.3 Pressure Tmin
# ASME B31.3 - 304.1.2 Straight Pipe Under Internal Pressure
P = 740  # Shell specification I, Design Pressure 740psig @ 100F
D = 6.625 # Outside Diameter
Y = .4 # Values of Coefficient Y for t < D/6 Table 304.1.1
E = 1.0 # Weld joint efficiency, 1.0 seamless
W = 1.0 # Weld Joint Strength Reduction Factor, W, Table 302.3.5
S = 20000 # ASTM A106 Grade B, Seamless @100F
c = 0.05 # Shell specification D, Corrosion allowance

t = (P * D) / (2*((S * E * W) + (P * Y)))
t = 0.120
tm = t + c # Note t = exclusive of corrosion allowance while tm is inclusive of corrosion allowance
tm = 0.170
