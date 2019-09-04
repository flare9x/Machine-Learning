# Linear Regression

using CSV
using Statistics
using Missings
using Plots
using StatsBase

# Read data
dataset = CSV.read("C:/Users/Andrew.Bannerman/Desktop/Machine Learning/Machine-Learning/Machine-Learning/linear_regression/Salary_Data.csv", header=true, types=[Float64, Float64])
dataset[:index] = collect(1:1:size(dataset,1)) # index the data
dataset = dataset[sample(axes(dataset, 1), size(dataset,1); replace = false, ordered = false), :] # Random shuffle the dataframe

# Feature Scaling
dataset[:Salary] = standardization(Float64.(dataset[:Salary]))
dataset[:YearsExperience] = standardization(Float64.(dataset[:YearsExperience]))

# Split data to train and test sets
split_ratio = 2/3
len = size(dataset,1)
split_no = Int64.(round(split_ratio * len))
train_set = dataset[1:split_no,:]
test_set = dataset[split_no+1:len,:]

# Visualize the data
gr() # We will continue onward using the GR backend
plot(train_set[:YearsExperience],train_set[:Salary],seriestype=:scatter,title="My Scatter Plot (Training Data)")

# Find the regression formula
x = Float64.(train_set[:YearsExperience])
y = Float64.(train_set[:Salary])

b1 = cov(x,y) / var(x) # b (slope)
b0 = mean(y) - b1 * mean(x) # a (intercept)

# Apply the regression formula to the training data
y_hat_train = fill(0.0,size(train_set,1)) # initialize array
y_hat_train .= b0 .+ (b1 .* x)
plot!(train_set[:YearsExperience],y_hat_train, smooth=true)
plot_annotation = "y = ",round(b0,digits=2)," + (",round(b1,digits=2)," * x)"
annotate!(-.5, 1.0, text(plot_annotation, :red, :left, 10))
training_set_residuals = y .- y_hat_train
r_2 = cor(x,y) ^2

# Using the same regression coefficients found on training data - plot the regression line on unseen test data
x = Float64.(test_set[:YearsExperience])
y = Float64.(test_set[:Salary])
y_hat_test = fill(0.0,size(test_set,1))
y_hat_test .= b0 .+ (b1 .* x)
plot(test_set[:YearsExperience],test_set[:Salary],seriestype=:scatter,title="My Scatter Plot (Test Data)")
plot!(test_set[:YearsExperience],y_hat_test, smooth=true)
plot_annotation = "y = ",round(b0,digits=2)," + (",round(b1,digits=2)," * x)"
annotate!(-.5, 1.0, text(plot_annotation, :red, :left, 10))
test_set_residuals = y .- y_hat_test
r_2 = cor(x,y) ^2

"""
```
lin_reg_ols(x_train::Array{T},y_train::Array{T},x_test::Array{T},y_test::Array{T})
```
Train the regression model on training data - derive coefficients b0 and b1
Apply training coefficients on test set data:: y_hat_test = b0 .+ (b1 .* x_test)
"""
function lin_reg_ols(x_train::Array{T},y_train::Array{T},x_test::Array{T},y_test::Array{T})::Array{T} where {T<:Real}
    let y_hat_train = size(x_train,1), y_hat_test = size(x_test,1)
    b1 = cov(x_train,y_train) / var(x_train) # b (slope)
    b0 = mean(y_train) - b1 * mean(x_train) # a (intercept)
    coefficients = b1,b0
    y_hat_train = b0 .+ (b1 .* x_train)
    # Test set
    y_hat_test = b0 .+ (b1 .* x_test)
    return y_hat_test
end
end

# Test function 
x_train = Float64.(train_set[:YearsExperience])
y_train= Float64.(train_set[:Salary])
x_test = Float64.(test_set[:YearsExperience])
y_test= Float64.(test_set[:Salary])

test_predictions = lin_reg_ols(x_train,y_train,x_test,y_test)
test_set_residuals = y_test .- test_predictions
r_2 = cor(x_test,y_test) ^2
