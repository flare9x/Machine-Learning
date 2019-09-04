# Machine Learning
# Data Preprocessing Array functions

using CSV
using Statistics
using Missings

"""
```
train_test_split(x::Array{T}; split_ratio::Float64=.7)
```
Split Data Into Train and Test Sets
"""
function train_test_split(x::Array{T}; split_ratio::Float64=.7)::Array{T} where {T<:Real}
    let len = length(len), split_no = Int64.(round(split_ratio * len))
    end
    train_set = data[1:split_no]
    test_set = data[split_no+1:len]
    return train_set,test_set
end

"""
```
standardization(x::Array{T})
```
Feature Scaling
standardization = x - mean(x) / standard deviation(x)
"""
function standardization(x::Array{T})::Array{T} where {T<:Real}
    let out = fill(NaN, size(x,1)), avg = mean(x), stdev = std(x)
    @inbounds for i = 1:size(x,1)
        out[i] = (x[i] - avg) / stdev
end
    return out
end
end

"""
```
normalization(x::Array{T})
```
Feature Scaling
normalization = x - min(x) / max(x) - min(x)
"""
function normalization(x::Array{T})::Array{T} where {T<:Real}
    let out = fill(NaN, size(x,1)), min = minimum(x), max = maximum(x)
    @inbounds for i = 1:size(x,1)
        out[i] = ((x[i] - min) / (max - min))
end
    return out
end
end

"""
```
data_fill_nan_inf(x::Array{T})
```
Convert NaN to Inf
"""
function data_fill_nan_inf(x::Array{T})::Array{T} where {T<:Real}
        @inbounds for i = 1:size(x,1)
        if (isnan(x[i]))
                x[i] = Inf
            end
end
return x
end

x = data_fill_nan_inf(x)

"""
```
fill_missing(x::Array{T}; method::String="mean")
```
Run function data_fill_nan_inf() this function thens replace Inf with the mean, median or previous values
"""
function fill_missing(x::Array{T}; method::String="mean")::Array{T} where {T<:Real}
    y = x[x .!= Inf] # Subset to remove InF, NaN
        @inbounds for i = 1:size(x,1)
        if method == "mean"
            avg = mean(y)
            if x[i] == Inf
                x[i] = avg
            end
        elseif method == "median"
            med = median(y)
            if x[i] == Inf
                x[i] = med
            end
        elseif method == "prev" && i >= 1
            prev = x[i-1]
            if x[i] == Inf
                x[i] = prev
            end
        elseif method == "prev" && i == 1
            prev = x[i+1]
            if x[i] == Inf
                x[i] = prev
            end
end
end
return x
end

# Data preprocessing - working with dataframes

# Read data
dataset = CSV.read("C:/Users/Andrew.Bannerman/Desktop/Machine Learning/Machine-Learning/Machine-Learning/data_preprocessing/Data.csv", header=true,types=[String, Float64, Float64, String])

# Replace missing values with the mean
using Missings
dataset[:Age] = collect(Missings.replace(dataset[:Age], mean(skipmissing(dataset[:Age]))))
dataset[:Salary] = collect(Missings.replace(dataset[:Salary], mean(skipmissing(dataset[:Salary]))))

# One hot encoding categroial variables
for i in unique(dataset[:Country])
    dataset[Symbol(i)] = ifelse.(dataset[:Country] .== i, 1, 0)
end

dataset[:Purchased] = ifelse.(dataset[:Purchased] .== "Yes", 1, 0)

# Feature Scaling - note in order to not have multiple variables over powering each other in the model
dataset[:Age] = standardization(dataset[:Age])
dataset[:Salary] = standardization(dataset[:Salary])

# Split dataframe to train and test sets
split_ratio = .7
len = size(dataset,1)
split_no = Int64.(round(split_ratio * len))
train_set = dataset[1:split_no,:]
test_set = dataset[split_no+1:len,:]
