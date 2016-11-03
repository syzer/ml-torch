
require 'nn'

function fill_funny(t, add)
  s = t:storage()
  for i = 1, s:size() do
    s[i] = ((i - 1) / s:size()) + add
  end
end

width = 6
height = 4
input_planes = 2
output_planes = 3
kw = 3
kh = 3

-- Linear

print("Linear")

input = torch.Tensor(1, 1, width, height)
fill_funny(input, 0.1)
net = nn.Sequential()
net:add(nn.View(width * height))
fc1 = nn.Linear(width * height, 2)
fill_funny(fc1.weight, 0.2)
fill_funny(fc1.bias, 0.3)
net:add(fc1)
net:forward(input)

print(net.output)

-- SpatialConvolution

print("SpatialConvolution")
input = torch.Tensor(1, input_planes, width, height)
fill_funny(input, 0.4)
conv1 = nn.SpatialConvolution(input_planes, output_planes, kw, kh)
fill_funny(conv1.weight, 0.5)
fill_funny(conv1.bias, 0.6)
net = nn.Sequential()
net:add(conv1)
net:forward(input)

print(net.output)

-- Linear & SpatialConvolution

print("SpatialConvolution & Linear")
input = torch.Tensor(1, input_planes, width, height)
fill_funny(input, 0.7)
conv1OutSize = (width - kw + 1) * (height - kh + 1)
conv1 = nn.SpatialConvolution(input_planes, output_planes, kw, kh)
fill_funny(conv1.weight, 0.8)
fill_funny(conv1.bias, 0.9)
fc1 = nn.Linear(conv1OutSize, 2)
fill_funny(fc1.weight, 1.0)
fill_funny(fc1.bias, 1.1)
net = nn.Sequential()
net:add(conv1)
net:add(nn.View(conv1OutSize))
net:add(fc1)
net:forward(input)

print(net.output)
