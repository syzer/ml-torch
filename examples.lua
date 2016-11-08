require 'nn'

function tensor_fill_funny(t, add)
  s = t:storage()
  for i = 1, s:size() do
    s[i] = ((i - 1) / s:size()) + add
  end
end

function tensor_print(t)
  s = t:storage()
  for i = 1, s:size() do
    io.write(string.format("%0.5f ", s[i]))
  end
  io.write("\n");
end

width = 6
height = 4
input_planes = 2
output_planes = 3
kw = 3
kh = 3

print("Linear")

input = torch.Tensor(1, 1, width, height)
tensor_fill_funny(input, 0.1)
net = nn.Sequential()
net:add(nn.View(width * height))
linear = nn.Linear(width * height, 2)
tensor_fill_funny(linear.weight, 0.2)
tensor_fill_funny(linear.bias, 0.3)
net:add(linear)
net:forward(input)
tensor_print(net.output)

print("SpatialConvolution")

input = torch.Tensor(1, input_planes, width, height)
tensor_fill_funny(input, 0.4)
spatconv = nn.SpatialConvolution(input_planes, output_planes, kw, kh)
tensor_fill_funny(spatconv.weight, 0.5)
tensor_fill_funny(spatconv.bias, 0.6)
net = nn.Sequential()
net:add(spatconv)
net:forward(input)
tensor_print(net.output)

print("ReLU")

input = torch.Tensor(2, 1, 2, 3)
tensor_fill_funny(input, -0.5)
net = nn.Sequential()
net:add(nn.ReLU())
net:forward(input)
tensor_print(net.output)

print("BatchNormalization")

input = torch.Tensor(3, 5)
tensor_fill_funny(input, 0.1)
net = nn.Sequential()
bn = nn.BatchNormalization(5, 1e-5, 0.1, false)
tensor_fill_funny(bn.running_mean, 0.2)
tensor_fill_funny(bn.running_var, 0.3)
net:add(bn)
net:evaluate()
net:forward(input)
tensor_print(net.output)

print("SpatialBatchNormalization")

input = torch.Tensor(2, 3, 2, 2)
tensor_fill_funny(input, 0.1)
sbn = nn.SpatialBatchNormalization(3, 1.0e-5, 0.1, false)
tensor_fill_funny(sbn.running_mean, 0.2)
tensor_fill_funny(sbn.running_var, 0.3)
net = nn.Sequential()
net:add(sbn)
net:evaluate()
net:forward(input)
tensor_print(net.output)

print("All")

input = torch.Tensor(1, input_planes, width, height)
tensor_fill_funny(input, 0.7)
spatconv_size = (width - kw + 1) * (height - kh + 1)
spatconv = nn.SpatialConvolution(input_planes, output_planes, kw, kh)
tensor_fill_funny(spatconv.weight, 0.8)
tensor_fill_funny(spatconv.bias, 0.9)
linear = nn.Linear(spatconv_size, 2)
tensor_fill_funny(linear.weight, 1.0)
tensor_fill_funny(linear.bias, 1.1)
net = nn.Sequential()
net:add(spatconv)
net:add(nn.View(spatconv_size))
net:add(linear)
net:forward(input)
tensor_print(net.output)

-- TODO save
