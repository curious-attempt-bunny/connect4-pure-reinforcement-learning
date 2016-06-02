require 'csvigo'
local csv = csvigo.load({path='states.training.transformed.csv', mode='raw'})
local raw = torch.Tensor(csv) -- 85 inputs, 1 target
local inputSize = raw:size(2)-1
local input = raw:narrow(2, 1, inputSize)
local target = raw:narrow(2, inputSize+1, 1)

local size = raw:size(1) -- ~42k examples
local validRatio = 0.15
local nValid = math.floor(size*validRatio)
local nTrain = size - nValid

local train = {}
for i=1,nTrain do
  train[i] = {input[i], target[i]}
end
function train:size() return nTrain end

local valid = {}
for i=1,nValid do
  valid[i] = {input[i+nTrain], target[i+nTrain]}
end
function valid:size() return nValid end


require 'nn'

mlp = nn.Sequential()
mlp:add(nn.Linear(inputSize,60))
mlp:add(nn.Sigmoid())
-- mlp:add(nn.Linear(60,30))
-- mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(60,1))

--print(mlp)
--print(valid)
-- mlp.add(nn.L1Penalty(0.5))

criterion = nn.MSECriterion() -- Mean Squared Error criterion
trainer = nn.StochasticGradient(mlp, criterion)
trainer:train(train) -- train using some examples
-- trainer:train(valid) -- train using some examples

-- -- local preditions = mlp:forward(valid)
-- for i=1,10 do
--   print(valid[i])
--   -- print(predictions[i])
--   print(mlp:forward(valid[i][1]))
--   print(valid[i][2][1])
-- end
-- -- trainer:train(valid)