require 'csvigo'
local csv = csvigo.load({path='states.training.transformed.csv', mode='raw'})
local raw = torch.Tensor(csv) -- 85 inputs, 1 target
local inputSize = raw:size(2)-1
local input = raw:narrow(2, 1, inputSize)
local target = raw:narrow(2, inputSize+1, 1)

local size = raw:size(1) -- ~42k examples
local validRatio = 0.15
local nValid = math.floor(size*validRatio)
local chunkSize = 1
nValid = nValid - (nValid % chunkSize)

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

-- consider: mlp:add(nn.L1Penalty(0.5))

criterion = nn.MSECriterion()

require 'optim'

local optimState = {learningRate=0.1}
local params, gradParams = mlp:getParameters()

local validInputs = torch.Tensor(nValid, inputSize)
local validOutputs = torch.Tensor(nValid)

for i=1,nValid do
  validInputs[i] = torch.Tensor(valid[i][1])
  validOutputs[i] = torch.Tensor(valid[i][2])
end

for epoch=1,25 do
  local chunks = nValid / chunkSize
  local shuffle_indexes = torch.randperm(nTrain)
  local batchTrainInputs = torch.Tensor(chunkSize, inputSize)
  local batchTrainOutputs = torch.Tensor(chunkSize) -- not sure why this had to be hardcoded

  for batchIndex=0,chunks-1 do
    for i=1,chunkSize do
      batchTrainInputs[i] = torch.Tensor(train[shuffle_indexes[batchIndex*chunkSize+i]][1])
      batchTrainOutputs[i] = torch.Tensor(train[shuffle_indexes[batchIndex*chunkSize+i]][2])
    end

    local function feval(params)
      gradParams:zero()

      local outputs = mlp:forward(batchTrainInputs)
      local loss = criterion:forward(outputs, batchTrainOutputs)
      local dloss_doutput = criterion:backward(outputs, batchTrainOutputs)
      mlp:backward(batchTrainInputs, dloss_doutput)

      -- print(loss)

      return loss,gradParams
    end

    optim.sgd(feval, params, optimState)
  end

  local outputs = mlp:forward(validInputs)
  local loss = criterion:forward(outputs, validOutputs)

  print(loss)
end
-- trainer = nn.StochasticGradient(mlp, criterion)
-- -- trainer.maxIteration = 100
-- trainer.learningRate = 0.1
-- trainer:train(train)
-- trainer:train(valid) -- lazy eye-ball validation for a different in training error

-- for i=1,10 do
--   print(valid[i])
--   -- print(predictions[i])
--   print(mlp:forward(valid[i][1]))
--   print(valid[i][2][1])
-- end
-- -- trainer:train(valid)