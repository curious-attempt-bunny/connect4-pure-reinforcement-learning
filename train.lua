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

local validInputs = torch.Tensor(nValid, inputSize)
local validOutputs = torch.Tensor(nValid)

for i=1,nValid do
  validInputs[i] = torch.Tensor(valid[i][1])
  validOutputs[i] = torch.Tensor(valid[i][2])
end

require 'nn'

local dropout = nn.Dropout()

mlp = nn.Sequential()
mlp:add(nn.Linear(inputSize,60))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(60,30))
-- mlp:add(dropout)
mlp:add(nn.Sigmoid())
-- mlp:add(nn.Linear(30,15))
-- mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(30,1))

-- iterations 25, learning rate 0.1

-- 85 --signmoid--> 60 --linear--> 1                                               (25 iterations, validation 0.094427700655113) better!
-- 85 --signmoid--> 60 --sigmoid--> 30 --linear--> 1                               (25 iterations, validation 0.055259071962564) better!
-- 85 --signmoid--> 60 --sigmoid--> 30 --sigmoid--> 15 --linear--> 1               (25 iterations, validation 0.050595296822482) better!  +/- 0.04
-- 85 --signmoid--> 60 --sigmoid--> 40 --sigmoid--> 30 --linear--> 1               (25 iterations, validation 0.056451917992587)   xx 
-- 85 --signmoid--> 60 --sigmoid with dropout--> 30 --sigmoid--> 15 --linear--> 1  (25 iterations, validation 0.11103154212837 )   xx
-- 85 --signmoid--> 50 --sigmoid--> 30 --sigmoid--> 15 --linear--> 1               (25 iterations, validation 0.067343755670655)   xx
-- 85 --signmoid--> 70 --sigmoid--> 30 --sigmoid--> 15 --linear--> 1               (25 iterations, validation 0.049547172093965) better ? +/- 0.06
-- 85 --signmoid--> 70 --sigmoid--> 40 --sigmoid--> 15 --linear--> 1               (25 iterations, validation 0.065719608630751)   xx

-- iterations 75, learning rate 0.1

-- 85 --signmoid--> 60 --sigmoid--> 30 --sigmoid--> 15 ---linear--> 1              (75 iterations, validation 0.047547653263386) better!
-- 85 --signmoid--> 60 --sigmoid--> 30 --linear--> 1                               (75 iterations, validation 0.049784894683693)   xx

-- iterations 75, learning rate 0.3 (terrible!)
-- iterations 75, learning rate 0.2 (still bad)
-- iterations 75, learning rate 0.1, learning rate decay 0.1 (0.046840735792506) better ?


-- consider: mlp:add(nn.L1Penalty(0.5))

criterion = nn.MSECriterion()

-- require 'optim'

-- local optimState = {learningRate=0.1}
-- local params, gradParams = mlp:getParameters()

-- for epoch=1,25 do
--   local chunks = nValid / chunkSize
--   local shuffle_indexes = torch.randperm(nTrain)
--   local batchTrainInputs = torch.Tensor(chunkSize, inputSize)
--   local batchTrainOutputs = torch.Tensor(chunkSize) -- not sure why this had to be hardcoded

--   for batchIndex=0,chunks-1 do
--     for i=1,chunkSize do
--       batchTrainInputs[i] = torch.Tensor(train[shuffle_indexes[batchIndex*chunkSize+i]][1])
--       batchTrainOutputs[i] = torch.Tensor(train[shuffle_indexes[batchIndex*chunkSize+i]][2])
--     end

--     local function feval(params)
--       gradParams:zero()

--       local outputs = mlp:forward(batchTrainInputs)
--       local loss = criterion:forward(outputs, batchTrainOutputs)
--       local dloss_doutput = criterion:backward(outputs, batchTrainOutputs)
--       mlp:backward(batchTrainInputs, dloss_doutput)

--       -- print(loss)

--       return loss,gradParams
--     end

--     optim.sgd(feval, params, optimState)
--   end

--   local outputs = mlp:forward(validInputs)
--   local loss = criterion:forward(outputs, validOutputs)

--   print(loss)
-- end
trainer = nn.StochasticGradient(mlp, criterion)
function trainer:hookIteration(iteration, currentError)
  dropout.train = false
  local outputs = mlp:forward(validInputs)
  local loss = criterion:forward(outputs, validOutputs)
  dropout.train = true
  
  print(iteration, currentError, loss)
end

trainer.maxIteration = 1000
trainer.learningRate = 0.1
trainer.learningRateDecay = 0.1
trainer.verbose = false
trainer:train(train)

-- for i=1,10 do
--   print(valid[i])
--   -- print(predictions[i])
--   print(mlp:forward(valid[i][1]))
--   print(valid[i][2][1])
-- end
-- -- trainer:train(valid)