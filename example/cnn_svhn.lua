
torch    = require 'torch'
autograd = require 'autograd'
dist     = require 'distributions'
image    = require 'image'
util     = require 'autograd.util'
optim    = require 'optim'
lossFuns = require 'autograd.loss'



function ready_files()

	local dirs = {}
	table.insert(dirs, '../data/1/')
	table.insert(dirs, '../data/2/')
	table.insert(dirs, '../data/3/')
	table.insert(dirs, '../data/4/')
	table.insert(dirs, '../data/5/')
	table.insert(dirs, '../data/6/')
	table.insert(dirs, '../data/7/')
	table.insert(dirs, '../data/8/')
	table.insert(dirs, '../data/9/')
	table.insert(dirs, '../data/10/')

	local files  = {}
	local labels = {}

	for k, dir in pairs (dirs) do 

		for file in paths.files(dir) do

			if file:find('jpg' .. '$') then

				print(file)
				table.insert(files, paths.concat(dir,file))
				table.insert(labels, k)
			end
		end
	end
	
	return files, labels
end

local files, labels = ready_files()

for k, v in pairs( files ) do
   		print(k, v)
		print(labels[k])
end



function shuffle(files, labels)
	
  	local new_files = {}
  	local new_labels = {}
  	for i=1, #labels do
    		ran = math.random(#new_labels+1)	
    		table.insert(new_files, ran, files[i])
    		table.insert(new_labels, ran, labels[i])	
  	end
  	return new_files, new_labels

end

local files, labels = shuffle(files, labels)


function load_images(files, labels)

	local trainData = {
				x=torch.Tensor(100,32,32),
			   	y=torch.Tensor(100),
				size = 100
			  }

	local testData = {
				x=torch.Tensor(100,32,32),
			   	y=torch.Tensor(100),
				size = 100
			}

	for k, filename in  pairs(files) do

		num_image = image.load(filename)
		--print(num_image:size(1))
		print(num_image:size()) -- 3x32x32

		local gray_image = torch.Tensor(32, 32)
		local r = num_image[1]
		local g = num_image[2]
		local b = num_image[3]
		
		gray_image = gray_image:add(0.21, r)
		gray_image = gray_image:add(0.72, g)
		gray_image = gray_image:add(0.07, b)
		
		trainData.x[k] = gray_image
		testData.x[k] = gray_image


		
		trainData.y[{k}] = labels[k]
		testData.y[{k}] = labels[k]
		
	end

	local classes = {'1','2','3','4','5','6','7','8','9','10'}

	trainData.y = util.oneHot(trainData.y)
	testData.y = util.oneHot(testData.y)	
	
	return trainData, testData, classes

end

trainData, testData, classes = load_images(files, labels)
print(trainData.size)

local inputSize = trainData.x[1]:nElement()
print (inputSize) --32x32

local confusionMatrix = optim.ConfusionMatrix(classes)

local predict,f,params

local reshape = autograd.nn.Reshape(1,32,32)

local conv1, acts1, pool1, conv2, acts2, pool2, flatten, linear
local params = {}

conv1, params.conv1 = autograd.nn.SpatialConvolutionMM(1, 16, 5, 5)
acts1 = autograd.nn.Tanh()
pool1 = autograd.nn.SpatialMaxPooling(2, 2, 2, 2)

-- 32-5 +1 ==> 28
-- 28/2    ==> 14

conv2, params.conv2 = autograd.nn.SpatialConvolutionMM(16, 16, 5, 5)
acts2 = autograd.nn.Tanh()
pool2, params.pool2 = autograd.nn.SpatialMaxPooling(2, 2, 2, 2)

-- 14 - 5 + 1 ==> 10
-- 10/2       ==> 5 

flatten = autograd.nn.Reshape(16*5*5)
linear,params.linear = autograd.nn.Linear(16*5*5, 10)

--params = autograd.util.cast(params, 'float') -- error!!


-- Define our network
function predict(params, input, target)
   local h1 = pool1(acts1(conv1(params.conv1, reshape(input))))
   local h2 = pool2(acts2(conv2(params.conv2, h1)))
   local h3 = linear(params.linear, flatten(h2))
   local out = util.logSoftMax(h3)
   return out
end

-- Define our loss function
function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = lossFuns.logMultinomialLoss(prediction, target)
   return loss, prediction
end

-- Define our parameters
torch.manualSeed(0)

local df = autograd(f, {optimize=true})

-- Train a neural network
for epoch = 1,20 do
   print('Training Epoch #'..epoch)

   for i = 1,trainData.size do
      
      -- Next sample:
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10)

      -- Grads:
      local grads, loss, prediction = df(params,x,y)

      -- Update weights and biases
      for iparam = 1,2 do
         params.conv1[iparam]  = params.conv1[iparam] - grads.conv1[iparam] * 0.01
         params.conv2[iparam]  = params.conv2[iparam] - grads.conv2[iparam] * 0.01
         params.linear[iparam] = params.linear[iparam] - grads.linear[iparam] * 0.01
      end

      -- Log performance:
      --print(prediction[1], y[1])
      confusionMatrix:add(prediction[1], y[1])
      if i % 100 == 0 then
         print(confusionMatrix)
         confusionMatrix:zero()
      end
      
   end
   
end
