% submission.m
% Student number: 200412739
% Code only - no essay
% setup - Topic 6 VLE
T = 500;
Nu = 100;
b = 0.1;
c = 0.2;
tau = 17;

% generate first 20 random numbers
xi = rand(1, 20)';

% generate next 530 using the mackey-glass equation as in topic 6 VLE code
for t = 20:T+49
    xi(t + 1) = xi(t) + c*xi(t - tau)/(1 + xi(t - tau).^10) - b*xi(t);
end

% data setup
data = xi(100:T);
% normalise data as in topic 6 VLE code
data_norm = 2.0*((data-min(data))/(max(data)-min(data))-0.5);

% set the value for sig to 1
sig = 1;

% set the learning rate
learning_rate = 0.00001;
% set the bias value (the "1" input in the diagram of the NN in brief)
bias = 1;

% exploring the network and observing visually, it was found that the 
% ouput sum weight output with 4 nodes and exp output of 3 nodes 
% gave the best and consistent results
% number of hidden nodes that contributes towards the output of sum
sum_node_count = 4;
% number of hidden nodes that contributes towards the output of exp
exp_node_count = 3;

% the number of nodes in the hidden layer
hidden_node_count = sum_node_count + exp_node_count;

% the weights for input - hidden layer
% three inputs: x[t-1], x[t-2] and the bias
w1 = rand(hidden_node_count, 3);

% the weights for the hidden - output layer for sum (mean) & exp(variance)
% generated randomly - has a significant effect on performance
w2_sum = rand(1, sum_node_count);
w2_exp = rand(1, exp_node_count);

% exploring different values for epochs from 200 to 1500,
% visually 1000 epochs gave the best results
for epoch = 1:1000
    % go through all the data until last two
    % the current value (i) and (i+1) are used for the input
    % the value at (i + 2) is used as the target, therefore
    % if all the data is used there will be an index out of bounds
    for i = 1:size(data_norm) - 2
        % get the lagged values from the normalised data
        x_t1 = data_norm(i);
        x_t2 = data_norm(i+1);
        % the target is actually 2 indexes ahead
        target = data_norm(i + 2);
        
        % forward pass
        inputs = [x_t1; x_t2; bias];

        % multiply the weights with the inputs
        % apply the hyperblic tangent activation function as described
        % the outputs for the hidden layer
        hidden = tanh(w1 * inputs);

        % the outputs of the hidden layer going to the sum 
        hidden_sum = hidden(1:sum_node_count);
        % the outputs of the hidden layer going to the exp 
        hidden_exp = hidden(sum_node_count + 1:end);
        
        % calculate the sum output node value uing the hidden sum weights
        % bias is omitted knowingly as it was not described in
        % the coursework brief for this node
        output_sum = w2_sum * hidden_sum;
        
        % calculate the exp output value applying the exponent function
        % to the weight multiplied by the output of the hidden exp weights
        % bias is omitted knowingly as it was not described in
        % the coursework brief for this node
        output_exp = exp(w2_exp * hidden_exp);

        % backpropagation
        % calculate error
        error = target - output_sum;

        % back pass for sum
        % find the gradient for the errors in the output sum layer
        % by plugging into the differentiated log likelihood function
        % set to zero
        bout_sum = error / sig;
        
        % update sig using the output of the predicted value
        sig = output_exp;
        
        % propagate values back to the hidden nodes that output to the
        % sum outout node
        bp_sum = w2_sum' * bout_sum;

        % back pass for exp
        % repeat as for the sum version but with the differentiated log
        % likelihood function for sigma
        bout_exp = ((error * error)/sig - 1)/2;
        % propagate values back
        bp_exp = w2_exp' * bout_exp;

        % combine bp from sum and exp to get one vector that
        % contains all values to represent the hidden nodes
        bp_all = vertcat(bp_sum, bp_exp);
        
        % calculate the gradient for the input-hidden layer weights
        bh_all = (1.0 - hidden.^2).*bp_all;
        % get the weight deltas for hidden - output 
        dw2_sum = bout_sum * hidden_sum';
        dw2_exp = bout_exp * hidden_exp';
        % udpate the weights hidden - oputput
        w2_sum = w2_sum + learning_rate*dw2_sum;
        w2_exp = w2_exp + learning_rate*dw2_exp;
         
        % update weights for hidden layer (w1)
        % get the weight delta for input - hidden layer
        dw1 = bh_all * inputs';
        % update the weights for input - hidden layer
        w1 = w1 + learning_rate*dw1;
    end
end

% single forward pass for plots
% this calculates and stores the data for the prediction of all the 
% data values
for i = 1:size(data_norm) - 2
    % get inputs
    x_t1 = data_norm(i);
    x_t2 = data_norm(i+1);
    inputs = [x_t1; x_t2; bias];
    
    % calculate outputs of hidden layer (input - hidden)
    hidden = tanh(w1 * inputs);
    
    % get the outputs of the hidden layer for the sum output
    hidden_sum = hidden(1:sum_node_count);
    
    % get the predicted output
    output_sum = w2_sum * hidden_sum;
    
    % get the predicted variance
    output_exp = exp(w2_exp * hidden_exp);

    % store the approximations in outputs_sum
    outputs_sum(i) = output_sum;
    % store the variances in outputs_exp
    outputs_exp(i) = output_exp;
    % store the real value in targets_sum
    targets_sum(i) = data_norm(i+2);
end

% calculate confidence intervals
conf1 = outputs_sum + 1.645*outputs_exp;
conf2 = outputs_sum - 1.645*outputs_exp;

plot([1:size(outputs_sum')],outputs_sum,'r',...
    [1:size(targets_sum')],targets_sum,'b',...
    [1:size(conf1')], conf1, 'g',...
    [1:size(conf2')], conf2, 'g');
legend('Approximation', 'Data values', 'Confidence intervals')
title('Mackay-Glass time series')
