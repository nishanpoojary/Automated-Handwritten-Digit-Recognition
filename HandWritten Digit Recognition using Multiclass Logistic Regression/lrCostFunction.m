function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== CODE HERE ======================
%               Set J to the cost.
%               Computing the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%               Vectorised Implementation

h = sigmoid(X*theta);

% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
shift_theta = theta(2:end);
theta_reg = [0;shift_theta];

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

% grad_zero = (1/m)*X(:,1)'*(h-y);
% grad_rest = (1/m)*(shift_x'*(h - y)+lambda*shift_theta);
% grad      = cat(1, grad_zero, grad_rest);

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);








% =============================================================

grad = grad(:);

end
