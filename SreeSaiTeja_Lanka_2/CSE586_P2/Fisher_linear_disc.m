function[w] = Fisher_linear_disc(x_n, K, t)
l = unique(t);
[a,b] = size(x_n);
%initialize the variance within vector
var_w = zeros(b,b);
n = a;
for i = 1:size(l,1)
    x_k = x_n(t == l(i), :);
    class_size = size(x_k,1);
    m_k(i,:) = mean(x_k) * class_size;
    var_k = cov(x_k)*class_size;
    var_w = var_w + var_k;
end
var_b = cov(m_k)*n;
final  = var_w\var_b;
%convert to eigen values and vectors
[vectors, values] = eig(final);
%descending sort based on values and get indexes
[sorted_values, indx]  = sort(values, 'descend');
w = vectors(:, indx(1:K-1));
end
