function y = d_func(x)
y =  (x >= -1 & x <= 0) | x >= 1;
y = sign(y-0.5);
end