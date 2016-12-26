function map = main()
ratio_array = linspace(1,10,10);
map = zeros(10,1);
for i=1:10
   [ ~, ~, map_tmp] = DPSH(32, 'cifar10', ratio_array(i)*0.1);
   map(i) = map_tmp;
end
end
