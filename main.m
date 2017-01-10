function map = main()
ratio_array = [0.05, 0.1, 0.2];
batchsize = [32, 32, 32];
t = [0.5, 1, 2, 5, 10];
l = length(t);
%eta = ones(1,l)*0.1;
map = zeros(4,l);
for i=1:3
   for j=1:l
   [ ~, ~, map_tmp] = transfer_hash(32, 'svhn', 'mnist', t(j), 0.1, ratio_array(i), batchsize(i));
   map(i+1,j) = map_tmp;
   map(1,j)=t(j);
   end
end
fileID = fopen(['results/', 'mnist2svhn005_02.log'], 'w');
fprintf(fileID, '%6s\t%6s\t%6s\t%6s\n', 'tmep', 'map005', 'map_01', 'map_02');
fprintf(fileID, '%6d\t%2.4f\t%2.4f\t%2.4f\n', map);
fclose(fileID);
end
