function map = main()
ratio_array = [0.1, 0.2, 0.4];
batchsize = [32, 32, 64];
N = length(batchsize);
t = [2];%[ 0.5, 1, 2, 5, 10, 20];
% eta = [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1];
% eta= [0.2, 0.2, 0.2];
mu = [0.01, 0.1, 0.5, 1, 2, 10];
l = length(mu);
map = zeros(4,l);
for i=1:N
   for j=1:l
     [ ~, ~, map_tmp] = transfer_hash(32, 'svhn', 'mnist', 2, 0.5, mu(j), ratio_array(i), batchsize(i));
     map(i+1,j) = map_tmp;
     map(1,j)=mu(j);
   end
end
fileID = fopen(['results/', 'mnist2svhn005_02_eta2.log'], 'w');
fprintf(fileID, '%6s\t%6s\t%6s\t%6s\n', 'mu', 'map005', 'map_01', 'map_02');
fprintf(fileID, '%4.2f\t%2.4f\t%2.4f\t%2.4f\n', map);
fclose(fileID);
end
