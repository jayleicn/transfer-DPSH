function map = main()
codelen = 32; % m = 2*codelen
alpha = 0.01;
ratio_array = [0.1, 0.2, 0.4];
batchsize = [32, 32, 64];
N = length(batchsize);
t = 2;%[ 0.5, 1, 2, 5, 10, 20];
%eta = [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1];
% eta= [0.2, 0.2, 0.2];
eta = 0.5;
%mu_1 = [0.001, 0.01, 0.1, 0.5, 1, 10];
mu_1 = 0.5;
mu_2 = 0.1;
l = length(eta);
map = zeros(4,l);
for i=1:N
   for j=1:l
     [ ~, ~, map_tmp] = transfer_dsh(32, 'svhn', 'mnist', t, 64, alpha, eta(j), mu_1, mu_2, ratio_array(i), batchsize(i));
     map(i+1,j) = map_tmp;
     map(1,j)=eta(j);
   end
end
fileID = fopen(['results_dsh/', 'mnist2svhn005_02_eta2.log'], 'w');
fprintf(fileID, '%6s\t%6s\t%6s\t%6s\n', 'eta', 'map01', 'map_02', 'map_04');
fprintf(fileID, '%2.4f\t%2.4f\t%2.4f\t%2.4f\n', map);
fclose(fileID);
end
