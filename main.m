function map = main()
ratio_array = [0.1, 0.2];
t = [1, 2, 5, 10, 20, 50];
map = zeros(3,6);
for i=1:2
   for j=1:6
   [ ~, ~, map_tmp] = transfer_hash(32, 'svhn', 'mnist', t(j), ratio_array(i));
   map(i+1,j) = map_tmp;
   map(1,j)=t(j);
   end
end
fileID = fopen(['results/', 'mnist2svhn01_02.log'], 'w');
fprintf(fileID, '%6s\t%6s\t%6s', 'tmep', 'map_01', 'map_02');
fprintf(fileID, '%6d\t%4.2f\t%4.2f', map);
fclose(fileID);
end
