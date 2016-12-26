% Evaluate the performance of the deep hash model
% data_set: retrieve set
% dataset_L: retrieve set ground truth
% test_data: query set
% test_L: query set ground truth
% retrieve set 59,000 -- query set 1,000, final
function [map,B_dataset,B_test] = test(net, dataset_L, test_L,data_set, test_data ) 
    S = compute_S(dataset_L,test_L) ; 
    [B_dataset, B_test] = compute_B (data_set,test_data,net) ;
    map = return_map (B_dataset, B_test, S) ;
end