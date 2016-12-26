function [B_dataset,B_test,map] = DPSH(codelens,dataset_name, ratio)
    %% download data and pre-trained CNN from the web
    % download_data; % use "run download_data.m" seperately is prefered,
    % since it takes a lot of time
    %% prepare the dataset % best run once
    if ~exist([dataset_name,'.mat'])
        data_prepare(dataset_name);
    end
    load([dataset_name,'.mat']);

    %% vary training data size
    % ratio = 1.0;
    train_data_tmp = [];
    train_L_tmp = [];
    for label=0:9
        index = find(train_L==label);
        N = size(index,1);
        perm = randperm(N);
        index = index(perm);
        data = train_data(:,:,:,index(1:ceil(N*ratio)));
        labels = train_L(index(1:ceil(N*ratio)));
        train_data_tmp = cat(4,train_data_tmp,data);    
        train_L_tmp = cat(1,train_L_tmp,labels);
    end
    train_data = train_data_tmp;
    train_L = train_L_tmp;   

    %% load the pre-trained CNN and source model
    net = load('/home/jielei/data/model/imagenet-vgg-f.mat');
    net = vl_simplenn_tidy(net);
    net_source = load('/home/jielei/project/s_new/transfer-DPSH/results/mnist_net.mat');
    net_source = net_source.net;
    net_source = vl_simplenn_tidy(net_source);

    %% initialization
    maxIter = 90;
    lambda = 10;
    t=1.0; % temperature of soft_target
    eta=0.1; % weight of soft_target
    %lr = logspace(-2,-6,maxIter); with maxIter =150
    lr(1:80) =  0.01;
    lr(81:90) = 0.001;

    totalTime = tic;
    net = net_structure(net, codelens);
    U = zeros(size(train_data,4),codelens);
    B = zeros(size(train_data,4),codelens);

    %% saving
    if ~exist('results', 'dir')
        mkdir('results');
    end
    dir_time = [dataset_name, '-', num2str(codelens), '-', datestr(now, 'dd-mmm-yyyy-HH:MM:SS')];
    mkdir(['results/', dir_time]);
    fileID = fopen(['results/', dir_time, '/loss.log'],'w');  
    fprintf(fileID, '%.2f', 'ratio'); 
    fprintf(fileID,'%6s %12s %10s\n','iter','loss', 'lr');
    fclose(fileID);

    fileID = fopen(['results/', dir_time, '/map.log'],'w');    
    fprintf(fileID,'%6s %4s\n','iter','map');
    fclose(fileID);

    %% training train (X1, L1, U, B, net, net_source, t, lambda, eta, iter, lr, loss_iter) 
    for iter = 1: maxIter
        loss_iter = 0;
        [net, U, B, loss_iter] = train(train_data,train_L,U,B,net, net_source, t, lambda, eta, iter, lr(iter), loss_iter);
        fileID = fopen(['results/', dir_time, '/loss.log'], 'a'); % append
        fprintf(fileID, '%6d %12.2f %10d\n', [iter; loss_iter; lr(iter)]);
        fclose(fileID);
        %validating
        if mod(iter, 5) == 0
            map_val = test(net, retrieve_val_L, val_L, retrieve_val, val_data );
            fprintf('current validation MAP is %.2f\n', map_val);
            fileID = fopen(['results/', dir_time, '/map.log'], 'a'); % append
            map_iter = [iter; map_val];
            fprintf(fileID, '%6d %4.2f\n', map_iter);
            fclose(fileID);
        end
    end
 
    %% testing
    [map,B_dataset,B_test] = test(net, retrieve_test_L, test_L, retrieve_test, test_data );
    fileID = fopen(['results/', dir_time, '/map.log'], 'a'); % append
    map_iter = [0; map];
    fprintf(fileID, '%6d %4.2f\n', map_iter);
    fclose(fileID);
    save(['./results/', dir_time, '/codes_res', '.mat'], 'B_dataset','B_test','map');
    save(['./results/', dir_time, '/net', '.mat'], 'net');
    totalTime=toc(totalTime);
    fprintf('Total elapsed time is %4.2f seconds', totalTime);
end
