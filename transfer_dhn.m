function [B_dataset,B_test,map] = transfer_dhn(codelens, dataset_t, dataset_s, t, lambda, eta, mu_1, mu_2, ratio,  batchsize)
    if ~exist([dataset_t,'.mat'])
        data_prepare(dataset_t);
    end
    dataset_target = load([dataset_t,'.mat']);

    if ~exist([dataset_s,'_dhn_U0.mat'])
        source_data_prep(dataset_s);
    end
    dataset_source = load([dataset_s,'_dhn_U0.mat']);

    %% vary training data size
    % ratio = 1.0;
    train_data_t = []; % target
    train_L_t = [];
    s_1 = RandStream('mt19937ar','Seed',1); % random seed for training data.
    for label=0:9
        % target
        index_t = find(dataset_target.train_L==label);
        N = size(index_t,1);
        perm = randperm(s_1,N);
        index_t = index_t(perm);
        data = dataset_target.train_data(:,:,:,index_t(1:ceil(N*ratio)));
        labels = dataset_target.train_L(index_t(1:ceil(N*ratio)));
        train_data_t = cat(4,train_data_t,data);  
        train_L_t = cat(1,train_L_t,labels);       
        % source
        index_s = find(dataset_source.U0_L==label);
        U0_source{label+1} = dataset_source.U0_source(index_s,:); % 10 * W*H*Code_len*N
    end 

    %% load the pre-trained CNN and source model
    net = load('/home/jielei/data/model/imagenet-vgg-f.mat');
    net = vl_simplenn_tidy(net);

    %% initialization
    maxIter = 140;
    lr(1:120) =  0.01;
    lr(121:140) = 0.001;

    totalTime = tic;
    net = net_structure(net, codelens);
    U = zeros(size(train_data_t,4),codelens);
    B = zeros(size(train_data_t,4),codelens);
    W = ones(10,500)./500; % initialized as the average

    %% saving
    if ~exist('results_dhn', 'dir')
        mkdir('results_dhn');
    end
    dir_time = [dataset_t, dataset_s, '-', num2str(codelens), '-', datestr(now, 'dd-mmm-yyyy-HH:MM:SS')];
    mkdir(['results_dhn/', dir_time]);
    fileID = fopen(['results_dhn/', dir_time, '/loss.log'],'w');  
    fprintf(fileID, '%.2f', 'ratio'); 
    fprintf(fileID,'%6s %12s %10s\n','iter','loss', 'lr');
    fclose(fileID);

    fileID = fopen(['results_dhn/', dir_time, '/map.log'],'w');    
    fprintf(fileID,'%6s %4s\n','iter','map');
    fclose(fileID);

    fileID = fopen(['results_dhn/', dir_time, '/parameters.log'],'w');    
    fprintf(fileID,'%s \n','codelens, dataset_t, dataset_s, t, lambda, eta, mu_1, mu_2,ratio,  batchsize');
    fprintf(fileID,'%d \t %s \t %s \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %d\n', codelens, dataset_t, dataset_s, t, lambda, eta, mu_1, mu_2, ratio,  batchsize);
    fclose(fileID);
    
    s_2 = RandStream('mt19937ar','Seed',2);  % random seed to shuffle
    for iter = 1: maxIter
        loss_iter = 0;
        [net, U, B, W, loss_iter] = train(U,B,W, s_2, train_data_t,train_L_t, net, U0_source, t, lambda, eta, mu_1, mu_2, iter, lr(iter), loss_iter, batchsize); %dataset_source.train_data, train_idx_s, net_source,
        fileID = fopen(['results_dhn/', dir_time, '/loss.log'], 'a'); % append
        fprintf(fileID, '%6d %10.4f %10d\n', [iter; loss_iter; lr(iter)]);
        fclose(fileID);
        %validating
        if mod(iter, 20) == 0
            map_val = test(net, dataset_target.retrieve_val_L, dataset_target.val_L, dataset_target.retrieve_val, dataset_target.val_data );
            fprintf('current validation MAP is %.2f\n', map_val);
            fileID = fopen(['results_dhn/', dir_time, '/map.log'], 'a'); % append
            map_iter = [iter; map_val];
            fprintf(fileID, '%6d %2.4f\n', map_iter);
            fclose(fileID);
        end
    end
 
    %% testing
    [map,B_dataset,B_test] = test(net, dataset_target.retrieve_test_L, dataset_target.test_L, dataset_target.retrieve_test, dataset_target.test_data );
    fileID = fopen(['results_dhn/', dir_time, '/map.log'], 'a'); % append
    map_iter = [0; map];
    fprintf(fileID, '%6d %2.4f\n', map_iter);
    fclose(fileID);
    save(['./results_dhn/', dir_time, '/codes_res', '.mat'], 'B_dataset','B_test','map', 'W');
    save(['./results_dhn/', dir_time, '/net', '.mat'], 'net');
    totalTime=toc(totalTime);
    fprintf('Total elapsed time is %4.2f seconds', totalTime);
end
