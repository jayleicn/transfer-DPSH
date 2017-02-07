% D_s = [U_s, y_s]
function source_data_prep(dataset_s) % train_data, train_L  *,*,*,5000
    % load dataset
    if ~exist([dataset_s,'.mat'])
        data_prepare(dataset_s);
    end
    dataset_source = load([dataset_s,'.mat']);
    X_s = dataset_source.train_data;

    % load source pretrained network
    % net_source = load('/home/jielei/project/s_new/DPSH-IJCAI16/results/raw-exp/mnist-32-10-Jan-2017-01:10:23/net.mat'); % 0.99
    net_source = net_source.net;
    net_source = vl_simplenn_tidy(net_source);

    % load vgg ==> in order to load the image parameters
    net = load('/home/jielei/data/model/imagenet-vgg-f.mat');
    net = vl_simplenn_tidy(net);

    bs = 256; %batchsize
    N = length(dataset_source.train_L);
    U0_source = [];
    for j = 0:ceil(N/bs)-1
        ids = (1+j*bs):min((j+1)*bs,N);
        ims = X_s(:,:,:,ids);
        ims_ = single(ims); % note: 0-255 range, single precision
        ims_ = imresize(ims_, net.meta.normalization.imageSize(1:2)); % resize 32 --> 224
        ims_ = ims_ - repmat(net.meta.normalization.averageImage,1,1,1,size(ims_,4));
        ims_ = gpuArray(ims_); % 224*224*3*128  

        %% run the CNN
        res_source = vl_simplenn(net_source, ims_);
        tmp_U0_source = squeeze(gather(res_source(end).x))'; % only source will add temperatue
        U0_source = cat(1, U0_source, tmp_U0_source);
    end
    U0_L = dataset_source.train_L;
    save([dataset_s,'_dhn_U0.mat'], 'U0_source', 'U0_L');
end
