function data_prepare(dataset_name)
X = [];
L = [];
switch dataset_name
    case 'cifar10'
        for i=1:5
            clear data labels batch_label;
            load(['/home/jielei/data/cifar-10-batches-mat/data_batch_' num2str(i) '.mat']);
            data = reshape(data',[32,32,3,10000]);
            data = permute(data,[2,1,3,4]);
            X = cat(4,X,data); % 32*32*3*50000
            L = cat(1,L,labels); % 1*50000 in {0,...,9}
        end
        clear data labels;
        load('/home/jielei/data/cifar-10-batches-mat/test_batch.mat');
        data=reshape(data',[32,32,3,10000]);
        data = permute(data,[2,1,3,4]);
        X = cat(4,X,data); % 32*32*3*(50000+10000)
        L = cat(1,L,labels); % 1*(50000+10000) in {0,...,9}
    case 'svhn'
        svhn_train = load('/home/jielei/data/digits/svhn/svhn_train_32x32.mat');
        X = cat(4,X,svhn_train.X); 
        L = cat(1,L,mod(squeeze(svhn_train.y),10)); 

        svhn_test = load('/home/jielei/data/digits/svhn/svhn_test_32x32.mat');
        X = cat(4,X,svhn_test.X); 
        L = cat(1,L,mod(squeeze(svhn_test.y),10));
    case 'mnist'
        images_2d = loadMNISTImages('/home/jielei/data/digits/mnist/train-images-idx3-ubyte');
        images_2d = reshape(images_2d, [size(images_2d,1),size(images_2d,2),1,size(images_2d,3)]);
        images_3d = repmat(images_2d, [1,1,3,1]); 
        labels = loadMNISTLabels('/home/jielei/data/digits/mnist/train-labels-idx1-ubyte');
        X = cat(4,X,images_3d); 
        L = cat(1,L,labels); 

        images_2d = loadMNISTImages('/home/jielei/data/digits/mnist/t10k-images-idx3-ubyte');
        images_2d = reshape(images_2d, [size(images_2d,1),size(images_2d,2),1,size(images_2d,3)]);
        images_3d = repmat(images_2d, [1,1,3,1]); 
        labels = loadMNISTLabels('/home/jielei/data/digits/mnist/t10k-labels-idx1-ubyte');
        X = cat(4,X,images_3d);
        L = cat(1,L,labels); 
    otherwise
        warning('Unexpected dataset name.')
end


test_data = [];
test_L = [];
val_data = [];
val_L = [];
retrieve_test=[];%data_set = [];
retrieve_test_L=[];%dataset_L = [];
retrieve_val=[];
retrieve_val_L=[];
train_data = [];
train_L = [];
for label=0:9
    index = find(L==label);
    N = size(index,1);
    perm = randperm(N);
    index = index(perm);

    data = X(:,:,:,index(1:100));    
    labels = L(index(1:100));
    test_data = cat(4,test_data,data); 
    test_L = cat(1,test_L,labels);

    data = X(:,:,:,index(101:end)); 
    labels = L(index(101:end));
    retrieve_test = cat(4,retrieve_test,data);
    retrieve_test_L = cat(1,retrieve_test_L,labels);
    
    data = X(:,:,:,index(101:200));    
    labels = L(index(101:200));
    val_data = cat(4,val_data,data);  
    val_L = cat(1,val_L,labels);

    data = X(:,:,:,index(201:2000)); 
    labels = L(index(201:2000));
    retrieve_val = cat(4,retrieve_val,data);  
    retrieve_val_L = cat(1,retrieve_val_L,labels);
    
    data = X(:,:,:,index(201:700));
    labels = L(index(201:700));
    train_data = cat(4,train_data,data);    
    train_L = cat(1,train_L,labels);
end
save(strcat(dataset_name, '.mat'),'test_data','test_L',...
                    'retrieve_test','retrieve_test_L',...
                    'val_data','val_L',...
                    'retrieve_val','retrieve_val_L',...
                    'train_data','train_L');
end

