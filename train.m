function [net, U, B, loss_iter] = train (U, B, s_2, X_t, L_t, net, X_s, Idx_s, net_source, t, lambda, eta, iter, lr, loss_iter, batchsize, lossOption) 
    N = size(X_t,4); % 5000
    index = randperm(s_2,N);
    for j = 0:ceil(N/batchsize)-1
        batch_time=tic;
        %% random select a minibatch
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));
        S = calcNeighbor (L_t, ix, 1:N); % #batchsize * N in {1,0}, similarity matrix
        %% load target
        im = X_t(:,:,:,ix);
        im_ = single(im); % note: 0-255 range, single precision
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)); % resize 32 --> 224
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4));
        im_ = gpuArray(im_); % 224*224*3*128 
        res = vl_simplenn(net, im_);
        U0 = squeeze(gather(res(end).x))'; % res(end).x is the net output, batchsize * codelen
        U(ix,:) = U0 ; % update relative rows
        B(ix,:) = sign(U0);  % update relative rows

        %% load source
        ids =  cat(1, Idx_s{ix});
        if ~strcmp(lossOption{2}, '10-distill')
            tmp_ids = [1:length(ids)/10] * 10;
            ids = ids(tmp_ids);
        end

        ims = X_s(:,:,:,ids);
        ims_ = single(ims); % note: 0-255 range, single precision
        ims_ = imresize(ims_, net.meta.normalization.imageSize(1:2)); % resize 32 --> 224
        ims_ = ims_ - repmat(net.meta.normalization.averageImage,1,1,1,size(ims_,4));
        ims_ = gpuArray(ims_); % 224*224*3*128  

        %% run the CNN
        res_source = vl_simplenn(net_source, ims_);
        U0_source = squeeze(gather(res_source(end).x))'; % only source will add temperatue
        if strcmp(lossOption{2}, '10-distill')
            tmp = [];
            for i = 1:length(ids)/10
                tmp(i,:) = sum(U0_source(i*10-9:i*10,:));
            end
            U0_source = tmp;
        end


        if strcmp(lossOption{1}, 'soft-only')
            loss_hard = 0;
            dJdU = 0;
            switch lossOption{2}
                case 'l2-norm'
                    loss_soft = (U0-U0_source).^2; % L2-norm
                    dJdU_soft = 2*(U0_source - U0)/size(ix,2); % L2-norm
                case {'1-distill', '10-distill'}
                    softmax_U0 = softmax(U0')';
                    softmax_U0_source = softmax(U0_source' ./t)';
                    loss_soft = -softmax_U0_source.*log(softmax_U0 + 1e-30); % cross_entropy
                    dJdU_soft = softmax_U0_source - softmax_U0/size(ix,2); % cross_entropy
                otherwise
                    disp('Err for lossOption field 2');
            end

        elseif strcmp(lossOption{1}, 'soft-hard')
            T = U0 * U' / 2;
            A = 1 ./ (1 + exp(-T)); 
            bN = size(ix, 2) * N;
            loss_hard_1 = -S.*T + log1p(exp(-T)) + T;
            loss_hard_2 = lambda*((U0-sign(U0)).^2); % log(1+exp(-x)) + x
            loss_hard = (sum(loss_hard_1(:)) + sum(loss_hard_2(:)))/bN;
            dJdU = ((S - A) * U - 2*lambda*(U0-sign(U0)))/bN; % hard
            switch lossOption{2}
                case 'l2-norm'
                    loss_soft = (U0-U0_source).^2; % L2-norm
                    dJdU_soft = 2*(U0_source - U0)/size(ix,2); % L2-norm
                case {'1-distill', '10-distill'}
                    softmax_U0 = softmax(U0')';
                    softmax_U0_source = softmax(U0_source' ./t)';
                    loss_soft = -softmax_U0_source.*log(softmax_U0 + 1e-30); % cross_entropy
                    dJdU_soft = t*t*(softmax_U0_source - softmax_U0)/size(ix,2); % cross_entropy
                otherwise
                    disp('Err for lossOption field 2');
            end
        else
            disp('Err input for lossOption field 1');
        end

        loss_soft = sum(loss_soft(:))/size(ix,2);
        loss_batch = loss_hard + eta*loss_soft;
        loss_iter = loss_iter + loss_batch;
        dJdU = dJdU + eta*dJdU_soft;
        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)]));
        res = vl_simplenn( net, im_, dJdoutput);
        %% update the parameters of CNN
        net = update(net , res, lr);
        batch_time = toc(batch_time);
        fprintf(' iter %d loss %.2f = %.2f + %.2f batch %d/%d (%.1f img/s) ,lr is %d\n', iter, loss_batch, loss_hard, eta*loss_soft, j+1,ceil(size(X_t,4)/batchsize), batchsize/ batch_time,lr) ;
    end
    loss_iter = loss_iter/ceil(N/batchsize);
end	
