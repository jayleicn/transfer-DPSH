function [net, U, B, loss_iter] = train (U, B, X_t, L_t, net, X_s, L_s, net_source, t, lambda, eta, iter, lr, loss_iter, batchsize) 
    N = size(X_t,4); % 5000
    index = randperm(N);
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
        %% load source
        ims = X_s(:,:,:,ix);
        ims_ = single(ims); % note: 0-255 range, single precision
        ims_ = imresize(ims_, net.meta.normalization.imageSize(1:2)); % resize 32 --> 224
        ims_ = ims_ - repmat(net.meta.normalization.averageImage,1,1,1,size(ims_,4));
        ims_ = gpuArray(ims_); % 224*224*3*128         
        %% run the CNN
        res = vl_simplenn(net, im_);
        res_source = vl_simplenn(net_source, ims_);
        U0 = squeeze(gather(res(end).x))'; % res(end).x is the net output, batchsize * codelen
        U0_source = squeeze(gather(res_source(end).x))'; %./ t; % only source will add temperatue
        softmax_U0 = softmax(U0')';
        softmax_U0_source = softmax(U0_source' ./t)';
        U(ix,:) = U0 ; % update relative rows
        B(ix,:) = sign(U0);  % update relative rows
        T = U0 * U' / 2;
        A = 1 ./ (1 + exp(-T)); 
        bN = size(ix, 2) * N;
        loss_hard_1 = -S.*T + log1p(exp(-T)) + T;
        loss_hard_2 = lambda*((U0-sign(U0)).^2); % log(1+exp(-x)) + x
        loss_hard = (sum(loss_hard_1(:)) + sum(loss_hard_2(:)))/bN;
        loss_soft = -softmax_U0_source.*log(softmax_U0 + 1e-30); % cross_entropy
        %loss_soft = (U0-U0_source).^2; % L2-norm
        loss_soft = t*t*sum(loss_soft(:))/size(ix,2);
        loss_batch = loss_hard + eta*loss_soft;
        loss_iter = loss_iter + loss_batch;
        % Note!!! that this is actually -dJdU.
        dJdU = ((S - A) * U - 2*lambda*(U0-sign(U0)))/bN; % hard
        % dJdU_soft = 2*(U0_source - U0)/size(ix,2); % L2-norm
        dJdU_soft = t*t*(softmax_U0_source - softmax_U0)/size(ix,2); % cross_entropy
        dJdU = dJdU + eta*dJdU_soft; % hard_traget + soft_target   
        % dJdU_norm = norm(dJdU, 'fro');
        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)]));
        res = vl_simplenn( net, im_, dJdoutput);
        %% update the parameters of CNN
        net = update(net , res, lr);
        batch_time = toc(batch_time);
        fprintf(' iter %d loss %.2f = %.2f + %.2f batch %d/%d (%.1f img/s) ,lr is %d\n', iter, loss_batch, loss_hard, eta*loss_soft, j+1,ceil(size(X_t,4)/batchsize), batchsize/ batch_time,lr) ;
    end
    loss_iter = loss_iter/ceil(N/batchsize);
end	

% function Y = softmax_customized(X) % X is a NxC matrix
% E = exp(bsxfun(@minus, X, max(X, [], 2)));
% L = sum(E,2);
% Y = bsxfun(@rdivide, E, L);
% end
