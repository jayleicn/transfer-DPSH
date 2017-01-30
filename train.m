function [net, U, B, loss_iter] = train (U, B, W, X_t, L_t, net, U0_source, U0_L, t, lambda, eta, iter, lr, loss_iter, batchsize) %X_s, Idx_s, net_source,
    N = size(X_t,4); % 5000
    index = randperm(N);
    codelen = size(U0_source,2);
    for j = 0:ceil(N/batchsize)-1
        batch_time=tic;
        %% random select a minibatch
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));
        S = calcNeighbor (L_t, ix, 1:N); % #batchsize * N in {1,0}, similarity matrix
        %% load target
        labels = U0_L(ix);
        im = X_t(:,:,:,ix);
        im_ = single(im); % note: 0-255 range, single precision
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)); % resize 32 --> 224
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4));
        im_ = gpuArray(im_); % 224*224*3*128 
        res = vl_simplenn(net, im_);
        U0 = squeeze(gather(res(end).x))'; % res(end).x is the net output, batchsize * codelen
        U(ix,:) = U0 ; % update relative rows
        B(ix,:) = sign(U0);  % update relative rows

        T = U0 * U' / 2;
        A = 1 ./ (1 + exp(-T)); 
        bN = size(ix, 2) * N;
        loss_hard_1 = -S.*T + log1p(exp(-T)) + T;
        loss_hard_2 = lambda*((U0-sign(U0)).^2); % log(1+exp(-x)) + x
        loss_hard = (sum(loss_hard_1(:)) + sum(loss_hard_2(:)))/bN;
        dJdU = ((S - A) * U - 2*lambda*(U0-sign(U0)))/bN; % hard

        for i = 0:9
            cls_weighted_U0_source(i+1,:) = U0_source{i} .* repmat(W(i+1)', 1, codelen)
        end
        weighted_U0_source = cls_weighted_U0_source(labels+1,:) % batchsize * codelen
        
        softmax_U0 = softmax(U0')';
        softmax_weighted_U0_source = softmax(weighted_U0_source' ./t)';
        loss_soft = -softmax_weighted_U0_source.*log(softmax_U0 + 1e-30); % cross_entropy
        dJdU_soft = t*t*(softmax_weighted_U0_source - softmax_U0)/size(ix,2); % cross_entropy

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
