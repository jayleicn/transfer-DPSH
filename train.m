function [net, U, B, W, loss_iter] = train (U, B, W, s_2, X_t, L_t, net, U0_source, t, m, alpha, eta, mu_1, mu_2, iter, lr, loss_iter, batchsize) %X_s, Idx_s, net_source,
    N = size(X_t,4); % 5000 * ratio
    index = randperm(s_2, N); 
    codelen = size(U0_source{1},2);
    for j = 0:ceil(N/batchsize)-1
        batch_time=tic;
        %% random select a minibatch
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));
        S = 1 - calcNeighbor (L_t, ix, 1:N); % #batchsize * N in {1,0}, similarity matrix
        %% load target
        im = X_t(:,:,:,ix);
        im_ = single(im); % note: 0-255 range, single precision
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)); % resize 32 --> 224
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4));
        im_ = gpuArray(im_); % 224*224*3*128 
        res = vl_simplenn(net, im_);
        U0 = squeeze(gather(res(end).x))'; % res(end).x is the net output, batchsize * codelen
        U(ix,:) = U0; % update relative rows
        B(ix,:) = sign(U0);  % update relative rows

        % T = U0 * U' / 2;
        % A = 1 ./ (1 + exp(-T)); 
        % bN = size(ix, 2) * N;
        % loss_hard_1 = -S.*T + log1p(exp(-T)) + T;
        % loss_hard_2 = lambda*((U0-sign(U0)).^2); % log(1+exp(-x)) + x
        % loss_hard = (sum(loss_hard_1(:)) + sum(loss_hard_2(:)))/bN;
        % dJdU = ((S - A) * U - 2*lambda*(U0-sign(U0)))/bN; % hard

        % [begin] compute loss_hard and gradient  -- only in minibatch
        S = S(:,ix);
        cur_b = size(ix,2);
        cur_b2 = cur_b^2;
        loss_1 = 0;
        loss_2 = 0;
        clear dJdU_1; % matrix
        clear dJdU_2; % matrix
        for i=1:size(ix,2)
            A = repmat(U0(i,:), cur_b, 1) - U0; % N * codelen
            A_sum_square = sum(A.^2, 2); % N*1
            loss_1 = loss_1 + 0.5 * (1 - S(i,:)) * A_sum_square;
            loss_2 = loss_2 + 0.5 * S(i,:) * max(m - A_sum_square, 0);
            dJdU_1(i,:) = 2 * (1 - S(i,:)) * A;
            dJdU_2(i,:) = - 2 * S(i,:) * (A .* repmat((m - A_sum_square) > 0, 1, size(U0, 2)));  
        end
        loss_1 = loss_1 / cur_b2;
        loss_2 = loss_2 / cur_b2;
        dJdU_1 = dJdU_1 ./ cur_b2;
        dJdU_2 = dJdU_2 ./ cur_b2;
        
        U0_3 = abs(abs(U0)-1);
        loss_3 = alpha * sum(U0_3(:)) / cur_b;
        dJdU_3 = 2 * alpha * d_func(U0) ./ cur_b;

        loss_hard = loss_1 + loss_2 + loss_3;
        dJdU = - dJdU_1 - dJdU_2 - dJdU_3; 
        % [end] compute loss_hard and gradient

        % averged sum [source]
        for i = 0:9
            tmp = U0_source{i+1} .* repmat(500.*W(i+1,:)', 1, codelen);
            cls_weighted_U0_source(i+1,:) = sum(tmp, 1) ./ 500;
        end
        labels = L_t(ix) + 1; % this should be L_t, from target set.
        weighted_U0_source = cls_weighted_U0_source(labels,:); % batchsize * codelen
        
        % update neural nets & compute loss
        softmax_U0 = softmax(U0')';
        Q = weighted_U0_source./t;
        P = softmax(Q')';
        loss_soft = -P.*log(softmax_U0 + 1e-30); % cross_entropy
        dJdU_soft = t*t*(P - softmax_U0)/size(ix,2); % cross_entropy, batchsize * codelen


        % sum-to-one constraint, else using l1-norm
        batchW = W(labels,:);
        abs_batchW = abs(batchW);
        sum_batchW = sum(batchW, 2);
        square_batchW = sum_batchW .* sum_batchW;
        loss_soft = ( sum(loss_soft(:)) + mu_1*( sum(square_batchW(:)) + size(ix,2) - 2*sum(sum_batchW(:)) ) + mu_2*sum(abs_batchW(:)) )/size(ix,2); % actually this is incorrect, since many different W(i,:)

        loss_batch = loss_hard + eta*loss_soft;
        loss_iter = loss_iter + loss_batch;
        dJdU = dJdU + eta*dJdU_soft;
        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)]));
        res = vl_simplenn( net, im_, dJdoutput);

        %% update the parameters of CNN
        net = update(net , res, lr);

        % [begin] Update W  
        % How? update from W(1,:) to W(10,:), in sequence.
        % for single first
        % [batchsize * code_len]
        updateW = true;
        if updateW 
            lr_w = 0.01;
            dJdP = -log(softmax_U0 + 1e-30); % [batchsize * code_len]
            for i =1:length(labels)
               single_dPdQ = repmat(P(i,:)',1,codelen) .* eye(codelen) - P(i,:)'*P(i,:); % code_len * code_len
               single_dQdW = U0_source{labels(i)}'; % should be[ code_len x n_per_cls ]
               dJdW(i,:) = dJdP(i,:) * single_dPdQ * single_dQdW; % (batchsize x) [1 x codelen] * [ codelen x codelen ] * [ codelen x  n_per_cls ]
            end

            for i = 1:10
               num_i = sum(labels==i);
               if num_i
                  cls_dJdW(i,:) = - sum(dJdW(find(labels==i),:), 1)/num_i - mu_1*2*(sum(W(i,:))*ones(size(W(i,:)))-1) - mu_2*sign(W(i,:));
               else
                  cls_dJdW(i,:) = zeros(1,500); % - sign(W(i,:)); % do not add l1 norm, since no update for it
               end
            end
            W = W + lr_w .* cls_dJdW; % W [ 10 x n_per_cls ], note cls_dJdW is negative derative
            W(W<0) = 0; % non-negative constraint
        end
        % [end] Update W 

        batch_time = toc(batch_time);
        fprintf(' iter %d loss %.2f = %.2f + %.2f batch %d/%d (%.1f img/s) ,lr is %d\n', iter, loss_batch, loss_hard, eta*loss_soft, j+1,ceil(size(X_t,4)/batchsize), batchsize/ batch_time,lr) ;
    end
    loss_iter = loss_iter/ceil(N/batchsize);
end	
