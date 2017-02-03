function [net, U, B, W, loss_iter] = train (U, B, W, s_2, X_t, L_t, net, U0_source, t, lambda, eta, mu, iter, lr, loss_iter, batchsize) %X_s, Idx_s, net_source,
    N = size(X_t,4); % 5000 * ratio
    index = randperm(s_2, N); 
    codelen = size(U0_source{1},2);
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

        T = U0 * U' / 2;
        A = 1 ./ (1 + exp(-T)); 
        bN = size(ix, 2) * N;
        loss_hard_1 = -S.*T + log1p(exp(-T)) + T;
        loss_hard_2 = lambda*((U0-sign(U0)).^2); % log(1+exp(-x)) + x
        loss_hard = (sum(loss_hard_1(:)) + sum(loss_hard_2(:)))/bN;
        dJdU = ((S - A) * U - 2*lambda*(U0-sign(U0)))/bN; % hard

        % averged sum [source]
        for i = 0:9
            tmp = U0_source{i+1} .* repmat(500.*W(i+1,:)', 1, codelen);
            cls_weighted_U0_source(i+1,:) = sum(tmp, 1) ./ 500;
        end
        labels = L_t(ix); % this should be L_t, from target set.
        weighted_U0_source = cls_weighted_U0_source(labels+1,:); % batchsize * codelen
        
        % update neural nets & compute loss
        softmax_U0 = softmax(U0')';
        Q = weighted_U0_source./t;
        P = softmax(Q')';
        loss_soft = -P.*log(softmax_U0 + 1e-30); % cross_entropy
        dJdU_soft = t*t*(P - softmax_U0)/size(ix,2); % cross_entropy


        % sum-to-one constraint, else using l1-norm
        sum_to_one = true; 
        if sum_to_one
            batchW = W(labels+1,:);
            sum_batchW = sum(batchW, 2);
            square_sum_batchW = sum_batchW .* sum_batchW;
            loss_soft = ( sum(loss_soft(:))  + mu*(sum(square_sum_batchW(:))+size(ix,2) - 2*sum_batchW ) )/size(ix,2); % actually this is incorrect, since many different W(i,:)
        else
            batchW = abs(W(labels+1,:));
            loss_soft = ( sum(loss_soft(:))  + mu*sum(batchW(:)) )/size(ix,2); % actually this is incorrect, since many different W(i,:)
        end

        loss_batch = loss_hard + eta*loss_soft;
        loss_iter = loss_iter + loss_batch;
        dJdU = dJdU + eta*dJdU_soft;
        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)]));
        res = vl_simplenn( net, im_, dJdoutput);
        %% update the parameters of CNN
        net = update(net , res, lr);

        % Update W begin 
        % How? update from W(1,:) to W(10,:), in sequence.
        % for single first
        % [batchsize * code_len]
        updateW = true;
        if updateW 
            lr_w = 0.01;
            dJdP = -log(softmax_U0 + 1e-30); % [batchsize * code_len]
            for i =1:length(labels)
               single_dPdQ = repmat(P(i,:)',1,codelen) .* eye(codelen) - P(i,:)'*P(i,:); % code_len * code_len
               single_dQdW = U0_source{labels(i)+1}'; % should be[ code_len x n_per_cls ]
               dJdW(i,:) = dJdP(i,:) * single_dPdQ * single_dQdW; % (batchsize x) [1 x codelen] * [ codelen x codelen ] * [ codelen x  n_per_cls ]
            end

            for i = 1:10
               num_i = sum(labels==i);
               if num_i
                  if sum_to_one
                      % cls_dJdW(i,:) = - sum(dJdW(find(labels==i),:), 1)/num_i - mu * abs(sign(W(i,:);
                      cls_dJdW(i,:) = - sum(dJdW(find(labels==i),:), 1)/num_i - 2 * mu * (W(i,:) - 1);
                  else
                      cls_dJdW(i,:) = - sum(dJdW(find(labels==i),:), 1)/num_i - mu * sign(W(i,:)); % add l1 norm
                  end
               else
                  cls_dJdW(i,:) = zeros(1,500); % - sign(W(i,:)); % do not add l1 norm, since no update for it
               end
            end
            W = W + lr_w .* cls_dJdW; % W [ 10 x n_per_cls ], note cls_dJdW is negative derative
            W(W<0) = 0; % non-negative constraint
        end
        % Update W end

        batch_time = toc(batch_time);
        fprintf(' iter %d loss %.2f = %.2f + %.2f batch %d/%d (%.1f img/s) ,lr is %d\n', iter, loss_batch, loss_hard, eta*loss_soft, j+1,ceil(size(X_t,4)/batchsize), batchsize/ batch_time,lr) ;
    end
    loss_iter = loss_iter/ceil(N/batchsize);
end	
