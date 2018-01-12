%
    clear;
    xmean = cell(1,50);
    ymean = cell(1,50);
for f1 = 1:20
    
    clearvars -except xmean ymean f1;
    clc;
    close;
    set(0,'DefaultFigureWindowStyle','docked');
    load highway.mat;
    frames = cell(1,364);
    fprintf('f1 =   %d\n',f1);
    obj_size = 30;
    thresh = 30;
    colormap(gray);
    %%
    
    % averageing background image 
    % we have taken 20 frames for averageing background
     
    avg_back = double(rgb2gray(mov(65).cdata));
    %%
    
    Object = 0;
    temp = 1;
    j = 1;

    start = j;
    on(1) = start;
    M = 3500;
    h = size(avg_back,1)/2;
    w = size(avg_back,2)/2;
    Phi = randn(M,w*h);    
    z = zeros(w*h,size(mov,2));
    N_tar = 0;
    

%%
    flag = 0;
    flaga = 0;
    count = 0;
    x = [];
    for i = 1:size(mov,2)
        img_temp = double(rgb2gray(mov(i).cdata));
        object = imabsdiff(avg_back,img_temp);
        object((find(object < 0.01*min(min(object))))) = 0;    
        object = abs(object);
        object(find(object > 0.4*max(max(object)))) = 255;   
        object1 = object;
        
        subplot(121);imagesc(object);
        object = object > thresh;
        object = bwareaopen(object,obj_size,8);
        object = downsample(object,2);
        object = downsample(object',2)';
        
        object1 = downsample(object1,2);
        object1 = downsample(object1',2)';
        
        frames{i} = object1./max(max(object1));
        
        x_temp = [];
        if(count== 0) 
        x_temp  = distance2(object);
        else
         count = count - 1;
        end    
        if(~isempty(x_temp))
            num = size(x_temp,1);
            count = 8;
            for k = N_tar+1:num+N_tar
                on(k) = i;
                N_tar = N_tar + 1;
            end
            
        end
        x = [x; x_temp];
        z(:,i) = reshape(frames{i},w*h,1);
    end

%%    
    
    x = round(x(:,:));
    T = size(mov,2); %# of time steps
    Q = 18; %process noise variance
    R = 30; %observation noise variance

    loc = zeros(w*h,2);
    cnt = 1;
    for i = 1:w
        for j = 1:h
            loc(cnt,1) = i;
            loc(cnt,2) = j;
            cnt = cnt + 1;
        end
    end

    
    f = [1    0     1     0     0;
        0     1     0     1     0;
        0     0     1     0     0;
        0     0     0     1     0;
        0     0     0     0     1];

    kw = 10;

    start_pts           = [x];

    tracking            = zeros(1,N_tar);
    labels              = 1:100; %associate each filter with a start point;
    N                   = 200; %# of particles per target
    N_all               = N_tar*N;
    P                   = ones(N,1)/N;
    xp_cell = cell(1,N_tar);
    x0 = zeros(5,N_tar);
    P_cell = cell(1,N_tar);
    
     for i = 1:N_tar
         x1              = start_pts(labels(i),:);
         xp_cell{i}      = [repmat(x1,N,1) 0*ones(N,1) 0*ones(N,1) (kw)+3*randn(N,1)];
         x0(:,i)         = [x1 0 0 kw]'; %measurement of position
         P_cell{i}       = P; %particles and weights
         xav_cell{i}     = repmat((sum(repmat(P,1,5).*xp_cell{i})'),1,on(i)+3);
     end


    off = 364*ones(1,N_tar);

    
    on_thresh = 500;
    turn_on = zeros(3,1);
    for t = 2 : T
        (fprintf('Time=%d\n',t));
        clf;

        for i = 1:N_tar
            xp           = xp_cell{i};
            P            = P_cell{i};
            xav          = xav_cell{i};



                 %resampling
                A            = ones(N,1)*rand(1,N); 
                B            = cumsum(P)*ones(1,N);
                k            = sum (A > B) + 1;
               
                if(off(i) <= t )
                    continue;
                end    
                if (on(i) <= t)
                    xp = (f*repmat(xav(:,t-1),1,N))' + [sqrt(Q)*randn(N,2) 0.25*randn(N,2) 0.5*randn(N,1)]; %prediction
                else
                    continue;
                end    
                if(xav(1,t-1) > w || xav(2,t-1) > h || isnan(xav(1,t-1)) || isnan(xav(1,t-1)) || xav(1,t-1) < 0)
                off(i) = t;
                end
                  
                xp_xy        = xp(:,1:2);

                %optimize the dist2 mtx
                box_size     = 4;
                xcloudmin    = min(xp_xy(:,1))-box_size*kw;
                xcloudmax    = max(xp_xy(:,1))+box_size*kw;
                ycloudmin    = min(xp_xy(:,2))-box_size*kw;
                ycloudmax    = max(xp_xy(:,2))+3*kw;
                close_idx    = find(loc(:,1) < xcloudmax & loc(:,1) > xcloudmin & loc(:,2) < ycloudmax & loc(:,2) > ycloudmin);%find the relative indexes related to the square that we have selected
                %6864 * 1
                z_local      = zeros(size(z,1),1);
                z_local(close_idx) = z(close_idx,t); %take only that intensity values of orignal image
                y            = Phi*z_local;
                %y = 3500 *1;
                loc_close    = loc(close_idx,:);    %convert relative indexes to cartesian indexes
                dist2        = inf*ones(N,size(loc,1));
                dist2(:,close_idx) = L2_distance(xp_xy',loc_close'); %200*25344   %find the distance of each particle from above selected pixels in original image
                zp = exp(-dist2.^2./repmat(xp(:,5),1,w*h));%quasi inversion %estimeted pixel intensity 

                yp           = Phi*zp';

                temp_sum     = -.5 * sum((y*ones(1,N)-yp).^2)/R;
                    temp_sum     = temp_sum-max(temp_sum);

                P            = P(k).*exp(temp_sum)'; %weight update     %p(yk/ski)


                P            = P/sum(P);
                xav(:,t)     = sum(repmat(P,1,5).*xp); %state estimation update
                
                clf
                hold on;
                  imagesc(reshape(z(:,t),h,w));
                plot(xav(1,:),xav(2,:),'g.-')
                
                fprintf('(%d,%d)\n',uint8(xav(1,t)),uint8(xav(2,t)));
                axis ij
                pause(0.0001);
                hold off;


            xp_cell{i}   = xp;
            P_cell{i}    = P;
            xav_cell{i}  = xav;

        end

        
        
    end
    
    
    for g = 1:N_tar
            if(f1 == 1)
            xmean{1,g}(f1,:) = xav_cell{1,g}(1,:);
            ymean{1,g}(f1,:) = xav_cell{1,g}(2,:); 
            break;
            end
        if(size(xav_cell{1,g},2) == size(xmean{1,g},2))
            xmean{1,g}(f1,:) = xav_cell{1,g}(1,:);
            ymean{1,g}(f1,:) = xav_cell{1,g}(2,:);
        elseif(size(xav_cell{1,g},2) < size(xmean{1,g},2))
            xmean{1,g}(f1,:) = xmean{1,g}((f1-1),:);
            ymean{1,g}(f1,:) = ymean{1,g}((f1-1),:);
        else
             for temp = 1:f1-1
                     if(size(xmean{1,g},2) < size(xav_cell{1,g},2))
                         xmean{1,g}(temp,1:size(xav_cell{1,g},2)) = xav_cell{1,g}(1,:);
                         ymean{1,g}(temp,1:size(xav_cell{1,g},2)) = xav_cell{1,g}(2,:);
                     end
             end    
          xmean{1,g}(f1,1:size(xav_cell{1,g},2)) = xav_cell{1,g}(1,:);
          ymean{1,g}(f1,1:size(xav_cell{1,g},2)) = xav_cell{1,g}(2,:);   
        end
        
    
    end

end
save('final_CPF_f20_M3500','xmean','ymean','f1','M','kw','box_size','Q','R','N');    
for g = 1:N_tar
xmean{1,g} = sum(xmean{1,g},1)/f1;
ymean{1,g} = sum(ymean{1,g},1)/f1;
end
save('final_avg_CPF_f20_M3500','xmean','ymean','f1','M','kw','box_size','Q','R','N');    
for i = 2:364
    im = rgb2gray(mov(i).cdata);
    im = downsample(im,2);
    im = downsample(im',2)';
    hold on;
   
    imagesc(im);
    for k=1:N_tar
    if i <= size(xmean{1,k},2)    
    plot(xmean{1,k}(1,1:i),ymean{k}(1,1:i),'g-');
    end
    end
    axis ij;
    pause (1/30);
    hold off;
end
