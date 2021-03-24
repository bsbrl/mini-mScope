function [Tf,Rbar,A,imseq,A0,max_frame] = filtering(av,imax,jmax,tmax,imseq,behaviors,mask_imgpixels,roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi,behavior_index,A,fil,fscale,tstart,tend,vidtype)
%filtering(av,imseq(1).(behaviors{p}).tmax,imseq(1).(behaviors{p}).(field_name),behaviors,mask_imgpixels,roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi)
switch calculation_type
    case 'Temporal Filter1' %global correction
            wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
            steps = tmax;%total frames
           
            
            set(wt,'Units', 'normalized');
            % Change the size of the figure
            set(wt,'Position', [0.35 0.4 0.3 0.08]);

            for t = 1:tmax
                im = squeeze(imseq(:,:,t));
                Tf(1).(behaviors{behavior_index}).mask_roi(t,1) = sum(im(sub2ind(size(im),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)))/(mask_imgpixels(2).X*mask_imgpixels(2).Y);
%                 for j = background_roi + 1 + analyze_roi
%                    Tf(1).(behaviors{behavior_index}).ROI(t,j) = sum(im(sub2ind(size(im),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels)))/(roi_pixels(j).X*roi_pixels(j).Y);                 
%                 end
            end
            Rbar(1).(behaviors{behavior_index}) = sum(Tf(1).(behaviors{behavior_index}).mask_roi)/tmax;

            A = nan(imax,jmax,tmax);
            for t = 1:tmax
                im = squeeze(imseq(:,:,t));
                A(:,:,t) = (im - (av * (Tf(1).(behaviors{behavior_index}).mask_roi(t)/Rbar(1).(behaviors{behavior_index}))))./av;
                if mod(t,20)==0
                    waitbar(t/steps,wt,sprintf('Temporal filtering and removing global fluctuations for frame %1.0f/%1.0f',t,steps))
                end
                %A(:,:,t) = im - Tf(1).mask_roi(t)/Rbar;
            end

            imseq = []; %reduce memory usage by eliminating 
            close(wt)
%             toc
            A0 = [];
            max_frame = [];
            
    case 'Temporal Filter2' %no global correction, temporal pixel filter
        wt=waitbar(0,'Temporal Filtering Data');%progress bar to see how code processsteps=len;
        steps=imax;%total frames
        switch vidtype
            case 'blueframes'
                %%
%                 x = [150.0000  127.0000  128.0000  142.0000  106.0000  120.0000];
%                 y = [37.0000   77.0000  105.0000   91.0000  124.0000  175.0000];
%                 desired_orders = [2,4];
%                 passband_orders = [0.5,1,2];
%                 ticks = {'M1','RSG','HL','FL','BC','VC'};
% 
%     %                 tmp = squeeze(imseq(size(imseq,1)/2,size(imseq,2)/2,:));
%                 for j = 1:length(x)
%                     for k = 1:length(desired_orders)
%                         for i = 1:length(passband_orders)
%                             d = designfilt('bandpassiir','FilterOrder',desired_orders(k), ...
%                                 'PassbandFrequency1',0.1,'PassbandFrequency2',5, ...
%                                 'SampleRate',15,'PassbandRipple',passband_orders(i));
% 
%                             tmp = squeeze(imseq(x(j),y(j),:));
%                             mean1 = mean(tmp);
%                             tmp2 = filtfilt(d,tmp);
%                             mean2 = mean(tmp2);
%                             shift = mean1-mean2;
%                             figure;
%                             plot(tmp,'b','LineWidth',1)
%                             hold on
%                             plot(tmp2+shift,'r','LineWidth',1.5)
%                             titletxt = strcat(num2str(d.FilterOrder),{' '},'order',{' '},d.FrequencyResponse,{' '},d.DesignMethod,{' '},'filter, passband ripple of',{' '},num2str(d.PassbandRipple),',',ticks{j});
%                             title(titletxt{1})
%                             xlabel('Frame number')
%                             ylabel('Center FOV pixel intensity')
%                             box off
%                             set(gcf,'color','w');
%                             set(gca,'FontName','Arial','FontSize',14,'LineWidth',1)
%                             legend('Original Trace','Filtered Trace')
%                             saveas(gcf,strcat(folder,titletxt{1},'.jpeg'))
%                             saveas(gcf,strcat(folder,titletxt{1},'.fig'))
%                             close;
%                         end
% 
%                     end
% 
%                 end
            
%                 d = designfilt('bandpassiir','FilterOrder',4, ...
%                 'PassbandFrequency1',0.1,'PassbandFrequency2',5, ...
%                 'SampleRate',15)
%                 %%
% 
%                 for i = 1:imax
%                     for j = 1:jmax
%                         mean1 = mean(squeeze(imseq(i,j,:)));
%                         imseq(i,j,:) = filtfilt(d,squeeze(imseq(i,j,:)));
%                         mean2 = mean(squeeze(imseq(i,j,:)));
%                         shift = mean1-mean2;
%                         imseq(i,j,:) = squeeze(imseq(i,j,:)) + shift;
%                     end
%                     if mod(i,20)==0
%                         waitbar(i/steps,wt,sprintf('Temporal filtering pixels for pixel row %1.0f/%1.0f',i,steps))
%                     end
% 
%                 end
                
            case 'greenframes'
%                 d = designfilt('bandpassiir','FilterOrder',4, ...
%                 'PassbandFrequency1',0.02,'PassbandFrequency2',0.08, ...
%                 'SampleRate',15,'PassbandRipple',0.01);
                d = designfilt('lowpassiir','FilterOrder',2, ...
                'PassbandFrequency',0.15, ...
                'SampleRate',15,'PassbandRipple',0.2)
                
%                 imshow(uint8(av))
%                 hold on
%                 for j = 1:6
%                     [x(j),y(j)] = getpts();
%                     plot(x(j),y(j),'*r')  
%                 end
%%
%                 x = [150.0000  127.0000  128.0000  142.0000  106.0000  120.0000];
%                 y = [37.0000   77.0000  105.0000   91.0000  124.0000  175.0000];
%                 desired_orders = [2,4];
%                 passband_orders = [0.1,0.2,1];
%                 ticks = {'M1','RSG','HL','FL','BC','VC'};
%             
% %                 tmp = squeeze(imseq(size(imseq,1)/2,size(imseq,2)/2,:));
%                 for j = 1:length(x)
%                     for k = 1:length(desired_orders)
%                         for i = 1:length(passband_orders)
%                             d = designfilt('bandpassiir','FilterOrder',desired_orders(k), ...
%                                 'PassbandFrequency1',0.02,'PassbandFrequency2',0.08, ...
%                                 'SampleRate',15,'PassbandRipple',passband_orders(i));
% %                             d = designfilt('lowpassiir','FilterOrder',desired_orders(k), ...
% %                                 'PassbandFrequency',0.15, ...
% %                                 'SampleRate',15,'PassbandRipple',passband_orders(i));
%                             
%                             tmp = squeeze(imseq(x(j),y(j),:));
%                             mean1 = mean(tmp);
%                             tmp2 = filtfilt(d,tmp);
%                             mean2 = mean(tmp2);
%                             shift = mean1-mean2;
%                             figure;
%                             plot(tmp,'b','LineWidth',1)
%                             hold on
%                             plot(tmp2+shift,'r','LineWidth',1.5)
%                             titletxt = strcat(num2str(d.FilterOrder),{' '},'order',{' '},d.FrequencyResponse,{' '},d.DesignMethod,{' '},'filter, passband ripple of',{' '},num2str(d.PassbandRipple),',',ticks{j});
%                             title(titletxt{1})
%                             xlabel('Frame number')
%                             ylabel('Center FOV pixel intensity')
%                             box off
%                             set(gcf,'color','w');
%                             set(gca,'FontName','Arial','FontSize',14,'LineWidth',1)
%                             legend('Original Trace','Filtered Trace')
%                             saveas(gcf,strcat(folder,titletxt{1},'.jpeg'))
%                             saveas(gcf,strcat(folder,titletxt{1},'.fig'))
%                             close;
%                         end
%                         
%                     end
%                     
%                 end

                %%
                for i = 1:imax
                    for j = 1:jmax
                        mean1 = mean(squeeze(imseq(i,j,:)));
                        imseq(i,j,:) = filtfilt(d,squeeze(imseq(i,j,:)));
                        mean2 = mean(squeeze(imseq(i,j,:)));
                        shift = mean1-mean2;
                        imseq(i,j,:) = squeeze(imseq(i,j,:)) + shift;
                    end
                    if mod(i,20)==0
                        waitbar(i/steps,wt,sprintf('Temporal filtering pixels for pixel row %1.0f/%1.0f',i,steps))
                    end

                end
        end
        
        close(wt)
        av = mean(imseq(:,:,tstart:tend),3);
        zero_vals = find(av == 0);
        av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)
        
        %% Detrend data and calculate DFF
%         wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
%         steps = imax;%total frames
%         set(findall(wt),'Units', 'normalized');
%         % Change the size of the figure
%         set(wt,'Position', [0.35 0.4 0.3 0.08]);
%         A = nan(imax,jmax,tmax);
%         for i = 1:imax
%             for j = 1:jmax
%                 temp = av(i,j);
%                 A(i,j,:) = (squeeze(imseq(i,j,:))-temp)/temp;
%             end
% 
%             if mod(i,20)==0
%                 waitbar(i/steps,wt,sprintf('Detrending pixels for pixel row %1.0f/%1.0f',i,steps))
%             end
%         end 
%         close(wt)
        A = imseq;
        imseq = [];
        
        toc
        A0 = [];
        max_frame = [];
            
    case 'Spatial Filter' %spatially filter DFF trace
        A0 = nan(imax,jmax,tmax);                      % initializes matrix; same size as A

        if fil > 0

            weight = nan(1+2*fil,1+2*fil);             % initializes weighted filtering matrix
            [s,q] = deal(1+fil,1+fil);                 % specifying center of weight

            for m = 1:size(weight,1)                   % looping through indices in wi-wj
                for n = 1:size(weight,2)
                    d = pdist([s q; m n]);             % distance from (i,j) to each point in wi-wj
                    weight(m,n) = 1/(1+(d/fscale))^2;  % weight of pixel in wi-wj
                end
            end

            wi = nan(imax,2);
            wp = nan(imax,2);
            for i = 1:imax
                wi(i,:) = [i-min(fil,i-1) i+min(fil,imax-i)];    % vertical pixel range for filtering
                wp(i,:) = [s-min(fil,i-1) s+min(fil,imax-i)];    % vertical submatrix of weight
            end

            wj = nan(jmax,2);
            wq = nan(jmax,2);
            for j = 1:jmax
                wj(j,:) = [j-min(fil,j-1) j+min(fil,jmax-j)];    % horizontal pixel range for filtering
                wq(j,:) = [q-min(fil,j-1) q+min(fil,jmax-j)];    % horizontal submatrix of weight
            end


            wt=waitbar(0,'Spatial Filtering Data');%progress bar to see how code processsteps=len;
            steps=tmax;%total frames

            for t = 1:tmax
                im = squeeze(A(:,:,t));
                filim = nan(imax,jmax);                                
                for i = 1:imax
                    if or(i-fil<=0,i+fil>imax)
                        len = wp(i,1):wp(i,2);
                        for j = 1:jmax
                            temporary = im(wi(i,1):wi(i,2),wj(j,1):wj(j,2));
                            if or(j-fil<=0,j+fil>jmax)
                                wid = wq(j,1):wq(j,2);
                                total = weight(len,wid).*temporary;
                            else
                                total = weight(len,:).*temporary;
                            end
                            filim(i,j) = sum(sum(total))/sum(sum(weight));
                        end
                    else
                        for j = 1:jmax
                            temporary = im(wi(i,1):wi(i,2),wj(j,1):wj(j,2));
                            if or(j-fil<=0,j+fil>jmax)
                                wid = wq(j,1):wq(j,2);
                                total = weight(:,wid).*temporary;
                            else
                                total = weight.*temporary;
                            end
                            filim(i,j) = sum(sum(total))/sum(sum(weight));
                        end
                    end
                end
                A0(:,:,t) = filim;
                max_frame(t) = max(max(filim));

                if mod(t,20)==0
                    waitbar(t/steps,wt,sprintf('Spatial filtering data for frame %1.0f/%1.0f',t,steps))
                end
            end
            close(wt)

        else
            A0 = A;
        end
%         toc
end

end
            