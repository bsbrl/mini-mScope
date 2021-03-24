% Biosensing and Biorobotics Laboratory
% Samantha Linn, Daniel Surinach
% February 2020
% dF/F heat mapping in spontaneous behavior

%% Variable initialization

clear;
clc;
close all;


tstart = 5; %start frame
tend = 'full'; %end frame, 'full' = last frame
%frac = 1;
fil = 5;      % spatial filtering pixel radius (integer), LP used 2
fscale = 2;   % filtering weight (larger number = greater weight)
bin = 0.1;     % binning fraction (0 to 1, 1 = no binning)
folder = 'K:\Barnes Maze\Barnes Maze Mouse Trial Data\GCaMP\9-17_acquisition_t1\calcium\no moco\';
files = dir(fullfile(folder,'*.avi'));
vid = length(files);
tee = nan(vid,1);

stim_time = 5;%stimulus experiment details
stim_duration = 1;

scoring_files = 'n';
if strcmp(scoring_files,'y')==1
    xlsfiles = dir(fullfile(folder,'*.xlsx'));
    behaviors = {'moving_frames','still_frames','touch_frames','notouch_frames','grooming_frames','rearing_frames'};%total list of behaviors
    desired_behaviors = [1,2,5,6]; %desired behaviors from above
    
    scoring = struct('time',{});%,'frames',{},'moving_frames',{},'still_frames',{},'touch_frames',{});
    scoring_name = dir(fullfile(folder,'*.xlsx'));
    scoring(1).time = xlsread(strcat(folder,scoring_name.name));
    for j = 1:length(desired_behaviors)
        scoring(1).(behaviors{desired_behaviors(j)}).time = scoring(1).time(:,desired_behaviors(j));
        scoring(1).(behaviors{desired_behaviors(j)}).fullframes = [] ; %initialize frame


    end
    
end

paths = {'General Info','DFF Excel','Pixel Correlations','DFF Plots','DFF Movie'};
path0 = 'General Info';
path1 = 'DFF Excel';
path2 = 'Pixel Correlations';
path3 = 'DFF Plots';
path4 = 'DFF Movie';

folder_objects = dir(folder);        % all items in folder
foldercount = 0;

for j = 1:length(folder_objects) %check object names to see if previous analysis exists
   if strcmp(folder_objects(j).name,path1) == 1
        foldercount = foldercount + 1;
   elseif strcmp(folder_objects(j).name,path2) == 1
       foldercount = foldercount + 1;
   elseif strcmp(folder_objects(j).name,path3) == 1
       foldercount = foldercount+1;
   else
        overwrite_prompt{1} = 'y';
   end
   
   if strcmp(folder_objects(j).name,'brain_mask.jpeg') == 1 %if mask of brain exists
       mask_exists = 'y';
   else
       mask_exists = 'n';
   end
   
end

if foldercount == 3 %3 folders of prev analysis exist
    prompt = {'Do you wish to overwrite the current analysis in this folder? (y/n)?'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'New Analysis Prompt';
    overwrite_prompt = inputdlg(prompt,dlgtitle);
end

if strcmp(overwrite_prompt{1},'y') == 1 %if new analysis desired
    for p = 1:size(paths,2)
       mkdir(strcat(folder,paths{p}));
    end
    %mkdir(strcat(folder,path1));
    %mkdir(strcat(folder,path2));
    %mkdir(strcat(folder,path3));

    for ex = 1:vid
        tic
        fprintf('\nAnalyzing video %1.0f/%1.0f with name below\n',ex,vid)
        files(ex).name
        vidtype = strsplit(files(ex).name,{'.avi',' '});
        vidtype = vidtype{end-1};
        %% .avi --> 3D grayscale matrix

        v = VideoReader(strcat(folder,files(ex).name));            % reads video
        if strcmp(scoring_files,'y') == 1
            fps = v.FrameRate;
            set_fpsvec = ones(fps,1);
            for j = 1:length(desired_behaviors)
                for i = 1:size(scoring(1).time,1)
                    temp = scoring(1).(behaviors{desired_behaviors(j)}).time(i) * set_fpsvec;
                    scoring(1).(behaviors{desired_behaviors(j)}).fullframes = [scoring(1).(behaviors{desired_behaviors(j)}).fullframes;temp];%vector of elements in time list concatenated into frames 
                end
                scoring(1).(behaviors{desired_behaviors(j)}).ind = find(scoring(1).(behaviors{desired_behaviors(j)}).fullframes >= 3); %desired tracked behavior
                
            end
        end
        
        
        k = 0;
        wt=waitbar(0,'Extracting Frames');%progress bar to see how code processsteps=len;
        

        while hasFrame(v)                       % loops through video frames
            im = readFrame(v);                  % reads frame
            if k == 0 %first frame
                im_size = imresize(im,bin,'bilinear');
                imax = size(im_size,1);% frame height in units pixels
                jmax = size(im_size,2);% frame width in units pixels
                tmax = round(v.Duration*v.FrameRate);
                imseq = nan(imax,jmax,tmax);  % number of frames in video
                steps=tmax;%total frames
            end
                
            
            if strcmp(mask_exists,'n') == 1
               wt2 = waitbar(0,'Draw a mask over the brain region (exclude non GCaMP areas)');

               mask_imgvid = VideoWriter(strcat(folder,paths{1},'\mask_img.avi'));
               mask_imgvid.FrameRate = v.FrameRate;
               open(mask_imgvid);

               for j = 1:50
                   writeVideo(mask_imgvid,im);
               end
               close(mask_imgvid)

               rawvideo = VideoReader(strcat(folder,paths{1},'\mask_img.avi'));
               frame = 1;

               background_roi =0;
               analyze_roi = 1;
               seedpixel_roi = 'n';
               exclude_roi = 'y';

               imgname = strcat(folder,paths{1},'\mask_roi.jpeg');
               stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
               casenum = 0;
               data = 0;
               blueorgreenonly = {'blue','b'};

               frame_color = {};
               [frame_color] = findframe(rawvideo,frame,stopframe,frame_color,0,0,'',blueorgreenonly); %extract which frames are green and blue

               if isempty(frame_color.black)==1
                   frame_color.black = 0;
               end

                %rawvideo = VideoReader(strcat(folder,files(ex).name));
               rawvideo = VideoReader(strcat(folder,paths{1},'\mask_img.avi'));%VideoReader(strcat(folder,files(ex).name));
               [mask_imgpixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,data,frame_color); %set user-defined rois
               close(wt2)

               mask_exists = 'y';
               close all
               
               if size(im,3) > 1
                   imbw = rgb2gray(im);
               else
                   imbw = im;
               end
            
               imbw = imresize(imbw,bin);
               masked_img = uint8(255*ones(size(imbw)));

               masked_img(sub2ind(size(masked_img),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = imbw(sub2ind(size(imbw),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels));
               masked_img(sub2ind(size(masked_img),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels)) = 255;
               save(strcat(folder,paths{1},'\mask_roi.mat'),'mask_imgpixels');
               
               prompt = {'Is this an anesthetized mouse (y/n)?'};%let user choose if blue or green frame (manual segmentation)
               dlgtitle = 'Filter Decider';
               filter_prompt = inputdlg(prompt,dlgtitle);
               
               filter_prompt = filter_prompt{1};
            end
            
            
            k = k+1;
            if size(im,3)>1
                imseq(:,:,k) = imresize(double(rgb2gray(im)),bin);% converts every frame to B/W 3D matrix
            else
                imseq(:,:,k) = imresize(double(im),bin);
            end
            
            if k == 1
                if size(im,3)>1
                    imshow(imresize(rgb2gray(im),bin))
                else
                    
                    imshow(imresize(im,bin))
                end
                imwrite(imresize(im,bin),strcat(folder,path0,'\ bwim2.jpeg')) %store first image for overlay
                %saveas(gcf,strcat(folder,name,' bwim.jpeg'))
            end

            if mod(k,20)==0
                waitbar(k/steps,wt,sprintf('Extracting frame data for frame %1.0f/%1.0f',k,steps))%progres bar
            end
        end
        close(wt)

%         imseq = imresize(imseq,bin);            % binning B/W 3D matrix
    %     imseq = imseq(:,:,round(tmin1*v.FrameRate):round(tmin2*v.FrameRate));
        imax = size(imseq,1);                   % dimension height
        jmax = size(imseq,2);                   % dimension width
        tmax = size(imseq,3);                   % dimension time
        
        %% frequency wavelet transforms
%         [wavelet,f] = cwt(squeeze(imseq(round(imax/3),round(jmax/3),:)),v.FrameRate);

        %% svd noise decomposition
        
        imseq = reshape(imseq,[imax*jmax,tmax]);
        pause(1)
        [U,S,V] = svd(imseq);
        
        %% zscore avg map
        close all;
        if strcmp(scoring_files,'y') == 1
            tmp = reshape(imseq,[imax*jmax,tmax]);
            zscore_mat = zeros(size(tmp));
            for j = 1:size(tmp,1) %iterate through pixels and calc zscore
                zscore_mat(j,:) = zscore(tmp(j,:));
            end
            zscore_mat = reshape(zscore_mat,[imax,jmax,tmax]);
            for j = 1:length(desired_behaviors)
                figure(j)
                extra_frames = logical(scoring(1).(behaviors{desired_behaviors(j)}).ind <= size(zscore_mat,3));
                if nnz(extra_frames) ~= length(extra_frames) %non zero elements found i.e. extra frames
                    scoring(1).(behaviors{desired_behaviors(j)}).ind = scoring(1).(behaviors{desired_behaviors(j)}).ind(extra_frames);
                end
                    
                zscore_struct.(behaviors{desired_behaviors(j)}) = mean(zscore_mat(:,:,scoring(1).(behaviors{desired_behaviors(j)}).ind),3);
                colormap('jet');
                pic = imagesc(zscore_struct.(behaviors{desired_behaviors(j)}));
                caxis([min(min(zscore_struct.(behaviors{desired_behaviors(j)}))),max(max(zscore_struct.(behaviors{desired_behaviors(j)})))])
                colorbar
                set(gca,'xtick',[])
                set(gca,'ytick',[])
                title(behaviors{desired_behaviors(j)})
            end
        end
        
        
        %% Denoising test background ratio division
        if filter_prompt == 'y' %anesthetized animal, different filters
            for i = 1:imax
                for j = 1:jmax
                    imseq(i,j,:) = smooth(squeeze(imseq(i,j,:)),5);
                end
            end
        else %else awake animal, more filters possible
            close all;
    %         noise_pix = squeeze(imseq(1,1,:));
            noise_pix = zeros(tmax,1);
            imshow(imbw)
            [xbkg,ybkg] = getpts;
            xbkg = round(xbkg);
            ybkg = round(ybkg);
            close;
    %         figure(1);
    %         legend_name = {};
            for j = 1:length(xbkg)
    %             legend_name{j} = strcat('noise_pix',num2str(j));
                if j == 1
                    base_noisescale = squeeze(imseq(ybkg(j),xbkg(j),:));
    %                 plot(base_noisescale);
    %                 hold on
                    noise_pix = noise_pix + base_noisescale;

                else
                    clear ratio
                    curr_noise = squeeze(imseq(ybkg(j),xbkg(j),:));
                    ratio = base_noisescale./curr_noise;

                    scaled_noise = curr_noise * mean(ratio); %scale noise 
    %                 plot(scaled_noise);
    %                 hold on
                    noise_pix = noise_pix + scaled_noise;
                end
            end
            noise_pix = noise_pix/j;
            
            %% denoise scaled zscore
%             for j = 1:length(xbkg)
%                 curr_noise = zscore(squeeze(imseq(ybkg(j),xbkg(j),:)));
%                 plot(curr_noise)
%                 hold on
%                 noise_pix = noise_pix + curr_noise;
%             end
%             noise_pix = noise_pix/length(xbkg);
%             plot(noise_pix,'r')
%             
%             %%
%             d = designfilt('bandpassiir','FilterOrder',4, ...
%                             'PassbandFrequency1',0.1,'PassbandFrequency2',4, ...
%                             'SampleRate',v.FrameRate)
%             time = 1:length(noise_pix);
%             
%             for i = 1:imax
%                 for j = 1:jmax
%                     sig_pix = zscore(squeeze(imseq(i,j,:)));
%                     denoise_sig = (sig_pix - noise_pix)+1;
%                     
%                     mean1 = mean(denoise_sig);
%                     freq_sig = filtfilt(d,denoise_sig);
%                     mean2 = mean(freq_sig);
%                     shift = mean1-mean2;
%                     freq_sig = freq_sig + shift;
% 
%     %                 plot(time,freq_sig)
% 
%                     time_sig = smooth(freq_sig,5);
%                     
%                     imseq(i,j,:) = time_sig;
%                 end
%             end
            
            %%
    %         plot(noise_pix);
    %         legend_name{end+1} = 'mean_noise';
    %         legend(legend_name)

            d = designfilt('bandpassiir','FilterOrder',4, ...
                            'PassbandFrequency1',0.1,'PassbandFrequency2',5, ...
                            'SampleRate',v.FrameRate)
            time = 1:length(noise_pix);
            for i = 1:imax
                for j = 1:jmax
                    sig_pix = squeeze(imseq(i,j,:));

    %                 figure;
    %                 plot(time,noise_pix,'k',time,sig_pix,'b')

                    count = 1;
                    clear ratio

                    for k = 1:length(noise_pix)
                        ratio(count,1) = sig_pix(k)/noise_pix(k);
                        count = count+1;
                    end

    %                 figure;
                    scaled_noise = noise_pix*mean(ratio);
    %                 plot(time,scaled_noise,'k',time,sig_pix,'b')

                    denoise_sig = sig_pix ./ scaled_noise;
%                     denoise_sig = (sig_pix - scaled_noise)+mean(sig_pix); %cancel noise, mean shift
    %                 figure;
    %                 plot(time,denoise_sig)
    %                 hold on



                    mean1 = mean(denoise_sig);
                    freq_sig = filtfilt(d,denoise_sig);
                    mean2 = mean(freq_sig);
                    shift = mean1-mean2;
                    freq_sig = freq_sig + shift;

    %                 plot(time,freq_sig)

                    time_sig = smooth(freq_sig,2);
                    imseq(i,j,:) = time_sig;
    %                 plot(time,time_sig)

                    dff_sig = (time_sig - mean(time_sig))/mean(time_sig);
    %                 figure
    %                 plot(time,dff_sig)
                end
            end
        end
        %% smoothen over time
%         for i = 1:imax
%             for j = 1:jmax
%                 imseq(i,j,:) = smooth(squeeze(imseq(i,j,:))./noise_pix,5);
%             end
%         end
        %% Denoising test Jill minifast
%         v = VideoReader(strcat(folder,files(ex).name));
%         im = read(v,1);
%         fused_imgvid = VideoWriter(strcat(folder,paths{1},'\dff_fusedimg.avi'));
%         fused_imgvid.FrameRate = v.FrameRate;
%         open(fused_imgvid );
%         for j = 1:50
%             writeVideo(fused_imgvid,im);
%         end
%         close(fused_imgvid)
% 
%         %rawvideo = VideoReader(strcat(folder,files(ex).name));
%         rawvideo = VideoReader(strcat(folder,paths{1},'\dff_fusedimg.avi'));
%         frame = 1;
% 
%         prompt = {'Background ROIs'};%let user choose if blue or green frame (manual segmentation)
%         dlgtitle = 'ROI Selection For Analysis';
%         roi_input = inputdlg(prompt,dlgtitle,[1 60]);
%         background_roi = str2double(roi_input{1});
%         analyze_roi = 1;
%         seedpixel_roi = 'n';
%         exclude_roi = 'n';
% 
%         imgname = strcat(folder,paths{1},'\roi_selection.jpeg');
%         stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
%         casenum = 0;
%         data = 0;
%         blueorgreenonly = {'blue','b'};
% 
%         frame_color = {};
%         [frame_color] = findframe(rawvideo,frame,stopframe,frame_color,0,0,'',blueorgreenonly); %extract which frames are green and blue
%         if isempty(frame_color(1).black)==1
%             frame_color(1).black = 0;
%         end
% 
%         %rawvideo = VideoReader(strcat(folder,files(ex).name));
%         rawvideo = VideoReader(strcat(folder,paths{1},'\dff_fusedimg.avi'));%VideoReader(strcat(folder,files(ex).name));
%         [roi_pixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,0,frame_color); %set user-defined rois
%         close all
%         [A,F] = background_noise_correction(roi_pixels,background_roi,imseq);
        
        %% Raw data --> detrended
        A = nan(imax,jmax,tmax);
       % av = mean(imseq,3);
        %av = mean(imseq(:,:,1:round(frac*tmax)),3);
%         wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
%         steps=imax;%total frames
        
        switch vidtype
            case 'bluevideo'
                d = designfilt('bandpassiir','FilterOrder',4, ...
                'PassbandFrequency1',0.1,'PassbandFrequency2',4, ...
                'SampleRate',v.FrameRate);
            case 'greenvideo'
                d = designfilt('bandpassiir','FilterOrder',4, ...
                'PassbandFrequency1',0.02,'PassbandFrequency2',0.08, ...
                'SampleRate',v.FrameRate);
        end

%         close;
%         tmp = squeeze(imseq(1,1,:));
%         mean1 = mean(tmp)
%         tmp2 = filtfilt(d,tmp);
%         mean2 = mean(tmp2)
%         plot(tmp,'g');
%         hold on
%         shift = mean1 - mean2;
%         plot(tmp2 + shift,'b');
        %% freq filter imseq
%         for i = 1:imax
%             for j = 1:jmax
%                 mean1 = mean(squeeze(imseq(i,j,:)));
%                 imseq(i,j,:) = filtfilt(d,squeeze(imseq(i,j,:)));
%                 mean2 = mean(squeeze(imseq(i,j,:)));
%                 shift = mean1-mean2;
%                 imseq(i,j,:) = squeeze(imseq(i,j,:)) + shift;
%             end
%             if mod(i,20)==0
%                 waitbar(i/steps,wt,sprintf('Detrending pixels for pixel row %1.0f/%1.0f',i,steps))
%             end
%             
%         end
%         close(wt)
        if strcmp(tend,'full') == 1
            tend = tmax;
        end
        av = mean(imseq(:,:,tstart:tend),3);
%         av = mean(mean(av));
        %%
        imseq_mask = zeros(imax,jmax,tmax);
        for t = 1:size(imseq,3) %introduce NaN to border outside mask for video rendering
            tempdata = squeeze(imseq(:,:,t));
            tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = NaN;
            tempdata(sub2ind(size(tempdata),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels)) = NaN;
            imseq_mask(:,:,t) = tempdata;

        end
        
        %%
        zero_vals = find(av == 0);
        av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)
        %% traditional df/f
        if filter_prompt == 'y'
            wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
            steps=imax;%total frames
            for i = 1:imax
                for j = 1:jmax
                    temp = av(i,j);
    %                 temp = av;
                    A(i,j,:) = (squeeze(imseq(i,j,:))-temp)/temp;
                end

                if mod(i,20)==0
                    waitbar(i/steps,wt,sprintf('Detrending pixels for pixel row %1.0f/%1.0f',i,steps))
                end
            end 
    %         clear imseq
            close(wt)
            toc
        %% global illum correction
        else
            wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
            steps=tmax;%total frames
            Tf = [];%fluorescent trace over time
    %         av = mean(A(:,:,tstart:tend),3);
            excludemask = zeros(size(av));
            excludemask(sub2ind(size(excludemask),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = 1;
            excludemask(sub2ind(size(excludemask),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels)) = 0;
            [yexclude_pixels,xexclude_pixels] = find(excludemask == 1);
            Yexclude = size(unique(yexclude_pixels),1);
            Xexclude = size(unique(xexclude_pixels),1);
            for t = 1:tmax
    %             im = squeeze(A(:,:,t));
                im = squeeze(imseq(:,:,t));
    %             Tf(t,1) = sum(im(sub2ind(size(im),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)))/(mask_imgpixels(2).X*mask_imgpixels(2).Y);
                Tf(t,1) = sum(im(sub2ind(size(im),yexclude_pixels,xexclude_pixels)))/(Yexclude*Xexclude);

            end
            Rbar = sum(Tf(:,1))/tmax;

            for t = 1:tmax
    %             im = squeeze(A(:,:,t));
                im = squeeze(imseq(:,:,t));
                A(:,:,t) = (im - (av * (Tf(t,1)/Rbar)))./av;
                if mod(t,20)==0
                    waitbar(t/steps,wt,sprintf('Removing global artifacts for frame %1.0f/%1.0f',t,steps))
                end
                %A(:,:,t) = im - Tf(1).mask_roi(t)/Rbar;
            end

    %         imseq = []; %reduce memory usage by eliminating 
            close(wt)
            toc
        end
        
        %% filtered plots
%         for i =  1:length(xbw)
%             figure(i)
%             plot(time,squeeze(A0_nothing(ybw(i),xbw(i),:)),time,squeeze(A0_globfilt(ybw(i),xbw(i),:)),time,squeeze(A0_bfilt(ybw(i),xbw(i),:)),time,squeeze(A0_bfilt_globfilt(ybw(i),xbw(i),:)),time,squeeze(A0_bfilt_4pts(ybw(i),xbw(i),:)),time,squeeze(A0_bfilt_globfilt_4pts(ybw(i),xbw(i),:)))
%             legend('no filt','global filt','backround filt','global n background filt','background filt 4 pts','global n back filt 4 pts')
%         end
        

        %% Apply weighted spatial filter to video

        A0 = nan(imax,jmax,tmax);                      % initializes matrix; same size as A

        if fil > 0

            weight = nan(1+2*fil,1+2*fil);             % initializes weighted filtering matrix
            [p,q] = deal(1+fil,1+fil);                 % specifying center of weight

            for m = 1:size(weight,1)                   % looping through indices in wi-wj
                for n = 1:size(weight,2)
                    d = pdist([p q; m n]);             % distance from (i,j) to each point in wi-wj
                    weight(m,n) = 1/(1+(d/fscale))^2;  % weight of pixel in wi-wj
                end
            end

            wi = nan(imax,2);
            wp = nan(imax,2);
            for i = 1:imax
                wi(i,:) = [i-min(fil,i-1) i+min(fil,imax-i)];    % vertical pixel range for filtering
                wp(i,:) = [p-min(fil,i-1) p+min(fil,imax-i)];    % vertical submatrix of weight
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

        tee(ex) = tmax;
        %% tiff stack
%         mkdir(strcat(folder,paths{1},'\tiff_stack'))
%         %tmp = rescale(A0,0,1);
%         tmp = A0;
%         imwrite(squeeze(tmp(:,:,1)),strcat(folder,paths{1},'\tiff_stack\','tst.tif'))
%         for t = 2:size(tmp,3)
%             imwrite(squeeze(tmp(:,:,t)),strcat(folder,paths{1},'\tiff_stack\','tst.tif'),'WriteMode','append');
%         end
%         name = strsplit(v.Name,'.avi');
%         name = name{1};
%         tiff_file = Tiff(strcat(folder,paths{1},'\tiff_stack\',name,'.tif'),'w');
%         setTag(tiff_file,'Compression',Tiff.Compression.None);
%         setTag(tiff_file,'ImageLength',imax);
%         setTag(tiff_file,'ImageWidth',jmax);
%         write(tiff_file,A0);
%         close(tiff_file)
        %%
        clear A stack
        
        %% dff avg map
        if strcmp(scoring_files,'y') == 1
            for j = 1:length(desired_behaviors)
                figure(j)
                extra_frames = logical(scoring(1).(behaviors{desired_behaviors(j)}).ind <= size(A0,3));
                if nnz(extra_frames) ~= length(extra_frames) %non zero elements found i.e. extra frames
                    scoring(1).(behaviors{desired_behaviors(j)}).ind = scoring(1).(behaviors{desired_behaviors(j)}).ind(extra_frames);
                end
                    
                dff_avg.(behaviors{desired_behaviors(j)}) = mean(A0(:,:,scoring(1).(behaviors{desired_behaviors(j)}).ind),3);
                colormap('jet');
                pic = imagesc(dff_avg.(behaviors{desired_behaviors(j)}));
                caxis([min(min(dff_avg.(behaviors{desired_behaviors(j)}))),max(max(dff_avg.(behaviors{desired_behaviors(j)})))])
                colorbar
                set(gca,'xtick',[])
                set(gca,'ytick',[])
                title(behaviors{desired_behaviors(j)})
            end
        end
            
            
        %% Calculate dF/F range with respect to time

        [max_val,max_ind] = max(max_frame(1:tee(ex))); %find time index for max in all frames
        figure;
        colormap('jet');
        pic = imagesc(squeeze(A0(:,:,max_ind)));
        caxis([-0.01,0.02])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(folder,path4,'\ dffmax_stillimage.jpeg'))
        saveas(pic,strcat(folder,path4,'\ dffmax_stillimage.fig'))
        pause(1);
        close Figure 1
        
        [max_scale,max_ind] = sort(squeeze(max(max(A0)))); %get max frame in time series for an estimated DFF scale bar
        [min_scale,min_ind] = sort(squeeze(min(min(A0)))); %get min frame in time series for estimated DFF scale bar for video writing

        %% Output movie
        %%
        figure;
        colorscaling_input{1} = 'n';
        %a = mean(min_scale); %take mean of entire time series and set as min value for estimated colorbar
        %b = mean(max_scale);
        a = -0.02;
        b = 0.02; %15%DFF assumption
        %%
        idealframe = max_ind(find(max_scale > b,1)); %find first instance where a frame is greater than set average
        if isempty(idealframe) == 1
            [~,idealframe] = max(max_scale);
        end
%         idealframe = 3000;
        clear A0_smooth;
%         A0_smooth =A0;
        A0_smooth = zeros(size(A0));
        for i = 1:imax
            for j = 1:jmax
                A0_smooth(i,j,:) = smooth(A0(i,j,:),3);
            end
        end 
%         A0_smooth = A0;
        
        %A0_smooth = reshape(smooth(A0,5),size(A0)); %smoothen data for video writing
        name = strsplit(v.name,'.avi');
        name = strcat('\',name{1});
        count = 1;

        while strcmp(colorscaling_input{1},'n') == 1
            dffv = VideoWriter(strcat(folder,path4,name,' dff.avi'));
            dffv.FrameRate = v.FrameRate;
            open(dffv);

            time = linspace(0,v.Duration,min(tee));
            

            for t = 1:size(A0_smooth,3) %introduce NaN to border outside mask for video rendering
                tempdata = squeeze(A0_smooth(:,:,t));
                tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = NaN;
                tempdata(sub2ind(size(tempdata),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels)) = NaN;
                A0_smooth(:,:,t) = tempdata;
                
                imagesc(squeeze(A0_smooth(:,:,t)));
                axis off;
                colormap('jet');
                caxis([a b])
                colorbar
                frame = getframe(gcf);
                writeVideo(dffv,frame);
            end
            close(dffv)
            close all;
            figure;
            colormap('jet');
            pic = imagesc(A0_smooth(:,:,idealframe)); %show what current colorbar is set to in a test frame
            caxis([a b])
            colorbar
            set(gca,'xtick',[])
            set(gca,'ytick',[])


            prompt = {'Is output image sufficiently colorized (y/n)?'};
            dlgtitle = 'Color Image Transparency';
            colorscaling_input = inputdlg(prompt,dlgtitle);

            if strcmp(colorscaling_input,'n')==1
               colormapeditor

               f = figure('Renderer','painters','Position',[500,500,100,100]);
               h = uicontrol('String','Done',...
                'Position',[10 10 50 50],...
                'Callback', 'set(gcbf, ''Name'', datestr(now))');
               waitfor(f, 'Name');
               close(f)
               saveas(pic,strcat(folder,path4,'\dffAvgMax_stillframe.jpeg'))
               saveas(pic,strcat(folder,path4,'\dffAvgMax_stillframe.fig'))
               close all

               openfig(strcat(folder,path4,'\dffAvgMax_stillframe.fig'));
               C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
               jet_colors = C.Colormap;% get colorbar range values
               lim = C.Limits;%get limits of colorbar

               a = lim(1);
               b = lim(2);
               
               close(figure(1))
               count = count+1; %ensure video writing occurs once only
               
%                for t = 1:size(A0_smooth,3)
%                     imagesc(squeeze(A0_smooth(:,:,t)));
%                     axis off;
%                     colormap('jet');
%                     caxis([a b])
%                     colorbar
%                     frame = getframe(gcf);
% 
%                     if t == 1
%                        C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
%                        jet_colors = C.Colormap;% get colorbar range values
%                        lim = C.Limits;%get limits of colorbar
%                     end
% 
%                     writeVideo(dffv,frame);
%                 end
%                 close(dffv)
            end
            close(figure(1))

%             if strcmp(colorscaling_input,'y')==1 && count == 1
%                     for t = 1:size(A0_smooth,3)
%                         imagesc(squeeze(A0_smooth(:,:,t)));
%                         axis off;
%                         colormap('jet');
%                         caxis([a b])
%                         colorbar
%                         frame = getframe(gcf);
% 
%                         if t == 1
%                            C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
%                            jet_colors = C.Colormap;% get colorbar range values
%                            lim = C.Limits;%get limits of colorbar
%                         end
% 
%                         writeVideo(dffv,frame);
%                     end
%                     close(dffv)
%             end

        end
        close(figure(1))
        close(dffv)


        %% output movie actual size

        z2 = VideoWriter(strcat(folder,path4,name,' dff_actualsize.avi'));
        z2.FrameRate = v.FrameRate;
        open(z2);
        bw_brain_img = imread(strcat(folder,path0,'\ bwim2.jpeg'));%read image to get image size (imax*jmax)
        del = 0; %color addition
        v = VideoReader(strcat(folder,files(ex).name));
        jet_colors = jet;  
        for t = 1:size(A0,3)

            currentframe = squeeze(A0_smooth(:,:,t));%squeeze to 2d
            [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

            %single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
            single_frame = imresize(rgb2gray(read(v,t)),bin);
            [colorimage] = colorimage_gen(single_frame,newcolormap);

            for k = 1:3 %create video with masked image applied, if desired
               colorimage(sub2ind(size(colorimage),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels,k*ones(size(mask_imgpixels(2).ynot_pixels,1),1))) = 1;
               colorimage(sub2ind(size(colorimage),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels,k*ones(size(mask_imgpixels(1).excluderoi.y_pixels,1),1))) = 1;
            end

            writeVideo(z2,colorimage);

            if t == idealframe
                imwrite(colorimage,strcat(folder,path4,name,'_dffAvgmax_stillframe_correctdim_fullcolor.jpeg'))
            end

        end
        close(z2)
        toc
        close all
        
    end
end
close all

%%
v = VideoReader(strcat(folder,files(1).name));
im = rgb2gray(read(v,1));
fused_imgvid = VideoWriter(strcat(folder,paths{1},'\dff_fusedimg.avi'));
fused_imgvid.FrameRate = v.FrameRate;
open(fused_imgvid);
for j = 1:50
    writeVideo(fused_imgvid,im);
end
close(fused_imgvid)

%rawvideo = VideoReader(strcat(folder,files(ex).name));
rawvideo = VideoReader(strcat(folder,paths{1},'\dff_fusedimg.avi'));
frame = 1;

prompt = {'Analyze ROIs (at least 1)','Draw seed pixels within ROIs? (y/n)','Exclude any ROIs from Analysis? (y/n)'};%let user choose if blue or green frame (manual segmentation)
dlgtitle = 'ROI Selection For Analysis';
roi_input = inputdlg(prompt,dlgtitle,[1 60]);
background_roi = 0;
analyze_roi = str2double(roi_input{1});
seedpixel_roi = roi_input{2};
exclude_roi = roi_input{3};

imgname = strcat(folder,paths{1},'\roi_selection.jpeg');
stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
casenum = 0;
data = 0;
blueorgreenonly = {'blue','b'};

frame_color = {};
[frame_color] = findframe(rawvideo,frame,stopframe,frame_color,0,0,'',blueorgreenonly); %extract which frames are green and blue
if isempty(frame_color(1).black)==1
    frame_color(1).black = 0;
end

%rawvideo = VideoReader(strcat(folder,files(ex).name));
rawvideo = VideoReader(strcat(folder,paths{1},'\dff_fusedimg.avi'));%VideoReader(strcat(folder,files(ex).name));
[roi_pixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,0,frame_color); %set user-defined rois
close all;
%%
mean_intensity = struct('roi',{});
for t = 1:size(A0,3)
    currentframe = squeeze(A0_smooth(:,:,t));
    for j = 1:(background_roi+1+analyze_roi)
        intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
        mean_intensity(1).roi(j,t) = mean(intensity);%last element in struct contains average mean data
        if strcmp(seedpixel_roi,'y') == 1
           for p = 1:length(roi_pixels(j).yseed) %iterate through chosen seed pixels
              intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).xseed(p),roi_pixels(j).yseed(p)));
              mean_intensity(1).seedpixel(j).trace(p,t) = mean(intensity); %p is seed pixel, j is roi, t is frame
           end

        end
    end

end
close all;
figure_count = 1;
clear running_maxval running_minval
count = 1;
for j = background_roi+2:size(mean_intensity(1).roi,1) %iterate through rois (first few are background ROIs)
    %first j index is average dff trace
    figure(j)
    hold on
    %plot_color = [1 0 0]; %red

    [val,~] = max(mean_intensity(1).roi(j,1:end-5));
    running_maxval(1,j-background_roi-1) = val;

    [val,~] = min(mean_intensity(1).roi(j,1:end-5));
    running_minval(1,j-background_roi-1) = val;

    plot(time(1:end-5),100*mean_intensity(1).roi(j,1:end-5));
    hold on
    title(strcat('Mean DFF For ROI ',num2str(j-background_roi-1)))
    
    if strcmp(seedpixel_roi,'y') == 1
        figure(size(mean_intensity(1).roi,1)+count);
        for q = 1:size(mean_intensity(1).seedpixel(j).trace,1)
            
            plot(time(1:end-5),100*mean_intensity(1).seedpixel(j).trace(q,1:end-5))
            hold on
            titletxt = strcat('Pixel DFF For Seedpixel(s)',{' '},'Within ROI',{' '},num2str(j-background_roi-1));
            title(titletxt)
            legend_txt(count).txt{q} = strcat('Seed pixel ',num2str(q));
            
            %ylim([-15,5])
        end
        legend(legend_txt(count).txt)

    end
    count = count+1;
%         axis([0 v.Duration -0.01,0.02])
%         xlabel('Time (s)')
%         ylabel('DFF')
%         title(strcat('DFF For ROI ',num2str(j-background_roi)))
%         hold on
%         line([stim_time stim_time],[-0.01,0.02],'Color','black','LineStyle','--')
%         line([stim_time+stim_duration stim_time+stim_duration],[-0.01,0.02],'Color','black','LineStyle','--')
end
   

[val,ind] = max(running_maxval);
roimax_val = val;
roimax_ind = ind;

[val,ind] = min(running_minval);
roimin_val = val;
roimin_ind = ind;


%% write individual ROI movie 
close all;
roi_dff = 1;
t_saveframe = [4,7];%time segment to save ROI movie from (in seconds)
desired_data = {'blue'}; %choose desired roi data writers
%desired_data = {'green'};
a = 0;
b = 0.01;%scale for heatmap
if roi_dff == 1
    for p = 1:size(desired_data,2)
        mkdir(strcat(folder,paths{1},'\time series evolution','\',desired_data{p}));
        for j = 1:background_roi + analyze_roi + 1
            %z2 = VideoWriter(strcat(folder,paths{1},'\',desired_data{p},'_dffavgROI',num2str(j),'_actualsize.avi'));
            %z2.FrameRate = v.FrameRate;
            %open(z2);
            bw_brain_img = imread(strcat(folder,paths{1},'\ bwim2.jpeg'));%read image to get image size (imax*jmax)
            del = 0; %color addition
            tmp_img = bw_brain_img;
            for k = 1:3
                tmp_img(sub2ind(size(tmp_img),roi_pixels(j).ynot_pixels,roi_pixels(j).xnot_pixels,k*ones(size(roi_pixels(j).ynot_pixels,1),1))) = 255;
                
            end
            imwrite(tmp_img,strcat(folder,paths{1},'\time series evolution\',desired_data{p},'\BWROI',num2str(j),'.jpeg'))
            
            for t = 1:min(tee)
                if t_saveframe(1) <= time(t) && time(t) <= t_saveframe(end)
                    switch desired_data{p}
                        case 'blue'
                            %currentframe = reshape(data.blue(:,t),[imax,jmax]);%squeeze to 2d
                            currentframe = squeeze(A0_smooth(:,:,t));
                            color_arg = 'bluevideo';    
                    end
                    %currentframe = squeeze(hemocorr_dff(:,:,t));%squeeze to 2d
                    [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

                    single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
                    [colorimage] = colorimage_gen(single_frame,newcolormap);

                    
                    for k = 1:3 %create video with masked image applied, if desired
                        if j == 1
                            if p == 3
                                %%
                                colorimage(sub2ind(size(colorimage),mask_imgpixels(2).excluderoi.ynot_pixels,mask_imgpixels.(color_arg)(2).excluderoi.xnot_pixels,k*ones(size(mask_imgpixels.(color_arg)(2).excluderoi.xnot_pixels,1),1))) = 1;
                            else
                                colorimage(sub2ind(size(colorimage),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels,k*ones(size(mask_imgpixels(2).ynot_pixels,1),1))) = 1;
                                colorimage(sub2ind(size(colorimage),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels,k*ones(size(mask_imgpixels(1).excluderoi.y_pixels,1),1))) = 1;
                            end
                        else
                            colorimage(sub2ind(size(colorimage),roi_pixels(j).ynot_pixels,roi_pixels(j).xnot_pixels,k*ones(size(roi_pixels(j).ynot_pixels,1),1))) = 1;
                        end
                    end
                    
                    imwrite(colorimage,strcat(folder,paths{1},'\time series evolution\',desired_data{p},'\dffavgROI',num2str(j),'_time_evolution_',num2str(time(t)),' sec.jpeg')) 
                    
                    tempdata = currentframe;
                    if j == 1
                        if p == 3
                            tempdata(sub2ind(size(tempdata),mask_imgpixels(2).excluderoi.ynot_pixels,mask_imgpixels(2).excluderoi.xnot_pixels)) = NaN;
                        else
                            tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = NaN;
                            tempdata(sub2ind(size(tempdata),mask_imgpixels(1).excluderoi.y_pixels,mask_imgpixels(1).excluderoi.x_pixels)) = NaN;
                        end
                    else
                       tempdata(sub2ind(size(tempdata),roi_pixels(j).ynot_pixels,roi_pixels(j).xnot_pixels)) = NaN; 
                    end
                    imagesc(tempdata)
                    axis off;
                    colormap('jet');
                    caxis([a b])
                    colorbar
                    box off
                    set(gcf,'color','w');
                    set(gca,'FontName','Arial','FontSize',14,'LineWidth',1)
                    
                    frame = getframe(gcf);
                    imwrite(frame.cdata,strcat(folder,paths{1},'\time series evolution\',desired_data{p},'\LARGEdffavgROI',num2str(j-1),'_time_evolution_',num2str(time(t)),' sec.jpeg'));
                    
                end

            end
            %close(z2)
            
        end
        
    end
    
end
close all;