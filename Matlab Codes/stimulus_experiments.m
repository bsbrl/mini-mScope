% Biosensing and Biorobotics Laboratory
% Daniel Surinach, Samantha Linn
% March 2021
% dF/F heat mapping in stimulus experiments
% calculates dF/F for each video in set
% and then one averaged dF/F video using all the trials
% generates plots for dF/F as well and statistical tests for stimulus
% periods

%% Variable initialization

clear;
clc;
close all;

% tmin1 = 8;    % start frame
% tmin2 = 12;   % end frame
tstart = 5; %start frame
tend = 'full'; %end frame, 'full' = last frame

%frac = 1;
fil = 5;      % spatial filtering pixel radius (integer), LP used 2
fscale = 2;   % filtering weight (larger number = greater weight)
bin = 0.1;     % binning fraction (0 to 1, 1 = no binning)
%input directory folder
folder = 'O:\My Drive\BSBRL DATA REPOSITORY\PROJECTS\MESOSCOPE\Calcium&Behavior Camera Data For Sharing\SKylar intact skull\skylar_intact_mouse41_t1_Rvisual\merge\segmented_intactawakevisual_stim\blue\';
files = dir(fullfile(folder,'*.avi'));%.avi file types
vid = length(files);
tee = nan(vid,1);
t_saveframe = [4,8]; %time scale to save avg dff video to show heat map evolution

stim_time = 5;%stimulus experiment details
stim_duration = 1;

paths = {'dffGeneral','dffRange','dffMax','dffzscore'};%mkdir output paths

scoring_files = 'n';%seed pixel analysis for stimulus experiments

folder_objects = dir(folder);        % all items in folder
foldercount = 0;

for j = 1:length(folder_objects) %check object names to see if previous analysis exists
   if strcmp(folder_objects(j).name,paths{1}) == 1
        foldercount = foldercount + 1;
   elseif strcmp(folder_objects(j).name,paths{2}) == 1
       foldercount = foldercount + 1;
   elseif strcmp(folder_objects(j).name,paths{3}) == 1
       foldercount = foldercount+1;
   else
        overwrite_prompt{1} = 'y';
   end
   
   if strcmp(folder_objects(j).name,'brain_mask.jpeg') == 1 %if mask of brain exists
       mask_exists = 'y';
   else
       mask_exists = 'n';%check for previous defined mask
   end
   
end

if foldercount == 3 %3 folders of prev analysis exist
    prompt = {'Do you wish to overwrite the current analysis in this folder? (y/n)?'};
    dlgtitle = 'New Analysis Prompt';
    overwrite_prompt = inputdlg(prompt,dlgtitle);
end

if strcmp(overwrite_prompt{1},'y') == 1 %if new analysis desired
    %else if videos already calculated, load them later
    for p = 1:size(paths,2) %make new directories for processed data
       mkdir(strcat(folder,paths{p}));
       if strcmp(paths{p},'dffGeneral') == 1
           mkdir(strcat(folder,paths{p},'\time series evolution'));
       end
    end

    for ex = 1:vid %iterate through individual videos
        tic
        fprintf('\nAnalyzing video %1.0f/%1.0f with name below\n',ex,vid)
        files(ex).name

        %% .avi --> 3D grayscale matrix

        v = VideoReader(strcat(folder,files(ex).name));            % reads video

        imax = v.Height;                        % frame height in units pixels
        jmax = v.Width;                         % frame width in units pixels
        tmax = round(v.Duration*v.FrameRate);          % number of frames in video

        imseq = nan(imax,jmax,tmax);            % initializes matrix; same size as video
        k = 0;
        wt=waitbar(0,'Extracting Frames');%progress bar to see how code processsteps=len;
        steps=tmax;%total frames

        while hasFrame(v)                       % loops through video frames
            im = readFrame(v);                  % reads frame
            
            if strcmp(mask_exists,'n') == 1 %generate mask over image
                %exclude borders of brain in calcium channel
                %exclude midline as well
               mask_imgvid = VideoWriter(strcat(folder,paths{1},'\mask_img.avi'));%generates mask object
               mask_imgvid.FrameRate = v.FrameRate;
               open(mask_imgvid);

               for j = 1:10
                   writeVideo(mask_imgvid,im);
               end
               close(mask_imgvid)

               rawvideo = VideoReader(strcat(folder,paths{1},'\mask_img.avi'));
               frame = 1;
                
               %no background roi, 1 analyze roi since we have the mask
               %drawing. No seedpixels desired. Exclude midline from mask
               %as well. 
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

               wt2 = waitbar(0,'Draw a mask over the brain region (exclude non GCaMP areas)');

                %rawvideo = VideoReader(strcat(folder,files(ex).name));
               %get the pixel information for the areas in side and outside
               %the mask
               rawvideo = VideoReader(strcat(folder,paths{1},'\mask_img.avi'));%VideoReader(strcat(folder,files(ex).name));
               [mask_imgpixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,data,frame_color); %set user-defined rois
               close(wt2)

               mask_exists = 'y';
               close all

               imbw = imresize(rgb2gray(im),bin);
               masked_img = uint8(255*ones(size(imbw)));

               masked_img(sub2ind(size(masked_img),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = imbw(sub2ind(size(imbw),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels));
               save(strcat(folder,paths{1},'\mask_roi.mat'),'mask_imgpixels');

            end
            
            
            k = k+1;
            imseq(:,:,k) = double(rgb2gray(im));% converts every frame to B/W 3D matrix
            if k == 1
                imshow(imresize(rgb2gray(im),bin))
                imwrite(imresize(im,bin),strcat(folder,'bwim2.jpeg')) %store first image for overlay
                imwrite(im,strcat(folder,'bwim.jpeg')); %full size image
                %saveas(gcf,strcat(folder,name,' bwim.jpeg'))
            end

            if mod(k,20)==0
                waitbar(k/steps,wt,sprintf('Extracting frame data for frame %1.0f/%1.0f',k,steps))
            end
        end
        close(wt)

        imseq = imresize(imseq,bin);            % binning B/W 3D matrix
    %     imseq = imseq(:,:,round(tmin1*v.FrameRate):round(tmin2*v.FrameRate));
        imax = size(imseq,1);                   % dimension height
        jmax = size(imseq,2);                   % dimension width
        tmax = size(imseq,3);                   % dimension time
        
        %% smoothen temporally if desired
        for i = 1:imax
            for j = 1:jmax
                imseq(i,j,:) = smooth(squeeze(imseq(i,j,:)),5);
            end
        end

        %% Denoising test - background ratio division
%         for i = 1:imax
%             for j = 1:jmax
%                 imseq(i,j,:) = smooth(squeeze(imseq(i,j,:)),5);
%             end
%         end
%         if ex == 1
%             close all;
%                 noise_pix = squeeze(imseq(1,1,:));
%             
%             imshow(uint8(im))
%             [xbkg,ybkg] = getpts;
% 
%             xbkg = round(xbkg*bin);
%             ybkg = round(ybkg*bin);
%         end
%         close all;
%         noise_pix = zeros(tmax,1);
%             figure(1);
%             legend_name = {};
%         for j = 1:length(xbkg)
%                 legend_name{j} = strcat('noise_pix',num2str(j));
%             if j == 1
%                 base_noisescale = squeeze(imseq(ybkg(j),xbkg(j),:));
%                     plot(base_noisescale);
%                     hold on
%                 noise_pix = noise_pix + base_noisescale;
% 
%             else
%                 clear ratio
%                 curr_noise = squeeze(imseq(ybkg(j),xbkg(j),:));
%                 ratio = base_noisescale./curr_noise;
% 
%                 scaled_noise = curr_noise * mean(ratio); %scale noise 
%                     plot(scaled_noise);
%                     hold on
%                 noise_pix = noise_pix + scaled_noise;
%             end
%         end
%         noise_pix = noise_pix/j;
%         
%             plot(noise_pix);
%             legend_name{end+1} = 'mean_noise';
%             legend(legend_name)
% 
%         d = designfilt('bandpassiir','FilterOrder',4, ...
%                         'PassbandFrequency1',0.1,'PassbandFrequency2',4, ...
%                         'SampleRate',v.FrameRate)
%         time = 1:length(noise_pix);
%         for i = 1:imax
%             for j = 1:jmax
%                 sig_pix = squeeze(imseq(i,j,:));
% 
%                     figure;
%                     plot(time,noise_pix,'k',time,sig_pix,'b')
% 
%                 count = 1;
%                 clear ratio
% 
%                 for k = 1:length(noise_pix)
%                     ratio(count,1) = sig_pix(k)/noise_pix(k);
%                     count = count+1;
%                 end
% 
%                     figure;
%                 scaled_noise = noise_pix*mean(ratio);
%                     plot(time,scaled_noise,'k',time,sig_pix,'b')
% 
%                 denoise_sig = sig_pix ./ scaled_noise;
%                 denoise_sig = (sig_pix - scaled_noise)+mean(sig_pix); %cancel noise, mean shift
%                     figure;
%                     plot(time,denoise_sig)
%                     hold on
% 
% 
% 
%                 mean1 = mean(denoise_sig);
%                 freq_sig = filtfilt(d,denoise_sig);
%                 mean2 = mean(freq_sig);
%                 shift = mean1-mean2;
%                 freq_sig = freq_sig + shift;
% 
%                     plot(time,freq_sig)
% 
%                 time_sig = smooth(freq_sig,5);
%                 time_sig = smooth(denoise_sig,5);
%                 imseq(i,j,:) = time_sig;
%                     plot(time,time_sig)
% 
%                 dff_sig = (time_sig - mean(time_sig))/mean(time_sig);
%                     figure
%                     plot(time,dff_sig)
%             end
%         end
        
        %% zscore of raw imseq data
        tmp = reshape(imseq,[imax*jmax,tmax]);
        zscore_mat = zeros(size(tmp));
        for j = 1:size(tmp,1) %iterate through pixels and calc zscore
            zscore_mat(j,:) = zscore(tmp(j,:));
        end

        %% global illum correction for free behavior
%         av = mean(imseq(:,:,1:round(frac*tmax)),3);
%         Tf = [];
%         roi_pixels = [];
%         Rbar = [];
%         
%         [Tf,Rbar,A,imseq,~,~] = filtering(av,imax,jmax,tmax,imseq,{'stim'},mask_imgpixels,roi_pixels,'Temporal Filter1',Tf,Rbar,background_roi,analyze_roi,1,0,fil,fscale,1,0,'blueframes');
        %% frequency filter for free behavior
%         d = designfilt('bandpassiir','FilterOrder',4, ...
%                 'PassbandFrequency1',0.1,'PassbandFrequency2',7, ...
%                 'SampleRate',15)
%         wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
%         steps=imax;%total frames
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
%         if strcmp(tend,'full') == 1
%             tend = tmax;
%         end
%         av = mean(imseq(:,:,1:round(frac*tmax)),3);
%         zero_vals = find(av == 0);
%         av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)
        %% Raw data --> detrended

        A = nan(imax,jmax,tmax);
        x = 1:tmax;
        
%         vars = load(strcat(folder,'avgdffmatrix.mat'));
%         av = vars.av;
%         
%         av = mean(imseq(:,:,1:round(frac*tmax)),3);
        if strcmp(tend,'full') == 1
            tend2 = tmax;
        else
            tend2 = tend;
        end

        av = mean(imseq(:,:,tstart:tend2),3);%calculate Fo for entire time series
        zero_vals = find(av == 0);
        av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)

        wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
        steps=imax;%total frames

        for i = 1:imax
            for j = 1:jmax
                temp = av(i,j);%traditional dF/F measurement
                A(i,j,:) = (squeeze(imseq(i,j,:))-temp)/temp;
            end

            if mod(i,20)==0
                waitbar(i/steps,wt,sprintf('Detrending pixels for pixel row %1.0f/%1.0f',i,steps))
            end
        end 
        
        clear imseq tmp
        close(wt)
        
        %% Apply weighted spatial filter to video

        A0 = zeros(imax,jmax,tmax);                      % initializes matrix; same size as A

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
        
        %%
        tee(ex) = tmax;%store max time for this video trial
        filename = split(files(ex).name,'.');
        name = strcat('\',filename{1});
        stack = reshape(A0,[imax*jmax tmax]);
        csvwrite(strcat(folder,paths{1},name,' dff.csv'),stack);
        %save(strcat(folder,paths{1},name,' dff.mat'),'stack');
        
        %csvwrite(strcat(folder,paths{4},name,' zscore.csv'),zscore_mat);
        clear A stack

        %% Calculate dF/F range with respect to time

        Range = range(A0,3);
        Max = max(A0,[],3);
        csvwrite(strcat(folder,paths{2},name,' dffrange.csv'),Range);
        csvwrite(strcat(folder,paths{3},name,' dffmax.csv'),Max);

        figure;
        colormap('jet');
        pic = imagesc(Range);
        caxis([0 .04])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(folder,paths{2},name,' dffrange.jpeg'))
        saveas(pic,strcat(folder,paths{2},name,' dffrange.fig'))
        pause(1);
        close Figure 1
        clear Range

        figure;
        colormap('jet');
        pic = imagesc(Max);
        caxis([0 .03])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(folder,paths{3},name,' dffmax.jpeg'))
        saveas(pic,strcat(folder,paths{3},name,' dffmax.fig'))
        pause(1);
        close Figure 1
        clear Range

        [max_val,max_ind] = max(max_frame(1:tee(ex))); %find time index for max in all frames
        figure;
        colormap('jet');
        pic = imagesc(squeeze(A0(:,:,max_ind)));
        caxis([-0.01,0.02])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(folder,paths{3},name,' dffmax_stillimage.jpeg'))
        saveas(pic,strcat(folder,paths{3},name,' dffmax_stillimage.fig'))
        pause(1);
        close Figure 1

        %% Output movie for dF/F 
        dffv = VideoWriter(strcat(folder,paths{1},name,' dff.avi'));
        dffv.FrameRate = v.FrameRate;
        open(dffv);
        
        A0_smooth = zeros(size(A0)); %smoothen for video rendering
        for i = 1:imax
            for j = 1:jmax
                A0_smooth(i,j,:) = smooth(A0(i,j,:),3);
            end
        end 
%         A0_smooth = reshape(smooth(A0,3),size(A0)); %smoothen data for video writing
        for t = 1:tmax %generate mask and omit data outside of mask drawn before
            tempdata = squeeze(A0_smooth(:,:,t));
            tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = 0;
            A0_smooth(:,:,t) = tempdata;
            
            im = imagesc(squeeze(A0_smooth(:,:,t))); %smoothen data during video writing
            axis off;
            colormap('jet');
            caxis([0 .02]);
            colorbar
            frame = getframe(gcf);
            writeVideo(dffv,frame);
        end
        close(dffv)

        %% repeat above but in a reals-scaled video
        C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
        jet_colors = C.Colormap;% get colorbar range values
        lim = C.Limits;%get limits of colorbar

        z2 = VideoWriter(strcat(folder,paths{1},name,' dff2.avi'));
        z2.FrameRate = v.FrameRate;
        open(z2);
        bw_brain_img = imread(strcat(folder,'bwim2.jpeg'));%read image to get image size (imax*jmax)
        a = lim(1); %get colorbar limits
        b = lim(2);
        del = 0; %color addition
        for t = 1:tmax
            %take data and rescale between 0 and 1 
            %and generate a pseudocolor map with real video dimensions
            currentframe = squeeze(A0_smooth(:,:,t));%squeeze to 2d
            [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

            single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
            [colorimage] = colorimage_gen(single_frame,newcolormap);

            writeVideo(z2,colorimage);

        end
        close(z2)
        toc
        close all
    end
    close all
    save(strcat(folder,'workspace.mat'),'tee','folder','paths','imax','jmax','tmax','mask_imgpixels','stim_time','stim_duration','jet_colors','files','vid','t_saveframe','masked_img','bin','name');

end


%% load vars if previous analysis has been chosen
vars = load(strcat(folder,'workspace.mat'));
varname = fields(vars);
for j = 1:length(varname)
    if strcmp(varname{j},'folder') ~= 1
        assignin('base',varname{j},vars.(varname{j}));%take struct element and assign var to it
    end
end
v = VideoReader(strcat(folder,files(1).name));
% stim_time = 5;%stimulus experiment details
% stim_duration = 1;
%% Average dF/F images and dF/F movie
tic
avg_count = 0;
if vid ~= 1
    
    R = nan(imax,jmax,vid);
    M = nan(imax,jmax,vid);
    runsum = zeros(imax*jmax,min(tee));
    Zscore = zeros(imax*jmax,min(tee));
    
    files1 = dir(fullfile(folder,paths{1},'*.csv'));%load files
    files2 = dir(fullfile(folder,paths{2},'*.csv'));
    files3 = dir(fullfile(folder,paths{3},'*.csv'));
    files4 = dir(fullfile(folder,paths{4},'*.csv'));
    
    for i = 1:length(files1)
        if strcmp(files1(i).name,'average_dff.csv') == 1
            avg_count = 1;
        end
    end
    
    if avg_count == 1
        runsum = table2array(readtable(strcat(folder,paths{1},'\average_dff.csv')));
        runsum = reshape(runsum,[imax,jmax,min(tee)]);
    else
        for ex = 1:vid %calcualte running average dF/F for each stimulus trial
            temp = table2array(readtable(strcat(folder,paths{1},'\',files1(ex).name)));
            data = temp(:,1:min(tee));
            clear temp
            runsum = (1/ex).*data + ((ex-1)/ex).*runsum;
            clear data

            R(:,:,ex) = table2array(readtable(strcat(folder,paths{2},'\',files2(ex).name)));

            M(:,:,ex) = table2array(readtable(strcat(folder,paths{3},'\',files3(ex).name)));

            %Zscore(:,:,ex) = table2array(readtable(strcat(folder,paths{4},'\',files4(ex).name)));

        end
        runsum = reshape(runsum,[imax,jmax,min(tee)]);
        %Zscore = reshape(Zscore,[imax,jmax,min(tee)]);
        rfinal = mean(R,3);
        mfinal = mean(M,3);
        csvwrite(strcat(folder,paths{2},'\dffAvgRange.csv'),rfinal);
        csvwrite(strcat(folder,paths{3},'\dffAvgMax.csv'),mfinal);

        figure;
        colormap('jet');
        pic = imagesc(rfinal);
        caxis([0 .04])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(folder,paths{2},'\dffAvgRange.jpeg'))
        saveas(pic,strcat(folder,paths{2},'\dffAvgRange.fig'))
        pause(1);
        close Figure 1

        figure;
        colormap('jet');
        pic = imagesc(mfinal);
        caxis([0 .03])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(folder,paths{3},'\dffAvgMax.jpeg'))
        saveas(pic,strcat(folder,paths{3},'\dffAvgMax.fig'))
        pause(1);
        close Figure 1 


        stack = reshape(runsum,[imax*jmax min(tee)]);
        csvwrite(strcat(folder,paths{1},'\average_dff.csv'),stack);
        %save(strcat(folder,paths{1},'\average_dff.mat'),'stack');

    %     stack = reshape(Zscore,[imax*jmax min(tee)]);
    %     csvwrite(strcat(folder,paths{4},'\average_dffzscore.csv'),stack);
    %     save(strcat(folder,paths{4},'\average_dffzscore.mat'),'stack');
        clear stack max_frame avg
    end
    
    [max_val,max_ind] = max(max(max(runsum))); %find max value in average video set
    
    
    figure;
    
    colorscaling_input{1} = 'n'; %colorbar limits for average dF/F movie
    a = -0.01;
    b = 0.02;
    %%
%     runsum_smooth = reshape(smooth(runsum,3),size(runsum));
    runsum_smooth = zeros(size(runsum)); %smoothen avg video
    for i = 1:imax
        for j = 1:jmax
            runsum_smooth(i,j,:) = smooth(runsum(i,j,:),3);
        end
    end 
    
    
    while strcmp(colorscaling_input{1},'n') == 1 %write output dF/F avg video
        dffavg = VideoWriter(strcat(folder,paths{1},'\dffAvg.avi'));
        dffavg.FrameRate = v.FrameRate;
        open(dffavg);
        time = linspace(0,min(tee)/v.FrameRate,min(tee));
        count = 1;
        colorbarcount = 1;
        
        for t = 1:size(runsum_smooth,3)
            tempdata = squeeze(runsum_smooth(:,:,t));
            tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = 0;
            runsum_smooth(:,:,t) = tempdata;
            
            imagesc(squeeze(runsum_smooth(:,:,t)));
            axis off;
            colormap('jet');
            caxis([a b])
            colorbar
            frame = getframe(gcf);
            writeVideo(dffavg,frame);
            
%             if round(stim_time*v.Framerate) <= t && t <= round((stim_time+stim_duration)*v.FrameRate)
%                 if count == 1
%                     start_frame = t;
%                 end
%                 avg(count) = mean(mean(squeeze(runsum_smooth(:,:,t))));
%                 count = count+1;
%             end
            if stim_time<=time(t) && time(t)<=stim_time+stim_duration %calculate frames within desired stimulus time
                if count == 1
                    start_frame = t;
                end
               avg(count) = mean(mean(squeeze(runsum_smooth(:,:,t))));
               count = count+1;
            end
        end
        [avgval,avgind] = max(avg);
        avgind = avgind + start_frame;
        %avgind = 15;
        
        close all;
        figure;
        colormap('jet');
        pic = imagesc(runsum_smooth(:,:,avgind));
        caxis([a b])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        
        close(dffavg) %generate movie and ask if colorbar limits work
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

           saveas(pic,strcat(folder,paths{3},'\dffAvgMax_stillframe.jpeg'))
           saveas(pic,strcat(folder,paths{3},'\dffAvgMax_stillframe.fig'))
           close all
           
           openfig(strcat(folder,paths{3},'\dffAvgMax_stillframe.fig'));
           C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
           jet_colors = C.Colormap;% get colorbar range values
           lim = C.Limits;%get limits of colorbar

           a = lim(1);
           b = lim(2);
        end
        close(figure(1))
        
    end
    
    
    %% write avg video with actual size
    
    z2 = VideoWriter(strcat(folder,paths{1},'\dffavg_actualsize_fullcolor.avi'));
    z2.FrameRate = v.FrameRate;
    open(z2);
    bw_brain_img = imread(strcat(folder,'bwim2.jpeg'));%read image to get image size (imax*jmax)

    del = 0; %color addition

    for t = 1:min(tee)

        currentframe = squeeze(runsum_smooth(:,:,t));%squeeze to 2d
        [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

        single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
        [colorimage] = colorimage_gen(single_frame,newcolormap);


        for k = 1:3 %create video with masked image applied, if desired
           colorimage(sub2ind(size(colorimage),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels,k*ones(size(mask_imgpixels(2).ynot_pixels,1),1))) = 1;
        end


        writeVideo(z2,colorimage);

        if t == avgind
            imwrite(colorimage,strcat(folder,paths{3},'\dffAvgmax_stillframe_correctdim_fullcolor.jpeg'))
        end

        if t_saveframe(1) <= time(t) && time(t) <= t_saveframe(end)
           imwrite(colorimage,strcat(folder,paths{1},'\time series evolution','\dffavg_time_evolution_fullcolor_',num2str(time(t)),' sec.jpeg')) 
        end

    end
    
    colorscaling_input{1} = 'n';
    
    while strcmp(colorscaling_input{1},'n') == 1
        
        z2 = VideoWriter(strcat(folder,paths{1},'\dffavg_actualsize.avi'));
        z2.FrameRate = v.FrameRate;
        open(z2);
        bw_brain_img = imread(strcat(folder,'bwim2.jpeg'));%read image to get image size (imax*jmax)
        del = 0; %color addition

        for t = 1:min(tee)

            currentframe = squeeze(runsum_smooth(:,:,t));%squeeze to 2d
            [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

            single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
            [colorimage] = colorimage_gen(single_frame,newcolormap);
            

            for k = 1:3 %create video with masked image applied, if desired
               colorimage(sub2ind(size(colorimage),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels,k*ones(size(mask_imgpixels(2).ynot_pixels,1),1))) = 1;
            end
            

            writeVideo(z2,colorimage);

            if t == avgind
                imwrite(colorimage,strcat(folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg'))
            end

            if t_saveframe(1) <= time(t) && time(t) <= t_saveframe(end)
               imwrite(colorimage,strcat(folder,paths{1},'\time series evolution','\dffavg_time_evolution_',num2str(time(t)),' sec.jpeg')) 
            end

        end
        
        figure;
        imshow(imread(strcat(folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg')));
        prompt = {'Is output image sufficiently colorized to only show high DF/F? (y/n)?'};%let user choose if blue or green frame (manual segmentation)
        dlgtitle = 'Color Image Transparency';
        colorscaling_input = inputdlg(prompt,dlgtitle);
        
        if strcmp(colorscaling_input,'n')==1
           prompt = {strcat('Enter new scale bar (x,y) for color map (current is:',num2str(a),'-',num2str(b),')')};
           dlgtitle = 'Color Image Colormap';
           colormapoutput = inputdlg(prompt,dlgtitle);
           colormapoutput = strsplit(colormapoutput{1},',');
           a = str2double(colormapoutput{1});
           b = str2double(colormapoutput{2});
        end
        close Figure 1
        close(z2)
    end
    

    %% roi selection from dff video
    %set transparency of overlayed grayscale image and peak dF/F heat map
    %then choose desired rois in peak areas for further analysis
    
    figure(1)
    dff_maxstillframe = imread(strcat(folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg'));
    imshow(masked_img)%show rgb image to overlay to color
    hold on
    pic2 = imshow(dff_maxstillframe);%overlay heat map image to rgb (to help avoid blood vessels)
    transparency_number = 0.3;
    alpha = transparency_number*ones(size(pic2));%set opacity of color image over bw image
    set(pic2,'AlphaData',alpha)
    
    prompt = {'Sufficient transparency of color image (y/n)?'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'Color Image Transparency';
    transparency_input = inputdlg(prompt,dlgtitle);
    
    while strcmp(transparency_input{1},'n') == 1
        close all;
        if strcmp(transparency_input{1},'n') == 1
            prompt = {'Set transparency of color image as a decimal (0.3 is default)'};
            dlgtitle = 'Alpha Transparency Number';
            transparency_number = inputdlg(prompt,dlgtitle);
            transparency_number = str2double(transparency_number{1});
        end
        
%         f = figure(1);
%         screensize = get(groot,'Screensize');%store screensize of image to get overlay working via remote desktop too
%         width = size(masked_img,1);height = size(masked_img,2);
%         set(f,'Position',[round(screensize(3)/2) round(screensize(4)/2) width height])
%         set(f,'Renderer','ZBuffer')
        figure(1)
        
        imshow(masked_img)%show rgb image to overlay to color
        hold on
        pic2 = imshow(dff_maxstillframe);%overlay heat map image to rgb (to help avoid blood vessels)
        alpha = transparency_number*ones(size(pic2));%set opacity of color image over bw image
        set(pic2,'AlphaData',alpha)
        %newpic = getframe;
%         if bin == 1
%             newpic = getframe(figure(1));
%         else
%             newpic = getframe;%get fused image
%         end
%         newpic = pic2.CData;
%         close all;
%         imshow(newpic.cdata)
        
        prompt = {'Sufficient transparency of color image (y/n)?'};%let user choose if blue or green frame (manual segmentation)
        dlgtitle = 'Color Image Transparency';
        transparency_input = inputdlg(prompt,dlgtitle);
    end
    
    export_fig(figure(1),strcat(folder,paths{1},name,'_dffmax_stillimage_fused.png'));
    %imwrite(newpic.cdata,strcat(folder,paths{1},name,' dffmax_stillimage_fused.jpeg'));%save fused image as jpeg
    fused_img = imread(strcat(folder,paths{1},name,'_dffmax_stillimage_fused.png'));
    fused_img = imresize(fused_img,[v.Height,v.Width]); %pixelate and resize image in case binned (easier roi extraction)
    %figure(2)
    %imshowpair(newpic.cdata,tst,'montage');
    
    fused_imgvid = VideoWriter(strcat(folder,paths{1},'\dff_fusedimg.avi'));
    fused_imgvid.FrameRate = v.FrameRate;
    open(fused_imgvid);
    for j = 1:50
        writeVideo(fused_imgvid,fused_img);
    end
    close(fused_imgvid)
    
    %rawvideo = VideoReader(strcat(folder,files(ex).name));
    rawvideo = VideoReader(strcat(folder,paths{1},'\dff_fusedimg.avi'));
    frame = 1;
    
    prompt = {'Analyze ROIs (at least 1)','Draw seed pixels within ROIs? (y/n)'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'ROI Selection For Analysis';
    roi_input = inputdlg(prompt,dlgtitle,[1 60]);
    background_roi = 0;
    analyze_roi = str2double(roi_input{1});
    seedpixel_roi = roi_input{2};
    exclude_roi = 'n';
    
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
    
    rawvideo = VideoReader(strcat(folder,paths{1},'\dff_fusedimg.avi'));%VideoReader(strcat(folder,files(ex).name));
    [roi_pixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,0,frame_color); %set user-defined rois
    
    %iterate through each ROI and calculate the mean dF/F from all the
    %pixels located within the ROI. Populate to mean_intensity struct
    %starting with the average dF/F trial loaded here
    mean_intensity = struct('roi',{});
    for t = 1:size(runsum,3)
        currentframe = squeeze(runsum(:,:,t));
        for j = 1:(background_roi+1+analyze_roi)
            intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
            mean_intensity(vid+1).roi(j,t) = mean(intensity);%last element in struct contains average mean data
            if strcmp(seedpixel_roi,'y') == 1
            
               for p = 1:length(roi_pixels(j).yseed) %iterate through chosen seed pixels
                  intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).xseed(p),roi_pixels(j).yseed(p)));
                  mean_intensity(vid+1).seedpixel(j).trace(p,t) = mean(intensity); %p is seed pixel, j is roi, t is frame
               end

            end
        end
        
        
        
    end
end  
toc

%% seed pixel analysis
%only perform if scoring files provided
%usually during awake stimulus experiments 
dff_roi = struct;
if strcmp(scoring_files,'y')==1
    %behaviors = {'nostim_frames1','nostim_frames2','nostim_frames3','nostim_frames4','stim_frames5','stim_frames6','stim_frames7','nostim_frames8','nostim_frames9','nostim_frames10'};
    %desired_behaviors = 1:10;
    behaviors = {'nostim_frames','stim_frames'};
    desired_behaviors = 2;
    
    scoring = struct('time',{});%,'frames',{},'moving_frames',{},'still_frames',{},'touch_frames',{});
    scoring_name = strcat(folder,'scoring.xlsx');
    scoring(1).time = xlsread(scoring_name);
    for j = 1:length(desired_behaviors)
        scoring(1).(behaviors{desired_behaviors(j)}).time = scoring(1).time(:,desired_behaviors(j));
        scoring(1).(behaviors{desired_behaviors(j)}).fullframes = [] ; %initialize frame
    end
    
    fps = rawvideo.FrameRate;
    set_fpsvec = ones(round(fps),1);
    for j = 1:length(desired_behaviors)
        for i = 1:size(scoring(1).time,1)
            temp = scoring(1).(behaviors{desired_behaviors(j)}).time(i) * set_fpsvec;
            scoring(1).(behaviors{desired_behaviors(j)}).fullframes = [scoring(1).(behaviors{desired_behaviors(j)}).fullframes;temp];%vector of elements in time list concatenated into frames 
        end
        scoring(1).(behaviors{desired_behaviors(j)}).ind = find(scoring(1).(behaviors{desired_behaviors(j)}).fullframes >= 1); %desired tracked behavior
        
        extra_frames = logical(scoring(1).(behaviors{desired_behaviors(j)}).ind <= size(runsum,3));
        if nnz(extra_frames) ~= length(extra_frames) %non zero elements found i.e. extra frames
            scoring(1).(behaviors{desired_behaviors(j)}).ind = scoring(1).(behaviors{desired_behaviors(j)}).ind(extra_frames);
        end
    end


    %%
    im2 = rgb2gray(bw_brain_img);
    tmp = zeros(size(im2));
    tmp(sub2ind(size(tmp),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = im2(sub2ind(size(im2),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels));%initialize as mask roi

    for k = 1:size(mask_imgpixels(1).excluderoi,2)
        tmp(sub2ind(size(tmp),mask_imgpixels(1).excluderoi(k).y_pixels,mask_imgpixels(1).excluderoi(k).x_pixels)) = 0;%assign undesired ROIs to 0
    end

    [mask_imgpixels(2).excluderoi.ynot_pixels,mask_imgpixels(2).excluderoi.xnot_pixels] = find(tmp == 0);%find undesired ROIs and store
    [mask_imgpixels(2).excluderoi.y_pixels,mask_imgpixels(2).excluderoi.x_pixels] = find(tmp~=0);%find desired ROIs and store

    im2 = rgb2gray(imread(strcat(folder,'bwim.jpeg')));
    tmp = zeros(size(im2));
    tmp(sub2ind(size(tmp),mask_imgpixels(2).fullypixel,mask_imgpixels(2).fullxpixel)) = im2(sub2ind(size(im2),mask_imgpixels(2).fullypixel,mask_imgpixels(2).fullxpixel));%initialize as mask roi

    for k = 1:size(mask_imgpixels(1).excluderoi.fullypixel,2)
        tmp(sub2ind(size(tmp),mask_imgpixels(1).excluderoi(k).fullypixel,mask_imgpixels(1).excluderoi(k).fullxpixel)) = 0;%assign undesired ROIs to 0
    end

    [mask_imgpixels(2).excluderoi.not_fullypixel,mask_imgpixels(2).excluderoi.not_fullxpixel] = find(tmp == 0);%find undesired ROIs and store
    [mask_imgpixels(2).excluderoi.fullypixel,mask_imgpixels(2).excluderoi.fullxpixel] = find(tmp~=0);%find desired ROIs and store
    %%
    mkdir(strcat(folder,paths{3},'\Circle Seed Pixel Points'))
    for k = 1:length(desired_behaviors)
        titletxt = strsplit(behaviors{desired_behaviors(k)},'_');
        titletxt = strcat('\nPerforming seed pixel correlation for',{' '},titletxt{1},{' '},titletxt{2},{' '},'behavior\n');
        fprintf(titletxt{1}) 
        field_ind = [4]; %location of desired fields i.e. bluedff, green dff, or hemodff within data struct
        moving_clustercase = 'MC'; %Small cluster (SC) or moving window cluster (MC)
        close all;

        desired_fields = {'blue'};
        imseq(1).(behaviors{k}).imax = imax;
        imseq(1).(behaviors{k}).jmax = jmax;
        for z = 1:length(desired_fields)%iterate through desired fields

            if k == 1
                mkdir(strcat(folder,paths{3},'\',desired_fields{z})); %make new subfolder for desired trace types
            end
            %% seed pixel analysis
            for j = 2:(background_roi + 1 + analyze_roi) %iterate through rois - blue trace
        %         wt2=waitbar(0,'Seed pixel analysis for points within ROIs');%progress bar to see how code processsteps=len;
        %         steps2 = size(roi_pixels(j).circle_points.xpoints,1)+1;%total frames
        %         set(findall(wt2),'Units', 'normalized');
        %         % Change the size of the figure
        %         set(wt2,'Position', [0.35 0.4 0.3 0.08]);
        %%
                calculation_type = 'seed pixel analysis';

                [dff_roi] = seedpixel_aug(roi_pixels,mask_imgpixels,j,scoring,behaviors,runsum,desired_behaviors(k),dff_roi,imseq,background_roi,analyze_roi,folder,paths,calculation_type,field_ind(z)-3,moving_clustercase,desired_fields{z});


                figure(1)
                plot(dff_roi.(behaviors{desired_behaviors(k)})(field_ind(z)-3).seedpixel(j).centroid_r)
                hold on
                plot(mean(dff_roi.(behaviors{desired_behaviors(k)})(field_ind(z)-3).seedpixel(j).cluster_r,2))
                hold on
                plot(mean(dff_roi.(behaviors{desired_behaviors(k)})(field_ind(z)-3).seedpixel(j).movingcluster(1).rmean,2))

                titletxt = strsplit(behaviors{desired_behaviors(k)},'_');
                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                titletxt = strcat('Centroid seed pixel coefficient comparison for ROI',{' '},num2str(j-1),{' '},'with',{' '},titletxt);
                title(titletxt{1});
                xlabel('Pixel Number')
                ylabel('Pearsons Correlation Coefficient Value')
                legend('Whole Trace Method','Cluster Method','Moving Window Cluster Method')
                ylim([-1,1])
                saveas(figure(1),strcat(folder,paths{3},'\',desired_fields{z},'\',titletxt{1},'.jpeg'));
                saveas(figure(1),strcat(folder,paths{3},'\',desired_fields{z},'\',titletxt{1},'.fig'));
                close all;

            end
            close all;

        end
        toc    

    end
end
%% snr measurements for the entire frame 
close all
snr = zeros(imax,jmax);
for i = 1:imax
    for j = 1:jmax
        snr(i,j) = (max(runsum(i,j,:))-mean(runsum(i,j,:)))/std(runsum(i,j,:));
    end
end
imagesc(snr)
colormap('jet')
colorbar
set(gca,'xtick',[])
set(gca,'ytick',[])
saveas(figure(1),strcat(folder,paths{1},'\snr.jpeg'))
%% find mean intensity for each ROI in video series
%iterate through each ROI and calculate the mean dF/F from all the
%pixels located within the ROI. Populate to mean_intensity struct
%for each separate stimulus trial

ytitle = 'DFF%';
switch ytitle
    case 'DFF%'
        mult_factor = 100;
    case 'zscore'
        mult_factor = 1;
end
for ex = 1:vid
    temp = table2array(readtable(strcat(folder,paths{1},'\',files1(ex).name)));
    data = temp(:,1:min(tee));
    time_frames = reshape(data,[imax,jmax,min(tee)]);
    
    for t = 1:size(time_frames,3)
        count = 1;
        currentframe = squeeze(time_frames(:,:,t));
        for j = 1:(background_roi+1+analyze_roi)
            intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
            mean_intensity(ex).roi(j,t) = mean(intensity);%contains data from each roi
            for p = 1:length(roi_pixels(j).yseed) %iterate through chosen seed pixels
               intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).xseed(p),roi_pixels(j).yseed(p)));
               mean_intensity(ex).seedpixel(j).trace(p,t) = mean(intensity); %p is seed pixel, j is roi, t is frame
            end
        end
        
        
    end

end

%% calculate max peak dF/F for each ROI within stimulus time
close all;
stim_frame_range = round((stim_time*v.FrameRate)):round(((stim_time+stim_duration)*v.FrameRate));
stim_data = struct('stim_period',{},'max_stimval',{},'ttest_data',{});

for ex = 1:size(mean_intensity,2) %include extra "vid" because avg appended to mean list above
   for j = 1:(background_roi+1+analyze_roi)
      stim_data(ex).stim_period(j,:) = mean_intensity(ex).roi(j,stim_frame_range); 
      [val,~] = max(stim_data(ex).stim_period(j,:));
      stim_data(ex).max_stimval(j,1) = val;
      for p = 1:length(roi_pixels(j).yseed) %iterate through chosen seed pixels
          stim_data(ex).seedpixel(p).stim_period(j,:) = mean_intensity(ex).seedpixel(j).trace(p,stim_frame_range);
          [val,~] = max(stim_data(ex).seedpixel(p).stim_period(j,:));
          stim_data(ex).seedpixel(p).max_stimval(j,1) = val;
          if ex < vid+1
              mean_intensity(vid+1).seedpixel(j).data(ex,:,p) = mean_intensity(ex).seedpixel(j).trace(p,:);
          else
              figure(j)
              plot(time,mult_factor*mean_intensity(vid+1).seedpixel(j).trace(p,:))
              hold on
              titletxt = strcat('DFF For Seedpixel',{' '},num2str(p)',{' '},'Within ROI',{' '},num2str(j));
              title(titletxt)
              
              err_data = mean_intensity(vid+1).seedpixel(j).trace(p,:);
              err_stdev = std(mult_factor*mean_intensity(vid+1).seedpixel(j).data(:,:,p));
              er = errorbar(time,mult_factor*err_data,err_stdev,err_stdev,'CapSize',20);
              er.Color = [0,0,0];
              er.LineStyle = 'none';
              
          end
      end
   end

end


%% plot mean DFF traces for each ROI
close all;
figure_count = 1;
clear running_maxval running_minval

for k = 1:vid+1 %iterate through all video sequences + avg dff
    for j = background_roi+2:size(mean_intensity(1).roi,1) %iterate through rois (first few are background ROIs)
        %first j index is average dff trace
        figure(j)
        if k == vid+1
            plot_color = [1 0 0]; %red
            %mean_intensity(k).roi(j,:) =
            %smooth(mean_intensity(k).roi(j,:),3);
        else
            plot_color = [0.8 0.8 0.8];
        end
        
        [val,~] = max(mean_intensity(k).roi(j,1:end-5));
        running_maxval(k,j-background_roi-1) = val;
        
        [val,~] = min(mean_intensity(k).roi(j,1:end-5));
        running_minval(k,j-background_roi-1) = val;
        
        if k < vid+1
            plot_var = smooth(mean_intensity(k).roi(j,1:end-5),1);
        else
            plot_var = mean_intensity(k).roi(j,1:end-5);
        end
        plot(time(1:end-5),mult_factor*plot_var,'Color',plot_color);
        hold on
        
         
        
%         axis([0 v.Duration -0.01,0.02])
%         xlabel('Time (s)')
%         ylabel('DFF')
%         title(strcat('DFF For ROI ',num2str(j-background_roi)))
%         hold on
%         line([stim_time stim_time],[-0.01,0.02],'Color','black','LineStyle','--')
%         line([stim_time+stim_duration stim_time+stim_duration],[-0.01,0.02],'Color','black','LineStyle','--')
    end
    
end

[val,ind] = max(running_maxval);
roimax_val = val;
roimax_ind = ind;

[val,ind] = min(running_minval);
roimin_val = val;
roimin_ind = ind;

for j = background_roi+2:size(mean_intensity(1).roi,1)
    figure(j)
    axis([0 v.Duration+0.4 mult_factor*roimin_val(j-background_roi-1),mult_factor*roimax_val(j-background_roi-1)])
    xlabel('Time (s)')
    ylabel(ytitle)
    title(strcat('DFF For ROI ',num2str(j-background_roi)))
    hold on
    line([stim_time stim_time],[mult_factor*roimin_val(j-background_roi-1),mult_factor*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--')
    line([stim_time+stim_duration stim_time+stim_duration],[mult_factor*roimin_val(j-background_roi-1),mult_factor*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--') 
    
    saveas(figure(j),strcat(folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials.jpeg'))
    saveas(figure(j),strcat(folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials.fig'))
    saveas(figure(j),strcat(folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials'),'epsc')
    
    if strcmp(seedpixel_roi,'y') == 1
        figure(j+10)
        hold on
        for q = 1:size(mean_intensity(vid+1).seedpixel(j).trace,1)
            if q == size(mean_intensity(vid+1).seedpixel(j).trace,1)
                filt_param = 1;
            else
                filt_param =1;
            end
            filt_trace = smooth(mean_intensity(vid+1).seedpixel(j).trace(q,1:end-5),filt_param);
            %plot(time(1:end-5),mult_factor*mean_intensity(1).seedpixel(j).trace(q,1:end-5))
            plot(time(1:end-5),mult_factor*filt_trace)
            hold on
            titletxt = strcat('DFF For Seedpixel(s)',{' '},'Within ROI',{' '},num2str(j));
            title(titletxt)
            legend_txt(j-background_roi-1).name{q} = strcat('Seed pixel ',num2str(q));
            %ylim([-15,5])
        end
        legend(legend_txt(j-background_roi-1).name);
        figure(j+10)
        titletxt = strcat('DFF For Seedpixel(s)',{' '},'Within ROI',{' '},num2str(j));

        saveas(figure(j+10),strcat(folder,paths{1},'\',titletxt{1},'.jpeg'))
        saveas(figure(j+10),strcat(folder,paths{1},'\',titletxt{1},'.fig'))
        saveas(figure(j+10),strcat(folder,paths{1},'\',titletxt{1}),'epsc')
     end
    
end

%% plot max value during stimulus period for each ROI
if analyze_roi > 1
    ipsi_contra_roipairs = [1,2];
    roiname = num2str(ipsi_contra_roipairs);
    roiname = strcat(roiname(1),'-',roiname(end));

    ipsi_contra_roipairs = ipsi_contra_roipairs + background_roi + 1;
    clear running_maxval running_minval
    for k = 1:vid+1 %iterate through all video sequences + avg dff
        for j = 1:size(ipsi_contra_roipairs,1)%background_roi+2:size(mean_intensity(1).roi,1) %iterate through rois (first few are background ROIs)
            %first j index is average dff trace
            figure(size(mean_intensity(1).roi,1)+j)
            if k == vid+1
                plot_color = [1 0 0]; %red
            else
                plot_color = [0.8 0.8 0.8];
            end

            [val,~] = max(stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1));
            running_maxval(k,j) = val;

            [val,~] = min(stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1));
            running_minval(k,j) = val;

            stim_data(vid+1).ttest_data(k,:) = stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1)';

    %         for q = 1:size(ipsi_contra_roipairs(j,:),2)
    %             for p = 1:length(roi_pixels(ipsi_contra_roipairs(j,q)).yseed) %iterate through chosen seed pixels
    %                 stim_data(vid+1).seedpixel(q).ttest_data(k,p) = stim_data(k).seedpixel(q).max_stimval(ipsi_contra_roipairs(j,p),1);
    %                 stim_data(vid+1).seedpixel(q).stim_period_tot(k,:,p) = stim_data(k).seedpixel(q).stim_period(ipsi_contra_roipairs(j,p),:);
    %             end
    %         end

            if k == vid+1
                stim_data(k).ttest_data(k,:) = mean(stim_data(k).ttest_data(1:end-1,:));
                stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1) = stim_data(k).ttest_data(k,:);

            end
            plot(mult_factor*stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1),'-*','Color',plot_color);
            hold on
    %         axis([0 v.Duration -0.01,0.02])
    %         xlabel('Time (s)')
    %         ylabel('DFF')
    %         title(strcat('DFF For ROI ',num2str(j-background_roi)))
    %         hold on
    %         line([stim_time stim_time],[-0.01,0.02],'Color','black','LineStyle','--')
    %         line([stim_time+stim_duration stim_time+stim_duration],[-0.01,0.02],'Color','black','LineStyle','--')
        end

    end

    [val,ind] = max(running_maxval(1:end-1));%find the trials with the lowest and highest DFF for the whole trial
    roimax_val = val;
    roimax_ind = ind;

    [val,ind] = min(running_minval(1:end-1));
    roimin_val = val;
    roimin_ind = ind;

    [h,p_val] = ttest(stim_data(vid+1).ttest_data(1:end-1,1),stim_data(vid+1).ttest_data(1:end-1,2)); %paired ttest without including last data point (avg of videos)
    stdev = std(mult_factor*stim_data(vid+1).ttest_data(1:end-1,:));

    fprintf('Max DFF for all trials is %4.4f occurs for the video listed below...Check if outliers are present\n',roimax_val)
    files(roimax_ind).name
    fprintf('Min DFF for all trials is %4.4f occurs for the video listed below...Check if outliers are present\n',roimin_val)
    files(roimin_ind).name

    fprintf('P-value for paired t-test between ipsilateral and contralateral for stimulus is %4.4f\n',p_val)

    fig_legend = {'Contralateral','Ipsilateral'};
    for j = 1:size(ipsi_contra_roipairs,1)
        figure(size(mean_intensity(1).roi,1)+j)
        ylim([mult_factor*roimin_val(j),mult_factor*roimax_val(j)])
        xlim([0.8,2.2])
        set(gca,'XTick',1:length(fig_legend),'XTickLabel',fig_legend)
        ylabel(ytitle)
        hold on

        er = errorbar([1,2],mult_factor*stim_data(vid+1).ttest_data(end,:),stdev,stdev,'CapSize',20);
        er.Color = [0,0,0];
        er.LineStyle = 'none';


        saveas(figure(size(mean_intensity(1).roi,1)+j),strcat(folder,paths{1},'\contra_ipsi_rois_',roiname,' peakdff_stimperiod.jpeg'))
        saveas(figure(size(mean_intensity(1).roi,1)+j),strcat(folder,paths{1},'\contra_ipsi_rois_',roiname,' peakdff_stimperiod'),'epsc')
        saveas(figure(size(mean_intensity(1).roi,1)+j),strcat(folder,paths{1},'\contra_ipsi_rois_',roiname,' peakdff_stimperiod.fig'))

    end
end

%% write dF/F for each ROI
%the rest of the frame will be omitted if you want to only show localized
%responses within each ROI
close all;
roi_dff = 1;
t_saveframe = [0,2];
desired_data = {'blue'}; %choose desired roi data writers
%desired_data = {'green'};
% a = 0;
% b = 0.015;
if roi_dff == 1
    for p = 1:size(desired_data,2)
        mkdir(strcat(folder,paths{1},'\time series evolution','\',desired_data{p}));
        for j = 1:background_roi + analyze_roi + 1
            %z2 = VideoWriter(strcat(folder,paths{1},'\',desired_data{p},'_dffavgROI',num2str(j),'_actualsize.avi'));
            %z2.FrameRate = v.FrameRate;
            %open(z2);
            bw_brain_img = imread(strcat(folder,'bwim2.jpeg'));%read image to get image size (imax*jmax)
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
                            currentframe = squeeze(runsum(:,:,t));
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