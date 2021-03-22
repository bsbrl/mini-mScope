% Biosensing and Biorobotics Laboratory
% Samantha Linn, Daniel Surinach
% February 2020
% dF/F heat mapping in stimulus experiments with averaging completed

%% Variable initialization

function stimulus_exp_hemocorr(bin,t_saveframe,stim_time,stim_duration,input_folder,analysis_type)
% tmin1 = 8;    % start frame
% tmin2 = 12;   % end frame
%frac = 1;
files = dir(fullfile(input_folder,'*.avi'));
if strcmp(analysis_type,'hemo_corr_avgdff')==1
    vid = 1; %only need 1 video for rendering purpose if DFF avg movie
else
    vid = length(files);
end
tee = nan(vid,1);
moviewrite = 0;

paths = {'dffGeneral','dffRange','dffMax'};
% path1 = 'dffGeneral';
% path2 = 'dffRange';
% path3 = 'dffMax';

folder_objects = dir(input_folder);        % all items in folder
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
       mask_exists = 'n';
   end
   
end

if foldercount == 3 %3 folders of prev analysis exist
    prompt = {'Do you wish to overwrite the current analysis in this folder? (y/n)?'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'New Analysis Prompt';
    overwrite_prompt = inputdlg(prompt,dlgtitle);
end

%% Average range dF/F image and dF/F movie
tic
if vid > 0
    
%     runsum = zeros(imax*jmax,min(tee));
    fprintf('Performing hemodynamic correction\n')
    if vid == 1 %csv avg files, no vid files
        img_files = dir(fullfile(input_folder,'*.jpeg')); %get jpeg files
        for p = 1:size(paths,2)
           mkdir(strcat(input_folder,paths{p}));
           if strcmp(paths{p},'dffGeneral') == 1
               mkdir(strcat(input_folder,paths{p},'\time series evolution'));
           end
        end
        for i = 1:length(img_files)
           im = imread(strcat(input_folder,img_files(i).name));
           vidtype = strsplit(img_files(i).name,{'.jpeg',' '});%find if blue or green video
           vidtype = vidtype{end-1};%get if greenvideo or bluevideo
           
           mask_imgvid = VideoWriter(strcat(input_folder,paths{1},'\',vidtype,'_mask_img.avi'));
           mask_imgvid.FrameRate = 15;
           open(mask_imgvid);

           for j = 1:50
               writeVideo(mask_imgvid,im);
           end
           close(mask_imgvid)


           rawvideo = VideoReader(strcat(input_folder,paths{1},'\',vidtype,'_mask_img.avi'));
           frame = 1;
           frame_color = {};
          
           blueorgreenonly = {'blue','b'};
           stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
           [frame_color] = findframe(rawvideo,frame,stopframe,frame_color,0,0,'',blueorgreenonly); %extract which frames are green and blue
           if isempty(frame_color.black)==1
               frame_color.black = 0;
           end
          [mask_imgpixels.(vidtype),~,~,~] = roi_gen(im,input_folder,paths,rawvideo,bin,frame_color,'y','blue','n','n');
          
          imbw = imresize(rgb2gray(im),bin);
          mask_type = im;
          masked_img.(vidtype) = uint8(255*ones(size(imresize(mask_type,bin))));
% 
          masked_img.(vidtype)(sub2ind(size(masked_img.(vidtype)),mask_imgpixels.(vidtype)(2).y_pixels,mask_imgpixels.(vidtype)(2).x_pixels)) = mask_type(sub2ind(size(mask_type),mask_imgpixels.(vidtype)(2).y_pixels,mask_imgpixels.(vidtype)(2).x_pixels));
          masked_img.(vidtype)(sub2ind(size(masked_img.(vidtype)),mask_imgpixels.(vidtype)(1).excluderoi.y_pixels,mask_imgpixels.(vidtype)(1).excluderoi.x_pixels)) = 255;
        end
        imax = size(imresize(im,bin),1);
        jmax = size(imresize(im,bin),2);
        im_resize = imread(strcat(input_folder,'bwim bluevideo.jpeg'));
        im_resize = imresize(im_resize,bin); 
        imwrite(im_resize,strcat(input_folder,' bwim2.jpeg'));
        

        files1 = dir(fullfile(input_folder,'*.csv'));
        vid = length(files1);
        %%
        data = struct;
        for ex = 1:vid
            splitname = strsplit(files1(ex).name,{'_',' ','.'});
            for k = 1:length(splitname)
                if strcmp(splitname{k},'bluevideo') == 1 %bluevideo
                    data(1).blue = table2array(readtable(strcat(input_folder,'\',files1(ex).name)));
                    tee(1) = size(data(1).blue,2);
                elseif strcmp(splitname{k},'greenvideo')==1
                    data(1).green = table2array(readtable(strcat(input_folder,'\',files1(ex).name)));
                    tee(2) = size(data(1).green,2);

    %                 for z = 1:size(data(1).green,1)
    %                    data(1).green(z,:) = sgolayfilt(data(1).green(z,:),5,21); 
    % 
    %                 end
                end

            end

            clear temp
            %runsum = (1/ex).*data + ((ex-1)/ex).*runsum;
            %clear data

        end
        data(1).blue = data(1).blue(:,1:min(tee));
        data(1).green = data(1).green(:,1:min(tee));
        v = VideoReader(strcat(input_folder,files(1).name));
        
    else
        files1 = dir(fullfile(input_folder,paths{1},'*.csv'));
        data = struct;
        for ex = 1:vid
            splitname = strsplit(files1(ex).name,{'_',' ','.'});
            for k = 1:length(splitname)
                if strcmp(splitname{k},'bluevideo') == 1 %bluevideo
                    temp = table2array(readtable(strcat(input_folder,paths{1},'\',files1(ex).name)));
                    data(1).blue = temp(:,1:min(tee));
                elseif strcmp(splitname{k},'greenvideo')==1
                    temp = table2array(readtable(strcat(input_folder,paths{1},'\',files1(ex).name)));
                    data(1).green = temp(:,1:min(tee));

    %                 for z = 1:size(data(1).green,1)
    %                    data(1).green(z,:) = sgolayfilt(data(1).green(z,:),5,21); 
    % 
    %                 end
                end

            end

            clear temp
            %runsum = (1/ex).*data + ((ex-1)/ex).*runsum;
            %clear data

        end
    end
    %%
    im2 = double(imresize(rgb2gray(im),bin));
    tmp = zeros(size(im2));
    tmp(sub2ind(size(tmp),mask_imgpixels.bluevideo(2).y_pixels,mask_imgpixels.bluevideo(2).x_pixels)) = im2(sub2ind(size(im2),mask_imgpixels.bluevideo(2).y_pixels,mask_imgpixels.bluevideo(2).x_pixels));%initialize as mask roi

    for m = 1:size(mask_imgpixels.greenvideo(1).excluderoi,2)
        tmp(sub2ind(size(tmp),mask_imgpixels.greenvideo(1).excluderoi(m).y_pixels,mask_imgpixels.greenvideo(1).excluderoi(m).x_pixels)) = 0;%assign undesired ROIs to 0
        tmp(sub2ind(size(tmp),mask_imgpixels.bluevideo(1).excluderoi(m).y_pixels,mask_imgpixels.bluevideo(1).excluderoi(m).x_pixels)) = 0;%assign undesired ROIs to 0
    end

    tmp(sub2ind(size(tmp),mask_imgpixels.greenvideo(2).ynot_pixels,mask_imgpixels.greenvideo(2).xnot_pixels)) = 0;

    [mask_imgpixels.hemocorrect(2).excluderoi.ynot_pixels,mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels] = find(tmp==0);
    [mask_imgpixels.hemocorrect(2).excluderoi.y_pixels,mask_imgpixels.hemocorrect(2).excluderoi.x_pixels] = find(tmp~=0);
    
%     files1 = dir(fullfile(folder,paths{1},'*.csv'));
    
    toc
    wt=waitbar(0,'Temporal Filtering Data');%progress bar to see how code processsteps=len;
    steps=imax;%total frames
%     d = designfilt('lowpassiir','FilterOrder',2, ...
%         'PassbandFrequency',0.15, ...
%         'SampleRate',15,'PassbandRipple',0.2)
    data.green = reshape(data.green,[imax jmax min(tee)]);
    %% filter design green channel
     d = designfilt('bandpassiir','FilterOrder',2, ...
        'PassbandFrequency1',0.02,'PassbandFrequency2',0.08, ...
        'SampleRate',15,'PassbandRipple',0.005)
    for i = 1:imax
        for j = 1:jmax
            data.green(i,j,:) = filtfilt(d,squeeze(data.green(i,j,:)));
        end
        if mod(i,20)==0
            waitbar(i/steps,wt,sprintf('Temporal filtering pixels for pixel row %1.0f/%1.0f',i,steps))
        end

    end
    close(wt)
    data.green = reshape(data.green,[imax*jmax min(tee)]);
    %hemocorr_dff = data(1).blue./data(1).green - 1; %Iblue/Ioblue / Igreen/Io green - 1 = DFF/F
    hemocorr_dff = data(1).blue - data(1).green;
    hemocorr_dff = reshape(hemocorr_dff,[imax jmax min(tee)]);
    data.blue = reshape(data.blue,[imax jmax min(tee)]);
    data.green = reshape(data.green,[imax jmax min(tee)]);
    
    %% filter design hemodynamic corrected channel
    wt=waitbar(0,'Temporal Filtering Data');%progress bar to see how code processsteps=len;
    steps=imax;%total frames
    d = designfilt('bandpassiir','FilterOrder',4, ...
        'PassbandFrequency1',0.1,'PassbandFrequency2',5, ...
        'SampleRate',15,'PassbandRipple',1)

    for i = 1:imax
        for j = 1:jmax
            hemocorr_dff(i,j,:) = filtfilt(d,squeeze(hemocorr_dff(i,j,:)));
        end
        if mod(i,20)==0
            waitbar(i/steps,wt,sprintf('Temporal filtering pixels for pixel row %1.0f/%1.0f',i,steps))
        end

    end
    close(wt)
    %%
    stack = reshape(hemocorr_dff,[imax*jmax min(tee)]);
    fprintf('Saving corrected data\n')
    csvwrite(strcat(input_folder,paths{1},'\average_dff.csv'),stack);
    %save(strcat(folder,paths{1},'\average_dff.mat'),'stack');
    toc
    clear stack max_frame avg
    [max_val,max_ind] = max(max(max(hemocorr_dff))); %find max value in average video set
    
    %%
    figure;
    
    colorscaling_input{1} = 'n';
    a = -0.02;
    b = 0.05;
    
    %runsum_smooth = reshape(smooth(runsum,3),size(runsum));
    while strcmp(colorscaling_input{1},'n') == 1
        dffavg = VideoWriter(strcat(input_folder,paths{1},'\dffAvg.avi'));
        dffavg.FrameRate = v.FrameRate;
        open(dffavg);
        time = linspace(0,v.Duration,min(tee));
        count = 1;
        colorbarcount = 1;
        
        for t = 1:size(hemocorr_dff,3)
            tempdata = squeeze(hemocorr_dff(:,:,t));
            tempdata(sub2ind(size(tempdata),mask_imgpixels.hemocorrect(2).excluderoi.ynot_pixels,mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels)) = 0;
%             tempdata(sub2ind(size(tempdata),mask_imgpixels.greenvideo(2).ynot_pixels,mask_imgpixels.greenvideo(2).xnot_pixels)) = 0;%change to zero, video writing easier
%             tempdata(sub2ind(size(tempdata),mask_imgpixels.greenvideo(1).excluderoi.y_pixels,mask_imgpixels.greenvideo(1).excluderoi.x_pixels)) = 0;
            hemocorr_dff(:,:,t) = tempdata;
        end
        
        
        for i = 1:imax
            for j = 1:jmax
                hemocorr_dff(i,j,:) = smooth(hemocorr_dff(i,j,:),3);
            end
        end 
                
        for t = 1:size(hemocorr_dff,3)
            tempdata = squeeze(hemocorr_dff(:,:,t));
            tempdata(sub2ind(size(tempdata),mask_imgpixels.hemocorrect(2).excluderoi.ynot_pixels,mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels)) = NaN;

%             tempdata(sub2ind(size(tempdata),mask_imgpixels.greenvideo(2).ynot_pixels,mask_imgpixels.greenvideo(2).xnot_pixels)) = NaN;
%             tempdata(sub2ind(size(tempdata),mask_imgpixels.greenvideo(1).excluderoi.y_pixels,mask_imgpixels.greenvideo(1).excluderoi.x_pixels)) = NaN;
            %hemocorr_dff(:,:,t) = tempdata;
            
            imagesc(tempdata);
            axis off;
            colormap('jet');
            caxis([a b])
            colorbar
            frame = getframe(gcf);
            writeVideo(dffavg,frame);

            if stim_time<=time(t) && time(t)<=stim_time+stim_duration
                if count == 1
                    start_frame = t;
                end
               avg(count) = mean(mean(squeeze(hemocorr_dff(:,:,t))));
               count = count+1;
            end
        end
        [avgval,avgind] = max(avg);
        avgind = avgind + start_frame;
        
        close all;
        figure;
        colormap('jet');
        pic = imagesc(hemocorr_dff(:,:,avgind));
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

           saveas(pic,strcat(input_folder,paths{3},'\dffAvgMax_stillframe.jpeg'))
           saveas(pic,strcat(input_folder,paths{3},'\dffAvgMax_stillframe.fig'))
           close all
           
           openfig(strcat(input_folder,paths{3},'\dffAvgMax_stillframe.fig'));
           C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
           jet_colors = C.Colormap;% get colorbar range values
           lim = C.Limits;%get limits of colorbar

           a = lim(1);
           b = lim(2);
        end
        close(figure(1))
        close(dffavg)
        
    end
    
    
    %% write avg video with actual size
    
    z2 = VideoWriter(strcat(input_folder,paths{1},'\dffavg_actualsize_fullcolor.avi'));
    z2.FrameRate = v.FrameRate;
    open(z2);

    bw_brain_img = imread(strcat(input_folder,' bwim2.jpeg'));%read image to get image size (imax*jmax)


    del = 0; %color addition
    jet_colors = jet;
    for t = 1:size(hemocorr_dff,3)
        
        
        currentframe = squeeze(hemocorr_dff(:,:,t));%squeeze to 2d
        [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

        single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
        [colorimage] = colorimage_gen(single_frame,newcolormap);

        for k = 1:3
            colorimage(sub2ind(size(colorimage),mask_imgpixels.hemocorrect(2).excluderoi.ynot_pixels,mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels,k*ones(size(mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels,1),1))) = 1;
        end
        
%         for k = 1:3 %create video with masked image applied, if desired
% %            colorimage(sub2ind(size(colorimage),mask_imgpixels.greenvideo(2).ynot_pixels,mask_imgpixels.greenvideo(2).xnot_pixels,k*ones(size(mask_imgpixels.greenvideo(2).ynot_pixels,1),1))) = 1;
% %            colorimage(sub2ind(size(colorimage),mask_imgpixels.greenvideo(1).excluderoi.y_pixels,mask_imgpixels.greenvideo(1).excluderoi.x_pixels,k*ones(size(mask_imgpixels.greenvideo(1).excluderoi.y_pixels,1),1))) = 1;
%         end


        writeVideo(z2,colorimage);

        if t == avgind
            imwrite(colorimage,strcat(input_folder,paths{3},'\dffAvgmax_stillframe_correctdim_fullcolor.jpeg'))
        end

        if t_saveframe(1) <= time(t) && time(t) <= t_saveframe(end)
           imwrite(colorimage,strcat(input_folder,paths{1},'\time series evolution','\dffavg_time_evolution_fullcolor_',num2str(time(t)),' sec.jpeg')) 
        end

    end
    
    colorscaling_input{1} = 'n';
    
    while strcmp(colorscaling_input{1},'n') == 1
        
        z2 = VideoWriter(strcat(input_folder,paths{1},'\dffavg_actualsize.avi'));
        z2.FrameRate = v.FrameRate;
        open(z2);
        bw_brain_img = imread(strcat(input_folder,' bwim2.jpeg'));%read image to get image size (imax*jmax)
        del = 0; %color addition

        for t = 1:min(tee)

            currentframe = squeeze(hemocorr_dff(:,:,t));%squeeze to 2d
            [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

            single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
            [colorimage] = colorimage_gen(single_frame,newcolormap);
            

            for k = 1:3 %create video with masked image applied, if desired
               colorimage(sub2ind(size(colorimage),mask_imgpixels.greenvideo(2).ynot_pixels,mask_imgpixels.greenvideo(2).xnot_pixels,k*ones(size(mask_imgpixels.greenvideo(2).ynot_pixels,1),1))) = 1;
               colorimage(sub2ind(size(colorimage),mask_imgpixels.greenvideo(1).excluderoi.y_pixels,mask_imgpixels.greenvideo(1).excluderoi.x_pixels,k*ones(size(mask_imgpixels.greenvideo(1).excluderoi.y_pixels,1),1))) = 1;
            end
            

            writeVideo(z2,colorimage);

            if t == avgind
                imwrite(colorimage,strcat(input_folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg'))
            end

            if t_saveframe(1) <= time(t) && time(t) <= t_saveframe(end)
               imwrite(colorimage,strcat(input_folder,paths{1},'\time series evolution','\dffavg_time_evolution_',num2str(time(t)),' sec.jpeg')) 
            end

        end
        
        figure;
        imshow(imread(strcat(input_folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg')));
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
    
    figure(1)
    dff_maxstillframe = imread(strcat(input_folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg'));
    imshow(masked_img.greenvideo)%show rgb image to overlay to color
    hold on
    pic2 = imshow(dff_maxstillframe);%overlay heat map image to rgb (to help avoid blood vessels)
    transparency_number = 0.3;
    alpha = transparency_number*ones(size(pic2));%set opacity of color image over bw image
    set(pic2,'AlphaData',alpha)

    %newpic = getframe;%get fused image
    %close all;
    %imshow(newpic.cdata)
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
        
        figure(1)
        imshow(bw_brain_img)%show rgb image to overlay to color
        hold on
        pic2 = imshow(dff_maxstillframe);%overlay heat map image to rgb (to help avoid blood vessels)
        alpha = transparency_number*ones(size(pic2));%set opacity of color image over bw image
        set(pic2,'AlphaData',alpha)

        %newpic = getframe;%get fused image
        %close all;clc
        %imshow(newpic.cdata)
        
        prompt = {'Sufficient transparency of color image (y/n)?'};%let user choose if blue or green frame (manual segmentation)
        dlgtitle = 'Color Image Transparency';
        transparency_input = inputdlg(prompt,dlgtitle);
    end
    %%
    %saveas(figure(1),strcat(folder,paths{1},name,' dffmax_stillimage_fused.png'));
    export_fig(strcat(input_folder,paths{1},'\dffmax_stillimage_fused.png'));
    %imwrite(newpic.cdata,strcat(folder,paths{1},name,' dffmax_stillimage_fused.jpeg'));%save fused image as jpeg
   
    fused_img = imread(strcat(input_folder,paths{1},'\dffmax_stillimage_fused.png'));
    fused_img = imresize(fused_img,[v.Height,v.Width]); %pixelate and resize image in case binned (easier roi extraction)
    %figure(2)
    %imshowpair(newpic.cdata,tst,'montage');
    
    roi_im = imbw;
    roi_im(sub2ind(size(roi_im),mask_imgpixels.bluevideo(2).ynot_pixels,mask_imgpixels.bluevideo(2).xnot_pixels)) = 0;
    roi_im = imresize(roi_im,[v.Height,v.Width]);
    fused_imgvid = VideoWriter(strcat(input_folder,paths{1},'\dff_fusedimg.avi'));
    fused_imgvid.FrameRate = v.FrameRate;
    open(fused_imgvid);
    for j = 1:100
        writeVideo(fused_imgvid,fused_img);
    end
    close(fused_imgvid)
    
    %rawvideo = VideoReader(strcat(folder,files(ex).name));
    rawvideo = VideoReader(strcat(input_folder,paths{1},'\dff_fusedimg.avi'));
    frame = 1;
    
    prompt = {'Analyze ROIs (at least 1)','Draw seed pixels within ROIs? (y/n)'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'ROI Selection For Analysis';
    roi_input = inputdlg(prompt,dlgtitle,[1 60]);
    background_roi = 0;
    analyze_roi = str2double(roi_input{1});
    seedpixel_roi = roi_input{2};
    exclude_roi = 'n';

    
    imgname = strcat(input_folder,paths{1},'\roi_selection.jpeg');
    stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
    casenum = 0;
    blueorgreenonly = {'blue','b'};
    
    frame_color = {};
	[frame_color] = findframe(rawvideo,frame,stopframe,frame_color,0,0,'',blueorgreenonly); %extract which frames are green and blue
    if isempty(frame_color(1).black)==1
        frame_color(1).black = 0;
    end
    
    %rawvideo = VideoReader(strcat(folder,files(ex).name));

    rawvideo = VideoReader(strcat(input_folder,paths{1},'\dff_fusedimg.avi'));%VideoReader(strcat(folder,files(ex).name));
    [roi_pixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,0,frame_color); %set user-defined rois
%%
    %intensity_vector(j).brightnessblue=corrected_framemat(sub2ind(size(corrected_framemat),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
    mean_intensity = struct('roi',{});
    roi_fields = fields(roi_pixels); %get elements of struct
    for p = 1:size(roi_fields,1)%check field elements
       if strcmp(roi_fields{p},'yseed') == 1 %if seed pixels chosen exist
           seeds_exist = 1;
       end
           
    end
    for t = 1:size(hemocorr_dff,3)
        currentframe = squeeze(hemocorr_dff(:,:,t));
        for j = 1:(background_roi+1+analyze_roi)
            intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
            mean_intensity(vid+1).roi(j,t) = mean(intensity);%last element in struct contains average mean data
            if j > 1 %looking at analyze rois
                if seeds_exist == 1
                   for p = 1:length(roi_pixels(j).yseed) %iterate through chosen seed pixels
                      intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).xseed(p),roi_pixels(j).yseed(p)));
                      mean_intensity(vid+1).seedpixel(j).trace(p,t) = mean(intensity); %p is seed pixel, j is roi, t is frame
                   end

                end
            end
        end
        
    end
end  
toc
%% find mean intensity for each ROI in video series
categories = fields(data);
for ex = 1:vid
%     temp = table2array(readtable(strcat(folder,path1,'\',files1(ex).name)));
%     data = temp(:,1:min(tee));
    %time_frames = reshape(data(1).(categories{ex}),[imax,jmax,min(tee)]);
    time_frames = data(1).(categories{ex});
    
    for t = 1:size(time_frames,3)
        currentframe = squeeze(time_frames(:,:,t));
        for j = 1:(background_roi+1+analyze_roi)
            intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
            mean_intensity(ex).roi(j,t) = mean(intensity);%contains data from each roi
            
            if j > 1 %looking at analyze rois
                if seeds_exist == 1
                   for p = 1:length(roi_pixels(j).yseed) %iterate through chosen seed pixels
                      intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).xseed(p),roi_pixels(j).yseed(p)));
                      mean_intensity(ex).seedpixel(j).trace(p,t) = mean(intensity); %p is seed pixel, j is roi, t is frame
                   end

                end
            end
            
        end

    end
    
end

%% plot mean DFF traces for each ROI
close all;
plot_color = [0 0 1; 0 1 0; 1 0 0];
legend_names = {'blue channel','green channel','hemo correct 1'};
name2 = {};
for k = 1:vid+1 %iterate through all video sequences + avg dff
    figure_count = 1;
    for j = background_roi+2:size(mean_intensity(1).roi,1) %iterate through rois (first few are background ROIs)
        %first j index is average dff trace
        figure(j)
%         if k == vid+1
%             plot_color = [1 0 0]; %red
%             %mean_intensity(k).roi(j,:) = smooth(mean_intensity(k).roi(j,:),3);
%         else
%             plot_color = [0.8 0.8 0.8];
%         end
        
        [val,~] = max(mean_intensity(k).roi(j,3:end-5));
        running_maxval(k,j-background_roi-1) = val;
        
        [val,~] = min(mean_intensity(k).roi(j,3:end-5));
        running_minval(k,j-background_roi-1) = val;
        
        endvec = size(mean_intensity(k).roi,2);
        if k ~= vid+1 %blue and green, turn F/Fo into DFF
            DFF = 0;
        else
            DFF = 0;%hemo is already DFF
        end
        
        if k == 1
            plotvar = smooth(mean_intensity(k).roi(j,3:endvec-5)-DFF,5);
        elseif k == 2
            plotvar = smooth(mean_intensity(k).roi(j,3:endvec-5)-DFF,5);
        else
            plotvar = smooth(mean_intensity(k).roi(j,3:endvec-5)-DFF,5);
        end
        
        plot(time(1,3:endvec-5),100*plotvar,'Color',plot_color(k,:));
        hold on
        %ylim([-10,10])
        
        if seeds_exist == 1
            for q = 1:size(mean_intensity(k).seedpixel(j).trace,1)
               figure(size(mean_intensity(1).roi,1) + figure_count)
               
               if k == 1
                   plotvar = smooth(mean_intensity(k).seedpixel(j).trace(q,3:endvec-5),5);
               elseif k == 2
                   plotvar = smooth(mean_intensity(k).seedpixel(j).trace(q,3:endvec-5),5);
               elseif k == 3
                   plotvar = smooth(mean_intensity(k).seedpixel(j).trace(q,3:endvec-5),5);
               end

               plot(time(1,3:endvec-5),100*plotvar,'Color',plot_color(k,:))
               hold on
               titletxt = strcat('DFF For Seedpixel',{' '},num2str(q)',{' '},'Within ROI',{' '},num2str(j));
               name2{figure_count} = titletxt{1};
               figure_count = figure_count + 1;
               %ylim([-15,5])
            end
            
        end
        
        
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
figure_count = 1;
for j = background_roi+2:size(mean_intensity(1).roi,1)
    figure(j)
    %axis([0 v.Duration+0.4 100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)])
    xlabel('Time (s)')
    ylabel('DFF (%)')
    title(strcat('Mean DFF For ROI ',num2str(j-background_roi)))
    hold on
%     line([stim_time stim_time],[100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--')
%     line([stim_time+stim_duration stim_time+stim_duration],[100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--') 
    legend(legend_names)
    box off
    set(gcf,'color','w');
    set(gca,'FontName','Arial','FontSize',14,'LineWidth',1)
    saveas(figure(j),strcat(input_folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials.jpeg'))
    saveas(figure(j),strcat(input_folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials.fig'))
    saveas(figure(j),strcat(input_folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials'),'epsc')
%     saveas(figure(j),strcat(folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials'),'epsc')
    
    if seeds_exist == 1
        for q = 1:size(mean_intensity(k).seedpixel(j).trace,1)
           figure(size(mean_intensity(1).roi,1) + figure_count)

           xlabel('Time (s)')
           ylabel('DFF (%)')
           titletxt = strcat('DFF For Seedpixel',{' '},num2str(q)',{' '},'Within ROI',{' '},num2str(j));
           title(name2{figure_count})
           
           legend(legend_names)
           box off
           set(gcf,'color','w');
           set(gca,'FontName','Arial','FontSize',14,'LineWidth',1)
           saveas(figure(size(mean_intensity(1).roi,1) + figure_count),strcat(input_folder,paths{1},'\ROI_',num2str(j),' dff_seedpixel',num2str(q),'.jpeg'))
           saveas(figure(size(mean_intensity(1).roi,1) + figure_count),strcat(input_folder,paths{1},'\ROI_',num2str(j),' dff_seedpixel',num2str(q),'.fig'))
           saveas(figure(size(mean_intensity(1).roi,1) + figure_count),strcat(input_folder,paths{1},'\ROI_',num2str(j),' dff_seedpixel',num2str(q)),'epsc')
           figure_count = figure_count + 1;
        end

    end
end

%% write roi dff
close all;
roi_dff = 1;

desired_data = {'blue','green','hemo'}; %choose desired roi data writers
%desired_data = {'green'};
a = 0;
b = 0.006;
if roi_dff == 1
    for p = 1:size(desired_data,2)
        mkdir(strcat(input_folder,paths{1},'\time series evolution','\',desired_data{p}));
        for j = 1:background_roi + analyze_roi + 1
            %z2 = VideoWriter(strcat(folder,paths{1},'\',desired_data{p},'_dffavgROI',num2str(j),'_actualsize.avi'));
            %z2.FrameRate = v.FrameRate;
            %open(z2);
            bw_brain_img = imread(strcat(input_folder,' bwim2.jpeg'));%read image to get image size (imax*jmax)
            del = 0; %color addition

            for t = 1:min(tee)
                if t_saveframe(1) <= time(t) && time(t) <= t_saveframe(end)
                    switch desired_data{p}
                        case 'blue'
                            %currentframe = reshape(data.blue(:,t),[imax,jmax]);%squeeze to 2d
                            currentframe = squeeze(data.blue(:,:,t));
                            color_arg = 'bluevideo';

                        case 'green'
                            %currentframe = reshape(data.green(:,t),[imax,jmax]);%squeeze to 2d
                            currentframe = squeeze(data.green(:,:,t));
                            color_arg = 'greenvideo';

                        case 'hemo'
                            %currentframe = squeeze(hemocorr_dff(:,:,t));%squeeze to 2d
                            currentframe = squeeze(hemocorr_dff(:,:,t));
                            color_arg = 'hemocorrect';
                            
                    end
                    %currentframe = squeeze(hemocorr_dff(:,:,t));%squeeze to 2d
                    [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

                    single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
                    [colorimage] = colorimage_gen(single_frame,newcolormap);

                    
                    for k = 1:3 %create video with masked image applied, if desired
                        if j == 1
                            if p == 3
                                %%
                                colorimage(sub2ind(size(colorimage),mask_imgpixels.(color_arg)(2).excluderoi.ynot_pixels,mask_imgpixels.(color_arg)(2).excluderoi.xnot_pixels,k*ones(size(mask_imgpixels.(color_arg)(2).excluderoi.xnot_pixels,1),1))) = 1;
                            else
                                colorimage(sub2ind(size(colorimage),mask_imgpixels.(color_arg)(2).ynot_pixels,mask_imgpixels.(color_arg)(2).xnot_pixels,k*ones(size(mask_imgpixels.(color_arg)(2).ynot_pixels,1),1))) = 1;
                                colorimage(sub2ind(size(colorimage),mask_imgpixels.(color_arg)(1).excluderoi.y_pixels,mask_imgpixels.(color_arg)(1).excluderoi.x_pixels,k*ones(size(mask_imgpixels.(color_arg)(1).excluderoi.y_pixels,1),1))) = 1;
                            end
                        else
                            colorimage(sub2ind(size(colorimage),roi_pixels(j).ynot_pixels,roi_pixels(j).xnot_pixels,k*ones(size(roi_pixels(j).ynot_pixels,1),1))) = 1;
                        end
                    end
                    
                    imwrite(colorimage,strcat(input_folder,paths{1},'\time series evolution\',desired_data{p},'\dffavgROI',num2str(j),'_time_evolution_',num2str(time(t)),' sec.jpeg')) 
                    
                    tempdata = currentframe;
                    if j == 1
                        if p == 3
                            tempdata(sub2ind(size(tempdata),mask_imgpixels.(color_arg)(2).excluderoi.ynot_pixels,mask_imgpixels.(color_arg)(2).excluderoi.xnot_pixels)) = NaN;
                        else
                            tempdata(sub2ind(size(tempdata),mask_imgpixels.(color_arg)(2).ynot_pixels,mask_imgpixels.(color_arg)(2).xnot_pixels)) = NaN;
                            tempdata(sub2ind(size(tempdata),mask_imgpixels.(color_arg)(1).excluderoi.y_pixels,mask_imgpixels.(color_arg)(1).excluderoi.x_pixels)) = NaN;
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
                    imwrite(frame.cdata,strcat(input_folder,paths{1},'\time series evolution\',desired_data{p},'\LARGEdffavgROI',num2str(j-1),'_time_evolution_',num2str(time(t)),' sec.jpeg'));
                    
                end

            end
            %close(z2)

        end
        
    end
    return
end
end