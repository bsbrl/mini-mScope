% Biosensing and Biorobotics Laboratory
% Samantha Linn, Daniel Surinach
% February 2020
% dF/F heat mapping in stimulus experiments with averaging completed

%% Variable initialization
function stimulus_experiments(frac,fil,fscale,bin,input_folder,t_saveframe,stim_time,stim_duration)
% tmin1 = 8;    % start frame
% tmin2 = 12;   % end frame

files = dir(fullfile(input_folder,'*.avi'));
vid = length(files);
tee = nan(vid,1);


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

if strcmp(overwrite_prompt{1},'y') == 1 %if new analysis desired
    for p = 1:size(paths,2)
       mkdir(strcat(input_folder,paths{p}));
       if strcmp(paths{p},'dffGeneral') == 1
           mkdir(strcat(input_folder,paths{p},'\time series evolution'));
       end
    end
%     mkdir(strcat(folder,path1));
%     mkdir(strcat(folder,path2));
%     mkdir(strcat(folder,path3));
%     mkdir(strcat(folder,path1,'\time series evolution')) %save still shots from video at desired times

    for ex = 1:vid
        tic
        fprintf('\nAnalyzing video %1.0f/%1.0f with name below\n',ex,vid)
        files(ex).name

        %% .avi --> 3D grayscale matrix

        v = VideoReader(strcat(input_folder,files(ex).name));            % reads video

        imax = v.Height;                        % frame height in units pixels
        jmax = v.Width;                         % frame width in units pixels
        tmax = round(v.Duration*v.FrameRate);          % number of frames in video

        imseq = nan(imax,jmax,tmax);            % initializes matrix; same size as video
        k = 0;
        wt=waitbar(0,'Extracting Frames');%progress bar to see how code processsteps=len;
        steps=tmax;%total frames

        while hasFrame(v)                       % loops through video frames
            im = readFrame(v);                  % reads frame
            
            if strcmp(mask_exists,'n') == 1
               mask_imgvid = VideoWriter(strcat(input_folder,paths{1},'\mask_img.avi'));
               mask_imgvid.FrameRate = v.FrameRate;
               open(mask_imgvid);

               for j = 1:10
                   writeVideo(mask_imgvid,im);
               end
               close(mask_imgvid)

               rawvideo = VideoReader(strcat(input_folder,paths{1},'\mask_img.avi'));
               frame = 1;

               background_roi =0;
               analyze_roi = 1;
               seedpixel_roi = 'n';
               exclude_roi = 'n';

               imgname = strcat(input_folder,paths{1},'\mask_roi.jpeg');
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
               rawvideo = VideoReader(strcat(input_folder,paths{1},'\mask_img.avi'));%VideoReader(strcat(folder,files(ex).name));
               [mask_imgpixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,data,frame_color); %set user-defined rois
               close(wt2)

               mask_exists = 'y';
               close all

               imbw = imresize(rgb2gray(im),bin);
               masked_img = uint8(255*ones(size(imbw)));

               masked_img(sub2ind(size(masked_img),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = imbw(sub2ind(size(imbw),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels));
               save(strcat(input_folder,paths{1},'\mask_roi.mat'),'mask_imgpixels');

            end
            
            
            k = k+1;
            imseq(:,:,k) = double(rgb2gray(im));% converts every frame to B/W 3D matrix
            if k == 1
                imshow(imresize(rgb2gray(im),bin))
                
                if ex == 1%first video, store a copy for hemo correction (vid length, framerate)
                    vid_color = strsplit(input_folder,'\'); %get video type from folder
                    vid_color = vid_color{end-1};
                    hemo_folder = strsplit(input_folder,vid_color);%go back one root
                    hemo_folder = strcat(hemo_folder{1},'hemo');
                    mkdir(hemo_folder)

                    tmpname1 = strcat(input_folder,'bwim2',{' '},vid_color,'video.jpeg');                  
                    tmpname2 = strcat(input_folder,'bwim',{' '},vid_color,'video.jpeg');                                      
                    copyfile(strcat(input_folder,files(ex).name),hemo_folder)
                end
                imwrite(imresize(im,bin),tmpname1{1}) %store first image for overlay
                imwrite(im,tmpname2{1}); %full size image
                copyfile(tmpname2{1},hemo_folder)
                if ex == 1 
                    
                end
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

        %% Raw data --> detrended

        A = nan(imax,jmax,tmax);
        x = 1:tmax;
       % av = mean(imseq,3);
        av = mean(imseq(:,:,1:round(frac*tmax)),3);
        zero_vals = find(av == 0);
        av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)

        wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
        steps=imax;%total frames

        for i = 1:imax
            for j = 1:jmax
                temp = av(i,j);
                A(i,j,:) = (squeeze(imseq(i,j,:))-temp)/temp;
            end

            if mod(i,20)==0
                waitbar(i/steps,wt,sprintf('Detrending pixels for pixel row %1.0f/%1.0f',i,steps))
            end
        end 
        clear imseq
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

        tee(ex) = tmax;
        filename = split(files(ex).name,'.');
        name = strcat('\',filename{1});
        stack = reshape(A0,[imax*jmax tmax]);
        csvwrite(strcat(input_folder,paths{1},name,' dff.csv'),stack);
        save(strcat(input_folder,paths{1},name,' dff.mat'),'stack');
        clear A stack

        %% Calculate dF/F range with respect to time

        Range = range(A0,3);
        Max = max(A0,[],3);
        csvwrite(strcat(input_folder,paths{2},name,' dffrange.csv'),Range);
        csvwrite(strcat(input_folder,paths{3},name,' dffmax.csv'),Max);

        figure;
        colormap('jet');
        pic = imagesc(Range);
        caxis([0 .04])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        saveas(pic,strcat(input_folder,paths{2},name,' dffrange.jpeg'))
        saveas(pic,strcat(input_folder,paths{2},name,' dffrange.fig'))
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
        saveas(pic,strcat(input_folder,paths{3},name,' dffmax.jpeg'))
        saveas(pic,strcat(input_folder,paths{3},name,' dffmax.fig'))
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
        saveas(pic,strcat(input_folder,paths{3},name,' dffmax_stillimage.jpeg'))
        saveas(pic,strcat(input_folder,paths{3},name,' dffmax_stillimage.fig'))
        pause(1);
        close Figure 1

        %% Output movie
        %%
        dffv = VideoWriter(strcat(input_folder,paths{1},name,' dff.avi'));
        dffv.FrameRate = v.FrameRate;
        open(dffv);
        
        A0_smooth = zeros(size(A0));
        for i = 1:imax
            for j = 1:jmax
                A0_smooth(i,j,:) = smooth(A0(i,j,:),3);
            end
        end 
%         A0_smooth = reshape(smooth(A0,3),size(A0)); %smoothen data for video writing
        for t = 1:tmax
            tempdata = squeeze(A0_smooth(:,:,t));
            tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = 0;
            A0_smooth(:,:,t) = tempdata;
            
            im = imagesc(squeeze(A0_smooth(:,:,t))); %smoothen data during video writing
            axis off;
            colormap('jet');
            caxis([-.01 .02]);
            colorbar
            frame = getframe(gcf);
            writeVideo(dffv,frame);
        end
        close(dffv)

        %%
        C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
        jet_colors = C.Colormap;% get colorbar range values
        lim = C.Limits;%get limits of colorbar

        z2 = VideoWriter(strcat(input_folder,paths{1},name,' dff2.avi'));
        z2.FrameRate = v.FrameRate;
        open(z2);
        bw_brain_img = imread(tmpname1{1});%read image to get image size (imax*jmax)
        a = lim(1); %get colorbar limits
        b = lim(2);
        del = 0; %color addition
        for t = 1:tmax

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
end
close all


%% Average range dF/F image and dF/F movie
tic
if vid ~= 1
    
    R = nan(imax,jmax,vid);
    M = nan(imax,jmax,vid);
    runsum = zeros(imax*jmax,min(tee));
    
    files1 = dir(fullfile(input_folder,paths{1},'*.csv'));
    files2 = dir(fullfile(input_folder,paths{2},'*.csv'));
    files3 = dir(fullfile(input_folder,paths{3},'*.csv'));
    
    for ex = 1:vid
        temp = table2array(readtable(strcat(input_folder,paths{1},'\',files1(ex).name)));
        data = temp(:,1:min(tee));
        clear temp
        runsum = (1/ex).*data + ((ex-1)/ex).*runsum;
        clear data
        
        R(:,:,ex) = table2array(readtable(strcat(input_folder,paths{2},'\',files2(ex).name)));
        
        M(:,:,ex) = table2array(readtable(strcat(input_folder,paths{3},'\',files3(ex).name)));
        
    end
    runsum = reshape(runsum,[imax,jmax,min(tee)]);

    rfinal = mean(R,3);
    mfinal = mean(M,3);
    csvwrite(strcat(input_folder,paths{2},'\dffAvgRange.csv'),rfinal);
    csvwrite(strcat(input_folder,paths{3},'\dffAvgMax.csv'),mfinal);
    
    figure;
    colormap('jet');
    pic = imagesc(rfinal);
    caxis([0 .04])
    colorbar
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    saveas(pic,strcat(input_folder,paths{2},'\dffAvgRange.jpeg'))
    saveas(pic,strcat(input_folder,paths{2},'\dffAvgRange.fig'))
    pause(1);
    close Figure 1
    
    figure;
    colormap('jet');
    pic = imagesc(mfinal);
    caxis([0 .03])
    colorbar
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    saveas(pic,strcat(input_folder,paths{3},'\dffAvgMax.jpeg'))
    saveas(pic,strcat(input_folder,paths{3},'\dffAvgMax.fig'))
    pause(1);
    close Figure 1 
    
    
    stack = reshape(runsum,[imax*jmax min(tee)]);
    tmpname = strcat(input_folder,paths{1},'\average_dff',{' '},vid_color,'video.csv');
    csvwrite(tmpname{1},stack);
    
    copyfile(tmpname{1},hemo_folder);%copy .csv
    
    %save(strcat(input_folder,paths{1},'\average_dff.mat'),'stack');
    clear stack max_frame avg
    [max_val,max_ind] = max(max(max(runsum))); %find max value in average video set
    
    
    figure;
    
    colorscaling_input{1} = 'n';
    a = -0.01;
    b = 0.02;
    
%     runsum_smooth = reshape(smooth(runsum,3),size(runsum));
    runsum_smooth = zeros(size(runsum));
    for i = 1:imax
        for j = 1:jmax
            runsum_smooth(i,j,:) = smooth(runsum(i,j,:),3);
        end
    end 
    
    
    while strcmp(colorscaling_input{1},'n') == 1
        dffavg = VideoWriter(strcat(input_folder,paths{1},'\dffAvg.avi'));
        dffavg.FrameRate = v.FrameRate;
        open(dffavg);
        time = linspace(0,v.Duration,min(tee));
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

            if stim_time<=time(t) && time(t)<=stim_time+stim_duration
                if count == 1
                    start_frame = t;
                end
               avg(count) = mean(mean(squeeze(runsum_smooth(:,:,t))));
               count = count+1;
            end
        end
        [avgval,avgind] = max(avg);
        avgind = avgind + start_frame;
        
        close all;
        figure;
        colormap('jet');
        pic = imagesc(runsum_smooth(:,:,avgind));
        caxis([a b])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        
        close(dffavg)
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
        
    end
    
    
    %% write avg video with actual size
    
    z2 = VideoWriter(strcat(input_folder,paths{1},'\dffavg_actualsize_fullcolor.avi'));
    z2.FrameRate = v.FrameRate;
    open(z2);
    bw_brain_img = imread(tmpname1{1});%read image to get image size (imax*jmax)

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
        bw_brain_img = imread(tmpname1{1});%read image to get image size (imax*jmax)
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
    return
    figure(1)
    dff_maxstillframe = imread(strcat(input_folder,paths{3},'\dffAvgmax_stillframe_correctdim.jpeg'));
    imshow(masked_img)%show rgb image to overlay to color
    hold on
    pic2 = imshow(dff_maxstillframe);%overlay heat map image to rgb (to help avoid blood vessels)
    transparency_number = 0.3;
    alpha = transparency_number*ones(size(pic2));%set opacity of color image over bw image
    set(pic2,'AlphaData',alpha)

    newpic = getframe;%get fused image
    close all;
    imshow(newpic.cdata)
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

        newpic = getframe;%get fused image
        close all;
        imshow(newpic.cdata)
        
        prompt = {'Sufficient transparency of color image (y/n)?'};%let user choose if blue or green frame (manual segmentation)
        dlgtitle = 'Color Image Transparency';
        transparency_input = inputdlg(prompt,dlgtitle);
    end

    imwrite(newpic.cdata,strcat(input_folder,paths{1},name,' dffmax_stillimage_fused.jpeg'));%save fused image as jpeg
    fused_img = imread(strcat(input_folder,paths{1},name,' dffmax_stillimage_fused.jpeg'));
    fused_img = imresize(fused_img,[v.Height,v.Width]); %pixelate and resize image in case binned (easier roi extraction)
    %figure(2)
    %imshowpair(newpic.cdata,tst,'montage');
    
    fused_imgvid = VideoWriter(strcat(input_folder,paths{1},'\dff_fusedimg.avi'));
    fused_imgvid.FrameRate = v.FrameRate;
    open(fused_imgvid);
    for j = 1:50
        writeVideo(fused_imgvid,fused_img);
    end
    close(fused_imgvid)
    
    %rawvideo = VideoReader(strcat(folder,files(ex).name));
    rawvideo = VideoReader(strcat(input_folder,paths{1},'\dff_fusedimg.avi'));
    frame = 1;
    
    prompt = {'Background ROIs','Analyze ROIs (at least 1)','Draw seed pixels within ROIs? (y/n)','Exclude any ROIs from Analysis? (y/n)'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'ROI Selection For Analysis';
    roi_input = inputdlg(prompt,dlgtitle,[1 60]);
    background_roi = str2double(roi_input{1});
    analyze_roi = str2double(roi_input{2});
    seedpixel_roi = roi_input{3};
    exclude_roi = roi_input{4};
    
    imgname = strcat(input_folder,paths{1},'\roi_selection.jpeg');
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
    rawvideo = VideoReader(strcat(input_folder,paths{1},'\dff_fusedimg.avi'));%VideoReader(strcat(folder,files(ex).name));
    [roi_pixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,0,frame_color); %set user-defined rois
    
    %intensity_vector(j).brightnessblue=corrected_framemat(sub2ind(size(corrected_framemat),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
    mean_intensity = struct('roi',{});
    for t = 1:size(runsum,3)
        currentframe = squeeze(runsum(:,:,t));
        for j = 1:(background_roi+1+analyze_roi)
            intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
            mean_intensity(vid+1).roi(j,t) = mean(intensity);%last element in struct contains average mean data
        end
        
    end
end  
toc
%% find mean intensity for each ROI in video series

for ex = 1:vid
    temp = table2array(readtable(strcat(input_folder,paths{1},'\',files1(ex).name)));
    data = temp(:,1:min(tee));
    time_frames = reshape(data,[imax,jmax,min(tee)]);
    
    for t = 1:size(time_frames,3)
        count = 1;
        currentframe = squeeze(time_frames(:,:,t));
        for j = 1:(background_roi+1+analyze_roi)
            intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
            mean_intensity(ex).roi(j,t) = mean(intensity);%contains data from each roi
        end
        
        
    end

end

%% calculate max peak for each ROI within stimulus time
stim_frame_range = round((stim_time*v.FrameRate)):round(((stim_time+stim_duration)*v.FrameRate));
stim_data = struct('stim_period',{},'max_stimval',{},'ttest_data',{});

for ex = 1:size(mean_intensity,2) %include extra "vid" because avg appended to mean list above
   for j = 1:(background_roi+1+analyze_roi)
      stim_data(ex).stim_period(j,:) = mean_intensity(ex).roi(j,stim_frame_range); 
      [val,~] = max(stim_data(ex).stim_period(j,:));
      stim_data(ex).max_stimval(j,1) = val;
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
            %mean_intensity(k).roi(j,:) = smooth(mean_intensity(k).roi(j,:),3);
        else
            plot_color = [0.8 0.8 0.8];
        end
        
        [val,~] = max(mean_intensity(k).roi(j,1:end-5));
        running_maxval(k,j-background_roi-1) = val;
        
        [val,~] = min(mean_intensity(k).roi(j,1:end-5));
        running_minval(k,j-background_roi-1) = val;
        
        plot(time(1:end-5),100*mean_intensity(k).roi(j,1:end-5),'Color',plot_color);
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
    axis([0 v.Duration+0.4 100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)])
    xlabel('Time (s)')
    ylabel('DFF (%)')
    title(strcat('DFF For ROI ',num2str(j-background_roi)))
    hold on
    line([stim_time stim_time],[100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--')
    line([stim_time+stim_duration stim_time+stim_duration],[100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--') 
    
    saveas(figure(j),strcat(input_folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials.jpeg'))
    saveas(figure(j),strcat(input_folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials.fig'))
    saveas(figure(j),strcat(input_folder,paths{1},'\ROI_',num2str(j),' meandff_alltrials'),'epsc')
    
end

%% plot max value during stimulus period for each ROI
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
        
        if k == vid+1
            stim_data(k).ttest_data(k,:) = mean(stim_data(k).ttest_data(1:end-1,:));
            stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1) = stim_data(k).ttest_data(k,:);

        end
        plot(100*stim_data(k).max_stimval(ipsi_contra_roipairs(j,:),1),'-*','Color',plot_color);
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

[h,p] = ttest(stim_data(vid+1).ttest_data(1:end-1,1),stim_data(vid+1).ttest_data(1:end-1,2)); %paired ttest without including last data point (avg of videos)
stdev = std(100*stim_data(vid+1).ttest_data(1:end-1,:));

fprintf('Max DFF for all trials is %4.4f occurs for the video listed below...Check if outliers are present\n',roimax_val)
files(roimax_ind).name
fprintf('Min DFF for all trials is %4.4f occurs for the video listed below...Check if outliers are present\n',roimin_val)
files(roimin_ind).name

fprintf('P-value for paired t-test between ipsilateral and contralateral for stimulus is %4.4f\n',p)

fig_legend = {'Contralateral','Ipsilateral'};
for j = 1:size(ipsi_contra_roipairs,1)
    figure(size(mean_intensity(1).roi,1)+j)
    ylim([100*roimin_val(j),100*roimax_val(j)])
    xlim([0.8,2.2])
    set(gca,'XTick',1:length(fig_legend),'XTickLabel',fig_legend)
    ylabel('DFF (%)')
    hold on
    
    er = errorbar([1,2],100*stim_data(vid+1).ttest_data(end,:),stdev,stdev,'CapSize',20);
    er.Color = [0,0,0];
    er.LineStyle = 'none';


    saveas(figure(size(mean_intensity(1).roi,1)+j),strcat(input_folder,paths{1},'\contra_ipsi_rois_',roiname,' peakdff_stimperiod.jpeg'))
    saveas(figure(size(mean_intensity(1).roi,1)+j),strcat(input_folder,paths{1},'\contra_ipsi_rois_',roiname,' peakdff_stimperiod'),'epsc')
    saveas(figure(size(mean_intensity(1).roi,1)+j),strcat(input_folder,paths{1},'\contra_ipsi_rois_',roiname,' peakdff_stimperiod.fig'))
   
end
end