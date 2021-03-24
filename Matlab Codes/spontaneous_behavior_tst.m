% Biosensing and Biorobotics Laboratory
% Daniel Surinach March 2021
% analysis of open field, free behavior that contains calcium and
% reflectance channel measurements for a hemodynamic correction. 
% Requres a behavior scoring file, alternately pulsing calcium/hemo video
% and the list of final frames that are common between the calcium/hemo
% video and the behavior camera (calculated previously)

%% Variable initialization

clear;
clc;
close all;
f = findall(0,'type','figure','tag','TMWWaitbar');
delete(f)%delete all waitbar!

% tmin1 = 8;    % start frame
% tmin2 = 12;   % end frame
%frac = 1;
tstart = 5; %start frame for F0 
tend = 'full'; %end frame, full = last frame
fil = 5;      % spatial filtering pixel radius (integer), LP used 2
fscale = 2;   % filtering weight (larger number = greater weight)
bin = 0.1;     % binning fraction (0 to 1, 1 = no binning)
write_colorvid = 0; %choose whether you want a dff movie (slows down code)
%Input folder directory and output folder where you want behavior scoring
%plots to save
folder = 'H:\spontaneous behavior\1-28-20_trial3\segmented_spontaneous_videos\Analysis\aug_col\';%analysis folder 
folder2 = 'E:\spontaneous behavior\Behavior Scoring\Paper revision\Scoring Summary\Behavior Scoring Plots\';%behavior scoring video folder
vid_files = dir(fullfile(folder,'*.avi'));
vid = length(vid_files);
behaviors = {'moving_frames','still_frames','touch_frames','notouch_frames','grooming_frames','rearing_frames'};%total list of behaviors
desired_behaviors = [1,2]; %desired behaviors from above
paths = {'General Info','DFF Excel','Pixel Correlations','DFF Plots','DFF Movie'};
%sub_categories = {'moving_frames','still_frames';'touch_frames','notouch_frames'};

scoring = struct('time',{});%,'frames',{},'moving_frames',{},'still_frames',{},'touch_frames',{});
scoring_name = dir(fullfile(folder,'*.xlsx'));
scoring(1).time = xlsread(strcat(folder,scoring_name.name));
for j = 1:length(desired_behaviors)
    scoring(1).(behaviors{desired_behaviors(j)}).time = scoring(1).time(:,desired_behaviors(j));
    scoring(1).(behaviors{desired_behaviors(j)}).fullframes = []; %initialize frame
end
    

rawvideo = VideoReader(strcat(folder,vid_files(1).name)); %blue/green unsegmented video
led_warmuptime = 120; %LED warmup time in seconds
fps = round(rawvideo.FrameRate);
led_warmuptime = round(led_warmuptime * fps); %get total frames to get to LED warmup time, add 1 to be outside of LED warmup frame

%scoring(1).frames = [];
set_fpsvec = ones(fps,1);
%initalize scoring structure that contains scoring for different behaviors
for j = 1:length(desired_behaviors)
    for i = 1:size(scoring(1).time,1)
        temp = scoring(1).(behaviors{desired_behaviors(j)}).time(i) * set_fpsvec;
        scoring(1).(behaviors{desired_behaviors(j)}).fullframes = [scoring(1).(behaviors{desired_behaviors(j)}).fullframes;temp];%vector of elements in time list concatenated into frames 
%     if j == 1
%         scoring_frames = [scoring_frames,scoring(1).time(j):scoring(1).time(j)+fps];
%     else
%         scoring_frames = [scoring_frames,scoring_frames(end):scoring_frames(end)+fps]; 
%     end
    end
    
end
%scoring_frames = unique(scoring_frames)';

v = VideoReader(strcat(folder,vid_files(1).name));
while hasFrame(v)
    im = readFrame(v);
    imshow(im)
    prompt = {'Is this a blue or green frame? (b/g)'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'Blue and green frame selection';
    user_input = inputdlg(prompt,dlgtitle);
    
    break
end
close;


if strcmp(user_input{1},'b')==1
    mod_operator = 0;%start consecutive frame search with even incremenets if first frame is blue
    %i.e. pairs are 1-2,3-4,5-6ect
    
else
    mod_operator = 1;%start consecutive frame search with odd increments
    %i.e. pairs are 2-3,4-5,6-7
    
end

meso_beh_data = load(strcat(folder,'meso_beh_commonframes.mat'));
meso_beh_frames = meso_beh_data.final_frames; %final common frames (no frame drop) between meso and behavior cameras
%% final scoring vector used for analysis 
%final list of frames that are common between the behavior scoring and the
%available frames that are not dropped and found in both the cmos and
%behavior camera. The scoring struct has nested variables for each
%behavior and is used for seed pixel map generation and dF/F plots
for j = 1:length(desired_behaviors) %iterate through behaviors 
    %find consensus in at lesat 3/4 behavior scorers to include that second
    %of scored data (1 sec = 30 frames of data here at 30 fps)
    scoring(1).(behaviors{desired_behaviors(j)}).scoredframes = find(scoring(1).(behaviors{desired_behaviors(j)}).fullframes >= 3) + led_warmuptime;
    scoring(1).(behaviors{desired_behaviors(j)}).unscoredframes = find(scoring(1).(behaviors{desired_behaviors(j)}).fullframes == 0) + led_warmuptime;
    
    printtext = strcat('Detected %4.4f ',{' '},behaviors{desired_behaviors(j)},' from manual scoring\n');
    fprintf(printtext{1},length(scoring(1).(behaviors{desired_behaviors(j)}).scoredframes));
    
    scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect = intersect(meso_beh_frames,[scoring(1).(behaviors{desired_behaviors(j)}).scoredframes;scoring(1).(behaviors{desired_behaviors(j)}).unscoredframes]);
    
    
    scoring(1).(behaviors{desired_behaviors(j)}).final_frames = [];
    for k = 2:length(scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect)
        if mod(scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect(k),2) == mod_operator && scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect(k) == scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect(k-1)+1
           scoring(1).(behaviors{desired_behaviors(j)}).final_frames = [scoring(1).(behaviors{desired_behaviors(j)}).final_frames,scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect(k),scoring(1).(behaviors{desired_behaviors(j)}).vid_intersect(k-1)];%populate final vector

        end

    end
    scoring(1).(behaviors{desired_behaviors(j)}).final_frames = sort(scoring(1).(behaviors{desired_behaviors(j)}).final_frames)' - led_warmuptime;%this vector contains all the usable frames for the data set
    
    scoring(1).(behaviors{desired_behaviors(j)}).frame_color = struct('blue',{},'green',{}','black',{},'user_input',{});
    scoring(1).(behaviors{desired_behaviors(j)}).frame_color(1).userinput = user_input{1};

    if mod(scoring(1).(behaviors{desired_behaviors(j)}).final_frames(1),2) == 0 %if first frame is even
        scoring(1).(behaviors{desired_behaviors(j)}).frame_color.green = scoring(1).(behaviors{desired_behaviors(j)}).final_frames(logical(mod(scoring(1).(behaviors{desired_behaviors(j)}).final_frames,2)));
        scoring(1).(behaviors{desired_behaviors(j)}).frame_color.blue = setdiff(scoring(1).(behaviors{desired_behaviors(j)}).final_frames,scoring(1).(behaviors{desired_behaviors(j)}).frame_color.green);

    else %else frame is odd
        scoring(1).(behaviors{desired_behaviors(j)}).frame_color.blue = scoring(1).(behaviors{desired_behaviors(j)}).final_frames(logical(mod(scoring(1).(behaviors{desired_behaviors(j)}).final_frames,2)));
        scoring(1).(behaviors{desired_behaviors(j)}).frame_color.green = setdiff(scoring(1).(behaviors{desired_behaviors(j)}).final_frames,scoring(1).(behaviors{desired_behaviors(j)}).frame_color.blue);
    end

    %downsample final scored list since it is scored relative to the entire
    %movie, which includes the LED warmup time. This is done to preserve
    %the list of common frames relative to the global timestamp file
    %originally stored from the Miniscope cmos software
    scoring(1).(behaviors{desired_behaviors(j)}).downsample = scoring(1).(behaviors{desired_behaviors(j)}).scoredframes - led_warmuptime;
    
    scoring(1).(behaviors{desired_behaviors(j)}).downsample = intersect(scoring(1).(behaviors{desired_behaviors(j)}).downsample,scoring(1).(behaviors{desired_behaviors(j)}).frame_color.blue);

    for k = 1:length(scoring(1).(behaviors{desired_behaviors(j)}).downsample)%doesnt matter if blue or green taken, since only every other frame considered
        scoring(1).(behaviors{desired_behaviors(j)}).ind(k,1) = find(scoring(1).(behaviors{desired_behaviors(j)}).downsample(k) == scoring(1).(behaviors{desired_behaviors(j)}).frame_color.blue);
    end
    
    fprintf('After comparing to frame drops in miniscope CMOS and behavior cam and downsampling to %2.0fFPS...\n',v.FrameRate/2)
    printtext = strcat('Detected %4.4f ',{' '},behaviors{desired_behaviors(j)},' from manual scoring\n');
    fprintf(printtext{1},length(scoring(1).(behaviors{desired_behaviors(j)}).ind));
    
    figure(j)
    scoring(1).(behaviors{desired_behaviors(j)}).summary = zeros(size(scoring.(behaviors{desired_behaviors(j)}).fullframes));%combine behavior frames for plotting
    scoring(1).(behaviors{desired_behaviors(j)}).summary(scoring(1).(behaviors{desired_behaviors(j)}).downsample) = 1; %assign scored variable to 1
    plot(scoring(1).(behaviors{desired_behaviors(j)}).summary)
    xlabel('Frame Count')
    ylabel('Scoring Criteria')
    titletxt = strsplit(behaviors{desired_behaviors(j)},'_');
    titletxt = strcat('Behavior Scoring Frames For',{' '},titletxt{1},{' '},titletxt{2},{' '},'Behavior');
    title(titletxt{1});
    ylim([0 1.5])
    
    saveas(figure(j),strcat(folder,titletxt{1},'.jpeg'));
    saveas(figure(j),strcat(folder,titletxt{1},'.fig'));
    saveas(figure(j),strcat(folder,titletxt{1}),'epsc');
    
    tempvar = scoring(1).(behaviors{desired_behaviors(j)}).summary;
    tempname = strsplit(scoring_name.name,'.xlsx');
    mkdir(strcat(folder2,behaviors{desired_behaviors(j)}));
    titletxt = strcat(folder2,behaviors{desired_behaviors(j)},'\',titletxt{1},{' '},'For Mouse',{' '},tempname{1});
    save(strcat(titletxt{1},'.mat'),'tempvar','-v7.3');%save to.mat file for further editing
    
    saveas(figure(j),strcat(titletxt{1},'.jpeg'));
    saveas(figure(j),strcat(titletxt{1},'.fig'));
    saveas(figure(j),titletxt{1},'epsc');
    
    tempvar = scoring(1).(behaviors{desired_behaviors(j)}).downsample;
    tempname = strsplit(scoring_name.name,'.xlsx');
    titletxt = strsplit(behaviors{desired_behaviors(j)},'_');
    titletxt = strcat('Behavior Scoring Frames For',{' '},titletxt{1},{' '},titletxt{2},{' '},'Behavior');
    titletxt = strcat(folder2,behaviors{desired_behaviors(j)},'\',titletxt{1},{' '},'Downsampled For Mouse',{' '},tempname{1});
    save(strcat(titletxt{1},'.mat'),'tempvar','-v7.3');%save to.mat file for further editing
    
    
    
    
end
close all;

%% initialize the free behavior data parameters
tee = nan(vid,1);
 

prompt = {'Do you wish to load data from a previous analysis? Enter foldername or n'};
dlgtitle = 'Load Old Data Prompt';
loaddata_prompt = inputdlg(prompt,dlgtitle);

if strcmp(loaddata_prompt,'n') == 1 %use root folder as dir
    subfolder_ext = '';
else
    subfolder_ext = loaddata_prompt{1};
end

folder_objects = dir(strcat(folder,subfolder_ext));        % all items in folder
foldercount = 0;

for j = 1:length(folder_objects) %check object names to see if previous analysis that can be updated exists
   if strcmp(folder_objects(j).name,paths{1}) == 1
        foldercount = foldercount + 1;
   elseif strcmp(folder_objects(j).name,paths{2}) == 1
       foldercount = foldercount + 1;
   else
        overwrite_prompt{1} = 'y';
   end
   
end

if strcmp(subfolder_ext,'') == 0 %make new folder directories and paths
    for j = 1:length(paths)
        paths{j} = strcat(subfolder_ext,'\',paths{j});
    end
end

if foldercount == 2 %2 required folders of prev analysis exist
    prompt = {'Do you wish to overwrite all the current analysis in this folder? (y/n)?'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'New Analysis Prompt';
    overwrite_prompt = inputdlg(prompt,dlgtitle);
    
    if strcmp(overwrite_prompt{1},'n')==1 %don't wish to completely overwrite analysis
        prompt = {'Do you wish to update some analysis or run another iteration? (update/iteration)?'};%let user choose if blue or green frame (manual segmentation)
        dlgtitle = 'Update Analysis Prompt';
        update_prompt = inputdlg(prompt,dlgtitle);
        
        prompt = {'Update mask? (y/n)','Update ROIs?(y/n)','Update raw trace data?(y/n)','Update binning?(y/n)','Refilter data?(y/n)'};%let user choose if blue or green frame (manual segmentation)
        %update mask draws mask again
        %update rois draws rois again
        %update bin runs spatial and temporal filter again at new window
        %update filter runs spatial and temp filters again at new parameter
        dlgtitle = 'Update Analysis Prompt';
        update_vars = inputdlg(prompt,dlgtitle);
        
    else %overwrite all analysis in folder
        update_vars{1} = 'y';
        update_vars{2} = 'y';
        update_vars{3} = 'y';
        update_vars{4} = 'y';
        update_vars{5} = 'y';
    end
elseif foldercount == 0 %no analysis done
    overwrite_prompt{1} = 'y';
    update_vars{1} = 'y';
    update_vars{2} = 'y';
    update_vars{3} = 'y';
    update_vars{4} = 'y';
    update_vars{5} = 'y';
end



count = 1;
if strcmp(overwrite_prompt{1},'y') == 1 || strcmp(update_prompt{1},'update')==1%if new or updated analysis desired
    
    for p = 1:size(paths,2)
       mkdir(strcat(folder,paths{p}));
       if strcmp(paths{p},'Pixel Correlations') == 1
           mkdir(strcat(folder,paths{p},'\Circle Seed Pixel Points'));
       end
    end
    save(strcat(folder,paths{1},'\scoring_summarydata.mat'),'scoring')

    
    v = VideoReader(strcat(folder,vid_files(1).name));            % reads video
    k = 0;
    wt=waitbar(0,'Extracting Frames');%progress bar to see how code processsteps=len;
    steps=round(v.Duration*v.Framerate);%total frames
    %imseq = struct('blueframes',{},'greenframes',{},'meanblueframe',{},'meangreenframe',{});
    for p = 1:length(desired_behaviors)
        imseq.(behaviors{desired_behaviors(p)}).blueindex = 1;
        imseq.(behaviors{desired_behaviors(p)}).greenindex = 1;
        imseq.(behaviors{desired_behaviors(p)}).teecount = 1;
    end
%     blueindex = 1;
%     greenindex = 1;
    %% calculate analysis from scratch or batch-load analysis in .csv files
    if strcmp(update_vars{3},'y') == 1 || strcmp(update_vars{4},'y') == 1%recompute raw trace data
        fprintf('\nExtracting video sequence data\n')
        tic
        while hasFrame(v)                       % loops through video frames
            im = readFrame(v); 
            k = k+1;
            for p = 1:length(desired_behaviors)%extract video sequence for each behavior separately
                if strcmp(update_vars{1},'y') == 1 || strcmp(update_vars{2},'y') == 1
                    %draw a mask around the brain and exclude vasculature
                    %can choose how often to draw this or load
                    if k == scoring(1).(behaviors{desired_behaviors(p)}).frame_color.blue(1) && p == 1%only need to draw ROIs once though
                        [mask_imgpixels.bluedff,roi_pixels,background_roi,analyze_roi] = roi_gen(im,folder,paths,v,bin,scoring(1).(behaviors{desired_behaviors(1)}).frame_color,update_vars{1},'blue',update_vars{2},update_vars{4});
                    elseif k == scoring(1).(behaviors{desired_behaviors(p)}).frame_color.green(1) && p == 1
                        close all;
                        [mask_imgpixels.greendff,~,~,~] = roi_gen(im,folder,paths,v,bin,scoring(1).(behaviors{desired_behaviors(1)}).frame_color,update_vars{1},'green',update_vars{2},update_vars{4});

                        %% intersect between blue and green selected FOVs for hemodynamic correction
                        im2 = double(imresize(rgb2gray(im),bin));
                        tmp = zeros(size(im2));
                        tmp(sub2ind(size(tmp),mask_imgpixels.bluedff(2).y_pixels,mask_imgpixels.bluedff(2).x_pixels)) = im2(sub2ind(size(im2),mask_imgpixels.bluedff(2).y_pixels,mask_imgpixels.bluedff(2).x_pixels));%initialize as mask roi

                        for m = 1:size(mask_imgpixels.greendff(1).excluderoi,2)
                            tmp(sub2ind(size(tmp),mask_imgpixels.greendff(1).excluderoi(m).y_pixels,mask_imgpixels.greendff(1).excluderoi(m).x_pixels)) = 0;%assign undesired ROIs to 0
                            tmp(sub2ind(size(tmp),mask_imgpixels.bluedff(1).excluderoi(m).y_pixels,mask_imgpixels.bluedff(1).excluderoi(m).x_pixels)) = 0;%assign undesired ROIs to 0
                        end

                        tmp(sub2ind(size(tmp),mask_imgpixels.greendff(2).ynot_pixels,mask_imgpixels.greendff(2).xnot_pixels)) = 0;

                        [mask_imgpixels.hemocorrect(2).excluderoi.ynot_pixels,mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels] = find(tmp==0);
                        [mask_imgpixels.hemocorrect(2).excluderoi.y_pixels,mask_imgpixels.hemocorrect(2).excluderoi.x_pixels] = find(tmp~=0);
                        save(strcat(folder,paths{1},'\workspace.mat'),'mask_imgpixels','roi_pixels','background_roi','analyze_roi');
                    end
                    
                    if strcmp(update_vars{2},'n') == 1 && count == 1 %load ROIs
                        vars = load(strcat(folder,paths{1},'\workspace.mat'),'roi_pixels','analyze_roi','background_roi');
                        roi_pixels = vars.roi_pixels;
                        background_roi = vars.background_roi;
                        analyze_roi = vars.analyze_roi;
                        count = 2;
                    end
                    
                elseif strcmp(update_vars{1},'n') == 1 && strcmp(update_vars{2},'n') == 1 && count == 1%load mask img
                    vars = load(strcat(folder,paths{1},'\workspace.mat'),'mask_imgpixels','roi_pixels','analyze_roi','background_roi');
                    mask_imgpixels = vars.mask_imgpixels;
                    roi_pixels = vars.roi_pixels;
                    background_roi = vars.background_roi;
                    analyze_roi = vars.analyze_roi;
                    count = 2;
                    
                    if strcmp(update_vars{4},'y') == 1 %update bin, update rois
                        [mask_imgpixels.bluedff,roi_pixels,background_roi,analyze_roi] = roi_gen(im,folder,paths,v,bin,scoring(1).(behaviors{desired_behaviors(1)}).frame_color,update_vars{1},'blue',update_vars{2},update_vars{4},vars);
                    end

                end
                

                if imseq.(behaviors{desired_behaviors(p)}).blueindex <= length(scoring(1).(behaviors{desired_behaviors(p)}).frame_color.blue)
                    if k == scoring(1).(behaviors{desired_behaviors(p)}).frame_color.blue(imseq.(behaviors{desired_behaviors(p)}).blueindex)
                        imseq.(behaviors{desired_behaviors(p)}).blueframes(:,:,imseq.(behaviors{desired_behaviors(p)}).blueindex) = double(imresize(rgb2gray(im),bin,'bilinear'));
                        if imseq.(behaviors{desired_behaviors(p)}).blueindex == 1
                            imseq.(behaviors{desired_behaviors(p)}).bluedff = imresize(rgb2gray(im),bin,'bilinear');
                        end
                        imseq.(behaviors{desired_behaviors(p)}).blueindex = imseq.(behaviors{desired_behaviors(p)}).blueindex + 1;
                        
                    end
                end

                if imseq.(behaviors{desired_behaviors(p)}).greenindex <= length(scoring(1).(behaviors{desired_behaviors(p)}).frame_color.green)
                    if k == scoring(1).(behaviors{desired_behaviors(p)}).frame_color.green(imseq.(behaviors{desired_behaviors(p)}).greenindex)
                        imseq.(behaviors{desired_behaviors(p)}).greenframes(:,:,imseq.(behaviors{desired_behaviors(p)}).greenindex) = double(imresize(rgb2gray(im),bin,'bilinear'));
                        if imseq.(behaviors{desired_behaviors(p)}).greenindex == 1
                            imseq.(behaviors{desired_behaviors(p)}).greendff = imresize(rgb2gray(im),bin,'bilinear');
                        end
                        imseq.(behaviors{desired_behaviors(p)}).greenindex = imseq.(behaviors{desired_behaviors(p)}).greenindex + 1;
                    end
                end

                if mod(k,20)==0
                    waitbar(k/steps,wt,sprintf('Extracting frame data for frame %1.0f/%1.0f',k,steps))
                end
            end


        end

    %     %calculate variance of raw sequences
    %    for p = 1:size(scoring(1).time,2)
    %        for q = 1:size(sub_categories,2)
    %            imseq(1).(behaviors{p}).(strcat(sub_categories{p,q},'_var')) = reshape(var(reshape(imseq(1).(behaviors{p}).blueframes(:,:,scoring(1).(behaviors{p}).(strcat(sub_categories{p,1},'_ind'))),[size(imseq(1).(behaviors{p}).blueframes(:,:,scoring(1).(behaviors{p}).(strcat(sub_categories{p,1},'_ind'))),1)*size(imseq(1).(behaviors{p}).blueframes(:,:,scoring(1).(behaviors{p}).(strcat(sub_categories{p,1},'_ind'))),2),size(imseq(1).(behaviors{p}).blueframes(:,:,scoring(1).(behaviors{p}).(strcat(sub_categories{p,1},'_ind'))),3)]),0,2),[size(imseq(1).(behaviors{p}).blueframes(:,:,scoring(1).(behaviors{p}).(strcat(sub_categories{p,1},'_ind'))),1),size(imseq(1).(behaviors{p}).blueframes(:,:,scoring(1).(behaviors{p}).(strcat(sub_categories{p,1},'_ind'))),2)]);
    %        end
    %    end
        close(wt)
        toc
        

        for p = 1:length(desired_behaviors)%extract video sequence for each behavior separately
            imseq_fields = fields(imseq(1).(behaviors{desired_behaviors(p)})); %get objects in imseq struct for sorting
            des_fields = [4,6]; %bluegrames green frames time series
            for ex = des_fields %blue and green video (unsegmented)
                field_name = imseq_fields{ex};
                titletxt = strsplit(behaviors{desired_behaviors(p)},'_');
                titletxt = strcat('\nSaving',{' '},field_name,{' '},'raw data from',{' '},titletxt{1},{' '},titletxt{2},{' '},'to external file\n');
                fprintf(titletxt{1})      
                name = strcat('\',behaviors{desired_behaviors(p)},'_fullrawdata_',field_name);
                imax = size(imseq.(behaviors{desired_behaviors(p)}).(field_name),1);
                jmax = size(imseq.(behaviors{desired_behaviors(p)}).(field_name),2);
                tmax = size(imseq.(behaviors{desired_behaviors(p)}).(field_name),3);
                imseq.(behaviors{desired_behaviors(p)}).(field_name) = reshape(imseq.(behaviors{desired_behaviors(p)}).(field_name),[imax*jmax,tmax]);
                csvwrite(strcat(folder,paths{1},name,'.csv'),imseq.(behaviors{desired_behaviors(p)}).(field_name));
                name = strcat('\only_',behaviors{desired_behaviors(p)},'_fullrawdata_',field_name);
                csvwrite(strcat(folder,paths{1},name,'.csv'),imseq.(behaviors{desired_behaviors(p)}).(field_name)(:,scoring.(behaviors{desired_behaviors(p)}).ind));
                imseq.(behaviors{desired_behaviors(p)}).(field_name) = reshape(imseq.(behaviors{desired_behaviors(p)}).(field_name),[imax,jmax,tmax]);

                toc
            end
        end
        
    else %load raw trace csv data
        %%
        if strcmp(update_vars{1},'n') == 1 && strcmp(update_vars{2},'n') == 1%load mask img
            vars = load(strcat(folder,paths{1},'\workspace.mat'),'mask_imgpixels','roi_pixels','background_roi','analyze_roi');
            mask_imgpixels = vars.mask_imgpixels;
            roi_pixels = vars.roi_pixels;
            background_roi = vars.background_roi;
            analyze_roi = vars.analyze_roi;
        end 
        
        for p = 1:length(desired_behaviors)%extract video sequence for each behavior separately
            if p == 1
                if strcmp(update_vars{1},'y') == 1 || strcmp(update_vars{2},'y') == 1
                    im = read(v,scoring(1).(behaviors{desired_behaviors(p)}).frame_color.blue(1));
                    [mask_imgpixels.bluedff,roi_pixels,background_roi,analyze_roi] = roi_gen(im,folder,paths,v,bin,scoring(1).(behaviors{desired_behaviors(1)}).frame_color,update_vars{1},'blue',update_vars{2});
                    close all;
                    im = read(v,scoring(1).(behaviors{desired_behaviors(p)}).frame_color.green(1));
                    [mask_imgpixels.greendff,~,~,~] = roi_gen(im,folder,paths,v,bin,scoring(1).(behaviors{desired_behaviors(1)}).frame_color,update_vars{1},'green',update_vars{2});

                    %% intersect between blue and green selected FOVs for hemodynamic correction
                    im2 = double(imresize(rgb2gray(im),bin));
                    tmp = zeros(size(im2));
                    tmp(sub2ind(size(tmp),mask_imgpixels.bluedff(2).y_pixels,mask_imgpixels.bluedff(2).x_pixels)) = im2(sub2ind(size(im2),mask_imgpixels.bluedff(2).y_pixels,mask_imgpixels.bluedff(2).x_pixels));%initialize as mask roi

                    for m = 1:size(mask_imgpixels.greendff(1).excluderoi,2)
                        tmp(sub2ind(size(tmp),mask_imgpixels.greendff(1).excluderoi(m).y_pixels,mask_imgpixels.greendff(1).excluderoi(m).x_pixels)) = 0;%assign undesired ROIs to 0
                        tmp(sub2ind(size(tmp),mask_imgpixels.bluedff(1).excluderoi(m).y_pixels,mask_imgpixels.bluedff(1).excluderoi(m).x_pixels)) = 0;%assign undesired ROIs to 0
                    end

                    tmp(sub2ind(size(tmp),mask_imgpixels.greendff(2).ynot_pixels,mask_imgpixels.greendff(2).xnot_pixels)) = 0;

                    [mask_imgpixels.hemocorrect(2).excluderoi.ynot_pixels,mask_imgpixels.hemocorrect(2).excluderoi.xnot_pixels] = find(tmp==0);
                    [mask_imgpixels.hemocorrect(2).excluderoi.y_pixels,mask_imgpixels.hemocorrect(2).excluderoi.x_pixels] = find(tmp~=0);
                    save(strcat(folder,paths{1},'\workspace.mat'),'mask_imgpixels','roi_pixels','background_roi','analyze_roi');
                end
            end
            
            imseq.(behaviors{desired_behaviors(p)}).blueindex = [];
            imseq.(behaviors{desired_behaviors(p)}).greenindex = [];
            imseq.(behaviors{desired_behaviors(p)}).teecount = 1;
            imseq.(behaviors{desired_behaviors(p)}).blueframes = [];
            imseq.(behaviors{desired_behaviors(p)}).greenframes = [];
            imseq_fields = fields(imseq(1).(behaviors{desired_behaviors(p)})); %get objects in imseq struct for sorting
            imbw = imresize(rgb2gray(im),bin);
            imax = size(imbw,1);
            jmax = size(imbw,2);

            for ex = 4:5 %blue and green video (unsegmented)
                field_name = imseq_fields{ex};
                titletxt = strsplit(behaviors{desired_behaviors(p)},'_');
                titletxt = strcat('\nLoading',{' '},field_name,{' '},'raw data from',{' '},titletxt{1},{' '},titletxt{2},'\n');
                fprintf(titletxt{1})      
                name = strcat('\',behaviors{desired_behaviors(p)},'_fullrawdata_',field_name);
                imseq.(behaviors{desired_behaviors(p)}).(field_name) = table2array(readtable(strcat(folder,paths{1},name,'.csv')));
                tmax = size(imseq.(behaviors{desired_behaviors(p)}).(field_name),2);
                imseq.(behaviors{desired_behaviors(p)}).(field_name) = reshape(imseq.(behaviors{desired_behaviors(p)}).(field_name),[imax,jmax,tmax]);
            end
        end
        
    end
    
    for p = 1:length(desired_behaviors)
    imseq_fields = fields(imseq(1).(behaviors{desired_behaviors(p)})); %get objects in imseq struct for sorting
        for ex = [4,6] %blue and green video (unsegmented)

            field_name = imseq_fields{ex};
            fprintf(string(strcat('\nAnalyzing ',{' '},field_name,' video from videofile below\n')))
            vid_files.name

            imseq(1).(behaviors{desired_behaviors(p)}).imax = size(imseq(1).(behaviors{desired_behaviors(p)}).(field_name),1);                   % dimension height
            imseq(1).(behaviors{desired_behaviors(p)}).jmax = size(imseq(1).(behaviors{desired_behaviors(p)}).(field_name),2);                   % dimension width
            imseq(1).(behaviors{desired_behaviors(p)}).tmax = size(imseq(1).(behaviors{desired_behaviors(p)}).(field_name),3);                   % dimension time
            

            imseq(1).(behaviors{desired_behaviors(p)}).(strcat(field_name,'_var')) = var(imseq.(behaviors{desired_behaviors(p)}).(field_name)(:,:,scoring.(behaviors{desired_behaviors(p)}).ind),0,3);
            figure(3);
            imagesc(imseq(1).(behaviors{desired_behaviors(p)}).(strcat(field_name,'_var')));
            %caxis([0,0.3])
            colormap(jet)
            colorbar
            titletxt = strsplit(behaviors{desired_behaviors(p)},'_');
            titletxt = strcat('Raw pixel variance',{' '},field_name,{' '},'trace for',{' '},titletxt{1},{' '},titletxt{2});
            title(titletxt{1})
            saveas(figure(3),strcat(folder,paths{1},'\',titletxt{1},'.jpeg'))
            saveas(figure(3),strcat(folder,paths{1},'\',titletxt{1},'.fig'))
            close(figure(3));
                
            %% Raw data --> detrended
            if strcmp(update_vars{3},'y') == 1 ||strcmp(update_vars{4},'y') == 1 || strcmp(update_vars{5},'y') == 1
                Tf = [];
                Rbar = [];
                titletxt = strsplit(behaviors{desired_behaviors(p)},'_');
                titletxt = strcat('\nDetrending data for',{' '},titletxt{1},{' '},titletxt{2},{' '},'behavior\n');
                fprintf(titletxt{1}) 
                %% temporally smoothen data if desired

                for i = 1:imseq(1).(behaviors{desired_behaviors(p)}).imax
                    for j = 1:imseq(1).(behaviors{desired_behaviors(p)}).jmax
                        imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(i,j,:) = smooth(squeeze(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(i,j,:)),5);
                    end
                end
                %% dff noise background test
%                 A = nan(imseq(1).(behaviors{desired_behaviors(p)}).imax,imseq(1).(behaviors{desired_behaviors(p)}).jmax,imseq(1).(behaviors{desired_behaviors(p)}).tmax);
%                 if strcmp(tend,'full') == 1
%                     tend2 = imseq(1).(behaviors{desired_behaviors(p)}).tmax;
%                 else
%                     tend2 = tend;
%                 end
%                 %av = mean(imseq(1).(behaviors{p}).(field_name)(:,:,1:round(frac*imseq(1).(behaviors{p}).tmax)),3);
%                 av = mean(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(:,:,tstart:tend2),3);
%                 zero_vals = find(av == 0);
%                 av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)
% 
%                 wt=waitbar(0,'Detrending Data');%progress bar to see how code processsteps=len;
%                 steps=imax;%total frames
%                 for i = 1:imseq(1).(behaviors{desired_behaviors(p)}).imax
%                     for j = 1:imseq(1).(behaviors{desired_behaviors(p)}).jmax
%                         temp = av(i,j);
%         %                 temp = av;
%                         A(i,j,:) = (squeeze(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(i,j,:))-temp)/temp;
%                     end
% 
%                     if mod(i,20)==0
%                         waitbar(i/steps,wt,sprintf('Detrending pixels for pixel row %1.0f/%1.0f',i,steps))
%                     end
%                 end 
%                 close(wt)
                
                %%
%                A = nan(imax,jmax,tmax);
%                av = 0;
%                [A,F] = background_noise_correction(roi_pixels,1,imseq(1).(behaviors{desired_behaviors(p)}).(field_name),field_name);

                if strcmp(tend,'full') == 1
                    tend2 = imseq(1).(behaviors{desired_behaviors(p)}).tmax;
                else
                    tend2 = tend;
                end
                %av = mean(imseq(1).(behaviors{p}).(field_name)(:,:,1:round(frac*imseq(1).(behaviors{p}).tmax)),3);
                av = mean(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(:,:,tstart:tend2),3);
                zero_vals = find(av == 0);
                av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)
                
                if strcmp(field_name,'blueframes')==1
                     %% denoise test with background pixels
%                     im_bkg = rgb2gray(im);
%                     imshow(im_bkg);
%                     hold on
%                     [xbkg,ybkg] = getpts;
%                     for i = 1:length(xbkg)
%                         plot(xbkg,ybkg,'*r')
%                     end
%                     xbkg = round(xbkg*bin);
%                     ybkg = round(ybkg*bin);
%                     
%                     figure;
%                     imshow(imbw)
%                     hold on
%                     for i = 1:length(xbkg)
%                         plot(xbkg,ybkg,'*r')
%                     end
%                     %% 
%                     noise_pix = zeros(imseq(1).(behaviors{desired_behaviors(p)}).tmax,1);
%                     for j = 1:length(xbkg)
%                         noise_pix = noise_pix + squeeze(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(ybkg(j),xbkg(j),:));
%                     end
%                     noise_pix = noise_pix/j;
%                     
%                         
%                     %%
% %                     noise_pix = zeros(imseq(1).(behaviors{desired_behaviors(p)}).tmax,1);
% %                     for j = 1:length(mask_imgpixels.bluedff(2).ynot_pixels)
% %                         noise_pix = noise_pix + squeeze(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(mask_imgpixels.bluedff(2).ynot_pixels(j),mask_imgpixels.bluedff(2).xnot_pixels(j),:));
% %                     end
% %                     noise_pix = noise_pix/j;
%                         
% %                     noise_pix = squeeze(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(1,1,:));   
%                     for i = 1:imseq(1).(behaviors{desired_behaviors(p)}).imax
%                         for j = 1:imseq(1).(behaviors{desired_behaviors(p)}).jmax
%                             imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(i,j,:) = smooth(squeeze(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(i,j,:))./noise_pix,5);
%                         end
%                     end
%                     av = mean(imseq(1).(behaviors{desired_behaviors(p)}).(field_name)(:,:,tstart:tend2),3);
%                     zero_vals = find(av == 0);
%                     av(zero_vals) = 1e-5; %change zero to small number to avoid division by zero (mostly in green frames)
                    
                    vidtype = 'bluedff';
                    calculation_type = 'Temporal Filter1'; %case switch type
                    [Tf,Rbar,A,imseq(1).(behaviors{desired_behaviors(p)}).(field_name),~,~] = filtering(av,imseq(1).(behaviors{desired_behaviors(p)}).imax,imseq(1).(behaviors{desired_behaviors(p)}).jmax,imseq(1).(behaviors{desired_behaviors(p)}).tmax,imseq(1).(behaviors{desired_behaviors(p)}).(field_name),behaviors,mask_imgpixels.(vidtype),roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi,desired_behaviors(p),0,fil,fscale,tstart,tend2,field_name);
      
                else
                    vidtype = 'greendff';
                    calculation_type = 'Temporal Filter1'; %case switch type
                    [Tf,Rbar,Apre,imseq(1).(behaviors{desired_behaviors(p)}).(field_name),~,~] = filtering(av,imseq(1).(behaviors{desired_behaviors(p)}).imax,imseq(1).(behaviors{desired_behaviors(p)}).jmax,imseq(1).(behaviors{desired_behaviors(p)}).tmax,imseq(1).(behaviors{desired_behaviors(p)}).(field_name),behaviors,mask_imgpixels.(vidtype),roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi,desired_behaviors(p),0,fil,fscale,tstart,tend2,field_name);
                    calculation_type = 'Temporal Filter2'; %case switch type
                    [Tf,Rbar,A,Apre,~,~] = filtering(av,imseq(1).(behaviors{desired_behaviors(p)}).imax,imseq(1).(behaviors{desired_behaviors(p)}).jmax,imseq(1).(behaviors{desired_behaviors(p)}).tmax,Apre,behaviors,mask_imgpixels.(vidtype),roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi,desired_behaviors(p),0,fil,fscale,tstart,tend2,field_name);
%                     calculation_type = 'Temporal Filter2'; %case switch type
%                     [Tf,Rbar,A,imseq(1).(behaviors{p}).(field_name),~,~] = filtering(av,imseq(1).(behaviors{p}).imax,imseq(1).(behaviors{p}).jmax,imseq(1).(behaviors{p}).tmax,imseq(1).(behaviors{p}).(field_name),behaviors,mask_imgpixels,roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi,p,0,fil,fscale,tstart,tend,field_name);

                end

                
                %% Apply weighted spatial filter to video
                titletxt = strsplit(behaviors{desired_behaviors(p)},'_');
                titletxt = strcat('\nSpatially filtering data for',{' '},titletxt{1},{' '},titletxt{2},{' '},'behavior\n');
                fprintf(titletxt{1}) 

                calculation_type = 'Spatial Filter';
                [~,~,~,~,A0,~] = filtering(av,imseq(1).(behaviors{desired_behaviors(p)}).imax,imseq(1).(behaviors{desired_behaviors(p)}).jmax,imseq(1).(behaviors{desired_behaviors(p)}).tmax,imseq(1).(behaviors{desired_behaviors(p)}).(field_name),behaviors,mask_imgpixels,roi_pixels,calculation_type,Tf,Rbar,background_roi,analyze_roi,desired_behaviors(p),A,fil,fscale,tstart,tend,field_name);
                
                fprintf('\nSaving data to external file\n')
                imseq(1).(behaviors{desired_behaviors(p)}).tee(imseq.(behaviors{desired_behaviors(p)}).teecount) = imseq(1).(behaviors{desired_behaviors(p)}).tmax;
                name = strcat('\',behaviors{desired_behaviors(p)},'_',field_name);
                stack = reshape(A0,[imseq(1).(behaviors{desired_behaviors(p)}).imax*imseq(1).(behaviors{desired_behaviors(p)}).jmax imseq(1).(behaviors{desired_behaviors(p)}).tmax]);
                csvwrite(strcat(folder,paths{2},name,' dff.csv'),stack);
                %save(strcat(folder,path1,name,' dff.mat'),'stack','-v7.3');
                clear A stack
                toc
                
                %% Calculate dF/F range with respect to time
                close all
                max_frame = squeeze(max(max(A0)))';
                [max_val,max_ind] = max(max_frame(1:imseq(1).(behaviors{desired_behaviors(p)}).tee(imseq.(behaviors{desired_behaviors(p)}).teecount))); %find time index for max in all frames
                imseq.(behaviors{desired_behaviors(p)}).teecount = imseq.(behaviors{desired_behaviors(p)}).teecount+1;
                figure;
                colormap('jet');
                pic = imagesc(squeeze(A0(:,:,max_ind)));
                caxis('auto')
                colorbar
                set(gca,'xtick',[])
                set(gca,'ytick',[])
                saveas(pic,strcat(folder,paths{5},name,' dffmax_stillimage.jpeg'))
                saveas(pic,strcat(folder,paths{5},name,' dffmax_stillimage.fig'))
                pause(1);
                close Figure 1

                [max_scale,max_ind] = sort(squeeze(max(max(A0)))); %get max frame in time series for an estimated DFF scale bar
                [min_scale,min_ind] = sort(squeeze(min(min(A0)))); %get min frame in time series for estimated DFF scale bar for video writing
                close all;
            else %load already filtered data
                fprintf('\nLoading DFF data\n')
                imseq(1).(behaviors{desired_behaviors(p)}).tee(imseq.(behaviors{desired_behaviors(p)}).teecount) = imseq(1).(behaviors{desired_behaviors(p)}).tmax;
                name = strcat('\',behaviors{desired_behaviors(p)},'_',field_name);
                A0 = table2array(readtable(strcat(folder,paths{2},name,' dff.csv')));
                A0 = reshape(A0,[imseq(1).(behaviors{desired_behaviors(p)}).imax,imseq(1).(behaviors{desired_behaviors(p)}).jmax,imseq(1).(behaviors{desired_behaviors(p)}).tmax]);
                
            end

            %% Output movie full colorbar
            if write_colorvid == 1
                figure;
                colorscaling_input{1} = 'n';
                %a = mean(min_scale); %take mean of entire time series and set as min value for estimated colorbar
                %b = mean(max_scale);
                a = -0.15;
                b = 0.15; %15%DFF assumption

                idealframe = max_ind(find(max_scale > b,1)); %find first instance where a frame is greater than set average

                clear A0_smooth;
                A0_smooth = reshape(smooth(A0,3),size(A0)); %smoothen data for video writing


                while strcmp(colorscaling_input{1},'n') == 1
                    dffv = VideoWriter(strcat(folder,paths{5},name,' dff.avi'));
                    dffv.FrameRate = v.FrameRate/2;
                    open(dffv);

                    time = linspace(0,v.Duration,min(tee));
                    count = 1;

                    for t = 1:size(A0_smooth,3) %introduce NaN to border outside mask for video rendering
                        tempdata = squeeze(A0_smooth(:,:,t));
                        tempdata(sub2ind(size(tempdata),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels)) = NaN;
                        A0_smooth(:,:,t) = tempdata;
                    end

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
                       saveas(pic,strcat(folder,paths{5},name,'_dffAvgMax_stillframe.jpeg'))
                       saveas(pic,strcat(folder,paths{5},name,'_dffAvgMax_stillframe.fig'))
                       close all

                       openfig(strcat(folder,paths{5},name,'\dffAvgMax_stillframe.fig'));
                       C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
                       jet_colors = C.Colormap;% get colorbar range values
                       lim = C.Limits;%get limits of colorbar

                       a = lim(1);
                       b = lim(2);
                    end

                    close(figure(1))

                    for t = 1:size(A0_smooth,3)
                        imagesc(squeeze(A0_smooth(:,:,t)));
                        axis off;
                        colormap('jet');
                        caxis([a b])
                        colorbar
                        frame = getframe(gcf);

                        if t == 1
                           C = findall(gcf,'type','ColorBar');%find colorbar in zscore projection
                           jet_colors = C.Colormap;% get colorbar range values
                           lim = C.Limits;%get limits of colorbar
                        end

                        writeVideo(dffv,frame);
                    end
                    close(dffv)

                end


                %% output movie actual size

                z2 = VideoWriter(strcat(folder,paths{5},name,' dff_actualsize.avi'));
                z2.FrameRate = v.FrameRate/2;
                open(z2);
                bw_brain_img = imread(strcat(folder,paths{1},'\ bwim2.jpeg'));%read image to get image size (imax*jmax)
                del = 0; %color addition
                wt=waitbar(0,'DFF movie actual size');%progress bar to see how code processsteps=len;
                steps=size(A0,3);%total frames
                for t = 1:size(A0,3)

                    currentframe = squeeze(A0_smooth(:,:,t));%squeeze to 2d
                    [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors);

                    single_frame = double(rgb2gray(bw_brain_img));%create imax*jmax image
                    [colorimage] = colorimage_gen(single_frame,newcolormap);

                    for k = 1:3 %create video with masked image applied, if desired
                       colorimage(sub2ind(size(colorimage),mask_imgpixels(2).ynot_pixels,mask_imgpixels(2).xnot_pixels,k*ones(size(mask_imgpixels(2).ynot_pixels,1),1))) = 1;
                    end

                    writeVideo(z2,colorimage);

                    if t == idealframe
                        imwrite(colorimage,strcat(folder,paths{5},name,'_dffAvgmax_stillframe_correctdim_fullcolor.jpeg'))
                    end
                    
                    if mod(t,20)==0
                        waitbar(t/steps,wt,sprintf('Writing DFF movie actual size, frame %1.0f/%1.0f',t,steps))
                    end

                end
                close(z2)
                close(wt)
                toc
                close all
            end
        end
    end
else %load an old data file and assign it values 
    
        prompt = {'Enter new analysis folder name'};
        dlgtitle = 'New Folder Name';
        newfolder_name = inputdlg(prompt,dlgtitle);
        mkdir(strcat(folder,newfolder_name{1}))
        for j = 1:length(paths)
            paths_tmp = strcat(newfolder_name{1},'\',paths{j});
            if j == 1 || j == 2 || j == 5
                copyfile(strcat(folder,paths{j}),strcat(folder,paths_tmp));
            else
                mkdir(strcat(folder,paths_tmp));
                if j == 3
                    mkdir(strcat(folder,paths_tmp,'\Circle Seed Pixel Points'));
                end
            end
            paths{j} = paths_tmp;
        end
        
        
%         files = dir(fullfile(strcat(folder,oldfolder_name)));%find subfolders in desired folder
%         files(ismember( {files.name}, {'.', '..'})) = [];%remove empty char
%         dirflags = [files.isdir];
%         subfolders = files(dirflags);
        vars = load(strcat(folder,paths{1},'\workspace.mat'));
        varname = fields(vars);
        for j = 1:length(varname)
            assignin('base',varname{j},vars.(varname{j}));%take struct element and assign var to it
        end

end
close all
clear A0 A0_smooth

% roi_pixels = vars.roi_pixels;

%% Hemodynamic Correction DFFblue - DFFgreen
fprintf('\nPerforming Hemodynamic Correction\n')
data = struct;
files1 = dir(fullfile(folder,paths{2},'*.csv'));
for j = 1:length(desired_behaviors)
    data(1).(behaviors{desired_behaviors(j)}).count = 1;
end

for i = 1:length(files1)%load processed csv files and compute hemodynamic correction
    files_split = strsplit(files1(i).name,{' ','.','_'});
    count = 1;
    for j = 1:length(desired_behaviors)
       behaviors_split = strsplit(behaviors{desired_behaviors(j)},'_');
       if strcmp(files_split{1},behaviors_split{1}) == 1 %if condition on name matches behavior
           data(1).(behaviors{desired_behaviors(j)}).ind(data(1).(behaviors{desired_behaviors(j)}).count) = i;%sort files by behavior type
           data(1).(behaviors{desired_behaviors(j)}).name{data(1).(behaviors{desired_behaviors(j)}).count} = files1(i).name;%store names to check files
           if strcmp(files_split{3},'blueframes') == 1%get blue frames from data
               temp = table2array(readtable(strcat(folder,paths{2},'\',files1(i).name)));
               data(1).(behaviors{desired_behaviors(j)}).bluedff = temp(:,1:min(imseq(1).(behaviors{desired_behaviors(j)}).tee));
               clear temp
           elseif strcmp(files_split{3},'greenframes') == 1%get green frames from data
               temp = table2array(readtable(strcat(folder,paths{2},'\',files1(i).name)));
               data(1).(behaviors{desired_behaviors(j)}).greendff = temp(:,1:min(imseq(1).(behaviors{desired_behaviors(j)}).tee));
%                for z = 1:size(data(1).(behaviors{j}).greendff,1)
%                    data(1).(behaviors{j}).greendff(z,:) = sgolayfilt(data(1).(behaviors{j}).greendff(z,:),5,21); 
%                end
               clear temp
           end
           data(1).(behaviors{desired_behaviors(j)}).count = data(1).(behaviors{desired_behaviors(j)}).count +1;
       end
    end
   

    
end
%%
for j = 1:length(desired_behaviors) %iterate through each behavior 
    %assign the data folder the names and values for the calcium/hemo
    %channels and the corrected behavior
    data(1).(behaviors{desired_behaviors(j)}).bluedff = reshape(data(1).(behaviors{desired_behaviors(j)}).bluedff,[imseq(1).(behaviors{desired_behaviors(j)}).imax imseq(1).(behaviors{desired_behaviors(j)}).jmax min(imseq(1).(behaviors{desired_behaviors(j)}).tee)]);
    data(1).(behaviors{desired_behaviors(j)}).greendff = reshape(data(1).(behaviors{desired_behaviors(j)}).greendff,[imseq(1).(behaviors{desired_behaviors(j)}).imax imseq(1).(behaviors{desired_behaviors(j)}).jmax min(imseq(1).(behaviors{desired_behaviors(j)}).tee)]);

    data(1).(behaviors{desired_behaviors(j)}).hemocorrect = data(1).(behaviors{desired_behaviors(j)}).bluedff - data(1).(behaviors{desired_behaviors(j)}).greendff;

    wt=waitbar(0,'Temporal Filtering Data');%progress bar to see how code processsteps=len;
    steps=imax;%total frames
    
    %generate a bandpass filter for the hemo corrected and calcium channel
    d = designfilt('bandpassiir','FilterOrder',4, ...
        'PassbandFrequency1',0.1,'PassbandFrequency2',1, ...
        'SampleRate',15,'PassbandRipple',2)

    for i = 1:imax
        for k = 1:jmax
            data(1).(behaviors{desired_behaviors(j)}).hemocorrect(i,k,:) = filtfilt(d,squeeze(data(1).(behaviors{desired_behaviors(j)}).hemocorrect(i,k,:)));
            data(1).(behaviors{desired_behaviors(j)}).bluedff(i,k,:) = filtfilt(d,squeeze(data(1).(behaviors{desired_behaviors(j)}).bluedff(i,k,:)));
        end
        if mod(i,20)==0
            waitbar(i/steps,wt,sprintf('Temporal filtering pixels for pixel row %1.0f/%1.0f',i,steps))
        end

    end
    close(wt)
%     data(1).(behaviors{j}).hemocorrect_method2 = data(1).bluedff./(1+data(1).greendff);
    titletxt = strsplit(behaviors{desired_behaviors(j)},'_');
    titletxt = strcat('\nSaving corrected data from',{' '},titletxt{1},{' '},titletxt{2},{' '},'to external file\n');
    fprintf(titletxt{1}) 
    data(1).(behaviors{desired_behaviors(j)}).hemocorrect = reshape(data(1).(behaviors{desired_behaviors(j)}).hemocorrect,[imseq(1).(behaviors{desired_behaviors(j)}).imax*imseq(1).(behaviors{desired_behaviors(j)}).jmax min(imseq(1).(behaviors{desired_behaviors(j)}).tee)]);
    titletxt = strcat(folder,paths{2},'\',behaviors{desired_behaviors(j)},{' '},'hemocorrected.csv');
    csvwrite(titletxt{1},data(1).(behaviors{desired_behaviors(j)}).hemocorrect);
    %titletxt = strcat(folder,path1,'\',behaviors{j},{' '},'average_dff.mat');
    %save(strcat(folder,path1,'\average_dff.mat'),'data','-v7.3');

    data(1).(behaviors{desired_behaviors(j)}).hemocorrect = reshape(data(1).(behaviors{desired_behaviors(j)}).hemocorrect,[imseq(1).(behaviors{desired_behaviors(j)}).imax imseq(1).(behaviors{desired_behaviors(j)}).jmax min(imseq(1).(behaviors{desired_behaviors(j)}).tee)]);
%     data(1).(behaviors{j}).bluedff = reshape(data(1).(behaviors{j}).bluedff,[imseq(1).(behaviors{j}).imax imseq(1).(behaviors{j}).jmax min(imseq(1).(behaviors{j}).tee)]);
%     data(1).(behaviors{j}).greendff = reshape(data(1).(behaviors{j}).greendff,[imseq(1).(behaviors{j}).imax imseq(1).(behaviors{j}).jmax min(imseq(1).(behaviors{j}).tee)]);

    %data(1).hemocorrect_smooth = reshape(smooth(data(1).(behaviors{j}).hemocorrect,3),size(data(1).(behaviors{j}).hemocorrect)); %smoothen data for video rendering
    toc
end

%% ROI DFF extraction for each behavior and channel
if isempty(background_roi) == 1 && isempty(analyze_roi) == 1
    background_roi = roi_pixels(1).baseroi - 1;
    analyze_roi = roi_pixels(1).analyzeroi;
end
dff_roi = struct;
field_ind = [4,5,6]; %location of desired fields i.e. bluedff, green dff, or hemodff within data struct
% field_ind = 4;
%clear dff_roi;
for p = 1:length(desired_behaviors)%iterate through behaviors
     desired_fields = fields(data.(behaviors{desired_behaviors(p)}));
     desired_fields = desired_fields(field_ind); %blue dff, hemo corrected - choose which fields you want to run pixel analyses

    for t = 1:size(data(1).(behaviors{desired_behaviors(p)}).(desired_fields{1}),3)
        for z = 1:length(desired_fields)
            currentframe = squeeze(data(1).(behaviors{desired_behaviors(p)}).(desired_fields{z})(:,:,t));
            for j = 2:(background_roi+1+analyze_roi)
                intensity = currentframe(sub2ind(size(currentframe),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels));
                dff_roi(1).(behaviors{desired_behaviors(p)})(field_ind(z)-3).roi(j,t) = mean(intensity);%last element in struct contains average mean data

                %dff_roi(1).(behaviors{p})(3).seedpixel(j).time(:,t) = currentframe(sub2ind(size(currentframe),roi_pixels(j).ynotcentroid,roi_pixels(j).xnotcentroid));

            end
        end
    end
end

%% mean DFF traces for behaviors
for k = 1:length(desired_behaviors)
    close all;
    plot_color = [0,0,1;0,1,0;1,0,0;1,0,1]; %dff blue,dff green, hemo correct (red), hemo correct 2 (magenta)
    legend_names = {'blue channel','green channel','hemo correct 1'};
    %time = (1:size(data.hemocorrect,3))*(1/v.FrameRate);
    time = 1:length(scoring.(behaviors{desired_behaviors(k)}).ind);
    clear running_maxval
    clear running_minval
    for p = 1:size(dff_roi(1).(behaviors{desired_behaviors(k)}),2) %iterate through all video sequences + avg dff

        for j = 2:size(dff_roi(1).(behaviors{desired_behaviors(k)})(p).roi,1)%background_roi+2:size(mean_intensity(1).roi,1) %iterate through rois (first few are background ROIs)
            %first j index is average dff trace
            figure(j)

            [val,~] = max(dff_roi(1).(behaviors{desired_behaviors(k)})(p).roi(j,scoring.(behaviors{desired_behaviors(k)}).ind));
            running_maxval(p,j) = val;
            %running_maxval(k,j-background_roi-1) = val;

            [val,~] = min(dff_roi(1).(behaviors{desired_behaviors(k)})(p).roi(j,scoring.(behaviors{desired_behaviors(k)}).ind));
            running_minval(p,j) = val;
            %running_minval(k,j-background_roi-1) = val;

            %mean_intensity(k).roi(j,:) = smooth(mean_intensity(k).roi(j,:),10);
            dff_roi(1).(behaviors{desired_behaviors(k)})(p).(strcat('roi_',behaviors{desired_behaviors(k)}))(j,1:length(scoring.(behaviors{desired_behaviors(k)}).ind)) = dff_roi(1).(behaviors{desired_behaviors(k)})(p).roi(j,scoring.(behaviors{desired_behaviors(k)}).ind);

            plot(100*smooth(dff_roi(1).(behaviors{desired_behaviors(k)})(p).(strcat('roi_',behaviors{desired_behaviors(k)}))(j,1:length(scoring.(behaviors{desired_behaviors(k)}).ind)),3),'Color',plot_color(p,:));
            hold on
         end

    end

    [val,ind] = max(running_maxval);
    roimax_val = val;
    roimax_ind = ind;

    [val,ind] = min(running_minval);
    roimin_val = val;
    roimin_ind = ind;

    for j = 2:size(dff_roi(1).(behaviors{desired_behaviors(k)})(p).roi)%background_roi+2:size(mean_intensity(1).roi,1)
        figure(j)
        %axis([0 time(end)+4 100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)])
        axis([0 time(end)+4 100*roimin_val(j),100*roimax_val(j)])
        xlabel('Frame Count')
        ylabel('DFF (%)')
        titletxt = strsplit(behaviors{desired_behaviors(k)},'_');
        titletxt = strcat(titletxt{1},{' '},titletxt{2});
        titletxt = strcat(titletxt{1},{' '},'DFF For ROI ',num2str(j-1));
        title(titletxt{1});
        %title(strcat('DFF For ROI ',num2str(j-background_roi)))
        legend(legend_names)
        hold on
    %         line([stim_time stim_time],[100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--')
    %         line([stim_time+stim_duration stim_time+stim_duration],[100*roimin_val(j-background_roi-1),100*roimax_val(j-background_roi-1)],'Color','black','LineStyle','--') 

        saveas(figure(j),strcat(folder,paths{4},'\',titletxt{1},'.jpeg'))
        saveas(figure(j),strcat(folder,paths{4},'\',titletxt{1},'.fig'))

    end
end


% v = VideoReader(strcat(folder,vid_files(1).name))
% [mask_imgpixels.bluedff,roi_pixels,~,~] = roi_gen(im,folder,paths,v,bin,scoring(1).(behaviors{desired_behaviors(1)}).frame_color,update_vars{1},'blue','y');
%% seed pixel correlation analyses
%moving seed pixel correlation coefficeint 
%dff_roi(1).(behaviors{p})(1).seedpixel(j).time(:,t) = currentframe(sub2ind(size(currentframe),roi_pixels(j).ynotcentroid,roi_pixels(j).xnotcentroid));
% f = findall(0,'type','figure','tag','TMWWaitbar')
% delaitbar!
close all;%delete all w
threshold_corr = 0.7; %threshold pearsons correlation number
for k = 1:length(desired_behaviors)
    titletxt = strsplit(behaviors{desired_behaviors(k)},'_');
    titletxt = strcat('\nPerforming seed pixel correlation for',{' '},titletxt{1},{' '},titletxt{2},{' '},'behavior\n');
    fprintf(titletxt{1}) 
    field_ind = [4]; %location of desired fields i.e. bluedff, green dff, or hemodff within data struct
    moving_clustercase = 'SC'; %Small cluster (SC) or moving window cluster (MC)
    close all;
    
    desired_fields = fields(data.(behaviors{desired_behaviors(k)}));
    desired_fields = desired_fields(field_ind); %blue dff, hemo corrected - choose which fields you want to run pixel analyses
    for z   = 1:length(desired_fields)%iterate through desired fields
        if k == 1
            mkdir(strcat(folder,paths{3},'\',desired_fields{z})); %make new subfolder for desired trace types
            mkdir(strcat(folder,paths{3},'\',desired_fields{z},'\whole trace'));
            mkdir(strcat(folder,paths{3},'\',desired_fields{z},'\cluster'));
            mkdir(strcat(folder,paths{3},'\',desired_fields{z},'\moving cluster'));
        end
        %% seed pixel analysis
        for j = 2:(background_roi + 1 + analyze_roi) %iterate through rois - blue trace
            %% seed pixel maps for the seeds chosen in each ROI
            %If you get dimension mismatch errors here, it is usually
            %because you have chosen an ROI or seed that is outside the
            %mask border originally drawn. you can skip that ROI (j index)
            %and continue to run this if desired
            calculation_type = 'seed pixel analysis';
  
            [dff_roi] = seedpixel_aug(roi_pixels,mask_imgpixels.(desired_fields{z}),j,scoring,behaviors,data(1).(behaviors{desired_behaviors(k)}).(desired_fields{z}),desired_behaviors(k),dff_roi,imseq,background_roi,analyze_roi,folder,paths,calculation_type,field_ind(z)-3,moving_clustercase,desired_fields{z});
            
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
            close(figure(1));
            
            %% concatenate seed pixel maps into one overlayed map
            if j == 2 %first ROI
                figure(10)
                imshow(imseq.(behaviors{desired_behaviors(k)}).(desired_fields{z}))
                hold on
                jet_colors = jet;
                color_len = length(2:(background_roi + 1 + analyze_roi));
                color_rows = round(linspace(1,size(jet_colors,1),color_len));
%                 final_idx = NaN(imax,jmax);
            end
            figure(10)
            [row_corr,col_corr] = find(dff_roi.(behaviors{desired_behaviors(k)})(field_ind(z)-3).seedpixel(j).movingcluster(1).seedmap >= threshold_corr);
            plot(col_corr,row_corr,'*','Color',jet_colors(color_rows(j-1),:))
            
%             final_idx(row_corr,col_corr) = j-1;
            

        end
        titletxt = strsplit(behaviors{desired_behaviors(k)},'_');
        titletxt = strcat(titletxt{1},{' '},titletxt{2});
        titletxt = strcat('All ROIs seed pixel map with r greater than',{' '},num2str(threshold_corr),{' '},titletxt);
        title(titletxt{1});
        saveas(figure(10),strcat(folder,paths{3},'\',desired_fields{z},'\',titletxt{1},'.jpeg'));
        saveas(figure(10),strcat(folder,paths{3},'\',desired_fields{z},'\',titletxt{1},'.fig'));
        close(figure(10));
%         tmp = reshape(final_idx,[imax,jmax]);
%         currentframe = tmp;%squeeze to 2d
%         jet_colors = jet;
%         [newcolormap] = colormap_gen(min(min(final_idx)),max(max(final_idx)),0,currentframe,jet_colors);
%         colorimage = colorimage_gen(imseq.(behaviors{desired_behaviors(k)}).(desired_fields{z}),newcolormap);
%         imshow(newcolormap);
        
        %% cross correlational matrix with all ROIs
        close all
        if analyze_roi > 1 %need 2 rois for cross corr mat
            calculation_type = 'cross correlation matrix';
            [dff_roi] = seedpixel_aug(roi_pixels,mask_imgpixels.(desired_fields{z}),j,scoring,behaviors,data(1).(behaviors{desired_behaviors(k)}).(desired_fields{z}),desired_behaviors(k),dff_roi,imseq,background_roi,analyze_roi,folder,paths,calculation_type,field_ind(z)-3,moving_clustercase,desired_fields{z});
        end
        
        
    end
    toc    
    
end

%% Save desired variables to workspace
save(strcat(folder,paths{1},'\workspace.mat'),'roi_pixels','mask_imgpixels','imseq','dff_roi','scoring','Tf','Rbar','av','background_roi','analyze_roi','bin','-v7.3')
%% DFF average map
close all;%delete all w
for k = 1:length(desired_behaviors)
    titletxt = strsplit(behaviors{desired_behaviors(k)},'_');
    titletxt = strcat('\nPerforming seed pixel correlation for',{' '},titletxt{1},{' '},titletxt{2},{' '},'behavior\n');
    fprintf(titletxt{1}) 
    field_ind = [4]; %location of desired fields i.e. bluedff, green dff, or hemodff within data struct
    moving_clustercase = 'SC'; %Small cluster (SC) or moving window cluster (MC)
    
    desired_fields = fields(data.(behaviors{desired_behaviors(k)}));
    desired_fields = desired_fields(field_ind); %blue dff, hemo corrected - choose which fields you want to run pixel analyses
    for z = 1:length(desired_fields)%iterate through desired fields
        dff_avg = mean(data(1).(behaviors{desired_behaviors(k)}).(desired_fields{z})(:,:,scoring(1).(behaviors{desired_behaviors(k)}).ind),3);
        
        figure(k);
        colormap('jet');
        pic = imagesc(dff_avg);
        caxis([-0.02,0.02])
        colorbar
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        hold off
        
        
       % [dff_avg] = dff_avgmap(roi_pixels,mask_imgpixels.(desired_fields{z}),j,scoring,behaviors,data(1).(behaviors{desired_behaviors(k)}).(desired_fields{z}),desired_behaviors(k),folder,paths,field_ind(z)-3,desired_fields{z});
    end
end






