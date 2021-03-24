%% Mask and ROI pixel generation function
%updated to shorten code in main function
%allows ROIs to be drawn or loaded from previous analyses
%Daniel Surinach March 2020
%%
function [mask_imgpixels,roi_pixels,background_roi,analyze_roi] = roi_gen(im,folder,paths,v,bin,frame_color,update_mask,mask_type,update_roi,drawroi)
%% mask image selection
frame = 1;
if strcmp(update_mask,'y') == 1
    imwrite(im,strcat(folder,paths{1},'\ bwim.jpeg')); %store unbinned first image
    imwrite(imresize(im,bin),strcat(folder,paths{1},'\ bwim2.jpeg')) %store binned first image for overlay

    mask_imgvid = VideoWriter(strcat(folder,paths{1},'\mask_img.avi'));
    mask_imgvid.FrameRate = v.FrameRate;
    open(mask_imgvid);

    for j = 1:10
       writeVideo(mask_imgvid,im);
    end
    close(mask_imgvid)

    rawvideo = VideoReader(strcat(folder,paths{1},'\mask_img.avi'));

    imshow(im)
    prompt = {'Exclude any ROIs from Analysis? (y/n)'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'ROI Selection For Analysis';
    roi_input = inputdlg(prompt,dlgtitle,[1 60]);

    background_roi =0;
    analyze_roi = 1;
    seedpixel_roi = 'n';
    exclude_roi = roi_input{1};
    close(figure(1));
    
    imgname = strcat(folder,paths{1},'\mask_roi.jpeg');
    stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
    casenum = 0;
    data = 0;

    if isempty(frame_color.black)==1
       frame_color.black = 0;
    end

    wt2 = waitbar(0,'Draw a mask over the brain region (exclude non GCaMP areas)');

    %Draw mask over the brain and exclude any points of glare or large
    %vasculature from the mask
    rawvideo = VideoReader(strcat(folder,paths{1},'\mask_img.avi'));
    [mask_imgpixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,data,frame_color); %set user-defined rois
    close(wt2)
    close all

    imbw = imresize(rgb2gray(im),bin);
    masked_img = uint8(255*ones(size(imbw)));

    masked_img(sub2ind(size(masked_img),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = imbw(sub2ind(size(imbw),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels));
    save(strcat(folder,paths{1},'\mask_roi.mat'),'mask_imgpixels');

    im2 = double(imresize(rgb2gray(im),bin));
    tmp = zeros(size(im2));
    tmp(sub2ind(size(tmp),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels)) = im2(sub2ind(size(im2),mask_imgpixels(2).y_pixels,mask_imgpixels(2).x_pixels));%initialize as mask roi

    for k = 1:size(mask_imgpixels(1).excluderoi,2)
        tmp(sub2ind(size(tmp),mask_imgpixels(1).excluderoi(k).y_pixels,mask_imgpixels(1).excluderoi(k).x_pixels)) = 0;%assign undesired ROIs to 0
    end

    [mask_imgpixels(2).excluderoi.ynot_pixels,mask_imgpixels(2).excluderoi.xnot_pixels] = find(tmp == 0);%find undesired ROIs and store
    [mask_imgpixels(2).excluderoi.y_pixels,mask_imgpixels(2).excluderoi.x_pixels] = find(tmp~=0);%find desired ROIs and store
    
    im2 = rgb2gray(im);
    tmp = zeros(size(im2));
    tmp(sub2ind(size(tmp),mask_imgpixels(2).fullypixel,mask_imgpixels(2).fullxpixel)) = im2(sub2ind(size(im2),mask_imgpixels(2).fullypixel,mask_imgpixels(2).fullxpixel));%initialize as mask roi

    for k = 1:size(mask_imgpixels(1).excluderoi.fullypixel,2)
        tmp(sub2ind(size(tmp),mask_imgpixels(1).excluderoi(k).fullypixel,mask_imgpixels(1).excluderoi(k).fullxpixel)) = 0;%assign undesired ROIs to 0
    end

    [mask_imgpixels(2).excluderoi.not_fullypixel,mask_imgpixels(2).excluderoi.not_fullxpixel] = find(tmp == 0);%find undesired ROIs and store
    [mask_imgpixels(2).excluderoi.fullypixel,mask_imgpixels(2).excluderoi.fullxpixel] = find(tmp~=0);%find desired ROIs and store
else %load mask
    vars = load(strcat(folder,paths{1},'\workspace.mat'),'mask_imgpixels');
    mask_imgpixels = vars.mask_imgpixels;
    switch mask_type
        case 'blue'
            mask_imgpixels = mask_imgpixels.bluedff; %extract blue mask
            roi_pixels = []; 
            analyze_roi = [];
            background_roi = [];
        case 'green'
            mask_imgpixels = mask_imgpixels.greendff; %extract green mask
            roi_pixels = []; 
            analyze_roi = [];
            background_roi = [];
 
    end
end
%% roi selection
if strcmp(update_roi,'y') == 1
    switch mask_type
        case 'blue'
            %%
            im_overlay = imread(strcat(folder,'brainatlas_meso_overlay.jpg'));
            im_overlay = imresize(im_overlay,[size(im,1),size(im,2)]);
            imtmp = im_overlay;
            for k = 1:size(imtmp,3) %iterate through color dimensions if exist
                imtmp(sub2ind(size(imtmp),mask_imgpixels(2).excluderoi.not_fullypixel,mask_imgpixels(2).excluderoi.not_fullxpixel,k*ones(size(mask_imgpixels(2).excluderoi.not_fullypixel,1),1))) = 0;
            end
            imshow(imtmp)
            %%

            % imtmp = imread(strcat(folder,'brainatlas_meso_overlay.jpg'));
            % imtmp = imresize(imtmp,[size(im,1),size(im,2)]);
            still_imgvid = VideoWriter(strcat(folder,paths{1},'\stillimg.avi'));
            still_imgvid.FrameRate = v.FrameRate;
            open(still_imgvid);
            for j = 1:10
                writeVideo(still_imgvid,imtmp);
            end
            close(still_imgvid)
            rawvideo = VideoReader(strcat(folder,paths{1},'\stillimg.avi'));
            
            %Draw ROIs for analysis and select seed pixels for seed pixel
            %maps
            prompt = {'Background ROIs','Analyze ROIs (at least 1)','Draw seed pixels within ROIs? (y/n)'};%let user choose if blue or green frame (manual segmentation)
            dlgtitle = 'ROI Selection For Analysis';
            roi_input = inputdlg(prompt,dlgtitle,[1 60]);
            background_roi = str2double(roi_input{1});
            analyze_roi = str2double(roi_input{2});
            seedpixel_roi = roi_input{3};
            exclude_roi = 'n';

            imgname = strcat(folder,paths{1},'\roi_selection.jpeg');
            stopframe = rawvideo.FrameRate*rawvideo.Duration;%size(runsum,3);
            casenum = 0;
            prev_analysis = 0;

            if isempty(frame_color.black)==1
                frame_color.black = 0;
            end         

            wt2 = waitbar(0,'Draw Desired ROIs For Analysis');

            %rawvideo = VideoReader(strcat(folder,files(ex).name));
            rawvideo = VideoReader(strcat(folder,paths{1},'\stillimg.avi'));%VideoReader(strcat(folder,files(ex).name));
            [roi_pixels,~,~,~] = roi_pix(rawvideo,frame,background_roi,analyze_roi,seedpixel_roi,exclude_roi,bin,imgname,stopframe,casenum,prev_analysis,frame_color); %set user-defined rois
            close(wt2)

        case 'green'
            roi_pixels = []; 
            analyze_roi = [];
            background_roi = [];
    end
    %saveas(gcf,strcat(folder,name,' bwim.jpeg'))
else
    if strcmp(drawroi,'y')
        vars = load(strcat(folder,paths{1},'\workspace.mat'),'roi_pixels');
        roi_pixels = vars.roi_pixels;
    else
        roi_pixels = [];
    end
end

end