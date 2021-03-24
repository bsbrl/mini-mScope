%% Frame segmentation function to write separate calcium/hemodynamic channel videos
%this function (manually) determines the blue and green frames as well as
%the black frames (cmos error frames at the start of trials) and prompts
%the user to select whether the first non-black frame is blue or green.
%From this input, the blue and green frames are calculated from the current
%selected frame to the end of the video since there is no frame switching
%with improved circuit designs. This function also has added features like
%outlier detection and correction for each pixel in a frame if noise/power
%spiking is present (simple outlier detection) and writes the corrected
%data to a new video that can be reanalyzed without the spikes present. 

%Daniel Surinach 03/20/2021
%%
function [frame_color] = findframe(rawvideo,frame,stopframe,frame_color,outlier,param2,vidname,blueorgreen_only)


wt=waitbar(0,'starting video analysis');%progress bar to see code process
steps=stopframe;%total frames

%intensity struct with channel data preallocation
intensity = struct('totalframe',{},'black',{},'blackind',{},'green',{},'greenind',{},'blue',{},'blueind',{},'time',{});
blackcount = 1;

count = 1;


if strcmp(blueorgreen_only{1},'segment') == 1 %if looking at an unsegmented video (blue/green alternate pulsing)
    while hasFrame(rawvideo) %read through all the frames
        single_frame = readFrame(rawvideo); %grab current frame

        if size(single_frame,3)==3
            single_frame = rgb2gray(single_frame);%convert to grayscale if needed
        end

        intensity(1).totalframe(frame,1) = mean(mean(single_frame));%store mean intensity value for each frame
        %serves as a quick check plot later to see if there are glaring errors
        %in the video collection so far 
        intensity(1).time(frame,1) = rawvideo.CurrentTime;%store time series of frames
        if frame == 1
            if outlier == 1 %added outlier analysis that can get rid of outliers
                %resulting from spikes in data due to wire/power noise
                intensity(1).allframe = zeros(size(single_frame,1)*size(single_frame,2),stopframe); %store each pixel for noise correction
            end
            if param2 == 1 %parameter to write a video of blue/green frames
                %phased out in later designs since blue%green frames yields a
                %video that can't be written to a regular visual video
                intensity(1).allframe = zeros(size(single_frame,1)*size(single_frame,2),stopframe);
            end

            xborder=[0;0;size(single_frame,2);size(single_frame,2);0];%grab borders of the whole unbinned image
            yborder=[0;size(single_frame,1);size(single_frame,1);0;0];

            roi_pixels(1).x_border=[0;0;size(single_frame,2);size(single_frame,2);0];%xpixel border data for polygon
            roi_pixels(1).y_border=[0;size(single_frame,1);size(single_frame,1);0;0];

            resized_image=roipoly(single_frame,roi_pixels(1).x_border,roi_pixels(1).y_border);%resize polygon 
            [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)

            X=max(xborder)-min(xborder);%length of pixels in x direction
            Y=max(yborder)-min(yborder);

            roi_pixels(1).x_borderfull=xborder;%grab pixels in row-format and store in a structure
            roi_pixels(1).y_borderfull=yborder;
            roi_pixels(1).Xfull=X;%store other information related to whole-frame ROI
            roi_pixels(1).Yfull=Y;


            roi_pixels(1).X=max(roi_pixels(1).x_border)-min(roi_pixels(1).y_border);
            roi_pixels(1).Y=max(roi_pixels(1).y_border)-min(roi_pixels(1).y_border);
            roi_pixels(1).x_pixels=xpixels;%x pixels between image border
            roi_pixels(1).y_pixels=ypixels;
        end



        if intensity(1).totalframe(frame) < 5  %if looking at mostly black frames
            if blackcount < 100
                fprintf('Black frames detected at frame %1.0f\n',frame)
                intensity(1).black(blackcount,1) = intensity.totalframe(frame);%store their frame location
                intensity(1).blackind(blackcount,1) = frame;
                blackcount = blackcount+1;
            end
        else
            if outlier == 1 %begin storing all the pixel data for the whole-frame ROI 
                intensity(1).allframe(:,frame) = single_frame(sub2ind(size(single_frame),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels));
            end

            if param2 == 1
                intensity(1).allframe(:,frame) = single_frame(sub2ind(size(single_frame),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels));
            end

            if count == 1
                %user_input = {'b'};
                figure(1)
                imshow(single_frame) %show first non-black image
                prompt = {'Is this a blue or green frame? (b/g)'};%let user choose if blue or green frame (manual segmentation)
                dlgtitle = 'Blue and green frame selection';
                user_input = inputdlg(prompt,dlgtitle);
                %totalframes = rawvideo.Duration*rawvideo.FrameRate;
                frame_color(1).userinput = user_input{1};

                if strcmp(user_input{1},'b')==1 %if frame shown is blue, store every 2nd frame after it as blue
                    intensity(1).blueind(:,1) = frame:2:stopframe;
                    intensity(1).greenind(:,1) = frame+1:2:stopframe;
                else %else repeat with green frames
                    intensity(1).greenind(:,1) = frame:2:stopframe;
                    intensity(1).blueind(:,1) = frame+1:2:stopframe;
                end
                count = 2;
                close(figure(1))

    %             if (outlier==0) && (param2==0)
    %                 break
    %             end
            end
        end



        if frame==stopframe
            break
        end

        if mod(frame,20)==0
            waitbar(frame/steps,wt,sprintf('LED Analysis for frame %1.0f/%1.0f',frame,steps))
        end

        frame=frame+1;%iterate to next frame


    end

    close(wt)

    % figure(1)
    % plot(intensity(1).time(frame_color(1).blue),intensity(1).totalframe(frame_color(1).blue),'b',...
    %     intensity(1).time(frame_color(1).green),intensity(1).totalframe(frame_color(1).green),'g')
    % xlabel('Time (s)')
    % ylabel('Mean Pixel Intensity')
    % title('Blue vs Green Frame Intensities For Entire FOV')
    % legend('Blue LED','Green LED')

    %% find frame drops and outliers by kmeans grouping
    %this auto-clusters calcium and hemodynamic data since they have 
    %distinctive illumination profiles. If frames are dropped, it usually
    %"switches" the calcium and reflectance signals in original trace data
    %this function finds points where there are switches and corrects 
    %the dropped frames 
    [idx,~,sumd,~] = kmeans(intensity.totalframe,2,'Distance','sqeuclidean','MaxIter',1000,'Replicates',100,'Options',statset('UseParallel',1),'Display','off');
    
    group1 = find(idx == 1);%two grouping function, calcium/hemodynamic channel
    group2 = find(idx == 2);
    
    if mean(intensity.totalframe(group1)) > mean(intensity.totalframe(group2))
        intensity.blueind = group1;
        intensity.greenind = group2;
        
        intensity.blue = intensity.totalframe(group1);
        intensity.green = intensity.totalframe(group2);
    else
        intensity.blueind = group2;
        intensity.greenind = group1;
        
        intensity.blue = intensity.totalframe(group2);
        intensity.green = intensity.totalframe(group1);
    end
    
    %find outliers in calcium/hemodynamic channels and correct with
    %linear interpolation if there are frame drops or power surges/dips
    [~,TFblue]= filloutliers(intensity.blue,'linear');
    intensity.blueoutlier = find(TFblue == 1); %find locations of additional outliers
    intensity.blueoutlier = intensity.blueind(intensity.blueoutlier);
    intensity.blueind = setdiff(intensity.blueind,intensity.blueoutlier);%remove any outliers (large frame drops)
    intensity.blue = intensity.totalframe(intensity.blueind);
    
    [~,TFgreen] = filloutliers(intensity.green,'linear');
    intensity.greenoutlier = find(TFgreen == 1);
    intensity.greenoutlier = intensity.greenind(intensity.greenoutlier);
    intensity.greenind = setdiff(intensity.greenind,intensity.greenoutlier);
    intensity.green = intensity.totalframe(intensity.greenind);
    
    %assign final frames for calcium/hemodynamic channels to output struct
%     intensity(1).green = intensity(1).totalframe(intensity(1).greenind);
    frame_color(1).green = intensity(1).greenind;%store to frame_color structure as output function
    frame_color(1).greenval = intensity(1).green;
    frame_color(1).greentime = intensity(1).time(intensity(1).greenind);

    frame_color(1).blue = intensity(1).blueind;
%     intensity(1).blue = intensity(1).totalframe(intensity(1).blueind);
    frame_color(1).blueval = intensity(1).blue;
    frame_color(1).bluetime = intensity(1).time(intensity(1).blueind);

    frame_color(1).black = intensity(1).blackind;
    

    %%
    % fprintf('\nNumber of green frames is %1.0f\n',size(intensity(1).greenind,1))
    % fprintf('Number of blue frames is %1.0f\n\n',size(intensity(1).blueind,1))
    % figure(1)
    % plot(intensity(1).time(intensity(1).greenind),intensity(1).green,'b',...
    %     intensity(1).time(intensity(1).blueind),intensity(1).blue,'g')
    % xlabel('Time (s)')
    % ylabel('Mean Pixel Intensity')
    % title('Blue vs Green Frame Intensities For Entire FOV')
    % legend('Blue LED','Green LED')

    %this function provides a more careful approach to dealing with power
    %surges. It is still in the development phase so it usually unused
    if param2 == 1 %if video writing blue%green desired (phased out)
        intensity(1).allgreen = intensity(1).allframe(:,intensity(1).greenind);
        intensity(1).allblue = intensity(1).allframe(:,intensity(1).blueind);

         if length(frame_color(1).blue) ~= length(frame_color(1).green)
                if length(frame_color(1).blue)>length(frame_color(1).green)
                    maxlength = length(frame_color(1).green);
                else
                    maxlength = length(frame_color(1).blue);
                end
         end


         tmp2 = (intensity(1).allblue(:,1:maxlength)./max(intensity(1).allblue(:,1:maxlength),[],2))./(intensity(1).allgreen(:,1:maxlength)./max(intensity(1).allgreen(:,1:maxlength),[],2));
         %tmp2 = (intensity(1).allblue(:,1:maxlength)./max(intensity(1).allblue(1:maxlength),[],2))./(intensity(1).allgreen(:,1:maxlength)./max(intensity(1).allgreen(1:maxlength),[],2));
         infans = find(tmp2==inf);
         tmp2(infans) = NaN;
         tmp2 = tmp2/max(max(tmp2));

         v = VideoWriter(vidname);
         open(v)
         wt = waitbar(0,'starting video analysis2');%progress bar to see how code processsteps=len;
         %stopframe = 3601;
         steps = size(tmp2,2);%total frames
         for j = 1:size(tmp2,2)
            tmpframe = zeros(size(single_frame,1),size(single_frame,2));
            tmpframe(sub2ind(size(single_frame),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels)) = tmp2(:,j);
            %tmpframe = uint8(round(tmpframe)*255);
            writeVideo(v,tmpframe)
            if mod(j,20)==0
                waitbar(j/steps,wt,sprintf('writing blue/green video %1.0f/%1.0f',j,steps))
            end
         end
         close(v)
         close(wt)
    end

    if outlier == 1 %if outlier correction desired 
        intensity(1).allgreen = intensity(1).allframe(:,intensity(1).greenind);%store all green frames
        intensity(1).allblue = intensity(1).allframe(:,intensity(1).blueind);%store all blue frames


        wt=waitbar(0,'green frame correction');%progress bar to see how code processsteps=len;
        %stopframe=round(rawvideo.FrameRate*rawvideo.Duration);
        steps=size(intensity(1).allgreen,1);%total frames


        for i = 1:size(intensity(1).allgreen,1)
            intensity(1).allgreen(i,:) = filloutliers(intensity(1).allgreen(i,:),'next');%iterate through each pixel in data set
            %replace any outliers detected with the next non-outlier point

            %intensity(1).allframe(i,:) = filloutliers(intensity(1).allframe(i,:),'next');
            if mod(i,10000)==0
                 waitbar(i/steps,wt,sprintf('correction for green frame pixel %1.0f/%1.0f',i,steps))
            end
        end
        close(wt)
        clear wt
        intensity(1).allframe(:,intensity(1).greenind) = intensity(1).allgreen;

        wt=waitbar(0,'blue frame correction');%progress bar to see how code processsteps=len;
        %stopframe=round(rawvideo.FrameRate*rawvideo.Duration);
        steps=size(intensity(1).allblue,1);%total frames

        for k = 1:size(intensity(1).allblue,1) %repeat with blue frames
            intensity(1).allblue(k,:) = filloutliers(intensity(1).allblue(k,:),'next');
            if mod(k,10000)==0
                 waitbar(k/steps,wt,sprintf('correction for blue frame pixel %1.0f/%1.0f',k,steps))
            end
        end
        close(wt)
        clear wt
        intensity(1).allframe(:,intensity(1).blueind) = intensity(1).allblue;


        wt=waitbar(0,'frame correction writing');%progress bar to see how code process steps=len;
        steps=size(intensity(1).allframe,2);%total frames
        tmp = strsplit(rawvideo.Name,'.avi');

        v = VideoWriter(strcat(tmp{1},'_corrected.avi')); %write a video
        %with the corrected blue and green frames and re-run main code with the
        %corrected data from this analysis (outliers removed) 
        open(v)

        for j = 1:size(intensity(1).allframe,2)
           temp_frame = zeros(size(single_frame,1),size(single_frame,2));
           temp_frame2 = intensity(1).allframe(:,j);

           temp_frame(sub2ind(size(single_frame),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels)) = temp_frame2;
           writeVideo(v,uint8(temp_frame))

           if mod(j,20)==0
                 waitbar(j/steps,wt,sprintf('frame writing correction  %1.0f/%1.0f',j,steps))
            end
        end
        close(wt)
        close(v)
    end


    if outlier == 1 %further glance at outlier analysis 
        %has some fine tuning that allows spikes beyond a threshold to be
        %detected and shows the mean trace of the plot before and after
        %correction for the user to observe. Usually these spikes are global in
        %nature and all pixels will show the spikes, so the mean intensity of
        %each frame can be used as a landmarker to see if the analysis has
        %worked or needs further processing 

        [outblue,~,~,~] = isoutlier(frame_color(1).blueval,'mean','ThresholdFactor',2);
        [blueoutlier,~]=find(outblue==1);

        [outgreen,~,~,~] = isoutlier(frame_color(1).greenval,'mean','ThresholdFactor',2);
        [greenoutlier,~]=find(outgreen==1);

        if isempty(blueoutlier) == 0
            fprintf('Spikes in blue frames greater than 3x mean detected for %1.0f instances \n',length(blueoutlier))

            figure(1)
            plot(intensity(1).time(frame_color(1).blue),intensity(1).totalframe(frame_color(1).blue),'g')
            hold on
            plot(intensity(1).time(frame_color(1).blue(blueoutlier)),frame_color(1).blueval(blueoutlier),'.b')
            legend('Raw Data','Detected dropped spikes')
            title('blue trace')

            bluedata = intensity(1).totalframe(frame_color(1).blue);


            tmp2blue = bluedata;


            consec_intblue = find(diff(blueoutlier)==1); %find consecutive integers if they exist
            consec_intsortblue = unique(blueoutlier([consec_intblue;consec_intblue+1]));

            filtered_blue = filloutliers(bluedata,'next');

            figure(2)
            plot(intensity(1).time(frame_color(1).blue),intensity(1).totalframe(frame_color(1).blue),'.g')
            hold on
            plot(intensity(1).time(frame_color(1).blue),filtered_blue,'b')


        end

        if isempty(greenoutlier) == 0
            fprintf('Spikes in green frames greater than 3x mean detected for %1.0f instances \n',length(greenoutlier))

            figure(3)
            plot(intensity(1).time(frame_color(1).green),intensity(1).totalframe(frame_color(1).green),'g')
            hold on
            plot(intensity(1).time(frame_color(1).green(greenoutlier)),frame_color(1).greenval(greenoutlier),'.b')
            legend('Raw Data','Data minus spikes')
            title('green trace')

            greendata = intensity(1).totalframe(frame_color(1).green);
            tmp2green = greendata;

            consec_intgreen = find(diff(greenoutlier)==1); %find consecutive integers if they exist
            consec_intsortgreen = unique(greenoutlier([consec_intgreen;consec_intgreen+1]));

            filtered_green = filloutliers(greendata,'next');

            figure(4)
            plot(intensity(1).time(frame_color(1).green),intensity(1).totalframe(frame_color(1).green),'.g')
            hold on
            plot(intensity(1).time(frame_color(1).green),filtered_green,'b')

        end
    end

    greentime = intensity(1).time(intensity(1).greenind);
    bluetime = intensity(1).time(intensity(1).blueind);


%     fprintf('\nNumber of green frames is %1.0f\n',size(intensity(1).greenind,1))
%     fprintf('Number of blue frames is %1.0f\n\n',size(intensity(1).blueind,1))

    %% output plot of final frames for calcium/reflectance channel
    figure(1) %plot mean intensity plots
    plot(intensity(1).time(intensity(1).greenind(2:end-5)),intensity(1).green(2:end-5),'g',...
        intensity(1).time(intensity(1).blueind(2:end-5)),intensity(1).blue(2:end-5),'b')
    ax = gca;
    ax.FontSize = 12;
    ax.FontName = 'Arial';
    xlabel('\fontname{Arial}\fontsize{14} Time (s)')
    ylabel('\fontname{Arial}\fontsize{14} Mean Pixel Intensity')
    title('Blue vs Green Frame Intensities For Entire FOV')
    legend('Green LED','Blue LED')
    
    %% general information for % change of mean intensity of raw frame data
    int_greendata = [min(intensity(1).green(2:end-5)),mean(intensity(1).green(2:end-5)),max(intensity(1).green(2:end-5))]
    %display the range of min,mean,max intensity data for blue and green frames
    %to make sure that no frame dropping or mixing has occured (i.e. one frame
    %is much brighter than the other so if the ranges switch somewhere in the
    %plot or in this output, the video may be unusable 
    int_bluedata = [min(intensity(1).blue(2:end-5)),mean(intensity(1).blue(2:end-5)),max(intensity(1).blue(2:end-5))]

    fprintf('\nMax %% loss for green frames is %4.4f\n',max([1-int_greendata(1)/int_greendata(2),1-int_greendata(3)/int_greendata(2)])*100)
    fprintf('Max %% loss for blue frames is %4.4f\n\n',max([1-int_bluedata(1)/int_bluedata(2),1-int_bluedata(3)/int_bluedata(3)])*100)
   
else % if only analyzing a blue video for example
    while hasFrame(rawvideo) %read through all the frames
        single_frame = readFrame(rawvideo); %grab current frame

        if size(single_frame,3)==3
            single_frame = rgb2gray(single_frame);%convert to grayscale if needed
        end

        intensity(1).totalframe(frame,1) = mean(mean(single_frame));%store mean intensity value for each frame
        %serves as a quick check plot later to see if there are glaring errors
        %in the video collection so far 
        intensity(1).time(frame,1) = rawvideo.CurrentTime;%store time series of frames
        if frame == 1
            if outlier == 1 %added outlier analysis that can get rid of outliers
                %resulting from spikes in data due to wire/power noise
                intensity(1).allframe = zeros(size(single_frame,1)*size(single_frame,2),stopframe); %store each pixel for noise correction
            end
            if param2 == 1 %parameter to write a video of blue/green frames
                %phased out in later designs since blue%green frames yields a
                %video that can't be written to a regular visual video
                intensity(1).allframe = zeros(size(single_frame,1)*size(single_frame,2),stopframe);
            end

            xborder=[0;0;size(single_frame,2);size(single_frame,2);0];%grab borders of the whole unbinned image
            yborder=[0;size(single_frame,1);size(single_frame,1);0;0];

            roi_pixels(1).x_border=[0;0;size(single_frame,2);size(single_frame,2);0];%xpixel border data for polygon
            roi_pixels(1).y_border=[0;size(single_frame,1);size(single_frame,1);0;0];

            resized_image=roipoly(single_frame,roi_pixels(1).x_border,roi_pixels(1).y_border);%resize polygon 
            [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)

            X=max(xborder)-min(xborder);%length of pixels in x direction
            Y=max(yborder)-min(yborder);

            roi_pixels(1).x_borderfull=xborder;%grab pixels in row-format and store in a structure
            roi_pixels(1).y_borderfull=yborder;
            roi_pixels(1).Xfull=X;%store other information related to whole-frame ROI
            roi_pixels(1).Yfull=Y;


            roi_pixels(1).X=max(roi_pixels(1).x_border)-min(roi_pixels(1).y_border);
            roi_pixels(1).Y=max(roi_pixels(1).y_border)-min(roi_pixels(1).y_border);
            roi_pixels(1).x_pixels=xpixels;%x pixels between image border
            roi_pixels(1).y_pixels=ypixels;
        end



        if intensity(1).totalframe(frame) < 5  %if looking at mostly black frames
            fprintf('Black frames detected at frame %1.0f\n',frame)
            intensity(1).black(blackcount,1) = intensity.totalframe(frame);%store their frame location
            intensity(1).blackind(blackcount,1) = frame;
            blackcount = blackcount+1;
        else
            if outlier == 1 %begin storing all the pixel data for the whole-frame ROI 
                intensity(1).allframe(:,frame) = single_frame(sub2ind(size(single_frame),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels));
            end

            if param2 == 1
                intensity(1).allframe(:,frame) = single_frame(sub2ind(size(single_frame),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels));
            end

            if count == 1
                user_input = {'b'};
%                 figure(1)
%                 imshow(single_frame) %show first non-black image
%                 prompt = {'Is this a blue or green frame? (b/g)'};%let user choose if blue or green frame (manual segmentation)
%                 dlgtitle = 'Blue and green frame selection';
%                 user_input = inputdlg(prompt,dlgtitle);
%                 %totalframes = rawvideo.Duration*rawvideo.FrameRate;
                frame_color(1).userinput = user_input{1};

                if strcmp(user_input{1},'b')==1 %if frame shown is blue, store every 2nd frame after it as blue
                    intensity(1).blueind(:,1) = frame:stopframe;
                else %else repeat with green frames
                    intensity(1).greenind(:,1) = frame:stopframe;
                end
                count = 2;
                close(figure(1))

            end

        end



        if frame==stopframe
            break
        end

        if mod(frame,20)==0
            waitbar(frame/steps,wt,sprintf('LED Analysis for frame %1.0f/%1.0f',frame,steps))
        end

        frame=frame+1;%iterate to next frame


    end
    close(wt)
    intensity.greenind = round(intensity.greenind);
    intensity.blueind = round(intensity.blueind);
    
    intensity(1).green = intensity(1).totalframe(intensity(1).greenind);
    frame_color(1).green = intensity(1).greenind;%store to frame_color structure as output function
    frame_color(1).greenval = intensity(1).green;
    frame_color(1).greentime = intensity(1).time(intensity(1).greenind);

    frame_color(1).blue = intensity(1).blueind;
    intensity(1).blue = intensity(1).totalframe(intensity(1).blueind);
    frame_color(1).blueval = intensity(1).blue;
    frame_color(1).bluetime = intensity(1).time(intensity(1).blueind);

    frame_color(1).black = intensity(1).blackind;
    
%     fprintf('\nNumber of green frames is %1.0f\n',size(intensity(1).greenind,1))
%     fprintf('Number of blue frames is %1.0f\n\n',size(intensity(1).blueind,1))


    %%
    %display the range of min,mean,max intensity data for blue and green frames
    %to make sure that no frame dropping or mixing has occured (i.e. one frame
    %is much brighter than the other so if the ranges switch somewhere in the
    %plot or in this output, the video may be unusable 
    
    if isempty(frame_color.blue) == 1
        user_input = {'b'};
%                 figure(1)
%                 imshow(single_frame) %show first non-black image
%                 prompt = {'Is this a blue or green frame? (b/g)'};%let user choose if blue or green frame (manual segmentation)
%                 dlgtitle = 'Blue and green frame selection';
%                 user_input = inputdlg(prompt,dlgtitle);
%                 %totalframes = rawvideo.Duration*rawvideo.FrameRate;
        frame_color(1).userinput = user_input{1};

        if strcmp(user_input{1},'b')==1 %if frame shown is blue, store every 2nd frame after it as blue
            intensity(1).blueind(:,1) = 1:stopframe;
            frame_color(1).blue = frame_color(1).black(6:end);
            frame_color(1).black = frame_color(1).black(1:5);
        else %else repeat with green frames
            intensity(1).greenind(:,1) = 1:stopframe;
        end
        
    end
%     if strcmp(user_input{1},'b')==1
%         figure(1) %plot mean intensity plots
%         plot(intensity(1).time(intensity(1).blueind(2:end-5)),intensity(1).blue(2:end-5),'b')
%         ax = gca;
%         ax.FontSize = 12;
%         ax.FontName = 'Arial';
%         xlabel('\fontname{Arial}\fontsize{14} Time (s)')
%         ylabel('\fontname{Arial}\fontsize{14} Mean Pixel Intensity')
%         title('Blue vs Green Frame Intensities For Entire FOV')
%         legend('Green LED','Blue LED')
%         
%         int_bluedata = [min(intensity(1).blue(2:end-5)),mean(intensity(1).blue(2:end-5)),max(intensity(1).blue(2:end-5))]
%         fprintf('Max %% loss for blue frames is %4.4f\n\n',max([1-int_bluedata(1)/int_bluedata(2),1-int_bluedata(3)/int_bluedata(3)])*100)
%         
% 
%     else
%         figure(1) %plot mean intensity plots
%         plot(intensity(1).time(intensity(1).greenind(2:end-5)),intensity(1).green(2:end-5),'g')
%         ax = gca;
%         ax.FontSize = 12;
%         ax.FontName = 'Arial';
%         xlabel('\fontname{Arial}\fontsize{14} Time (s)')
%         ylabel('\fontname{Arial}\fontsize{14} Mean Pixel Intensity')
%         title('Blue vs Green Frame Intensities For Entire FOV')
%         legend('Green LED','Blue LED')
%         
%         int_greendata = [min(intensity(1).green(2:end-5)),mean(intensity(1).green(2:end-5)),max(intensity(1).green(2:end-5))]
% 
%         fprintf('\nMax %% loss for green frames is %4.4f\n',max([1-int_greendata(1)/int_greendata(2),1-int_greendata(3)/int_greendata(2)])*100)
%     end
end
end