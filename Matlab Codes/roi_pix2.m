%% ROI selection from user-defined polygon shapes
%this function allows the user to select a set of background ROIs for a
%background correction algorithm and a set of analysis ROIs and stores the
%pixels in these regions for calculations to be used at other sections in
%the code. You can draw any number of input closed polygons
%and get information about their centroid and all the pixels located within
%the drawn rois. The first row is binned information for the entire FOV.
%Each subsequent ROI is what you draw. The last ROI is the unbinned ROI
%used for resizing the ROi at different binning if desired.
%This is similar to roi_pix, but with added functionality
%%
function [roi_pixels,Sij,roi_num,first_frame] = roi_pix2(rawvideo,frame,baseroi_num,analyzeroi_num,seedpixel_roinum,exclude_roi,bin_num,imgname,stopframe,casenum,data,frame_color)

%preallocate roipixels struct
roi_pixels=struct('x_border',{},'y_border',{},'y_pixels',{},'x_pixels',{},'fullxpixel',{},'fullypixel',{},...
'x_borderfull',{},'y_borderfull',{},'Xfull',{},'Yfull',{},'centroidxy',{},'X',{},'Y',{},'baseroi',{},'analyzeroi',{});
Sij=struct('roiblue',{},'roigreen',{});


while hasFrame(rawvideo) %read through all the frames
    single_frame = readFrame(rawvideo); %grab current frame
    single_framegraylarge = single_frame;
%     if size(single_frame,3)==3
%         single_framegraylarge=rgb2gray(single_frame); %convert rgb to grayscale
%     else
%         single_framegraylarge = single_frame;
%     end
    
    single_framegray=imresize(single_framegraylarge,bin_num,'bilinear');
    if frame == frame_color(1).black(end)+1%if frame==1 %looking at first frame for roi selection
         first_frame=single_framegraylarge;%store a copy of first frame
        
        if casenum==0 %if ROI is being selected for the first time 
            baseroi_num=baseroi_num+1;
            roi_num=baseroi_num+analyzeroi_num;%calculate the total number of roi's
            %equivalent to the background ROIs drawn by the user, the
            %analyze ROIs drawn by the user, and one whole-frame background
            %ROI used for an additional global correction 
            
            roi_pixels(1).baseroi=baseroi_num;
            roi_pixels(1).analyzeroi=analyzeroi_num;

            %added to extract the pixel data for a whole-frame ROI (unbinned) 
            roi_pixels(roi_num+1).x_border=[0;0;size(single_framegraylarge,2);size(single_framegraylarge,2);0];%xpixel border data for polygon
            roi_pixels(roi_num+1).y_border=[0;size(single_framegraylarge,1);size(single_framegraylarge,1);0;0];
            resized_image=roipoly(single_framegraylarge,roi_pixels(roi_num+1).x_border,roi_pixels(roi_num+1).y_border);%resize polygon 
            [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)
            roi_pixels(roi_num+1).x_pixels=xpixels;%x pixels between image border
            roi_pixels(roi_num+1).y_pixels=ypixels;
            %added to extract the pixel data for a whole-frame ROI (unbinned) 
            
            count=1;

            xborder=[0;0;size(single_framegraylarge,2);size(single_framegraylarge,2);0];%extract ROI for whole-frame (binned) ROI
            yborder=[0;size(single_framegraylarge,1);size(single_framegraylarge,1);0;0];

            roi_pixels(1).x_border=[0;0;size(single_framegray,2);size(single_framegray,2);0];%xpixel border data for polygon
            roi_pixels(1).y_border=[0;size(single_framegray,1);size(single_framegray,1);0;0];

            resized_image=roipoly(single_framegray,roi_pixels(1).x_border,roi_pixels(1).y_border);%resize polygon 
            [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)
            [ynotpixels,xnotpixels] = find(resized_image==0);
            
            X=max(xborder)-min(xborder);%length of pixels in x direction
            Y=max(yborder)-min(yborder);

            roi_pixels(1).x_borderfull=xborder;%store full border for reanalysis of same roi
            roi_pixels(1).y_borderfull=yborder;
            roi_pixels(1).Xfull=X;
            roi_pixels(1).Yfull=Y;


            roi_pixels(1).X=max(roi_pixels(1).x_border)-min(roi_pixels(1).y_border);
            roi_pixels(1).Y=max(roi_pixels(1).y_border)-min(roi_pixels(1).y_border);
            roi_pixels(1).x_pixels=xpixels;%x pixels between image border
            roi_pixels(1).y_pixels=ypixels; %y pixels in image, both are the most important feature from this analysis
            roi_pixels(1).ynot_pixels = ynotpixels;
            roi_pixels(1).xnot_pixels = xnotpixels; %pixels outside of ROI bounds
            
            resized_image = double(resized_image);
            resized_image(sub2ind(size(resized_image),size(resized_image,1)/2,size(resized_image,2))) = -1;%set centroid to -1
            [ynotcentroid,xnotcentroid] = find(resized_image>=0);%find points outside of centroid for seed pixel calc

            roi_pixels(1).ynotcentroid = ynotcentroid;
            roi_pixels(1).xnotcentroid = xnotcentroid;
            
            Sij(1).roiblue = double(single_framegray);%store the first Sij (explained in next function) value 
            Sij(roi_num+1).roiblue=double(single_framegraylarge);

            Sij(1).roigreen = double(single_framegray);
            Sij(roi_num+1).roigreen = double(single_framegraylarge);

            %Sij(1).roi = double(single_framegray);
            %Sij(roi_num+1).roi = double(single_framegraylarge);
            wt = waitbar(0,'Draw Desired ROIs For Analysis');
            steps = roi_num;
            
%             roi_im = figure;
%             imshow(first_frame) %begin ROI drawing 
%             hold on
        elseif casenum == 1 %if roi is preloaded at same binning
            roi_name = fields(data);
            if length(roi_name) > 1
                roi_pixels = data;
            else
                roi_pixels = data.(roi_name{1});
            end
            baseroi_num = roi_pixels(1).baseroi;
            analyzeroi_num = roi_pixels(1).analyzeroi;
            roi_num = baseroi_num + analyzeroi_num;
            
            Sij(1).roiblue = double(single_framegray);
            Sij(roi_num+1).roiblue=double(single_framegraylarge);

            Sij(1).roigreen = double(single_framegray);
            Sij(roi_num+1).roigreen = double(single_framegraylarge);
        elseif casenum == 3 || casenum == 4 %if roi is preloaded at different binning
            baseroi_num = baseroi_num + 1;
            roi_num = baseroi_num + analyzeroi_num;
            roi_pixels(1).baseroi=baseroi_num;
            roi_pixels(1).analyzeroi=analyzeroi_num;
            Sij = [];
            
        end
            roi_im = figure;
            imshow(first_frame) %begin ROI drawing 
            hold on
            for i=2:roi_num %iterate for desired roi #s - these are user drawn (above was for the whole-frame binned image
                if casenum == 0 || casenum == 3 || casenum == 4
                    if casenum == 0 %draw rois for first time
                        [roi,xborder,yborder]=roipoly; %select roi and extract border points
                    elseif casenum == 3 %import xy pixels inside ROI (no border defined)
                        %can be useful if pixels are given with no ROI
                        %borders
%                         tmp_img = zeros(size(im2));
                        ind = find(data==i-1);
                        if bin_num < 1
                            if size(first_frame,3) > 1
                                im2 = imresize(rgb2gray(first_frame),bin_num);
                            else
                                im2 = imresize(first_frame,bin_num);
                            end
                        else
                            if size(first_frame,3) > 1
                                im2 = rgb2gray(first_frame);
                            else
                                im2 = first_frame;
                            end
                        end
                        [row_full,col_full] = ind2sub([size(im2,1),size(im2,2)],ind);
%                         tmp_img(ind) = ind;
                        bound = boundary(col_full,row_full);
                        boundx = col_full(bound)/bin_num;
                        boundy = row_full(bound)/bin_num;
                        
                        [roi,xborder,yborder] = roipoly(first_frame,boundx,boundy);
%                         figure(7)
%                         imshow(first_frame)
%                         hold on
%                         plot(xborder,yborder)
%                         plot(col_full(k),row_full(k),'Color',jet_colors(des_rows(j),:),'LineWidth',1)
                    elseif casenum == 4 %resize an already drawn ROI to different binning
                        boundx = data(i-1).filt_bordercol/bin_num; %get border data
                        boundy = data(i-1).filt_borderrow/bin_num;
                        [roi,xborder,yborder] = roipoly(first_frame,boundx,boundy);
                    end

                    hold on
                    [ypixelsbig,xpixelsbig]=find(roi==1); %find x,y pixels within drawn borders
                    [not_ypixelsbig,not_xpixelsbig] = find(roi == 0);

                    [geom,~,~ ]=polygeom(xborder,yborder);%use function to find centroid of polygon
                    %useful primarily for labels to identify which roi corresponds
                    %to what data set. [geom(2),geom(3)]=[xcentroid,ycentroid]
                    resized_image=roipoly(single_framegray,round(xborder*bin_num),round(yborder*bin_num));%resize polygon 
                elseif casenum == 1 %load from previous drawing
                    resized_image=roipoly(single_framegray,round(roi_pixels(i).x_borderfull*bin_num),round(roi_pixels(i).y_borderfull*bin_num));%resize polygon 
                    xborder = roi_pixels(i).x_borderfull;
                    yborder = roi_pixels(i).y_borderfull;
                    geom = roi_pixels(i).geom;
                    xpixelsbig = roi_pixels(i).fullxpixel;
                    ypixelsbig = roi_pixels(i).fullypixel;
                    not_ypixelsbig = roi_pixels(i).not_fullypixel;
                    not_xpixelsbig = roi_pixels(i).not_fullxpixel;
                    roi_pixels(i).circle_points.ynotcircle = [];
                    roi_pixels(i).circle_points.xnotcircle = [];
                end

                
                [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)
                [ynotpixels,xnotpixels] = find(resized_image==0);
                
                X=max(xborder)-min(xborder);%length of pixels in x direction
                Y=max(yborder)-min(yborder);

                roi_pixels(i).x_borderfull=xborder;%store full border for reanalysis of same roi
                roi_pixels(i).y_borderfull=yborder;
                roi_pixels(i).Xfull=X;
                roi_pixels(i).Yfull=Y;
                

                roi_pixels(i).x_border=round(xborder*bin_num);%xpixel border data for polygon
                roi_pixels(i).y_border=round(yborder*bin_num);
                roi_pixels(i).X=round(X*bin_num);
                roi_pixels(i).Y=round(Y*bin_num);
                roi_pixels(i).x_pixels=xpixels;%x pixels between image border
                roi_pixels(i).y_pixels=ypixels;
                roi_pixels(i).xnot_pixels = xnotpixels;%x pixels outside roi
                roi_pixels(i).ynot_pixels = ynotpixels;
                roi_pixels(i).centroidxy=[round(geom(2)*bin_num),round(geom(3)*bin_num)];%centroid (x,y) of polygon
                
                %% this subnest draws a list of circles inscribed within ROis
                %this is useful to make well-defined circular polygons
                %and determine effects of choosing seed pixel points
                %in different parts of the ROI
                catvecx = [];%make circles within polygon to grab pixel points
                catvecy = [];
                dist1 = [];
                pointmesh = 100; %100 points between polygon vertices
                for p = 1:length(roi_pixels(i).x_border)-1
                    xpoints = linspace(roi_pixels(i).x_border(p+1),roi_pixels(i).x_border(p),pointmesh);%connect lines 
                    ypoints = linspace(roi_pixels(i).y_border(p+1),roi_pixels(i).y_border(p),pointmesh);
                    catvecx = [catvecx,xpoints];%concatenate x,y points into border vector
                    catvecy = [catvecy,ypoints];
                   
                   
                end
                catvecx = catvecx';
                catvecy = catvecy';
                roi_pixels(i).x_bordercoord = catvecx;
                roi_pixels(i).y_bordercoord = catvecy; %store full border coord 
                
                for p = 1:length(roi_pixels(i).x_bordercoord)
                    dist1(p,1) = sqrt((roi_pixels(i).x_bordercoord(p) - roi_pixels(i).centroidxy(1))^2 + (roi_pixels(i).y_bordercoord(p) - roi_pixels(i).centroidxy(2))^2);%distance formula from centroid to each border point in mesh
                end
                roi_pixels(i).centroid_borderdist = dist1; %distances from centroid to each border
                
                [~,ind] = min(dist1);
%                 plot(roi_pixels(i).x_border,roi_pixels(i).y_border,catvecx,catvecy,'.',roi_pixels(i).centroidxy(1),roi_pixels(i).centroidxy(2),'x')
%                 hold on
                radii = 2;%how many circles to draw to get points
                circle_points = 2; %how many points to grab from the circle radii
                tolerance = 0.7; %pixel tolerance to edge of circle
                dist2 = linspace((dist1(ind)-tolerance)/radii,(dist1(ind)-tolerance),radii);
                for k = 1:size(dist2,2)
                    [xcircle,ycircle] = circle_coord(roi_pixels(i).centroidxy(1),roi_pixels(i).centroidxy(2),dist2(k));%gen circle points
                    circle_ind = round(linspace(1,length(xcircle),circle_points+1));
                    xpoints = round(xcircle(circle_ind(1:end-1)));%x,y pixels to keep on circle borders (rounded)
                    ypoints = round(ycircle(circle_ind(1:end-1)));
                    
                    roi_pixels(i).circle_points.xfull(:,k) = xcircle;
                    roi_pixels(i).circle_points.yfull(:,k) = ycircle;
                    roi_pixels(i).circle_points.xpoints(:,k) = xpoints;%row is points, col is circle radius 
                    roi_pixels(i).circle_points.ypoints(:,k) = ypoints;
%                     plot(xcircle, ycircle);
                end
                
                for k = 1:size(roi_pixels(i).circle_points.xfull,2)+1 %iterate through polygon centroid and circle points
                    
                    if k == size(roi_pixels(i).circle_points.xfull,2)+1%looking at points not in centroid
                        tmp_image = double(resized_image);
                        tmp_image(sub2ind(size(tmp_image),roi_pixels(i).centroidxy(2),roi_pixels(i).centroidxy(1))) = -1;%set centroid to -1
                        [ynotcentroid,xnotcentroid] = find(tmp_image>=0);%find points outside of centroid for seed pixel calc
                
                        roi_pixels(i).ynotcentroid = ynotcentroid;
                        roi_pixels(i).xnotcentroid = xnotcentroid;
                    else %looking at points on circles centered around centroid and within the polygon
                        for p = 1:size(roi_pixels(i).circle_points.xpoints,1)
                            tmp_image = double(resized_image);
                            tmp_image(sub2ind(size(tmp_image),roi_pixels(i).circle_points.ypoints(p,k),roi_pixels(i).circle_points.xpoints(p,k))) = -1;%set centroid to -1
                            [ynotcircle,xnotcircle] = find(tmp_image>=0);%find points outside of circle edge for seed pixel calc
                            
                            roi_pixels(i).circle_points.ynotcircle(:,p,k) = ynotcircle;
                            roi_pixels(i).circle_points.xnotcircle(:,p,k) = xnotcircle;
                            
                            
                        end
                        
                    end
                    
                end
                if casenum == 0
                    waitbar((i-1)/(steps-1),wt,sprintf('Completed drawing for ROI %1.0f/%1.0f',i-1,steps-1))
                end
                
                roi_pixels(i).fullxpixel=xpixelsbig;
                roi_pixels(i).fullypixel=ypixelsbig;
                roi_pixels(i).not_fullxpixel=not_xpixelsbig;
                roi_pixels(i).not_fullypixel=not_ypixelsbig;
                
                roi_pixels(i).geom = geom;%save centroid of roi for labeling

                Sij(i).roiblue=zeros(size(single_framegray,1),size(single_framegray,2));%preallocate vector with x-pixel by y-pixel whole roi of brain                
                Sij(i).roigreen=zeros(size(single_framegray,1),size(single_framegray,2));
            

                plot(xborder,yborder,'g');%overlay roi over original image
                hold on
                plot(roi_pixels(i).centroidxy(1)/bin_num,roi_pixels(i).centroidxy(2)/bin_num,'+g')
                hold on
                text(round(geom(2)),round(geom(3)),num2str(i-1))%add roi# label in the middle of the polygon
                hold on
                plot_colors = jet;
                color_length = round(linspace(1,size(plot_colors,1),radii)); %plotting colors
                plot_colors = plot_colors(color_length,:);
                
                for k = 1:radii
                    plot(roi_pixels(i).circle_points.xpoints(:,k)/bin_num,roi_pixels(i).circle_points.ypoints(:,k)/bin_num,'*','Color',plot_colors(k,:))
                    hold on
                end
                %saveas(gcf,'tst.jpg')
                
                if strcmp(seedpixel_roinum,'y') == 1 %seed pixels desired
                    if casenum == 0 || casenum == 3|| casenum == 4
                        wt2 = waitbar(0,'Draw Seed Pixel Points within ROIs, Double Click on Last Seed to Exit');
                        [yseed,xseed] = getpts; %draw desired seed pixels with mouse clicker
                        roi_pixels(i).yseedfull = yseed;
                        roi_pixels(i).xseedfull = xseed;
                        
                        roi_pixels(i).yseed = round(yseed*bin_num);
                        roi_pixels(i).xseed = round(xseed*bin_num);
                        close(wt2)
                    elseif casenum == 1 %redraw seed pixels if previously not defined
                        yseed = roi_pixels(i).yseedfull;
                        xseed = roi_pixels(i).xseedfull;
                        roi_pixels(i).yseed = round(yseed*bin_num);
                        roi_pixels(i).xseed = round(xseed*bin_num);
                    end

                    plot(yseed,xseed,'cx',yseed,xseed,'r+','MarkerSize',10)
                    hold on
                end
                
                %roi_image=getframe;%store roi for visual purposes
                %first_frame=roi_image.cdata;%store overlayed roi for first frame 
                %matlab roipoly function does not store visual roi for different
                %iterations
                

                %hold off
                

                if i==roi_num %if finished grabbing selected number of rois
                    if casenum == 0
                        close(wt)
                    end
                    
%                     colormap('jet');
%                     colorbar;
%                     plot_colors = findall(gcf,'type','ColorBar');
%                     plot_colors = plot_colors.Colormap;
                    plot_colors = jet;
                    color_length = round(linspace(1,size(plot_colors,1),length(baseroi_num+1:roi_num))); %plotting colors
                    plot_colors = plot_colors(color_length,:);
                    
                    if strcmp(exclude_roi,'y') == 1 %select ROIs to exclude from analysis
                        if casenum == 0 || casenum == 3|| casenum ==4
                            roi_input = {'y'};
                            count = 1;
                            wt2 = waitbar(0,'Draw ROIs you wish to exclude from analysis');
                            while strcmp(roi_input{1},'y') == 1 %run until user stops
                                [roi,xborder,yborder]=roipoly; %select roi and extract border points

                                hold on

                                [ypixelsbig,xpixelsbig]=find(roi==1); %find x,y pixels within drawn borders
                                [not_ypixelsbig,not_xpixelsbig] = find(roi == 0);

                                [geom,~,~ ]=polygeom(xborder,yborder);%use function to find centroid of polygon
                                %useful primarily for labels to identify which roi corresponds
                                %to what data set. [geom(2),geom(3)]=[xcentroid,ycentroid]

                                resized_image=roipoly(single_framegray,round(xborder*bin_num),round(yborder*bin_num));%resize polygon 
                                [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)
                                [ynotpixels,xnotpixels] = find(resized_image==0);

                                X=max(xborder)-min(xborder);%length of pixels in x direction
                                Y=max(yborder)-min(yborder);

                                roi_pixels(1).excluderoi(count).x_borderfull=xborder;%store full border for reanalysis of same roi
                                roi_pixels(1).excluderoi(count).y_borderfull=yborder;
                                roi_pixels(1).excluderoi(count).Xfull=X;
                                roi_pixels(1).excluderoi(count).Yfull=Y;


                                roi_pixels(1).excluderoi(count).x_border=round(xborder*bin_num);%xpixel border data for polygon
                                roi_pixels(1).excluderoi(count).y_border=round(yborder*bin_num);
                                roi_pixels(1).excluderoi(count).X=round(X*bin_num);
                                roi_pixels(1).excluderoi(count).Y=round(Y*bin_num);
                                roi_pixels(1).excluderoi(count).x_pixels=xpixels;%x pixels between image border
                                roi_pixels(1).excluderoi(count).y_pixels=ypixels;
                                roi_pixels(1).excluderoi(count).xnot_pixels = xnotpixels;%x pixels outside roi
                                roi_pixels(1).excluderoi(count).ynot_pixels = ynotpixels;
                                roi_pixels(1).excluderoi(count).centroidxy=[round(geom(2)*bin_num),round(geom(3)*bin_num)];%centroid (x,y) of polygon

                                roi_pixels(1).excluderoi(count).fullxpixel=xpixelsbig;
                                roi_pixels(1).excluderoi(count).fullypixel=ypixelsbig;
                                roi_pixels(1).excluderoi(count).not_fullypixel=not_ypixelsbig;
                                roi_pixels(1).excluderoi(count).not_fullxpixel=not_xpixelsbig;

                                roi_pixels(1).excluderoi(count).geom = geom;%save centroid of roi for labeling
                                plot(xborder,yborder,'r');%overlay roi over original image
                                hold on
                                plot(roi_pixels(1).excluderoi(count).centroidxy(1)/bin_num,roi_pixels(1).excluderoi(count).centroidxy(2)/bin_num,'+r')
                                hold on
                                text(round(geom(2)),round(geom(3)),num2str(count))%add roi# label in the middle of the polygon
                                roi_image=getframe;%store roi for visual purposes
                                first_frame=roi_image.cdata;%store overlayed roi for first frame
                                %hold off

                                count = count+1;

                                prompt = {'Continue ROI Exlusion Selection? (y/n)'};%let user choose if blue or green frame (manual segmentation)
                                dlgtitle = 'ROI Exclusion Selection For Analysis';
                                roi_input = inputdlg(prompt,dlgtitle,[1 60]);

                            end
                            close(wt2)
                        elseif casenum == 1
                            for count = 1:length(roi_pixels(1).excluderoi) %iterate through selected exclusions
                                geom = roi_pixels(1).excluderoi(count).geom;
                                xborder = roi_pixels(1).excluderoi(count).x_borderfull;%store full border for reanalysis of same roi
                                yborder = roi_pixels(1).excluderoi(count).y_borderfull;
                                
                                resized_image=roipoly(single_framegray,round(xborder*bin_num),round(yborder*bin_num));%resize polygon 
                                [ypixels,xpixels]=find(resized_image==1);%find pixels in resized image (~1-1 mapping)
                                [ynotpixels,xnotpixels] = find(resized_image==0);
                                
                                X=max(xborder)-min(xborder);%length of pixels in x direction
                                Y=max(yborder)-min(yborder);
                                
                                roi_pixels(1).excluderoi(count).x_border=round(xborder*bin_num);%xpixel border data for polygon
                                roi_pixels(1).excluderoi(count).y_border=round(yborder*bin_num);
                                roi_pixels(1).excluderoi(count).X=round(X*bin_num);
                                roi_pixels(1).excluderoi(count).Y=round(Y*bin_num);
                                
                                roi_pixels(1).excluderoi(count).x_pixels=xpixels;%x pixels between image border
                                roi_pixels(1).excluderoi(count).y_pixels=ypixels;
                                roi_pixels(1).excluderoi(count).xnot_pixels = xnotpixels;%x pixels outside roi
                                roi_pixels(1).excluderoi(count).ynot_pixels = ynotpixels;
                                roi_pixels(1).excluderoi(count).centroidxy=[round(geom(2)*bin_num),round(geom(3)*bin_num)];%centroid (x,y) of polygon
                                
                                plot(xborder,yborder,'r');%overlay roi over original image
                                hold on
                                plot(roi_pixels(1).excluderoi(count).centroidxy(1)/bin_num,roi_pixels(1).excluderoi(count).centroidxy(2)/bin_num,'+r')
                                hold on
                                text(round(geom(2)),round(geom(3)),num2str(count))%add roi# label in the middle of the polygon
                                if count == length(roi_pixels(1).excluderoi)
                                    count = count + 1;
                                end
                            end
                        end
                    end
                    
                    %img=figure(1);%save original figure with rois overlayed on it
                    %imshow(first_frame)
                    saveas(roi_im,imgname)
                    first_frame = getimage;%grab current image
                    %close(img)

                    %imshow(first_frame)%show original image with plotted pixels within borders
                    hold on
                    for k=baseroi_num+1:roi_num
                        if k==baseroi_num+1 %if looking at whole brain roi
                            %color='.b';% assign a color
                        else%assign a different color so you can still see samller rois overlayed over wholebrain
                            %color='.g';
                        end
%                         plot(roi_pixels(k).fullxpixel,roi_pixels(k).fullypixel,color)

                        plot(roi_pixels(k).fullxpixel,roi_pixels(k).fullypixel,'Color',plot_colors(k-baseroi_num,:))
                        hold on
                    end
                    hold off
                    
                    figure(2);
                    imshow(imresize(first_frame,bin_num,'bilinear'));hold on;
                    %resize original image and show where the pixels in the
                    %resized image lie (prove ~1-1  mapping after binning for any roi)
                    for k=baseroi_num+1:roi_num
                        if k==baseroi_num+1
                            %color='.b';
                        else
                            %color='.g';
                        end
                        plot(roi_pixels(k).x_border,roi_pixels(k).y_border,'r')
                        hold on
                        %plot(roi_pixels(k).x_pixels,roi_pixels(k).y_pixels,color)
                        plot(roi_pixels(k).x_pixels,roi_pixels(k).y_pixels)%plot_colors(k-baseroi_num,:))
                        hold on
                        %plot(roi_pixels(k).fullxpixel,roi_pixels(k).fullypixel,'Color',plot_colors(k-baseroi_num,:))

                    end
                    
                    if strcmp(exclude_roi,'y') == 1
                        for k = 1:count-1
                            plot(roi_pixels(1).excluderoi(k).x_border,roi_pixels(1).excluderoi(k).y_border,'r')
                            hold on
                        end
                    end

                end
            end

    end
    frame = frame+1;
end

end

function [first_frame] = firstblueframe(r2,frame_color)% sub function
%grabs the first blue frame after a green frame if the video analysis
%starts with a green frame since ROIs should be drawn on a blue frame image
%(intensities are higher, more regions are visible) 
frame = 1;
while hasFrame(r2)
    single_frame = readFrame(r2); %grab current frame
    if size(single_frame,3)==3
        single_framegraylarge=rgb2gray(single_frame); %convert rgb to grayscale
    else
        single_framegraylarge = single_frame;
    end
    
    if frame == frame_color(1).black(end)+2%look at first frame after black frame if it is green, make it blue
       first_frame=single_framegraylarge;%store a copy of first frame
       break
    end
    frame = frame+1;
end


end

%% get points for circles with a given center and radius
function [xunit,yunit] = circle_coord(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
%                 h = plot(xunit, yunit);
%                 hold off
end