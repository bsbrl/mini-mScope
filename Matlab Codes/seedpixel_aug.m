%% Seed pixel map function
%this function calcualtes seed pixel maps with a moving time window by
%using pearson's correlation coefficient. The window length and size can be
%set below with "stepsize" and "increment" variables. The MCchoice will run
%a moving recursive window through the scored data given to it. The SC
%choice will run a moving window with no backwards recursion, thus is less
%computationally intensive to run

%%
function [dff_roi] = seedpixel_aug(roi_pixels,mask_imgpixels,roi_index,scoring,behaviors,dff_data,behavior_index,dff_roi,imseq,background_roi,analyze_roi,folder,paths,calculation_type,dff_type,cluster_case,fields_name)

switch calculation_type %choose whether you're running seed pixel analysis or ROI correlation map analysis
    case 'seed pixel analysis' %individual seed pixel map
        wt2=waitbar(0,'Seed pixel analysis for points within ROIs');%progress bar to see how code processsteps=len;
        steps2 = size(roi_pixels(roi_index).circle_points.xpoints,1)+1;%total frames
        set(wt2,'Units', 'normalized');
        % Change the size of the figure
        set(wt2,'Position', [0.35 0.4 0.3 0.08]);
        runcount = 1;
        for p = 1:size(roi_pixels(roi_index).circle_points.xpoints,1)+1%number of circle radii + centroid of ROI
            for q = 1:size(roi_pixels(roi_index).circle_points.xpoints,2)%number of points per circle radii
                if p == size(roi_pixels(roi_index).circle_points.xpoints,1)+1
                    if runcount == 1 %only need to run centroid once since only 1 centroid per ROI
                        %%
                        roi_fields = fields(roi_pixels);
                        for i = 1:size(roi_fields,1)
                            if strcmp(roi_fields{i},'xseed') == 1 %if seed pixels drawn manually
                                choose_seed = 1;
                            end
                        end
                        
                        temp_trace = zeros(size(mask_imgpixels(2).excluderoi.y_pixels,1)-1,size(dff_data,3)); %preallocate, mask pix x frames
                        tmp = zeros(size(dff_data,1),size(dff_data,2)); %vec to get rid of centroid pixel in analysis
                        tmp(sub2ind(size(tmp),mask_imgpixels(2).excluderoi.y_pixels,mask_imgpixels(2).excluderoi.x_pixels)) = 1;
                        if choose_seed == 1
                            seedpixel_trace = squeeze(dff_data(roi_pixels(roi_index).xseed(1),roi_pixels(roi_index).yseed(1),:));
                            tmp(sub2ind(size(tmp),roi_pixels(roi_index).xseed(1),roi_pixels(roi_index).yseed(1))) = 0; %remove centroid pixel
                        else
                            seedpixel_trace = squeeze(dff_data(roi_pixels(roi_index).centroidxy(2),roi_pixels(roi_index).centroidxy(1),:));
                            tmp(sub2ind(size(tmp),roi_pixels(roi_index).centroidxy(2),roi_pixels(roi_index).centroidxy(1))) = 0; %remove centroid pixel
                        end
%                         seedpixel_trace = squeeze(dff_data(roi_pixels(roi_index).centroidxy(2),roi_pixels(roi_index).centroidxy(1),:));
                        %temp_trace = zeros(size(dff_data,1)*size(dff_data,2)-1,size(dff_data,3));
                        
                        
                         
                        
                        [ynotseed,xnotseed] = find(tmp == 1);%find pixels not containing centroid and within mask and excluding ROIs
                        %[ynotseed,xnotseed] = find(tmp == 0);
                        for t = 1:size(dff_data,3)
                            currentframe = squeeze(dff_data(:,:,t));
                            temp_trace(:,t) = currentframe(sub2ind(size(currentframe),ynotseed,xnotseed));
                        end
                        wt3=waitbar(0,'Cluster seed pixel analysis');%progress bar to see how code processsteps=len;
                        steps3 = size(temp_trace,1);%total clusters
                        set(wt3,'Units', 'normalized');
                        % Change the size of the figure
                        set(wt3,'Position', [0.35 0.4 0.3 0.08]);
                        for i = 1:size(temp_trace,1) %iterate through every pixel not in ROI
                            %r = corrcoef(dff_roi(1).roi(j,scoring(1).moving_frames_ind),dff_roi(1).seedpixel(j).time(i,scoring(1).moving_frames_ind));
                            r = corrcoef(seedpixel_trace(scoring(1).(behaviors{behavior_index}).ind),temp_trace(i,scoring(1).(behaviors{behavior_index}).ind));
                            dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).centroid_r(i,1) = r(1,2);
                            
                            
                            %Set stepsize and increment based on which
                            %cluster choice is chosen. For MC, the window
                            %length is used to grab data of that length for
                            %pearson's correlation coefficient and then the
                            %increment is used to move to the next window
                            %length with a backwards recursive time window.
                            %For SC, the window size is larger and doesn't
                            %contain a recursive window step
                            start = 1;
                            if strcmp(cluster_case,'MC') == 1
                                stepsize = round(15);% step size of 1s (15fps)
                                increment = round(8); %increase stepsize by 0.5s (8fps)
                            elseif strcmp(cluster_case,'SC') == 1
                                stepsize = round(2*15); %stepsize of n*1s(fps)
                                increment = round(2*15); %increase stepsize to move in n*1s(fps) increment
                            end

                            groups = find(diff(scoring.(behaviors{behavior_index}).ind) > 2);%find members with more than 2 frame drops
                            groupsize = size(groups,1)+1;

                            for g = 1:size(groups,1)+1
                                if g == size(groups,1)+1%case for last cluster
                                    if isempty(groups) == 1 %no clusters i.e. one long continous behavior
                                        cluster = scoring.(behaviors{behavior_index}).ind;
                                    else
                                        cluster = scoring.(behaviors{behavior_index}).ind(groups(g-1)+1:end);%get clusters between scoring areas
                                        if size(scoring.(behaviors{behavior_index}).ind,1) - groups(end) < stepsize
                                            groupsize = size(groups,1);
                                        end
                                    end
                                else
                                    cluster = scoring.(behaviors{behavior_index}).ind(start:groups(g));
                                end
%                                         if i == 1
%                                             fprintf('Last cluster length %1.0f\n',length(cluster))
%                                         end
                                r = corrcoef(seedpixel_trace(cluster),temp_trace(i,cluster));
                                dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).cluster_r(i,g) = r(1,2);

                                window_start = 1;%first frame in cluster
                                window_count = 1;
                                
                                %calculate pearsons correlation coefficient
                                %for the current window length

                                while window_start + stepsize - 1 <= length(cluster)

                                    moving_cluster = cluster(window_start:window_start+stepsize-1); %moving window cluster
                                    if increment == 1
                                        window_start = window_start + increment;%account for single frame increment analysis
                                    else
                                        window_start = window_start + increment - 1;%increment start by stepsize
                                    end

                                    r = corrcoef(seedpixel_trace(moving_cluster),temp_trace(i,moving_cluster));%moving corr coeff between clusters in scoring behavior
                                    dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(g).r(i,window_count) = r(1,2);
                                    window_count = window_count+1;
                                end

                            end
                            
                            if mod(i,20)==0
                                waitbar(i/steps3,wt3,sprintf('Completed moving window pixel correlation for pixel %1.0f/%1.0f with %1.0f subclusters',i,steps3,size(groups,1)+1))
                            end
                        end
                        close(wt3)
                        

                        for g = 1:groupsize
                            if isempty(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(g).r) == 1 %no clusters
                                gcount = g+1;%start search in next element to see if cluster non-empty
                                dff_roi_empty = 'y';
                                while dff_roi_empty == 'y'%clusters of empty cases 
                                    if isempty(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(gcount).r) == 0
                                        dff_roi_empty = 'n';
                                        dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(1).rmean(:,g) = NaN*ones(size(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(gcount).r,1),1);
                                    end
                                    gcount = gcount+1;
                                end
                            else
                                dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(1).rmean(:,g) = mean(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(g).r,2);
                            end
                        end
                            
                        
                        %% calculates 3 options for seed pixel maps
                        %the first option looks at large clusters of time
                        %i.e. the correlation coefficient of the entire
                        %trace. The second cluster method looks at smaller
                        %epochs of time arising from different scoring
                        %epochs of time during mouse behavior. The third
                        %cluster methods looks at clusters within the
                        %second method to have corerlation coefficeints of
                        %~1 second epochs of time throughout the traces

                        for g = 1:3 %3 methods added for spatial map
                            if choose_seed == 1
                                txt = 'chosen seed';
                            else
                                txt = 'centroid';
                            end
                            if g == 2
                                var = nanmean(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).cluster_r,2);
                                titletxt = strsplit(behaviors{behavior_index},'_');
                                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                                titletxt = strcat('ROI',{' '},num2str(roi_index-background_roi-1),{' '},fields_name,{' '},'trace',{' '},titletxt{1},{' '},'cluster',{' '},txt,{' '},'seed pixel');
                                ext_txt = 'cluster';
                            elseif g == 3
                                var = nanmean(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(1).rmean,2);
                                titletxt = strsplit(behaviors{behavior_index},'_');
                                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                                titletxt = strcat('ROI',{' '},num2str(roi_index-background_roi-1),{' '},fields_name,{' '},'trace',{' '},titletxt{1},{' '},'moving cluster',{' '},txt,{' '},'seed pixel');
                                ext_txt = 'moving cluster';
                            else
                                var = dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).centroid_r;
                                titletxt = strsplit(behaviors{behavior_index},'_');
                                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                                titletxt = strcat('ROI',{' '},num2str(roi_index-background_roi-1),{' '},fields_name,{' '},'trace',{' '},titletxt{1},{' '},txt,{' '},'seed pixel');
                                ext_txt = 'whole trace';
                            end
                            
                            %generate seed pixel heat maps and exclude areas
                            %that are outside of the originally drawn mask
                            im = NaN*ones(imseq(1).(behaviors{behavior_index}).imax,imseq(1).(behaviors{behavior_index}).jmax);
                            if choose_seed == 1
                                im(sub2ind(size(im),roi_pixels(roi_index).xseed(1),roi_pixels(roi_index).yseed(1))) = 1;
                            else
                                im(sub2ind(size(im),roi_pixels(roi_index).centroidxy(2),roi_pixels(roi_index).centroidxy(1))) = 1;
                            end
                            im(sub2ind(size(im),ynotseed,xnotseed)) =  var;

                            if g == 1
%                                 dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).centroid_r.seedmap = im;
                            elseif g == 2
%                                 dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).cluster_r.seedmap = im;
                            else
                                dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).movingcluster(1).seedmap = im;
                            end
                            %im(sub2ind(size(im),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels)) = 1;
                            %im(sub2ind(size(im),roi_pixels(j).ynot_pixels,roi_pixels(j).xnot_pixels)) = dff_roi(1).seedpixel(j).rmove;

                            figure(roi_index)
                            colormap('jet');
                            pic = imagesc(im);
                            caxis([-0.6,1])
                            colorbar
                            set(gca,'xtick',[])
                            set(gca,'ytick',[])
                            hold on

                            if g == 3

                                if roi_index == background_roi + 2 %first ROI, create diagonal matrix
                                    dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(1).corrcoeff = eye(length(2:(background_roi + 1 + analyze_roi))); %initialize corr coeff
                                end


                                for j = 2:(background_roi + 1 + analyze_roi)%cross correlational matrix between seed pixels
                                    dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(1).corrcoeff(j-1,roi_index-1) = im(sub2ind(size(im),roi_pixels(j).xseed(1),roi_pixels(j).yseed(1)));    
                                end
                                
                                %% additional ways to represent the seed pixel results

%                                     currentrow = dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(1).(strcat('corrcoeff_',sub_categories{behavior_index,z}))(:,roi_index-1);
%                                     currentroi = find(currentrow == 1);
%                                     notroi = setdiff(1:length(currentrow),currentroi);
%                                     partitions(:,1) = linspace(-1,1,size(jet,1))';%range from -1 to 1 for corr coeff
%                                     partitions(:,2) = linspace(1,5,size(jet,1))';%range from 1 to 10 for line width
%                                     colors = gray;
%                                     %colors(linewidth_ind,:)
%                                     for j = 1:length(notroi)
%                                         [~,linewidth_ind] = min(abs(dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(1).(strcat('corrcoeff_',sub_categories{behavior_index,z}))(notroi(j),roi_index-1) - partitions(:,1)))
%                                         linewidth(j) = partitions(linewidth_ind,2);
% 
%                                         xroi_line = [roi_pixels(currentroi+1).yseed,roi_pixels(notroi(j)+1).yseed];
%                                         yroi_line = [roi_pixels(currentroi+1).xseed,roi_pixels(notroi(j)+1).xseed];
%                                         plot(xroi_line,yroi_line,'Color',colors(linewidth_ind,:),'LineWidth',3)
%                                         hold on
%                                     end

                            end
                            title(titletxt{1})
                            saveas(figure(roi_index),strcat(folder,paths{3},'\',fields_name,'\',ext_txt,'\',titletxt{1},'.jpeg'))
                            saveas(figure(roi_index),strcat(folder,paths{3},'\',fields_name,'\',ext_txt,'\',titletxt{1},'.fig'))
                            close(figure(roi_index))
                        end
                              
                        runcount = runcount+1;
                    end

                else
                    
                    seedpixel_trace = squeeze(dff_data(roi_pixels(roi_index).circle_points.ypoints(q,p),roi_pixels(roi_index).circle_points.xpoints(q,p),:));
                    temp_trace = zeros(size(mask_imgpixels(2).excluderoi.y_pixels,1)-1,size(dff_data,3)); %preallocate, mask pix x frames
                    tmp = zeros(size(dff_data,1),size(dff_data,2)); %vec to get rid of centroid pixel in analysis
                    tmp(sub2ind(size(tmp),mask_imgpixels(2).excluderoi.y_pixels,mask_imgpixels(2).excluderoi.x_pixels)) = 1;
                    tmp(sub2ind(size(tmp),roi_pixels(roi_index).circle_points.ypoints(q,p),roi_pixels(roi_index).circle_points.xpoints(q,p))) = 0; %remove centroid pixel 

                    [ynotseed,xnotseed] = find(tmp == 1);%find pixels not containing centroid and within mask and excluding ROIs
                    %[ynotseed,xnotseed] = find(tmp == 0);
%                     temp_trace = zeros(size(dff_data,1)*size(dff_data,2)-1,size(dff_data,3));
                    for t = 1:size(dff_data,3)
                        currentframe = squeeze(dff_data(:,:,t));
%                         if t == 1
%                             temp_trace_size = size(currentframe(sub2ind(size(currentframe),ynotseed,xnotseed)));
%                             temp_trace = zeros(temp_trace_size(1),size(dff_data,3));%preallocate pixel vector
% %                             temp_trace = zeros(size(mask_imgpixels(2).excluderoi.y_pixels,1)-1,size(dff_data,3)); %preallocate, mask pix x frames
%                         end
                        temp_trace(:,t) = currentframe(sub2ind(size(currentframe),ynotseed,xnotseed));
%                         temp_trace(:,t) = currentframe(sub2ind(size(currentframe),roi_pixels(roi_index).circle_points.ynotcircle(:,q,p),roi_pixels(roi_index).circle_points.xnotcircle(:,q,p)));
                    end

                    for i = 1:size(temp_trace,1) %iterate through every pixel not in ROI
                        %r = corrcoef(dff_roi(1).roi(j,scoring(1).moving_frames_ind),dff_roi(1).seedpixel(j).time(i,scoring(1).moving_frames_ind));
                        r = corrcoef(seedpixel_trace(scoring(1).(behaviors{behavior_index}).ind),temp_trace(i,scoring(1).(behaviors{behavior_index}).ind));

                        dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).circle_r(i,q,p) = r(1,2);
                    end

                    im = NaN*ones(imseq(1).(behaviors{behavior_index}).imax,imseq(1).(behaviors{behavior_index}).jmax);
                    im(sub2ind(size(im),roi_pixels(roi_index).circle_points.ypoints(q,p),roi_pixels(roi_index).circle_points.xpoints(q,p))) = 1;
                    im(sub2ind(size(im),ynotseed,xnotseed)) =  dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).circle_r(:,q,p);
%                         im(sub2ind(size(im),roi_pixels(roi_index).circle_points.ynotcircle(:,q,p),roi_pixels(roi_index).circle_points.xnotcircle(:,q,p))) =  dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(roi_index).(strcat('circle_r_',sub_categories{behavior_index,z}))(:,q,p);
                    %im(sub2ind(size(im),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels)) = 1;
                    %im(sub2ind(size(im),roi_pixels(j).ynot_pixels,roi_pixels(j).xnot_pixels)) = dff_roi(1).seedpixel(j).rmove;

                    figure(roi_index)
                    colormap('jet');
                    pic = imagesc(im);
                    caxis([-0.6,1])
                    colorbar
                    set(gca,'xtick',[])
                    set(gca,'ytick',[])
                    titletxt = strsplit(behaviors{behavior_index},'_');
                    titletxt = strcat(titletxt{1},{' '},titletxt{2});
                    titletxt = strcat('ROI',num2str(roi_index-background_roi-1),{' '},{' '},fields_name,{' '},'trace',{' '},titletxt{1},{' '},'circle',{' '},num2str(p),{' '},'seed pixel',{' '},num2str(q));
                    title(titletxt{1})
                    saveas(figure(roi_index),strcat(folder,paths{3},'\Circle Seed Pixel Points','\',titletxt{1},'.jpeg'))
                    saveas(figure(roi_index),strcat(folder,paths{3},'\Circle Seed Pixel Points','\',titletxt{1},'.fig'))
                    close(figure(roi_index))
                end

                waitbar(p/steps2,wt2,sprintf('Completed analysis for seed pixel %1.0f/%1.0f of seed pixel region %1.0f/%1.0f within ROI %1.0f/%1.0f',q,size(roi_pixels(roi_index).circle_points.xpoints,2),p,size(roi_pixels(roi_index).circle_points.xpoints,1)+1,roi_index-background_roi-1,background_roi + analyze_roi))

            end
        end
        close(wt2)
        
    case 'cross correlation matrix' %cross correlational matrix with ROIs
        close all;
        for j = background_roi+2:size(dff_roi(1).(behaviors{behavior_index})(dff_type).roi,1)
           ticks{j-background_roi-1} = strcat('ROI',num2str(j-1)) ;
           ticks2{j-background_roi-1} = strcat('ROI',num2str(j-1),'Seed') ;
        end

        %%
        [dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}),~,~,~] = corrcoef(dff_roi(1).(behaviors{behavior_index})(dff_type).(strcat('roi_',behaviors{behavior_index}))(2:end,:)','Rows','complete');%corr coeff for behavior 1
        %Rmin_still = min(min(Rb1,[],'omitnan'));
        dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.Rmin = min(min(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}),[],'omitnan'));

        start = 1;
        stepsize = 15;% step size of 1s (15fps)
        increment = 8; %increase stepsize by 0.5s (8fps)
        groups = find(diff(scoring.(behaviors{behavior_index}).ind) > 2);%find members with more than 2 frame drops
        groupsize = size(groups,1)+1;

        for g = 1:size(groups,1)+1
            if g == size(groups,1)+1%case for last cluster
                if isempty(groups) == 1 %no clusters i.e. one long continous behavior
                    cluster = scoring.(behaviors{behavior_index}).ind;
                else
                    cluster = scoring.(behaviors{behavior_index}).ind(groups(g-1)+1:end);%get clusters between scoring areas
                    if size(scoring.(behaviors{behavior_index}).ind,1) - groups(end) < stepsize
                        groupsize = size(groups,1);
                    end
                end
            else
                cluster = scoring.(behaviors{behavior_index}).ind(start:groups(g));
            end
            
            dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.cluster_r(:,:,g) = corrcoef(dff_roi(1).(behaviors{behavior_index})(dff_type).roi(2:end,cluster)','Rows','complete');%corr coeff for behavior 1
            
            window_start = 1;%first frame in cluster
            window_count = 1;

            while window_start + stepsize -1 <= length(cluster)

                moving_cluster = cluster(window_start:window_start+stepsize-1); %moving window cluster
                window_start = window_start + increment - 1;%increment start by stepsize

                dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(g).r(:,:,window_count) = corrcoef(dff_roi(1).(behaviors{behavior_index})(dff_type).roi(2:end,moving_cluster)','Rows','complete');%corr coeff for behavior 1

                window_count = window_count+1;
            end

        end
        dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.clustermean = mean(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.cluster_r,3);
        dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.clusterRmin = min(min(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.clustermean,[],'omitnan'));

        for g = 1:groupsize
            dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).rmean(:,:,g) = mean(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(g).r,3);
        end
        dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).rmean = mean(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).rmean,3);
        dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).Rmin = min(min(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).rmean,[],'omitnan'));

        for g = 1:3 %3 cases to plot
            if g == 2
                low = .05*floor(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.clusterRmin/.05);
                var = dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.clustermean;
                xytick = (.5 + (0:length(var)-1));
                titletxt = strsplit(behaviors{behavior_index},'_');
                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                titletxt = strcat(fields_name,{' '},'Corr Coeff Matrix For Cluster Method For',{' '},titletxt{1});
%                     cbound = [min([dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(strcat(sub_categories{behavior_index,z},'_clusterRmin')),dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(strcat(sub_categories{behavior_index,z},'_clusterRmin'))]),1];
            elseif g == 3
                low = .05*floor(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).Rmin/.05);
                var = dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.movingcluster(1).rmean;
                xytick = (.5 + (0:length(var)-1));
                titletxt = strsplit(behaviors{behavior_index},'_');
                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                titletxt = strcat(fields_name,{' '},'Corr Coeff Matrix For Moving Cluster Method For',{' '},titletxt{1});
%                     cbound = [min([dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(strcat(sub_categories{behavior_index,z},'_movingcluster'))(1).Rmin,dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(strcat(sub_categories{behavior_index,z},'_movingcluster'))(1).Rmin]),1];

            else
                low = .05*floor(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.Rmin/.05);
                var = dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index});
                xytick = (.5 + (0:length(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}))-1));
                titletxt = strsplit(behaviors{behavior_index},'_');
                titletxt = strcat(titletxt{1},{' '},titletxt{2});
                titletxt = strcat(fields_name,{' '},'Corr Coeff Matrix For',{' '},titletxt{1});
%                     cbound = [min([dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(strcat('Rmin_',sub_categories{behavior_index,z})),dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(strcat('Rmin_',sub_categories{behavior_index,1}))]),1];
            end

            figure(1)

            imagesc(var,[low 1]);
            set(gca,'XTick',1:size(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}),1))
            set(gca,'Ytick',1:size(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}),1))
            set(gca,'XTickLabel',xytick)
            set(gca,'YTickLabel',xytick)

            set(gca,'xticklabel',ticks,'yticklabel',ticks,'xaxisLocation','top')
            colormap(hot);
            caxis([-1,1])

            colorbar

            title(titletxt{1})
            saveas(figure(1),strcat(folder,paths{3},'\',fields_name,'\',titletxt{1},'.jpeg'))
            saveas(figure(1),strcat(folder,paths{3},'\',fields_name,'\',titletxt{1},'.fig'))
            close(figure(1))
        end


        low = .05*floor(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.clusterRmin/.05);
        var = dff_roi(1).(behaviors{behavior_index})(dff_type).seedpixel(1).corrcoeff;
        xytick = (.5 + (0:length(var)-1));
        titletxt = strsplit(behaviors{behavior_index},'_');
        titletxt = strcat(titletxt{1},{' '},titletxt{2});
        titletxt = strcat(fields_name,{' '},'Corr Coeff Matrix For Moving Cluster Seed Pixel For',{' '},titletxt{1});

        figure(2)

        imagesc(var,[low 1]);
        set(gca,'XTick',1:size(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}),1))
        set(gca,'Ytick',1:size(dff_roi(1).(behaviors{behavior_index})(dff_type).corrcoeff.(behaviors{behavior_index}),1))
        set(gca,'XTickLabel',xytick)
        set(gca,'YTickLabel',xytick)

        set(gca,'xticklabel',ticks2,'yticklabel',ticks2,'xaxisLocation','top')
        colormap(hot);
        caxis([-1,1])

        colorbar

        title(titletxt{1})
        saveas(figure(2),strcat(folder,paths{3},'\',fields_name,'\',titletxt{1},'.jpeg'))
        saveas(figure(2),strcat(folder,paths{3},'\',fields_name,'\',titletxt{1},'.fig'))
        close(figure(2))
            
end
        
        

end