%% take input image, limits, and colorbar data andn make a colormap
%used to overlay a colormap onto a grayscale image to make it into a
%pseudocolor map without resizing the image like imagsc does
%Daniel Surinach, March 2021

%%
function [newcolormap] = colormap_gen(a,b,del,currentframe,jet_colors)


currentframe = currentframe(1:end)'; %transpose frame data
colorbar_interval = linspace(a-del,b+del,size(jet_colors,1))';%make list with jet color range
newcolormap=zeros(size(currentframe,1),3); %preallocate newcolormap
sorted_intensityvec = struct('ind',{},'fullind',{});%vec with indices for partitioning data into jet colormap region

%takes in an input frame and rescales it between the limits desired. This
%will assign values in a matrix a row listed in the desired colorbar (i.e.
%jet colors for example) and rescale it between 0 and 1 for video writing
for j = 1:size(colorbar_interval,1)%index through partitions and find elements existing here
    tmpind = find(currentframe <= colorbar_interval(j));%find data less than current point
    sorted_intensityvec(j).fullind = tmpind; %store full indices
    if j > 1
        sorted_intensityvec(j).ind = setdiff(tmpind,sorted_intensityvec(j-1).fullind,'stable');%find current values in data set less than colorjet
        %and compare them to previous found colors and eliminate
        %duplicates

        if isempty(sorted_intensityvec(j).ind) == 0 %found a value in range of data set
           for colorind = 1:length(sorted_intensityvec(j).ind)%go through all elements in current set
              newcolormap(sorted_intensityvec(j).ind(colorind),:) = jet_colors(j-1,:);%assign j-1 (floor) color
           end
        end

    else %else j = 1, assign the first color scheme to it
        sorted_intensityvec(j).ind = tmpind;
        if isempty(sorted_intensityvec(j).ind) == 0 %found a value in range
           for colorind = 1:length(sorted_intensityvec(j).ind)
              newcolormap(sorted_intensityvec(j).ind(colorind),:) = jet_colors(j,:); 
           end

        end
    end
    
    if j == size(colorbar_interval,1) %final index
        if size(tmpind,1) < size(currentframe,1) 
            tmpind2 = find(currentframe > colorbar_interval(j));
            for colorind = 1:length(tmpind2)
                newcolormap(tmpind2(colorind),:) = jet_colors(j,:);
            end
        end
    end

end

%% color, old function
% scaled_vec=rescale(intensity_vector,maxmin(1),maxmin(2));%rescale to min and max f/fo across all frames
% max(scaled_vec)
% min(scaled_vec)
% n = size(scaled_vec,1);% number of colors
% %C = [0,0,0.5625;0.5,0,0];
% C = [0 0 1; 1 0 0];% color map with red and blue on the edges
% C_HSV = rgb2hsv(C);% convert to HSV for interpolation
% C_HSV_interp = interp1([0 n], C_HSV(:, 1), 1:n);% interpolate hue value
% C_HSV = [C_HSV_interp(:), repmat(C_HSV( 1, 2:3), n, 1)];% compose full HSV colormap
% C = hsv2rgb(C_HSV);% convert back to RGB
% size(C)
% [~,ind]=sort(scaled_vec);%numerically sort all values as they may be out of order, keep indices of where sorted numbers belong
% 
% newcolormap=zeros(size(scaled_vec,1),3);
% newcolormap(ind,:)=C;%assign colors from blue (low intensity) to red (high intensity)
% %depending on their index from when they were sorted
% %i.e. [1 -5 0]->sort->[-5,0,1] with sorting index [2,3,1]
% %assign blue to pos.2, inbetween to pos.3, blue to pos.1


%% additional functions for color mapping that can be used for formatting
% for i=1:numframes
%     for j=1:roi_num
%         temp=zeros(size(single_framegray,1),size(single_framegray,2));
%         temp(sub2ind(size(single_framegray),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels))=(double(single_framegray(sub2ind(size(single_framegray),roi_pixels(j).y_pixels,roi_pixels(j).x_pixels)))-fosij(j))/(fosij(j))*100;
%         intensity_vector(j).delf_time(:,:,frame)=temp;
%     end
% end
% 
% %
% imshow(uint8(Sij(1).roi));
% colorMap = [linspace(0,1,256)', zeros(256,2)];
% colormap(colorMap);
% colorbar;
% 
%  c=jet(size(roi_pixels(2).x_pixels,1));
% [sortedDistances,sortIndexes] = sort(intensity_vector(2).delf_time(:,:,1));
% xs = roi_pixels(2).fullxpixel(sortIndexes);
% ys = roi_pixels(2).fullypixel(sortIndexes);
% imshow(first_frame)
% hold on
% scatter(xs,ys,1,c)
% 
% imshow(imresize(first_frame,bin_num,'bilinear'))
% hold on
% c=jet(size(roi_pixels(2).x_pixels,1));
% c2=jet(size(roi_pixels(3).x_pixels,1));
% scatter(roi_pixels(2).x_pixels,roi_pixels(2).y_pixels,1,c)
% hold on
% scatter(roi_pixels(3).x_pixels,roi_pixels(3).y_pixels,1,c2)
% 
% figure(2)
% imshow(uint8(Sij(1).roi))
% hold on
% map = hsv(255); % Or whatever colormap you want.
% hold on
% rgbImage = ind2rgb(uint8(Sij(1).roi), map); % im is a grayscale or indexed image.
% hold on
% imshow(rgbImage)
% hold on
% colormap(map)
% colorbar
% caxis([-2 2])
% %
% A=uint8(Sij(1).roi);
% RA = imref2d(size(Sij(1).roi));
% B = uint8(Sij(2).roi);
% RB = imref2d(size(B));
% RB.XWorldLimits = RA.XWorldLimits;
% RB.YWorldLimits = RA.YWorldLimits;
% C = imfuse(A,B,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
% C = imresize(C,0.5);
% imshow(C)
% [D,RD] = imfuse(A,RA,B,RB,'ColorChannels',[1 2 0]);
% D = imresize(D,0.5);
% imshow(D)
% 
% 
% %
% taken from "doc surf"
% [X,Y,Z] = peaks(30);
% surfc(X,Y,Z)
% tmp=intensity_vector(3).delf_time(:,:,1);
% tmp2=zeros(size(Sij(1).roi,1),size(Sij(1).roi,2));
% tmp2(sub2ind(size(tmp2),roi_pixels(3).y_pixels,roi_pixels(3).x_pixels))=tmp;
% 
% imshow(tmp2)
% colormap
% axis([-3 3 -3 3 -10 5])
% number of colors
% n = 20;
% color map with red and blue on the edges
% C = [0 0 1; 1 0 0];
% convert to HSV for interpolation
% C_HSV = rgb2hsv(C);
% interpolate hue value
% C_HSV_interp = interp1([0 n], C_HSV(:, 1), 1:n);
% compose full HSV colormap
% C_HSV = [C_HSV_interp(:), repmat(C_HSV(2:3), n, 1)];
% C_HSV = [C_HSV_interp(:), repmat(C_HSV( 1, 2:3), n, 1)];
% convert back to RGB
% C = hsv2rgb(C_HSV);
% set colormap
% 
% colormap(C)
% colorbar
end