%% Take input grayscale image and overlay a pseudocolor map
%takes an input image and overlays a desired colormap onto it
%Daniel Surinach, March 2021
%%
function [colorimage] = colorimage_gen(single_frame,newcolormap)
colorimage=double(cat(3,single_frame,single_frame,single_frame))/double(max(max(single_frame)));%3D vector with grayscale dims

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



for k=1:3 %assign color to R G B in color scheme
    colorimage(sub2ind(size(colorimage),roi_pixels(1).y_pixels,roi_pixels(1).x_pixels,k*ones(size(roi_pixels(1).y_pixels,1),1)))=newcolormap(:,k,1);

end    
end

