#HorizonFinder
This a CUDA/C++ implementation of a horizon finder that uses a Canny edge detector and Hough transform. It is optimized for finding horizons in marine images. If it detects two horizons in close proximity of each other (such as the shoreline and skyline), it will choose the lower one. It prints out the detected horizon line in a text file called houghLines.txt in slope intercept form, where the first value is the slope and the second value is the y-intercept. The coordinates are in image space, with the origin (0, 0) at the top left corner of the image, and the point (width, height) at the bottom right corner of the image.


##Running the Detector
It reads a grayscale image as a file of pixel intensities. The grayscale intensities are organized by flattening the matrix of pixel values down to a 1D array of values in row-major order. The values are newline delimited. The Matlab function in preprocess.m will automatically generate a text file in the correct format from an image input.

The Matlab function showHorizon will display the result of running the horizon detector on the image, given the image as an input.

Run the horizon finder with ./houghTransform 'inputfile' 'imagewidth' 'imageheight'
where inputfile is the newline delimited flattened image.
