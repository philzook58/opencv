#include “opencv2/opencv.hpp” // joint “main” opencv header

// include opencv_contrib, hal etc. headers separately …



// make code wrist- and eyes-friendly

// … or put your code into cv[::nested_namespace] namespace

using namespace cv;

using namespace std;



// for most experiments you do not need full-scale GUI app.

// just do the things and display results with highgui

int main(int argc, char** argv) {

// dst is created automatically
	// dst is created automatically


Mat src = imread(argv[1]), dst;


imshow(“test”, src); // no need in namedWindows()

waitKey(); // do not forget it


return 0; // skip cleanup things

}