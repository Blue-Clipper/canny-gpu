#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
class ImageIO {
    public:
        ImageIO(string addr);
        ImageIO(int *imageLine, int rows, int cols);
        int getRows();
        int getCols();
        void getImageLine(int **ImageLine);
        bool imageWrite(string addr);
        int _rows;
        int _cols;
        int *_imageLine;
        cv::Mat _imageMat;
};