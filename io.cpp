#include "io.h"

ImageIO::ImageIO(string addr) {
    _imageMat = cv::imread(addr, cv::IMREAD_GRAYSCALE);
    _rows = _imageMat.rows;
    _cols = _imageMat.cols;
    _imageLine = new int[_rows * _cols];
    for(int i = 0; i < _rows; i ++) {
        for(int j = 0; j < _cols; j ++) {
            _imageLine[i * _cols + j] = (int)_imageMat.at<u_char>(i, j);
        }
    }
}

ImageIO::ImageIO(int *imageLine, int rows, int cols) {
    if(imageLine == NULL) {
        return;
    }
    _rows = rows;
    _cols = cols;
    _imageLine = new int[_rows * _cols];
    for(int i = 0; i < _rows * _cols; i ++) {
        _imageLine[i] = imageLine[i];
    }
    cv::Mat mat(rows, cols, CV_8UC1);
    for(int i = 0; i < _rows; i ++) {
        for(int j = 0; j < _cols; j ++) {
            if(_imageLine[i * _rows + j] > 255) {
                mat.at<u_char>(i, j) = 255;
            } else if (_imageLine[i * _rows + j] < 0) {
                mat.at<u_char>(i, j) = 0;
            } else {
                mat.at<u_char>(i, j) = (u_char)_imageLine[i * _cols + j];
            }
            
        }
    }
    _imageMat = mat;
}

int ImageIO::getRows() {
    return _rows;
}
int ImageIO::getCols() {
    return _cols;
}
void ImageIO::getImageLine(int **imageLine) {
    if(!_imageLine) {
        return;
    }
    *imageLine = new int[_rows * _cols];
    for(int i = 0; i < _rows * _cols; i ++) {
        *(*imageLine + i) = _imageLine[i];
    }
}
bool ImageIO::imageWrite(string addr) {
    if(_imageLine == NULL) {
        return false;
    }
    cv::imwrite(addr, _imageMat);
}

