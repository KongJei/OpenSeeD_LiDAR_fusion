#ifndef COLOR_MAP_H
#define COLOR_MAP_H

#include <opencv2/opencv.hpp>
#include <vector>

inline std::vector<cv::Vec3b> getColorMap() {
    std::vector<cv::Vec3b> color_map;

    // Define colors (BGR format)
    color_map.push_back(cv::Vec3b(0, 0, 0));         // 0: Black
    color_map.push_back(cv::Vec3b(0, 0, 255));       // 1: Red
    color_map.push_back(cv::Vec3b(0, 255, 0));       // 2: Green
    color_map.push_back(cv::Vec3b(255, 0, 0));       // 3: Blue
    color_map.push_back(cv::Vec3b(0, 255, 255));     // 4: Cyan
    color_map.push_back(cv::Vec3b(255, 255, 0));     // 5: Yellow
    color_map.push_back(cv::Vec3b(255, 0, 255));     // 6: Magenta
    color_map.push_back(cv::Vec3b(192, 192, 192));   // 7: Gray
    color_map.push_back(cv::Vec3b(128, 0, 0));       // 8: Dark Red
    color_map.push_back(cv::Vec3b(128, 128, 0));     // 9: Olive
    color_map.push_back(cv::Vec3b(0, 128, 0));       // 10: Dark Green
    color_map.push_back(cv::Vec3b(128, 0, 128));     // 11: Purple
    color_map.push_back(cv::Vec3b(0, 128, 128));     // 12: Teal
    color_map.push_back(cv::Vec3b(255, 128, 0));     // 13: Orange
    color_map.push_back(cv::Vec3b(255, 128, 128));   // 14: Light Orange
    color_map.push_back(cv::Vec3b(128, 255, 0));     // 15: Light Green
    color_map.push_back(cv::Vec3b(128, 255, 128));   // 16: Light Cyan
    color_map.push_back(cv::Vec3b(0, 255, 128));     // 17: Light Teal
    color_map.push_back(cv::Vec3b(0, 255, 255));     // 18: Light Cyan
    color_map.push_back(cv::Vec3b(255, 0, 128));     // 19: Light Red
    color_map.push_back(cv::Vec3b(255, 0, 255));     // 20: Light Magenta
    color_map.push_back(cv::Vec3b(128, 128, 128));   // 21: Medium Gray
    color_map.push_back(cv::Vec3b(255, 128, 128));   // 22: Light Orange
    color_map.push_back(cv::Vec3b(128, 255, 128));   // 23: Light Green
    color_map.push_back(cv::Vec3b(128, 128, 255));   // 24: Light Blue
    color_map.push_back(cv::Vec3b(255, 255, 128));   // 25: Light Yellow
    color_map.push_back(cv::Vec3b(255, 255, 255));   // 26: White
    color_map.push_back(cv::Vec3b(128, 0, 128));     // 27: Dark Purple
    color_map.push_back(cv::Vec3b(0, 128, 255));     // 28: Light Blue
    color_map.push_back(cv::Vec3b(128, 128, 0));     // 29: Dark Olive

    return color_map;
}

#endif // COLOR_MAP_H
