#pragma once

#include <glm/glm.hpp>

#include <string>

class Image
{
public:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};
