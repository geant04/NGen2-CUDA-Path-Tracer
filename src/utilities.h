#pragma once

#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define INV_PI            0.3183098861837906715377676381983307306404f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

#define RUSSIAN_ROULETTE_FLAG 1
#define RAYTRACE_DEBUG_FLAG 2
#define USE_BVH_FLAG 3

struct GuiDataSettings
{
    int settingFlags;
    int bounces;

    float focalLengthDOF = 5.0f;
    float apertureDOF = 2.0f;
    bool useDOF = false;

    bool useACES = false;

    bool useAA = true;
    bool usePartition = false;
    bool useMaterialSort = false;

    bool useRussianRoulette = false;
    bool useDebugShader = false;
    bool useBVH = true;
};

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0), settings{} {}
    int TracedDepth;
    GuiDataSettings settings;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
