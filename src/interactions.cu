#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#define DIFFUSION_PROFILE 1

__host__ __device__ glm::vec3 calculateHemisphereDirection(glm::vec3 normal, float cosTheta, float phi)
{
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return cosTheta * normal
        + cos(phi) * sinTheta * perpendicularDirection1
        + sin(phi) * sinTheta * perpendicularDirection2;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float cosTheta = sqrt(u01(rng)); // cos(theta)
    float phi = u01(rng) * TWO_PI;

    return calculateHemisphereDirection(normal, cosTheta, phi);
}

__host__ __device__ glm::vec3 calculateWalterGGXSampling(
    glm::vec3 normal,
    float roughness,
    thrust::default_random_engine &rng)
{
    //// https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
    //// Eq. 35, 36 for GGX sampling distribution
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u0 = u01(rng);
    float u1 = u01(rng);
    
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float cosTheta = sqrt((1.0f - u0) / (u0 * (alpha2 - 1.0f) + 1.0f)); 
    float sinTheta = sqrt(1 - cosTheta * cosTheta);
    float phi = TWO_PI * u1;

    return calculateHemisphereDirection(normal, cosTheta, phi);
}

// Have to include this stupid normal so the intellisense doesn't go crazy
__host__ __device__ glm::vec3 calculateRandomDirectionInSphere(float u1, float u2)
{
    float z = 2.0f * u1 - 1.0f;
    float phi = TWO_PI * u2;
    float r = sqrt(1.0f - z * z);

    return glm::vec3(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ float isotropicSampleDistance(
    float tFar, 
    float scatteringCoefficient, 
    float &weight,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float distance = -logf(u01(rng)) / scatteringCoefficient;

    if (distance >= tFar)
    {
        // pdf assignment that unfortunately doesn't exist
        return tFar;
    }

    weight = exp(-scatteringCoefficient * distance);
    return distance;
}

__host__ __device__ glm::vec3 isotropicTransmission(glm::vec3 absorptionCoefficient, float distance)
{
    return exp(-absorptionCoefficient * distance);
}

__host__ __device__ glm::vec3 isotropicSampleScatterDirection(float u1, float u2)
{
    return calculateRandomDirectionInSphere(u1, u2);
}

__host__ __device__ glm::vec3 henyeyGreensteinSampleDirection(float g, float u1, float u2, glm::vec3 wo)
{
    // Does u1, u2 need to be converted to -1,1 space? currently in 0,1 space
    // PBRT 4ed 11.3.1
    float sqTerm = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
    float cosThetaLocal = -1.0f / (2.0f * g)  * (1.0f + (g * g) - (sqTerm * sqTerm));

    // if g approx 0, our cosTheta is nearly isotropic (uniform random)
    if (abs(g)  < 0.01)
    {
        cosThetaLocal = 1.0f - 2.0f * u1;
    }

    float sinThetaLocal = sqrt(1.0f - cosThetaLocal * cosThetaLocal);
    float phi = TWO_PI * u2;

    // z = up vector
    float localX = glm::clamp(sinThetaLocal, -1.0f, 1.0f) * cos(phi);
    float localY = glm::clamp(sinThetaLocal, -1.0f, 1.0f) * sin(phi);
    float localZ = glm::clamp(cosThetaLocal, -1.0f, 1.0f);
    glm::vec3 localSphericalDirection = glm::vec3(localX, localY, localZ);

    // world normal, need to build ortho matrix
    glm::vec3 normal = wo;
    glm::vec3 tangent = glm::cross(abs(wo.x > 0.1f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0), normal);
    glm::vec3 bitangent = glm::cross(normal, tangent);

    return tangent * localX + bitangent * localY + normal * localZ;
}

// MASSIVE RAY SAMPLING FUNCTION!!!!!!!!!!!!!!!
__host__ __device__ void sampleRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 inDirection = glm::normalize(pathSegment.ray.direction);
    glm::vec3 outDirection;
    glm::vec3 halfVector = glm::normalize(inDirection + normal);

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Using Joe Schutte's Disney implementation for this
    float metallicWeight = m.metallic;
    float diffuseWeight = (1.0f - metallicWeight);
    float specularWeight = metallicWeight + diffuseWeight;

    float invSumWeight = 1.0f;

    float pDiffuse = diffuseWeight * invSumWeight;
    // float pSpecular = specularWeight * invSumWeight;

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, m.metallic);
    float pSpecular = glm::clamp((F0.x + F0.y + F0.z) / 3.0f, 0.02f, 0.98f);

    
    // Distance value used by transmission when within a medium
    float t = glm::length(intersect - pathSegment.ray.origin);
    float p = u01(rng);
    float epsilon = 0.0005f;
    glm::vec3 brdf = m.color;

    glm::vec3 wo = -inDirection;
    glm::vec3 wi;
    glm::vec3 intersectOffset;
    glm::vec3 diffuse_wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    glm::vec3 diffuseNormal = dot(normal, diffuse_wi) < 0.0f ? -diffuse_wi : diffuse_wi;

    if (pathSegment.medium != VACUUM)
    {
#if !DIFFUSION_PROFILE
        transmitMediumDiffusionBRDF(pathSegment, intersect, wo, wi, normal, t, m, rng, u01);
#else
        // https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering
        transmitMediumBRDF(pathSegment, intersect, wo, wi, normal, t, rng, u01);
#endif
        pathSegment.remainingBounces -= 1;
        return;
    }

    if (m.hasReflective)
    {
        // Specular GGX
        glm::vec3 microNormal = glm::normalize(calculateWalterGGXSampling(normal, m.roughness, rng));
        pathSegment.microNormal = microNormal;

        glm::vec3 specularDir = glm::reflect(inDirection, microNormal);

        float cosTheta = glm::max(dot(microNormal, wo), 0.0f);
        float f = fresnelDielectric(cosTheta, 1.0f, 1.45f);

        // metallic F experiments... YA BABY...
        glm::vec3 R0 = glm::mix(glm::vec3(0.04f), m.color, m.metallic);
        glm::vec3 metallicF = fresnelSchlick(R0, cosTheta);
        float metallicFavg = (0.2126f * metallicF.r + 0.7152f * metallicF.g + 0.0722f * metallicF.b);

        bool isSpecularBounce = p < glm::mix(f, metallicFavg, m.metallic);

        // Important brdf set. Don't remove this...
        brdf = glm::vec3(1.0f);
        if (m.metallic > 0.0f)
        {
            // Artifically set this true... 
            // Seems sus, but we'll work with it for now.
            // Unfortunately, this means our setup won't allow "half metal" materials to exist.
            isSpecularBounce = true; 
            brdf *= metallicF;
        }

        wi = glm::mix(
            diffuse_wi, 
            specularDir, 
            isSpecularBounce);
        brdf *= glm::mix(
            diffuseBRDF(wo, wi, normal, m), 
            // So this is most likely not correct at all, BUT:
            // Not multiplying specularBRDF with F so far gets me the closest result to Blender.
            // Now, I don't know why, but just an observation.
            dielectricSpecularBRDF(wo, wi, normal, microNormal, m), 
            isSpecularBounce);

        intersectOffset = normal * epsilon;
    }
    else if (m.subsurface > 0.0f)
    {
        glm::vec3 microNormal = glm::normalize(calculateWalterGGXSampling(normal, m.roughness, rng));
        normal = microNormal;

        float cosTheta = glm::max(0.0f, dot(normal, wo));
        float f = fresnelDielectric(cosTheta, 1.0f, 1.55f);

        if (u01(rng) < f)
        {
            pathSegment.ray.origin = intersect + normal * 0.001f;
            pathSegment.ray.direction = glm::reflect(-wo, normal);
        }
        else
        {
            wi = -wo;

            pathSegment.medium = ISOTROPIC;
            pathSegment.ray.origin = intersect + wi * 0.01f;
            pathSegment.ray.direction = wi;
        }
        
        pathSegment.remainingBounces -= 1;
        return;
    }
    else if (m.hasRefractive)
    {
        glm::vec3 microNormal = glm::normalize(calculateWalterGGXSampling(normal, m.roughness, rng));
        normal = microNormal;

        float cosThetaI = dot(normal, wo);        
        float etaA = 1.0f;
        float etaB = 1.55f;

        float rand = u01(rng);

        glm::vec3 R0 = glm::vec3((etaA - etaB) / (etaA + etaB));
        R0 = R0 * R0;
        glm::vec3 F = fresnelSchlick(R0, abs(cosThetaI));

        float f = fresnelDielectric(cosThetaI, etaA, etaB);

        brdf = glm::vec3(1.0f);

        if (rand < f)
        {
            wi = glm::reflect(glm::normalize(-wo), normal);
            //wi = glm::mix(wi, diffuseWi, m.roughness);
            brdf *= f;
        }
        else
        {
            // Transmissive material, use the specularBTDF
            float cosThetaI = dot(normal, wo);
            bool entering = cosThetaI > 0.0f;

            float eta = etaA / etaB;
            float iorRatio = (entering) ? eta : 1.0f / eta;

            wi = glm::refract(inDirection, (entering) ? normal : -normal, iorRatio);
            //wi = glm::mix(wi, (entering) ? -diffuseNormal : diffuseNormal, m.roughness);

            if (length(wi) < 0.01f)
            {
                wi = glm::reflect(inDirection, normal);
            }

            // absorptionMultiplier controls how much light absorbs color
            // Absorption is based on how far the light is within the medium.
            if (!entering)
            {
                float scatterDistance = t;
                glm::vec3 absorptionTint = exp(-scatterDistance * m.color * m.absorptionMultiplier);

                brdf *= absorptionTint;
            }

            brdf *= 1.0f - f;
        }

        pathSegment.ray.direction = wi;
        pathSegment.color *= brdf;
        pathSegment.ray.origin = intersect + wi * 0.001f;

        pathSegment.remainingBounces -= 1;
        return;
    }
    else
    {
        // Sample diffuse
        wi = diffuse_wi;
        brdf = diffuseBRDF(wo, wi, normal, m);
    }

    // Assign wi
    pathSegment.ray.direction = wi;

    // Assign intersect for the next bounce
    pathSegment.ray.origin = intersect + wi * epsilon; // wi * epsilon;

    // Yeah
    pathSegment.color *= brdf;

    pathSegment.remainingBounces -= 1;
}

__host__ __device__ glm::vec3 fresnelSchlick(glm::vec3 F0, float cosTheta)
{
    return F0 + (glm::vec3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

// Ripped straight from PBRT 3ed, thanks Google
__host__ __device__ float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // Potentially swap indices of refraction
    if(cosThetaI < 0.0f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = -cosThetaI;
    }

    // Compute cosTheta using Snell's law
    float sinThetaI = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Check for total internal reflection
    if(sinThetaT > 0.998f) {
        return 1;
    }

    float cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
}

__host__ __device__ glm::vec3 specularBTDF(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    const Material &m
)
{
    float cosThetaWo = dot(normal, wo);
    bool entering = cosThetaWo > 0;

    return glm::vec3(0.0f, 1.0f, 0.0f);
}

__host__ __device__ void transmitMediumDiffusionBRDF(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 wo,
    glm::vec3 in_wi,
    glm::vec3 normal,
    float t,
    const Material &m,
    thrust::default_random_engine &rng,
    thrust::uniform_real_distribution<float> &u01
)
{
    // TO DO: implement this one day
    return;
}

__host__ __device__ void transmitMediumBRDF(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    float t,
    thrust::default_random_engine &rng,
    thrust::uniform_real_distribution<float> &u01
)
{
#if 1
    float scatteringDistance = 0.5f;
    float scatteringCoefficient = 1.0f / scatteringDistance;
    float weight = 1.0f;
    float distance = isotropicSampleDistance(t, scatteringCoefficient, weight, rng);

    float absorptionAtDistance = 1.0f;
    glm::vec3 absorptionColor = glm::vec3(0.35f, 0.85f, 0.35f);
    glm::vec3 absorptionCoefficient = -log(absorptionColor) / absorptionAtDistance;

    // RichieSams has us create a "scatter event".
    if (distance < t)
    {
        glm::vec3 transmission = isotropicTransmission(absorptionCoefficient, distance);
        pathSegment.color *= transmission * weight;

        pathSegment.ray.origin += pathSegment.ray.direction * distance;
        pathSegment.ray.direction = henyeyGreensteinSampleDirection(-0.5f, u01(rng), u01(rng), -wo);
        //pathSegment.ray.direction = isotropicSampleScatterDirection(u01(rng), u01(rng));
    }
    // No scatter event, we've hit the surface and we're ready to leave!
    else
    {
        glm::vec3 transmission = exp(-absorptionCoefficient * t);
        pathSegment.color *= transmission;

        // No need to change ray direction, should be ok
        pathSegment.medium = VACUUM;
        pathSegment.ray.origin = intersect + normal * 0.001f;
    }

    return;
#endif
}

// Also known as Trowbridge-Reitz. Sorry. 
// https://pharr.org/matt/blog/2022/05/06/trowbridge-reitz
__host__ __device__ float GGXDistribution(float a2, float cosTheta)
{
    float cos2Theta = cosTheta * cosTheta;
    float denom = (a2 - 1.0f) * cos2Theta + 1.0f;

    return a2 / (PI * denom * denom);
}

__host__ __device__ float SmithGGX(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    float a2
)
{
#if 0
    float nDotI = dot(wi, normal);
    float nDotO = dot(wo, normal);

    // This combines SmithGGX(i, m) * SmithGGX(o, m)
    float denomIn = nDotI * sqrt(a2 + (1.0f - a2) * (nDotI * nDotI)) + 0.0001f;
    float denomOut = nDotO * sqrt(a2 + (1.0f - a2) * (nDotO * nDotO)) + 0.0001f;

    float out = ((2.0f * nDotI) / denomIn) * ((2.0f * nDotO) / denomOut);
#endif

#if 0
    // Schlick-Beckmann GGX, used in Karis's UE4 implementation
    // https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html#:~:text=Geometric%20Shadowing,%E2%8B%85vn%E2%8B%85v.
    float roughness = sqrt(a2);
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    
    float nDotL = dot(wi, normal);
    float nDotV = dot(wo, normal);

    float gSchlickL = nDotL / (nDotL * (1.0f - k) + k);
    float gSchlickV = nDotV / (nDotV * (1.0f - k) + k);

    float out = gSchlickL * gSchlickV;
#endif

#if 1
    // Uncorrelated G2 model, used by refs:
    // https://schuttejoe.github.io/post/ggximportancesamplingpart1/
    // https://media.gdcvault.com/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf

    float nDotL = dot(wi, normal);
    float nDotV = dot(wo, normal);

    float num = 2.0f * nDotL * nDotV;
    float denomV = nDotV * sqrt(a2 * (1.0f - a2) * (nDotL * nDotL));
    float denomL = nDotL * sqrt(a2 * (1.0f - a2) * (nDotV * nDotV));

    float out = num / (denomV + denomL);

    if((nDotV + nDotL) < 0.1f)
    {
        out = 0.0f;
    }
#endif

    return out;
}

__host__ __device__ glm::vec3 dielectricSpecularBRDF(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    glm::vec3 microNormal,
    const Material &m
)
{
    float nDotI = glm::max(dot(normal, wi), 0.0f);
    float nDotO = glm::max(dot(normal, wo), 0.0f);
    float nDotM = glm::max(dot(normal, microNormal), 0.0f);

    float mDotI = glm::max(dot(microNormal,wi), 0.0f);

    float roughness = glm::max(m.roughness, 0.0001f);
    float alpha = roughness * roughness;
    float alpha2 = glm::max(alpha * alpha, 0.02f);

    float G = glm::clamp(SmithGGX(wo, wi, microNormal, alpha2), 0.0f, 1.05f);

    // Adapted this from Schutte's specular BRDF simplification
    // https://schuttejoe.github.io/post/ggximportancesamplingpart1/
    // Evil clamping trick not by Schutte. Cool specular highlights take much longer to appear
    // as a consequence, but takes care of nasty fireflies.
    return glm::clamp(glm::vec3(G) * abs(dot(wo, microNormal)) / (nDotO * nDotM + 0.001f), 0.0f, 1.0f);
}

__host__ __device__ glm::vec3 diffuseBRDF(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    const Material &m
)
{
    return m.color;
}

