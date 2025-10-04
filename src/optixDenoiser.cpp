#include "OptixDenoiser.h"

void Denoiser::denoise(Image &inputImage)
{
	CUcontext cuCtx = 0;
	OptixDeviceContext context = nullptr;
	OptixDeviceContextOptions options = {};
	optixDeviceContextCreate(cuCtx, &options, &context);

	OptixDenoiserOptions denoiserOptions = {};
	OptixDenoiser denoiser = nullptr;
	optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser);

	OptixDenoiserSizes sizes;

	uint32_t width = inputImage.xSize;
	uint32_t height = inputImage.ySize;

	optixDenoiserComputeMemoryResources(denoiser, width, height, &sizes);

	CUdeviceptr state;
	cuMemAlloc(&state, sizes.stateSizeInBytes);

	CUdeviceptr scratch;
	cuMemAlloc(&scratch, sizes.withoutOverlapScratchSizeInBytes);

	optixDenoiserSetup(
		denoiser,
		0,
		width,
		height,
		state,
		sizes.stateSizeInBytes,
		scratch,
		sizes.withoutOverlapScratchSizeInBytes
	);

	// Copy to device...
	// OptiX uses different API
	CUdeviceptr dev_input;
	cuMemAlloc(&dev_input, sizeof(glm::vec3) * width * height);
	cuMemcpyHtoD(dev_input, inputImage.pixels, sizeof(glm::vec3) * width * height);

	CUdeviceptr dev_output;
	cuMemAlloc(&dev_output, sizeof(glm::vec3) * width * height);

	OptixImage2D inputLayer = {};
	inputLayer.data = dev_input;
	inputLayer.width = width;
	inputLayer.height = height;
	inputLayer.rowStrideInBytes = sizeof(glm::vec3) * width;
	inputLayer.pixelStrideInBytes = sizeof(glm::vec3);
	inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixImage2D outputLayer = {};
	outputLayer.data = dev_output;
	outputLayer.width = width;
	outputLayer.height = height;
	outputLayer.rowStrideInBytes = sizeof(glm::vec3) * width;
	outputLayer.pixelStrideInBytes = sizeof(glm::vec3);
	outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixDenoiserParams params = {};

	OptixDenoiserGuideLayer guideLayer = {}; // no albedo/normal
	OptixDenoiserLayer layer = { inputLayer, outputLayer };

	optixDenoiserInvoke(
		denoiser,
		0, // stream
		&params,
		state,
		sizes.stateSizeInBytes,
		&guideLayer,
		&layer,
		1, // num layers
		0, 0, // offset
		scratch,
		sizes.withoutOverlapScratchSizeInBytes
	);

	std::vector<glm::vec3> denoised(width * height);
	cuMemcpyDtoH(inputImage.pixels, dev_output, sizeof(glm::vec3) * width * height);
}