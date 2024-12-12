#pragma clang diagnostic push

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/system/hip/vector.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <stack>

// Function to make a dot
__host__ __device__ __inline__ float dot(const float3 &a, const float3 &b);

// Function to make a cross
__host__ __device__ __inline__ float3 cross(const float3 &a, const float3 &b);

// Function to return a length
__host__ __device__ __inline__ float length(const float3 &v);

// Function to normalize
__host__ __device__ __inline__ float3 normalize(const float3 &v);

struct Vec3 {
  float x, y, z;

  __host__ __device__ __inline__ Vec3 operator-(const Vec3 &v) const {
    return {x - v.x, y - v.y, z - v.z};
  }

  __host__ __device__ __inline__ Vec3 operator+(const Vec3 &v) const {
    return {x + v.x, y + v.y, z + v.z};
  }

  __host__ __device__ __inline__ Vec3 operator*(float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }

  __host__ __device__ __inline__ Vec3 operator*(const Vec3 &other) const {
    return Vec3(x * other.x, y * other.y, z * other.z);
  }

  __host__ __device__ __inline__ float dot(const Vec3 &v) const {
    return x * v.x + y * v.y + z * v.z;
  }

  __host__ __device__ __inline__ Vec3 cross(const Vec3 &v) const {
    return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
  }

  __host__ __device__ __inline__ Vec3() : x(0), y(0), z(0) {}

  __host__ __device__ __inline__ Vec3(float x_, float y_, float z_)
      : x(x_), y(y_), z(z_) {}

  __host__ __device__ __inline__ Vec3 init(float x, float y, float z) {
    Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
  }

  __host__ __device__ __inline__ Vec3 float3ToVec3(const float3 &f) {
    return Vec3(f.x, f.y, f.z);
  }

  __host__ __device__ __inline__ void normalize() {
    float length = sqrt(x * x + y * y + z * z);
    if (length > 0) {
      x /= length;
      y /= length;
      z /= length;
    }
  }
};

struct F3Triangle {
  float3 v0, v1, v2;
  int id;
};

struct BVHNode {
  float3 min, max;           // Min and max coordinates of the bounding box
  int leftChild, rightChild; // Indices of child nodes (-1 for leaves)
  int triangleIndex;
  // int splitAxis; //add
  int boxIndex;
};

// Method 1
struct BVHNodeSAH {
  float3 bounds_min;
  float3 bounds_max;
  int left, right;
  int triangleIndex;
  // int splitAxis;
  // int boxIndex;
};

struct F3Ray {
  float3 origin;
  float3 direction;
  float tMin;
  float tMax;
};

// AABB Method 1
// Function to calculate the AABB of a triangle
struct AABB {
  float3 min, max;
};

// AABB Method 1
// Structure to represent a BVH node AABB
struct BVHNodeAABB {
  AABB bounds;
  int leftChild;
  int rightChild;
  int firstTriangle;
  int triangleCount;
};

// SAH method 2
struct AABB_SAH2 {

  float3 min;
  float3 max;

  __host__ __device__ __inline__ void expand(const float3 &point) {
    min = make_float3(fminf(min.x, point.x), fminf(min.y, point.y),
                      fminf(min.z, point.z));
    max = make_float3(fmaxf(max.x, point.x), fmaxf(max.y, point.y),
                      fmaxf(max.z, point.z));
  }

  __host__ __device__ __inline__ void expand(const AABB_SAH2 &other) {
    expand(other.min);
    expand(other.max);
  }

  __host__ __device__ __inline__ AABB_SAH2()
      : min(make_float3(FLT_MAX, FLT_MAX, FLT_MAX)),
        max(make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)) {}
};

// SAH method 2
struct BVHNodeSAH2 {
  int left;
  int count;
  bool isLeaf;
  AABB_SAH2 bounds;
  __host__ __device__ __inline__ BVHNodeSAH2()
      : left(0), count(0), isLeaf(true) {}
  //__host__ __device__ __inline__ BVHNodeSAH2(int _left, int _count) :
  //left(_left), count(_count), isLeaf(_count == 1) {}
  //__host__ __device__ __inline__ BVHNodeSAH2(int _left = 0, int _count = 0) :
  //left(_left), count(_count), isLeaf(_count <= 1) {}

  __host__ __device__ __inline__ BVHNodeSAH2(int _left, int _count)
      : left(_left), count(_count), isLeaf(_count == 1), bounds() {}
};

// Camera for all configuation
struct Camera {
  float3 position; // Camera position
  float3 target;   // Point the camera is looking at
  float3 up;       // Up vector for the camera
  float fov;       // Field of view in radians
  float aspect;    // Aspect ratio (width/height)

  __device__ __inline__ void getRay(int x, int y, int width, int height,
                                    float3 *rayOrigin,
                                    float3 *rayDirection) const {
    // Calculate normalized device coordinates (NDC)
    float ndcX = (2.0f * (x + 0.5f) / width - 1.0f) * aspect;
    float ndcY = 1.0f - 2.0f * (y + 0.5f) / height;

    // Calculate the direction of the ray in world space
    float3 forward = make_float3(target.x - position.x, target.y - position.y,
                                 target.z - position.z);
    forward = normalize(forward);

    float3 right = cross(forward, up);
    right = normalize(right);

    float3 cameraUp = cross(right, forward);

    // Calculate the ray direction
    float3 horizontal = right * tan(fov / 2.0f);
    float3 vertical = cameraUp * tan(fov / 2.0f);

    *rayDirection = forward + horizontal * ndcX + vertical * ndcY;
    *rayDirection = normalize(*rayDirection);

    *rayOrigin = position;
  }
};

// Ligth parameters. Not implemented for instance
struct Light {
  float3 position;
  float3 color;
  float intensity;
};

// Material parameters. Not implemented for instance
struct Material {
  Vec3 albedo;
  float roughness;
  float metallic;
  __host__ __device__ __inline__ Material(Vec3 a, float r, float m)
      : albedo(a), roughness(r), metallic(m) {}
};

// Material plus parameters. Not implemented for instance
struct MaterialPlus {
  Vec3 albedo;
  float roughness;
  float metallic;
  float reflection;
  float transparency;
  float refractiveIndex;

  __host__ __device__ __inline__ MaterialPlus(Vec3 a, float r, float m,
                                              float refl = 0.0f,
                                              float trans = 0.0f,
                                              float refrIndex = 1.0f)
      : albedo(a), roughness(r), metallic(m), reflection(refl),
        transparency(trans), refractiveIndex(refrIndex) {}
};

// Tools for BVH Node view, save and load
void writeBVHNodes(const thrust::device_vector<BVHNode> &nodes);
void writeBVHNodesSAH1(const thrust::device_vector<BVHNodeSAH> &nodes);
void writeBVHNodesSAH2(const thrust::device_vector<BVHNodeSAH2> &nodes);
void loadBVH(const std::string &filename,
             thrust::device_vector<BVHNode> &nodes);
void saveBVH(const std::string &filename,
             const thrust::device_vector<BVHNode> &nodes);

// Save picture take from camera
void savePPM(const std::string &filename, unsigned char *data, int width,
             int height);

// Build differente BVH method with a gool to find the best stategy
void buildBVHWithTriangleVersion1(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes);
void buildBVHWithTriangleVersion2(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes);
void buildBVHWithTriangleVersion3(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes);
void buildBVHWithTriangleVersion4(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes);

void buildBVHWithTriangleVersion5(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes);

// AABB Method1
void buildBVH_AABB(thrust::device_vector<F3Triangle> &triangles,
                   thrust::device_vector<BVHNodeAABB> &nodes);

// SAH Method1
void buildBVHWithTriangleVersion3SAH(
    thrust::device_vector<F3Triangle> &triangles,
    thrust::device_vector<BVHNodeSAH> &nodes);

// SAH Method2
void buildBVHWithTriangleVersion3SAH2(
    thrust::device_vector<F3Triangle> &triangles,
    thrust::device_vector<BVHNodeSAH2> &nodes);

// rayTracing without take picture
__global__ void rayTracingKernel(BVHNode *nodes, F3Triangle *triangles,
                                 F3Ray *rays, int *hitResults, float *distance,
                                 float3 *intersectionPoint, int *hitId,
                                 int numRays);

// rayTracing build a picture
__global__ void rayTracingImgKernel(unsigned char *image, int width, int height,
                                    Camera camera, BVHNode *nodes,
                                    F3Triangle *triangles, int *hitResults,
                                    float *distance, float3 *intersectionPoint,
                                    int *hitId);

// This function call rayTracingImgKernel
void buildPicturRayTracingPPM(thrust::device_vector<F3Triangle> &triangles,
                              thrust::device_vector<BVHNode> &nodes,
                              Camera camera, int width, int height,
                              const std::string &filename, bool isSave);

// AABB method1 kernel
__global__ void rayTracingImgKernel_AABB(unsigned char *image, int width,
                                         int height, Camera camera,
                                         BVHNodeAABB *nodes,
                                         F3Triangle *triangles, int *hitResults,
                                         float *hitDistances,
                                         float3 *intersectionPoint, int *hitId);

// This function call rayTracingImgKernel_AABB
void buildPicturRayTracingPPM_AABB(thrust::device_vector<F3Triangle> &triangles,
                                   thrust::device_vector<BVHNodeAABB> &nodes,
                                   Camera camera, int width, int height,
                                   const std::string &filename, bool isSave);

// SAH method1 Kernel
__global__ void rayTracingImgKernelVersion3SAH(
    unsigned char *image, const int width, const int height,
    const Camera camera, const BVHNodeSAH *nodes, const F3Triangle *triangles,
    int *hitResults, float *distance, float3 *intersectionPoint, int *hitId);

// This function call rayTracingImgKernel_SAH
void buildPicturRayTracingPPM3_SAH(thrust::device_vector<F3Triangle> &triangles,
                                   thrust::device_vector<BVHNodeSAH> &nodes,
                                   Camera camera, int width, int height,
                                   const std::string &filename, bool isSave);

void buildPicturRayTracingPPM3_SAH2(
    thrust::device_vector<F3Triangle> &triangles,
    thrust::device_vector<BVHNodeSAH2> &nodes, Camera camera, int width,
    int height, const std::string &filename, bool isSave);

#pragma clang diagnostic pop