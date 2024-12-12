#include "bvhgpu.hpp"

#define SWAP(T, a, b)                                                          \
  do {                                                                         \
    T tmp = a;                                                                 \
    a = b;                                                                     \
    b = tmp;                                                                   \
  } while (0)

struct F3TriangleInit {
  float3 _v0, _v1, _v2;
  int _id;

  F3TriangleInit(float3 v0, float3 v1, float3 v2, int id)
      : _v0(v0), _v1(v1), _v2(v2), _id(id) {}

  __host__ __device__ __inline__ F3Triangle operator()(int idx) {
    F3Triangle t;
    t.v0 = _v0;
    t.v1 = _v1;
    t.v2 = _v2;
    t.id = _id;
    return t;
  }
};

struct Triangle {
  Vec3 v0, v1, v2;
  int id;
};

struct Box {
  Vec3 min;
  Vec3 max;
  int id;
};

struct BoxCentroid {
  float3 centroid;
  int index;
};

struct TriangleCentroid {
  float3 centroid;
  int index;
};

struct Rectangle {
  Vec3 v0, v1, v2, v3;
};

struct Ray {
  Vec3 origin, direction;
};

/*
struct Camera {
    float3 position;  // Camera position
    float3 target;    // Point the camera is looking at
    float3 up;        // Up vector for the camera
    float fov;        // Field of view in radians
    float aspect;     // Aspect ratio (width/height)

    __device__ __inline__ void getRay(int x, int y, int width, int height,
float3* rayOrigin, float3* rayDirection) const {
        // Calculate normalized device coordinates (NDC)
        float ndcX = (2.0f * (x + 0.5f) / width - 1.0f) * aspect;
        float ndcY = 1.0f - 2.0f * (y + 0.5f) / height;

        // Calculate the direction of the ray in world space
        float3 forward = make_float3(target.x - position.x, target.y -
position.y, target.z - position.z); forward = normalize(forward);

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
*/

__device__ __inline__ void getRay2(int x, int y, int width, int height,
                                   Camera camera, float3 *rayOrigin,
                                   float3 *rayDirection) {
  // Calculate normalized device coordinates (NDC)
  float ndcX = (2.0f * (x + 0.5f) / width - 1.0f) * camera.aspect;
  float ndcY = 1.0f - 2.0f * (y + 0.5f) / height;

  // Calculate the direction of the ray in world space
  float3 forward = make_float3(camera.target.x - camera.position.x,
                               camera.target.y - camera.position.y,
                               camera.target.z - camera.position.z);
  forward = normalize(forward);

  float3 right = cross(forward, camera.up);
  right = normalize(right);

  float3 cameraUp = cross(right, forward);

  // Calculate the ray direction
  float3 horizontal = right * tan(camera.fov / 2.0f);
  float3 vertical = cameraUp * tan(camera.fov / 2.0f);

  *rayDirection = forward + horizontal * ndcX + vertical * ndcY;
  *rayDirection = normalize(*rayDirection);

  *rayOrigin = camera.position;
}

// Function to make a dot
__host__ __device__ __inline__ float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Function to make a cross
__host__ __device__ __inline__ float3 cross(const float3 &a, const float3 &b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

// Function to return a length
__host__ __device__ __inline__ float length(const float3 &v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Function to normalize
__host__ __device__ __inline__ float3 normalize(const float3 &v) {
  float len = length(v);
  if (len > 0) {
    return float3(v.x / len, v.y / len, v.z / len);
  }
  return v;
}

// Function to write a float3
__host__ __device__ __inline__ void print_float3(const float3 &v) {
  printf("%f %f %f\n", v.x, v.y, v.z);
}

float3 toFloat3(const Vec3 &v) { return {v.x, v.y, v.z}; }

std::vector<F3Triangle> boxToTriangles(const Box &box, const int &id) {
  std::vector<F3Triangle> triangles;
  triangles.reserve(12);

  Vec3 vertices[8] = {
      {box.min.x, box.min.y, box.min.z}, {box.max.x, box.min.y, box.min.z},
      {box.max.x, box.max.y, box.min.z}, {box.min.x, box.max.y, box.min.z},
      {box.min.x, box.min.y, box.max.z}, {box.max.x, box.min.y, box.max.z},
      {box.max.x, box.max.y, box.max.z}, {box.min.x, box.max.y, box.max.z}};

  // Definition of the 12 triangles (2 per face)
  int indices[12][3] = {
      {0, 1, 2}, {0, 2, 3}, // Front face
      {1, 5, 6}, {1, 6, 2}, // Right face
      {5, 4, 7}, {5, 7, 6}, // Back face
      {4, 0, 3}, {4, 3, 7}, // Left face
      {3, 2, 6}, {3, 6, 7}, // Top face
      {4, 5, 1}, {4, 1, 0}  // Bottom face
  };

  for (int i = 0; i < 12; ++i) {
    F3Triangle tri;
    tri.v0 = toFloat3(vertices[indices[i][0]]);
    tri.v1 = toFloat3(vertices[indices[i][1]]);
    tri.v2 = toFloat3(vertices[indices[i][2]]);
    tri.id = id;
    triangles.push_back(tri);
    triangles.back().id = id;
  }
  return triangles;
}

// Intersection function between a ray and a plan
__host__ __device__ __inline__ std::optional<Vec3>
rayPlaneIntersect(const Ray &ray, const Vec3 &planePoint,
                  const Vec3 &planeNormal) {
  constexpr float epsilon = 1e-6f;

  float denom = planeNormal.dot(ray.direction);

  // Vérification si le rayon est parallèle au plan
  if (fabs(denom) < epsilon) {
    return {}; // Pas d'intersection
  }

  Vec3 p0l0 = planePoint - ray.origin;
  float t = p0l0.dot(planeNormal) / denom;

  // Check if the ray is parallel to the plane
  if (t < 0) {
    return {}; // The plan is behind the ray
  }

  // Calculate the intersection point
  return {ray.origin + ray.direction * t};
}

// Intersection function between a ray and a rectangle
__host__ __device__ __inline__ std::optional<Vec3>
rayRectangleIntersect(const Ray &ray, const Rectangle &rect) {
  // We must define the plane of the rectangle
  Vec3 edge1 = rect.v1 - rect.v0;
  Vec3 edge2 = rect.v3 - rect.v0;
  Vec3 normal = edge1.cross(edge2); // Normal of the rectangle

  constexpr float epsilon = 1e-6f;
  float denom = normal.dot(ray.direction);

  // Check if the ray is parallel to the plane
  if (fabs(denom) < epsilon) {
    return {}; // No intersection
  }

  // Calculation of the distance t at which the ray intersects the plane
  Vec3 p0l0 = rect.v0 - ray.origin;
  float t = p0l0.dot(normal) / denom;

  // Check if the intersection is in front of the ray
  if (t < 0) {
    return {}; // The rectangle is behind the ray
  }

  // The rectangle is behind the ray
  Vec3 intersectionPoint = ray.origin + ray.direction * t;

  // Check if the intersection point is inside the rectangle
  Vec3 c;

  // Check for the first side
  Vec3 edge00 = rect.v1 - rect.v0;
  Vec3 vp0 = intersectionPoint - rect.v0;
  c = edge00.cross(vp0);
  if (normal.dot(c) < 0)
    return {}; // The point is outside

  // Checking for the second side
  Vec3 edge01 = rect.v2 - rect.v1;
  Vec3 vp1 = intersectionPoint - rect.v1;
  c = edge01.cross(vp1);
  if (normal.dot(c) < 0)
    return {}; // The point is outside

  // Checking for the third side
  Vec3 edge02 = rect.v3 - rect.v2;
  Vec3 vp2 = intersectionPoint - rect.v2;
  c = edge02.cross(vp2);
  if (normal.dot(c) < 0)
    return {}; // The point is outside

  // Checking for the fourth side
  Vec3 edge03 = rect.v0 - rect.v3;
  Vec3 vp3 = intersectionPoint - rect.v3;
  c = edge03.cross(vp3);
  if (normal.dot(c) < 0)
    return {}; // The point is outside

  return intersectionPoint;
}

// Function to calculate the bounding box of a triangle
__host__ __device__ __inline__ void
calculateBoundingBox(const F3Triangle &triangle, float3 &min, float3 &max) {
  min = make_float3(fminf(fminf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                    fminf(fminf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                    fminf(fminf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
  max = make_float3(fmaxf(fmaxf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                    fmaxf(fmaxf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                    fmaxf(fmaxf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
}

// Function to build a simple BVH (medium construction method)
void buildBVHWithTriangleVersion1(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);

  // Initialize the sheets
  for (int i = 0; i < numTriangles; ++i) {
    BVHNode *raw_ptr = thrust::raw_pointer_cast(nodes.data());
    BVHNode &node = raw_ptr[numTriangles - 1 + i];

    calculateBoundingBox(triangles[i], node.min, node.max);
    node.triangleIndex = i;
    node.leftChild = node.rightChild = -1;
  }

  // Build the internal nodes
  for (int i = numTriangles - 2; i >= 0; --i) {
    BVHNode *raw_ptr = thrust::raw_pointer_cast(nodes.data());
    BVHNode &node = raw_ptr[i];
    int leftChild = 2 * i + 1;
    int rightChild = 2 * i + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    BVHNode leftNode = nodes[leftChild];
    BVHNode rightNode = nodes[rightChild];
    node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                           fminf(leftNode.min.y, rightNode.min.y),
                           fminf(leftNode.min.z, rightNode.min.z));

    node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                           fmaxf(leftNode.max.y, rightNode.max.y),
                           fmaxf(leftNode.max.z, rightNode.max.z));

    // if (i < 24) printf("Internal %d: min(%f,%f,%f) max(%f,%f,%f)\n", i,
    // node.min.x, node.min.y, node.min.z, node.max.x, node.max.y, node.max.z);
  }
}

void buildBVHWithTriangleVersion2(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);

  // Initialize the leaves in parallel
  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(numTriangles),
                   [triangles = thrust::raw_pointer_cast(triangles.data()),
                    nodes = thrust::raw_pointer_cast(nodes.data()),
                    numTriangles] __device__(int i) {
                     BVHNode &node = nodes[numTriangles - 1 + i];
                     calculateBoundingBox(triangles[i], node.min, node.max);
                     node.triangleIndex = i;
                     node.leftChild = node.rightChild = -1;
                   });

  // Build the internal nodes
  for (int i = numTriangles - 2; i >= 0; --i) {
    BVHNode *raw_ptr = thrust::raw_pointer_cast(nodes.data());
    BVHNode &node = raw_ptr[i];
    int leftChild = 2 * i + 1;
    int rightChild = 2 * i + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    const BVHNode &leftNode = raw_ptr[leftChild];
    const BVHNode &rightNode = raw_ptr[rightChild];

    // Vectorized min/max operations
    node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                           fminf(leftNode.min.y, rightNode.min.y),
                           fminf(leftNode.min.z, rightNode.min.z));

    node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                           fmaxf(leftNode.max.y, rightNode.max.y),
                           fmaxf(leftNode.max.z, rightNode.max.z));
  }
}

//...

__device__ __inline__ float3 minFloat3(const float3 &a, const float3 &b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __inline__ float3 maxFloat3(const float3 &a, const float3 &b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__global__ void initializeLeaves3(F3Triangle *triangles, BVHNode *nodes,
                                  int numTriangles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numTriangles) {
    BVHNode &node = nodes[numTriangles - 1 + i];
    node.min = make_float3(INFINITY, INFINITY, INFINITY);
    node.max = make_float3(-INFINITY, -INFINITY, -INFINITY);
    calculateBoundingBox(triangles[i], node.min, node.max);
    // assert(node.min.x <= node.max.x && node.min.y <= node.max.y && node.min.z
    // <= node.max.z);
    node.triangleIndex = i;
    node.leftChild = node.rightChild = -1;
    // if (i < 24) printf("Leaf %d: min(%f,%f,%f) max(%f,%f,%f)\n", i,
    // node.min.x, node.min.y, node.min.z, node.max.x, node.max.y, node.max.z);
  }
  __syncthreads();
}

__global__ void buildInternalNodes(BVHNode *nodes, int numTriangles,
                                   int level) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = 1 << level;
  int offset = stride - 1;

  if (idx < stride && idx + offset < numTriangles - 1) {
    int nodeIdx = idx + offset;
    BVHNode &node = nodes[nodeIdx];
    int leftChild = 2 * nodeIdx + 1;
    int rightChild = 2 * nodeIdx + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    const BVHNode &leftNode = nodes[leftChild];
    const BVHNode &rightNode = nodes[rightChild];

    node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                           fminf(leftNode.min.y, rightNode.min.y),
                           fminf(leftNode.min.z, rightNode.min.z));

    node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                           fmaxf(leftNode.max.y, rightNode.max.y),
                           fmaxf(leftNode.max.z, rightNode.max.z));

    // assert(node.min.x <= node.max.x && node.min.y <= node.max.y && node.min.z
    // <= node.max.z); if (nodeIdx < 24) printf("Internal %d: min(%f,%f,%f)
    // max(%f,%f,%f)\n", nodeIdx, node.min.x, node.min.y, node.min.z,
    // node.max.x, node.max.y, node.max.z);
  }
}

__global__ void buildInternalNodes3(BVHNode *nodes, int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx <= numTriangles - 2) {
    int nodeIdx = numTriangles - 2 - idx;
    // int nodeIdx = idx;
    BVHNode &node = nodes[nodeIdx];
    node.min = make_float3(INFINITY, INFINITY, INFINITY);
    node.max = make_float3(-INFINITY, -INFINITY, -INFINITY);

    int leftChild = 2 * nodeIdx + 1;
    int rightChild = 2 * nodeIdx + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    const BVHNode &leftNode = nodes[leftChild];
    const BVHNode &rightNode = nodes[rightChild];

    node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                           fminf(leftNode.min.y, rightNode.min.y),
                           fminf(leftNode.min.z, rightNode.min.z));

    node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                           fmaxf(leftNode.max.y, rightNode.max.y),
                           fmaxf(leftNode.max.z, rightNode.max.z));

    // assert(node.min.x <= node.max.x && node.min.y <= node.max.y && node.min.z
    // <= node.max.z); if (nodeIdx < 24) printf("Internal %d: min(%f,%f,%f)
    // max(%f,%f,%f)\n", nodeIdx, node.min.x, node.min.y, node.min.z,
    // node.max.x, node.max.y, node.max.z);
  }
  __syncthreads();
}

void buildBVHWithTriangleVersion3(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes) {
  int numTriangles = triangles.size();
  // printf("Nb Traingles=%i\n",numTriangles);
  nodes.resize(2 * numTriangles - 1);

  // Initialize leaves
  int blockSize = 512;
  int numBlocks = (numTriangles + blockSize - 1) / blockSize;
  initializeLeaves3<<<numBlocks, blockSize>>>(
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(nodes.data()), numTriangles);
  // hipDeviceSynchronize();
  // writeBVHNodes(nodes);
  //  Build internal nodes level by level

  buildInternalNodes3<<<numBlocks, blockSize>>>(
      thrust::raw_pointer_cast(nodes.data()), numTriangles);

  // hipDeviceSynchronize();
}
//...

//...
/*
struct Morton {
    uint32_t code;
    int index;
};

__device__ __inline__ inline uint32_t float_as_uint(float f) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = f;
    return converter.u;
}

// Kernel pour calculer les codes de Morton
__global__ void calculateMortonCodes(F3Triangle* triangles, Morton* mortonCodes,
int numTriangles, float3 sceneMin, float3 sceneExtent) { int idx = blockIdx.x *
blockDim.x + threadIdx.x; if (idx < numTriangles) { float3 centroid =
(triangles[idx].v0 + triangles[idx].v1 + triangles[idx].v2) / 3.0f; float3
normalized = (centroid - sceneMin) / sceneExtent; uint32_t x =
__float_as_uint(normalized.x) & 0x3FF; uint32_t y =
__float_as_uint(normalized.y) & 0x3FF; uint32_t z =
__float_as_uint(normalized.z) & 0x3FF; mortonCodes[idx].code = (x << 20) | (y <<
10) | z; mortonCodes[idx].index = idx;
    }
}


// Kernel pour construire le BVH
__global__ void buildBVH(Morton* sortedMortonCodes, BVHNode* nodes, F3Triangle*
triangles, int numTriangles) { int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles - 1) return;

    int commonPrefix = __clz(sortedMortonCodes[idx].code ^
sortedMortonCodes[idx+1].code); int direction = (commonPrefix -
__clz(sortedMortonCodes[idx].code ^ sortedMortonCodes[idx-1].code)) < 0 ? -1 :
1;

    int minPrefix = __clz(sortedMortonCodes[idx].code ^ sortedMortonCodes[idx +
direction].code); int maxPrefix = __clz(sortedMortonCodes[idx].code ^
sortedMortonCodes[idx - direction].code);

    int d = 2;
    while (__clz(sortedMortonCodes[idx].code ^ sortedMortonCodes[idx + d *
direction].code) > minPrefix) d *= 2;

    int delta = d / 2;
    while (delta > 0) {
        if (__clz(sortedMortonCodes[idx].code ^ sortedMortonCodes[idx + (d -
delta) * direction].code) > minPrefix) d -= delta; delta /= 2;
    }

    int j = idx + d * direction;
    int nodeIdx = idx + numTriangles - 1;
    int leftChild = min(idx, j);
    int rightChild = max(idx, j);

    nodes[nodeIdx].leftChild = leftChild;
    nodes[nodeIdx].rightChild = rightChild;

    if (leftChild < numTriangles) {
        int triIdx = sortedMortonCodes[leftChild].index;
        nodes[leftChild].min = nodes[leftChild].max = triangles[triIdx].v0;
        nodes[leftChild].min = minFloat3(nodes[leftChild].min,
triangles[triIdx].v1); nodes[leftChild].min = minFloat3(nodes[leftChild].min,
triangles[triIdx].v2); nodes[leftChild].max = maxFloat3(nodes[leftChild].max,
triangles[triIdx].v1); nodes[leftChild].max = maxFloat3(nodes[leftChild].max,
triangles[triIdx].v2); nodes[leftChild].triangleIndex = triIdx;
    }

    if (rightChild < numTriangles) {
        int triIdx = sortedMortonCodes[rightChild].index;
        nodes[rightChild].min = nodes[rightChild].max = triangles[triIdx].v0;
        nodes[rightChild].min = minFloat3(nodes[rightChild].min,
triangles[triIdx].v1); nodes[rightChild].min = minFloat3(nodes[rightChild].min,
triangles[triIdx].v2); nodes[rightChild].max = maxFloat3(nodes[rightChild].max,
triangles[triIdx].v1); nodes[rightChild].max = maxFloat3(nodes[rightChild].max,
triangles[triIdx].v2); nodes[rightChild].triangleIndex = triIdx;
    }

    nodes[nodeIdx].min = minFloat3(nodes[leftChild].min, nodes[rightChild].min);
    nodes[nodeIdx].max = maxFloat3(nodes[leftChild].max, nodes[rightChild].max);
}


void buildBVHWithTriangleVersion4(thrust::device_vector<F3Triangle>& triangles,
thrust::device_vector<BVHNode>& nodes) { int numTriangles = triangles.size();
    nodes.resize(2 * numTriangles - 1);

    // Calculer l'étendue de la scène
    F3Triangle* d_triangles = thrust::raw_pointer_cast(triangles.data());
    float3 sceneMin, sceneMax;
    // ... (calculer sceneMin et sceneMax)

    // Calculer les codes de Morton
    thrust::device_vector<Morton> mortonCodes(numTriangles);
    Morton* d_mortonCodes = thrust::raw_pointer_cast(mortonCodes.data());

    int blockSize = 256;
    int numBlocks = (numTriangles + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(calculateMortonCodes, dim3(numBlocks), dim3(blockSize),
0, 0, d_triangles, d_mortonCodes, numTriangles, sceneMin, sceneMax - sceneMin);

    // Trier les codes de Morton
    thrust::sort(thrust::device, mortonCodes.begin(), mortonCodes.end(),
                 [] __device__  (const Morton& a, const Morton& b) { return
a.code < b.code; });

    // Construire le BVH
    BVHNode* d_nodes = thrust::raw_pointer_cast(nodes.data());
    hipLaunchKernelGGL(buildBVH, dim3(numBlocks), dim3(blockSize), 0, 0,
d_mortonCodes, d_nodes, d_triangles, numTriangles);
}
*/

//..

struct Morton {
  uint32_t code;
  int index;
};

__device__ inline uint32_t float_as_uint(float f) {
  union {
    float f;
    uint32_t u;
  } converter;
  converter.f = f;
  return converter.u;
}

__device__ inline uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__global__ void calculateMortonCodes(const F3Triangle *triangles,
                                     Morton *mortonCodes, int numTriangles,
                                     float3 sceneMin, float3 sceneExtent) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    float3 centroid =
        (triangles[idx].v0 + triangles[idx].v1 + triangles[idx].v2) / 3.0f;
    float3 normalized = (centroid - sceneMin) / sceneExtent;
    // normalized = clamp(normalized, 0.0f, 1.0f);

    uint32_t x = expandBits((uint32_t)(normalized.x * 1024.0f));
    uint32_t y = expandBits((uint32_t)(normalized.y * 1024.0f));
    uint32_t z = expandBits((uint32_t)(normalized.z * 1024.0f));

    mortonCodes[idx].code = x | (y << 1) | (z << 2);
    mortonCodes[idx].index = idx;
  }
  __syncthreads();
}

__global__ void buildBVHLeaves(const Morton *sortedMortonCodes, BVHNode *nodes,
                               const F3Triangle *triangles, int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    int leafIdx = numTriangles - 1 + idx;
    int triIdx = sortedMortonCodes[idx].index;

    nodes[leafIdx].min =
        minFloat3(triangles[triIdx].v0,
                  minFloat3(triangles[triIdx].v1, triangles[triIdx].v2));
    nodes[leafIdx].max =
        maxFloat3(triangles[triIdx].v0,
                  maxFloat3(triangles[triIdx].v1, triangles[triIdx].v2));
    nodes[leafIdx].triangleIndex = triIdx;
    nodes[leafIdx].leftChild = nodes[leafIdx].rightChild = -1;
  }
  __syncthreads();
}

__global__ void buildBVHInternal(BVHNode *nodes, int numTriangles, int level) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = 1 << level;
  int offset = stride - 1;

  if (idx < stride && idx + offset < numTriangles - 1) {
    int nodeIdx = idx + offset;
    BVHNode &node = nodes[nodeIdx];
    int leftChild = 2 * nodeIdx + 1;
    int rightChild = 2 * nodeIdx + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    node.min = minFloat3(nodes[leftChild].min, nodes[rightChild].min);
    node.max = maxFloat3(nodes[leftChild].max, nodes[rightChild].max);
  }
  __syncthreads();
}

void buildBVHWithTriangleVersion4(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);

  F3Triangle *d_triangles = thrust::raw_pointer_cast(triangles.data());

  float3 sceneMin = thrust::transform_reduce(
      thrust::device, triangles.begin(), triangles.end(),
      [] __device__(const F3Triangle &tri) {
        return minFloat3(minFloat3(tri.v0, tri.v1), tri.v2);
      },
      make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
      [] __device__(const float3 &a, const float3 &b) {
        return minFloat3(a, b);
      });

  float3 sceneMax = thrust::transform_reduce(
      thrust::device, triangles.begin(), triangles.end(),
      [] __device__(const F3Triangle &tri) {
        return maxFloat3(maxFloat3(tri.v0, tri.v1), tri.v2);
      },
      make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX),
      [] __device__(const float3 &a, const float3 &b) {
        return maxFloat3(a, b);
      });

  float3 sceneExtent = sceneMax - sceneMin;

  // Calculate Morton codes
  thrust::device_vector<Morton> mortonCodes(numTriangles);
  Morton *d_mortonCodes = thrust::raw_pointer_cast(mortonCodes.data());

  int blockSize = 256;
  int numBlocks = (numTriangles + blockSize - 1) / blockSize;

  calculateMortonCodes<<<numBlocks, blockSize>>>(
      d_triangles, d_mortonCodes, numTriangles, sceneMin, sceneExtent);
  hipDeviceSynchronize();
  // Sort Morton codes
  thrust::sort(thrust::device, mortonCodes.begin(), mortonCodes.end(),
               [] __device__(const Morton &a, const Morton &b) {
                 return a.code < b.code;
               });

  // Build BVH leaves
  BVHNode *d_nodes = thrust::raw_pointer_cast(nodes.data());

  buildBVHLeaves<<<numBlocks, blockSize>>>(d_mortonCodes, d_nodes, d_triangles,
                                           numTriangles);
  hipDeviceSynchronize();
  // Build internal nodes
  for (int level = 0; (1 << level) < numTriangles; ++level) {
    int nodesAtThisLevel = 1 << level;
    numBlocks = (nodesAtThisLevel + blockSize - 1) / blockSize;
    buildBVHInternal<<<numBlocks, blockSize>>>(d_nodes, numTriangles, level);
    hipDeviceSynchronize();
  }
  hipDeviceSynchronize();
}

//..

__device__ __inline__ bool rayTriangleIntersect(const F3Ray &ray,
                                                const F3Triangle &triangle,
                                                float &t,
                                                float3 &intersectionPoint) {
  float3 edge1 = triangle.v1 - triangle.v0;
  float3 edge2 = triangle.v2 - triangle.v0;
  float3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);

  if (a > -1e-6 && a < 1e-6)
    return false;

  float f = 1.0f / a;
  float3 s = ray.origin - triangle.v0;
  float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  float3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  t = f * dot(edge2, q);

  // Calculate the intersection point
  if (t > 1e-6) {
    intersectionPoint = ray.origin + t * ray.direction;
    // printf("%f %f
    // %f\n",intersectionPoint.x,intersectionPoint.y,intersectionPoint.z); OK
  } else {
    intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  }

  return (t > 1e-6);
}

__device__ __inline__ __forceinline__ bool
rayTriangleIntersectVersion2(const F3Ray &ray, const F3Triangle &triangle,
                             float &t, float3 &intersectionPoint) {

  const float3 v0 = triangle.v0;
  const float3 edge1 = triangle.v1 - v0;
  const float3 edge2 = triangle.v2 - v0;
  const float3 pvec = cross(ray.direction, edge2);
  const float det = dot(edge1, pvec);
  if (fabsf(det) < 1e-6f)
    return false;
  const float invDet = __frcp_rn(det);
  const float3 tvec = ray.origin - v0;
  const float u = dot(tvec, pvec) * invDet;
  if (u < 0.0f || u > 1.0f)
    return false;
  const float3 qvec = cross(tvec, edge1);
  const float v = dot(ray.direction, qvec) * invDet;
  if (v < 0.0f || u + v > 1.0f)
    return false;
  t = dot(edge2, qvec) * invDet;
  if (t > 1e-6f) {
    intersectionPoint = ray.origin + t * ray.direction;
    return true;
  }

  intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  return false;
}

__global__ void rayTracingKernel(BVHNode *nodes, F3Triangle *triangles,
                                 F3Ray *rays, int *hitResults, float *distance,
                                 float3 *intersectionPoint, int *hitId,
                                 int numRays) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  F3Ray ray = rays[idx];
  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closesIntersectionId = -1;
  float3 closestIntersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  bool isView = false; // isView=true;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    BVHNode &node = nodes[nodeIdx];

    // Ray-box intersection test
    float tmin = (node.min.x - ray.origin.x) / ray.direction.x;
    float tmax = (node.max.x - ray.origin.x) / ray.direction.x;
    if (tmin > tmax)
      SWAP(float, tmin, tmax);

    float tymin = (node.min.y - ray.origin.y) / ray.direction.y;
    float tymax = (node.max.y - ray.origin.y) / ray.direction.y;
    if (tymin > tymax)
      SWAP(float, tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
      continue;

    if (tymin > tmin)
      tmin = tymin;
    if (tymax < tmax)
      tmax = tymax;

    float tzmin = (node.min.z - ray.origin.z) / ray.direction.z;
    float tzmax = (node.max.z - ray.origin.z) / ray.direction.z;
    if (tzmin > tzmax)
      SWAP(float, tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
      continue;

    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;

    if (tmax < 0)
      continue;

    int numIdNodeTrianggleIndex = node.triangleIndex;

    if (node.triangleIndex != -1) {
      // Sheet: test the intersection with the triangle
      float t;
      float3 intersectionPointT;
      if (rayTriangleIntersect(ray, triangles[node.triangleIndex], t,
                               intersectionPointT)) {

        // To view all intersections
        if (isView)
          printf("      Node Idx [%i] Num Ray[%i] <%f %f %f>\n", nodeIdx, idx,
                 intersectionPointT.x, intersectionPointT.y,
                 intersectionPointT.z);

        if (t < closestT) {
          closestT = t;
          closestTriangle = node.triangleIndex;
          closestIntersectionPoint = intersectionPointT;
          closesIntersectionId = triangles[numIdNodeTrianggleIndex].id;
          // printf("      NodeTriangleIndex=%i
          // %i\n",numIdNodeTrianggleIndex,triangles[numIdNodeTrianggleIndex].id);
        }
      }
    } else {
      // Internal node: add children to the stack
      stack[stackPtr++] = node.leftChild;
      stack[stackPtr++] = node.rightChild;
    }
  }

  hitResults[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closesIntersectionId;
  // if (closestTriangle!=-1) { printf("t=%f\n",closestT); } OK
}

void writeBVHNodes(const thrust::device_vector<BVHNode> &nodes) {
  std::vector<BVHNode> hostNodes(nodes.size());
  thrust::copy(nodes.begin(), nodes.end(), hostNodes.begin());

  std::cout << "BVH Nodes:" << std::endl;
  for (size_t i = 0; i < hostNodes.size(); ++i) {
    const BVHNode &node = hostNodes[i];
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "  Min: (" << node.min.x << ", " << node.min.y << ", "
              << node.min.z << ")" << std::endl;
    std::cout << "  Max: (" << node.max.x << ", " << node.max.y << ", "
              << node.max.z << ")" << std::endl;
    std::cout << "  Left Child: " << node.leftChild << std::endl;
    std::cout << "  Right Child: " << node.rightChild << std::endl;
    std::cout << "  Triangle Index: " << node.triangleIndex << std::endl;
    std::cout << std::endl;
  }
}

// Function that loads the BVH node
void loadBVH(const std::string &filename,
             thrust::device_vector<BVHNode> &nodes) {
  std::ifstream inFile(filename, std::ios::binary);

  // Returns an error message if the file does not exist.
  if (!inFile) {
    throw std::runtime_error("Could not open file for reading: " + filename);
  }

  int nodeCount;
  // Read the number of nodes
  inFile.read(reinterpret_cast<char *>(&nodeCount), sizeof(int));

  // Resize the device vector to hold the nodes
  nodes.resize(nodeCount);

  // Read each node
  for (int i = 0; i < nodeCount; ++i) {
    BVHNode *raw_ptr = thrust::raw_pointer_cast(nodes.data());
    BVHNode &node = raw_ptr[i];
    inFile.read(reinterpret_cast<char *>(&node.min), sizeof(float3));
    inFile.read(reinterpret_cast<char *>(&node.max), sizeof(float3));
    inFile.read(reinterpret_cast<char *>(&node.leftChild), sizeof(int));
    inFile.read(reinterpret_cast<char *>(&node.rightChild), sizeof(int));
    inFile.read(reinterpret_cast<char *>(&node.triangleIndex), sizeof(int));
    inFile.read(reinterpret_cast<char *>(&node.boxIndex), sizeof(int));
  }

  inFile.close();
}

// Function that saves the BVH node
void saveBVH(const std::string &filename,
             const thrust::device_vector<BVHNode> &nodes) {
  std::ofstream outFile(filename, std::ios::binary);

  // Returns an error message if the file does not exist.
  if (!outFile) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }

  int nodeCount = nodes.size();
  // Write the number of nodes
  outFile.write(reinterpret_cast<const char *>(&nodeCount), sizeof(int));

  // Write each node
  for (int i = 0; i < nodeCount; ++i) {
    const BVHNode &node = nodes[i];
    outFile.write(reinterpret_cast<const char *>(&node.min), sizeof(float3));
    outFile.write(reinterpret_cast<const char *>(&node.max), sizeof(float3));
    outFile.write(reinterpret_cast<const char *>(&node.leftChild), sizeof(int));
    outFile.write(reinterpret_cast<const char *>(&node.rightChild),
                  sizeof(int));
    outFile.write(reinterpret_cast<const char *>(&node.triangleIndex),
                  sizeof(int));
    outFile.write(reinterpret_cast<const char *>(&node.boxIndex), sizeof(int));
  }

  outFile.close();
}

__global__ void
rayTracingImgKernel(unsigned char *image, int width, int height, Camera camera,
                    BVHNode *nodes, F3Triangle *triangles, int *hitResults,
                    float *distance, float3 *intersectionPoint, int *hitId)

{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height)
    return;

  int x = idx % width;
  int y = idx / width;

  F3Ray ray;
  camera.getRay(x, y, width, height, &ray.origin, &ray.direction);

  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closesIntersectionId = -1;
  float3 closestIntersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  bool isView = false; // isView=true;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    BVHNode &node = nodes[nodeIdx];

    // Ray-box intersection test
    float tmin = (node.min.x - ray.origin.x) / ray.direction.x;
    float tmax = (node.max.x - ray.origin.x) / ray.direction.x;
    if (tmin > tmax)
      SWAP(float, tmin, tmax);

    float tymin = (node.min.y - ray.origin.y) / ray.direction.y;
    float tymax = (node.max.y - ray.origin.y) / ray.direction.y;
    if (tymin > tymax)
      SWAP(float, tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
      continue;

    if (tymin > tmin)
      tmin = tymin;
    if (tymax < tmax)
      tmax = tymax;

    float tzmin = (node.min.z - ray.origin.z) / ray.direction.z;
    float tzmax = (node.max.z - ray.origin.z) / ray.direction.z;
    if (tzmin > tzmax)
      SWAP(float, tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
      continue;

    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;

    if (tmax < 0)
      continue;

    int numIdNodeTrianggleIndex = node.triangleIndex;

    if (node.triangleIndex != -1) {
      // Sheet: test the intersection with the triangle
      float t;
      float3 intersectionPointT;
      if (rayTriangleIntersect(ray, triangles[node.triangleIndex], t,
                               intersectionPointT)) {

        // To view all intersections
        // if (isView) printf("      Node Idx [%i] Num Ray[%i] <%f %f
        // %f>\n",nodeIdx,idx,intersectionPointT.x,intersectionPointT.y,intersectionPointT.z);
        if (isView)
          printf("      Num Ray[%i] <%f %f %f>\n", idx, intersectionPointT.x,
                 intersectionPointT.y, intersectionPointT.z);

        if (t < closestT) {
          closestT = t;
          closestTriangle = node.triangleIndex;
          closestIntersectionPoint = intersectionPointT;
          closesIntersectionId = triangles[numIdNodeTrianggleIndex].id;
          // printf("      NodeTriangleIndex=%i
          // %i\n",numIdNodeTrianggleIndex,triangles[numIdNodeTrianggleIndex].id);
        }
      }
    } else {
      // Internal node: add children to the stack
      stack[stackPtr++] = node.leftChild;
      stack[stackPtr++] = node.rightChild;
    }
  }

  hitResults[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closesIntersectionId;

  // if (closestTriangle!=-1) { printf("t=%f\n",closestT); } OK

  if (hitId[idx] != -1) {
    int modeColor = 0;
    int value = 255;
    float d1 = length(camera.position - camera.target);
    float d2 = distance[idx];
    if (modeColor == 0) {
      value = 255.0f * (1.0f - d2 / (1.5f * d1));
      int ypx = (y * width + x) * 3;
      image[ypx] = value;
      image[ypx + 1] = value;
      image[ypx + 2] = value;
    }

    if (modeColor == 1) {
      float R, G, B;
      float d = 255.0f * d2 / d1;
      d = (fmax(fmin(d, 255.0f), 0.0f)) / 255.0f;
      R = 1.0 - d;
      B = d;
      G = sqrt(1.0 - B * B - R * R);
      int ypx = (y * width + x) * 3;
      image[ypx] = int(255.0f * R);
      image[ypx + 1] = int(255.0f * G);
      image[ypx + 2] = int(255.0f * B);
    }
  } else {
    int ypx = (y * width + x) * 3;
    image[ypx] = 0;
    image[ypx + 1] = 0;
    image[ypx + 2] = 0;
  }
}

//..

__device__ __inline__ inline float3 operator-(const float3 &a,
                                              const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Optimized Kernel
__global__ void rayTracingImgKernelVersion2(
    unsigned char *__restrict__ image, const int width, const int height,
    const Camera camera, const BVHNode *nodes, const F3Triangle *triangles,
    int *hitResults, float *distance, float3 *intersectionPoint, int *hitId) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height)
    return;

  const int x = idx % width;
  const int y = idx / width;

  F3Ray ray;
  camera.getRay(x, y, width, height, &ray.origin, &ray.direction);

  // constexpr int MAX_STACK_SIZE = 64;
  constexpr int MAX_STACK_SIZE = 8;
  int stack[MAX_STACK_SIZE];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closestIntersectionId = -1;
  float3 closestIntersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  bool isView = false; // isView=true;
  const float3 invDir = make_float3(
      1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

  while (stackPtr > 0) {
    const int nodeIdx = stack[--stackPtr];
    const BVHNode &node = nodes[nodeIdx];

    // Test d'intersection rayon-boîte optimisé avec slabs method
    float tmin = (node.min.x - ray.origin.x) * invDir.x;
    float tmax = (node.max.x - ray.origin.x) * invDir.x;
    if (tmin > tmax)
      SWAP(float, tmin, tmax);

    float tymin = (node.min.y - ray.origin.y) * invDir.y;
    float tymax = (node.max.y - ray.origin.y) * invDir.y;
    if (tymin > tymax)
      SWAP(float, tymin, tymax);

    tmin = max(tmin, tymin);
    tmax = min(tmax, tymax);

    float tzmin = (node.min.z - ray.origin.z) * invDir.z;
    float tzmax = (node.max.z - ray.origin.z) * invDir.z;
    if (tzmin > tzmax)
      SWAP(float, tzmin, tzmax);

    tmin = max(tmin, tzmin);
    tmax = min(tmax, tzmax);

    if (tmax < 0 || tmin > tmax || tmin > closestT)
      continue;

    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;

    if (tmax < 0)
      continue;

    if (node.triangleIndex != -1) {
      float t;
      float3 intersectionPointT;
      // if (rayTriangleIntersectVers2(ray, triangles[node.triangleIndex], t,
      // intersectionPointT))
      if (rayTriangleIntersectVersion2(ray, triangles[node.triangleIndex], t,
                                       intersectionPointT)) {

        // To view all intersections
        // if (isView) printf("      Node Idx [%i] Num Ray[%i] <%f %f
        // %f>\n",nodeIdx,idx,intersectionPointT.x,intersectionPointT.y,intersectionPointT.z);
        if (isView)
          printf("      Num Ray[%i] <%f %f %f>\n", idx, intersectionPointT.x,
                 intersectionPointT.y, intersectionPointT.z);

        if (t < closestT) {
          closestT = t;
          closestTriangle = node.triangleIndex;
          closestIntersectionPoint = intersectionPointT;
          closestIntersectionId = triangles[node.triangleIndex].id;
          // printf("      NodeTriangleIndex=%i
          // %i\n",numIdNodeTrianggleIndex,triangles[numIdNodeTrianggleIndex].id);
        }
      }
    } else if (stackPtr < MAX_STACK_SIZE - 1) {
      stack[stackPtr++] = node.leftChild;
      stack[stackPtr++] = node.rightChild;
    }
  }

  hitResults[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closestIntersectionId;

  if (hitId[idx] != -1) {
    int modeColor = 0;
    int value = 255;
    float d1 = length(camera.position - camera.target);
    float d2 = distance[idx];
    if (modeColor == 0) {
      value = 255.0f * (1.0f - d2 / (1.5f * d1));
      int ypx = (y * width + x) * 3;
      image[ypx] = value;
      image[ypx + 1] = value;
      image[ypx + 2] = value;
    }

    if (modeColor == 1) {
      float R, G, B;
      float d = 255.0f * d2 / d1;
      d = (fmax(fmin(d, 255.0f), 0.0f)) / 255.0f;
      R = 1.0 - d;
      B = d;
      G = sqrt(1.0 - B * B - R * R);
      int ypx = (y * width + x) * 3;
      image[ypx] = int(255.0f * R);
      image[ypx + 1] = int(255.0f * G);
      image[ypx + 2] = int(255.0f * B);
    }
  } else {
    int ypx = (y * width + x) * 3;
    image[ypx] = 0;
    image[ypx + 1] = 0;
    image[ypx + 2] = 0;
  }
}

__device__ __inline__ __forceinline__ float3 make_float3_fast(float x, float y,
                                                              float z) {
  return make_float3(__float2int_rn(x), __float2int_rn(y), __float2int_rn(z));
}

// Optimized Kernel
__global__ void rayTracingImgKernelVersion3(
    unsigned char *__restrict__ image, const int width, const int height,
    const Camera camera, const BVHNode *__restrict__ nodes,
    const F3Triangle *__restrict__ triangles, int *__restrict__ hitResults,
    float *__restrict__ distance, float3 *__restrict__ intersectionPoint,
    int *__restrict__ hitId) {

  __shared__ BVHNode sharedNodes[64]; // Ajuster la taille selon les besoins

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height)
    return;

  const int x = idx % width;
  const int y = idx / width;

  // On charge les premiers nœuds BVH dans la mémoire partagée
  if (threadIdx.x < 64) {
    sharedNodes[threadIdx.x] = nodes[threadIdx.x];
  }
  __syncthreads();

  F3Ray ray;
  camera.getRay(x, y, width, height, &ray.origin, &ray.direction);

  constexpr int MAX_STACK_SIZE = 64;
  int stack[MAX_STACK_SIZE];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closestIntersectionId = -1;
  float3 closestIntersectionPoint =
      make_float3_fast(INFINITY, INFINITY, INFINITY);
  const float3 invDir =
      make_float3_fast(__frcp_rn(ray.direction.x), __frcp_rn(ray.direction.y),
                       __frcp_rn(ray.direction.z));

  while (stackPtr > 0) {
    const int nodeIdx = stack[--stackPtr];
    const BVHNode &node = nodeIdx < 64 ? sharedNodes[nodeIdx] : nodes[nodeIdx];

    float tmin = fmaxf(fminf((node.min.x - ray.origin.x) * invDir.x,
                             (node.max.x - ray.origin.x) * invDir.x),
                       fmaxf((node.min.y - ray.origin.y) * invDir.y,
                             (node.max.y - ray.origin.y) * invDir.y));
    float tmax = fminf(fmaxf((node.min.x - ray.origin.x) * invDir.x,
                             (node.max.x - ray.origin.x) * invDir.x),
                       fminf((node.min.y - ray.origin.y) * invDir.y,
                             (node.max.y - ray.origin.y) * invDir.y));

    tmin = fmaxf(tmin, fminf((node.min.z - ray.origin.z) * invDir.z,
                             (node.max.z - ray.origin.z) * invDir.z));
    tmax = fminf(tmax, fmaxf((node.min.z - ray.origin.z) * invDir.z,
                             (node.max.z - ray.origin.z) * invDir.z));

    if (tmax < 0 || tmin > tmax || tmin > closestT)
      continue;

    if (node.triangleIndex != -1) {
      float t;
      float3 intersectionPointT;
      if (rayTriangleIntersect(ray, triangles[node.triangleIndex], t,
                               intersectionPointT)) {
        if (t < closestT) {
          closestT = t;
          closestTriangle = node.triangleIndex;
          closestIntersectionPoint = intersectionPointT;
          closestIntersectionId = triangles[node.triangleIndex].id;
        }
      }
    } else if (stackPtr < MAX_STACK_SIZE - 1) {
      stack[stackPtr++] = node.leftChild;
      stack[stackPtr++] = node.rightChild;
    }
  }

  hitResults[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closestIntersectionId;

  if (hitId[idx] != -1) {
    int modeColor = 0;
    float d1 = length(camera.position - camera.target);
    float d2 = distance[idx];
    int ypx = (y * width + x) * 3;

    if (modeColor == 0) {
      int value = __float2int_rn(255.0f * (1.0f - d2 / (1.5f * d1)));
      image[ypx] = image[ypx + 1] = image[ypx + 2] = value;
    } else if (modeColor == 1) {
      float d = fminf(fmaxf(255.0f * d2 / d1, 0.0f), 255.0f) / 255.0f;
      float R = 1.0f - d;
      float B = d;
      float G = sqrtf(1.0f - B * B - R * R);
      image[ypx] = __float2int_rn(255.0f * R);
      image[ypx + 1] = __float2int_rn(255.0f * G);
      image[ypx + 2] = __float2int_rn(255.0f * B);
    }
  } else {
    int ypx = (y * width + x) * 3;
    image[ypx] = image[ypx + 1] = image[ypx + 2] = 0;
  }
}

void savePPM(const std::string &filename, unsigned char *data, int width,
             int height) {
  std::ofstream file(filename, std::ios::binary);
  file << "P6\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<char *>(data), width * height * 3);
}

void buildPicturRayTracingPPM(thrust::device_vector<F3Triangle> &triangles,
                              thrust::device_vector<BVHNode> &nodes,
                              Camera camera, int width, int height,
                              const std::string &filename, bool isSave) {
  // Before using this function, the BVH must already be calculated and the
  // triangles must be in the device. Ray Tracing
  // const int threadsPerBlock = 256;
  const int threadsPerBlock = 512;
  const int numRays =
      width * height; // Total number of rays based on image dimensions
  int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

  //...
  thrust::device_vector<unsigned char> deviceImage(width * height * 3);
  thrust::device_vector<int> deviceHitResults(numRays);
  thrust::device_vector<float> deviceDistanceResults(numRays);
  thrust::device_vector<float3> deviceIntersectionPoint(numRays);
  thrust::device_vector<int> deviceIdResults(numRays);

  //...
  // rayTracingImgKernel<<<blocksPerGrid, threadsPerBlock>>>(
  // rayTracingImgKernelVersion3<<<blocksPerGrid, threadsPerBlock>>>(
  rayTracingImgKernelVersion2<<<blocksPerGrid, threadsPerBlock>>>(
      thrust::raw_pointer_cast(deviceImage.data()), width, height, camera,
      thrust::raw_pointer_cast(nodes.data()),
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(deviceHitResults.data()),
      thrust::raw_pointer_cast(deviceDistanceResults.data()),
      thrust::raw_pointer_cast(deviceIntersectionPoint.data()),
      thrust::raw_pointer_cast(deviceIdResults.data()));

  //...
  if (isSave) {
    thrust::host_vector<unsigned char> hostImage = deviceImage;
    savePPM(filename, hostImage.data(), width, height);
  }

  // Memory cleaning
  deviceHitResults.clear();
  deviceDistanceResults.clear();
  deviceIntersectionPoint.clear();
  deviceIdResults.clear();
  deviceImage.clear();
}

__device__ __inline__ Vec3 computeColor(const Vec3 &hitPoint,
                                        const Vec3 &normal, const Vec3 &viewDir,
                                        const Material &material,
                                        const Vec3 &lightPos,
                                        const Vec3 &lightColor) {
  float ambientStrength = 0.1f;                     // Force de l'ambient
  Vec3 ambient = material.albedo * ambientStrength; // Couleur ambiante

  // Direction de la lumière
  Vec3 lightDir = lightPos - hitPoint;
  lightDir.normalize();

  // Calcul de la lumière diffuse
  float diff = fmaxf(normal.dot(lightDir), 0.0f);
  Vec3 diffuse = material.albedo * diff; // Couleur diffuse

  // Calcul de la lumière spéculaire
  Vec3 reflectDir = lightDir - normal * (2.0f * normal.dot(lightDir));
  float spec =
      powf(fmaxf(reflectDir.dot(viewDir), 0.0f),
           material.roughness * 128); // Ajustement basé sur la rugosité
  Vec3 specular = lightColor * spec;  // Couleur spéculaire

  // Combinaison des contributions
  return ambient + diffuse + specular;
}

/********------******/

//#########################################################################################################################################################
// Version 5

// Function to calculate the height of the tree
int calculateTreeHeight(int numTriangles) {
  return static_cast<int>(std::ceil(std::log2(numTriangles)));
}

// Function to get the start index for a given level
int getStartIndexForLevel(int level, int numTriangles) {
  return std::max(0, (1 << level) - 1);
}

// Function to get the end index for a given level
int getEndIndexForLevel(int level, int numTriangles) {
  return std::min((1 << (level + 1)) - 1, numTriangles - 1);
}

// Construction of internal nodes
struct BuildInternalNodesFunctor {
  BVHNode *nodes;
  int numTriangles;

  __host__ __device__ __inline__ BuildInternalNodesFunctor(BVHNode *_nodes,
                                                           int _numTriangles)
      : nodes(_nodes), numTriangles(_numTriangles) {}

  __host__ __device__ __inline__ void operator()(int i) const {
    if (i >= numTriangles - 1)
      return;

    BVHNode &node = nodes[i];
    int leftChild = 2 * i + 1;
    int rightChild = 2 * i + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    const BVHNode &leftNode = nodes[leftChild];
    const BVHNode &rightNode = nodes[rightChild];

    node.min = make_float3(fminf(leftNode.min.x, rightNode.min.x),
                           fminf(leftNode.min.y, rightNode.min.y),
                           fminf(leftNode.min.z, rightNode.min.z));

    node.max = make_float3(fmaxf(leftNode.max.x, rightNode.max.x),
                           fmaxf(leftNode.max.y, rightNode.max.y),
                           fmaxf(leftNode.max.z, rightNode.max.z));
  }
};

void buildBVHWithTriangleVersion5(thrust::device_vector<F3Triangle> &triangles,
                                  thrust::device_vector<BVHNode> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);

  // Initialize the leaves in parallel
  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(numTriangles),
                   [triangles = thrust::raw_pointer_cast(triangles.data()),
                    nodes = thrust::raw_pointer_cast(nodes.data()),
                    numTriangles] __device__(int i) {
                     BVHNode &node = nodes[numTriangles - 1 + i];
                     calculateBoundingBox(triangles[i], node.min, node.max);
                     node.triangleIndex = i;
                     node.leftChild = node.rightChild = -1;
                   });

  // Synchronize to ensure all leaves are initialized. Very important !!!
  hipDeviceSynchronize();

  // Build the internal nodes level by level
  int treeHeight = calculateTreeHeight(numTriangles);
  BVHNode *raw_ptr = thrust::raw_pointer_cast(nodes.data());

  for (int level = treeHeight - 1; level >= 0; --level) {
    int startIdx = getStartIndexForLevel(level, numTriangles);
    int endIdx = getEndIndexForLevel(level, numTriangles);

    thrust::for_each(thrust::device, thrust::make_counting_iterator(startIdx),
                     thrust::make_counting_iterator(endIdx + 1),
                     BuildInternalNodesFunctor(raw_ptr, numTriangles));

    // Synchronize after each level. Very important !!!
    hipDeviceSynchronize();
  }
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
//#########################################################################################################################################################

//#########################################################################################################################################################
// AABB

// Function to calculate the AABB of a triangle
__device__ __inline__ AABB calculateAABB(const F3Triangle &triangle) {
  AABB aabb;
  aabb.min =
      make_float3(fminf(fminf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                  fminf(fminf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                  fminf(fminf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
  aabb.max =
      make_float3(fmaxf(fmaxf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                  fmaxf(fmaxf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                  fmaxf(fmaxf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
  return aabb;
}

struct CalculateAABB {
  __host__ __device__ __inline__ AABB
  operator()(const F3Triangle &triangle) const {
    AABB aabb;
    aabb.min =
        make_float3(fminf(fminf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                    fminf(fminf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                    fminf(fminf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
    aabb.max =
        make_float3(fmaxf(fmaxf(triangle.v0.x, triangle.v1.x), triangle.v2.x),
                    fmaxf(fmaxf(triangle.v0.y, triangle.v1.y), triangle.v2.y),
                    fmaxf(fmaxf(triangle.v0.z, triangle.v1.z), triangle.v2.z));
    return aabb;
  }
};

// Function to merge two AABBs
__device__ __inline__ AABB mergeAABB(const AABB &a, const AABB &b) {
  AABB result;
  result.min = make_float3(fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y),
                           fminf(a.min.z, b.min.z));
  result.max = make_float3(fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y),
                           fmaxf(a.max.z, b.max.z));
  return result;
}

__device__ __inline__ bool intersectAABB(const F3Ray &ray, const AABB &aabb) {
  float3 invDir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y,
                              1.0f / ray.direction.z);
  float3 t0 = make_float3((aabb.min.x - ray.origin.x) * invDir.x,
                          (aabb.min.y - ray.origin.y) * invDir.y,
                          (aabb.min.z - ray.origin.z) * invDir.z);
  float3 t1 = make_float3((aabb.max.x - ray.origin.x) * invDir.x,
                          (aabb.max.y - ray.origin.y) * invDir.y,
                          (aabb.max.z - ray.origin.z) * invDir.z);
  float tmin =
      fmaxf(fmaxf(fminf(t0.x, t1.x), fminf(t0.y, t1.y)), fminf(t0.z, t1.z));
  float tmax =
      fminf(fminf(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y)), fmaxf(t0.z, t1.z));
  return tmax >= tmin && tmin < ray.tMax && tmax > ray.tMin;
}

__device__ __inline__ bool
intersectTriangleVersion2(const F3Ray &ray, const F3Triangle &triangle,
                          float &t, float3 &intersectionPoint) {
  float3 edge1 = triangle.v1 - triangle.v0;
  float3 edge2 = triangle.v2 - triangle.v0;
  float3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);

  if (a > -1e-6f && a < 1e-6f)
    return false;
  float f = 1.0f / a;
  float3 s = ray.origin - triangle.v0;
  float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  float3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  t = f * dot(edge2, q);

  if (t > 1e-6f) {
    intersectionPoint = ray.origin + ray.direction * t;
  }
  return t > ray.tMin && t < ray.tMax;
}

__device__ __inline__ float3 calculateCentroid(const F3Triangle &triangle) {
  return make_float3((triangle.v0.x + triangle.v1.x + triangle.v2.x) / 3.0f,
                     (triangle.v0.y + triangle.v1.y + triangle.v2.y) / 3.0f,
                     (triangle.v0.z + triangle.v1.z + triangle.v2.z) / 3.0f);
}

struct MergeAABB {
  __host__ __device__ __inline__ AABB operator()(const AABB &a,
                                                 const AABB &b) const {
    AABB result;
    result.min.x = fminf(a.min.x, b.min.x);
    result.min.y = fminf(a.min.y, b.min.y);
    result.min.z = fminf(a.min.z, b.min.z);
    result.max.x = fmaxf(a.max.x, b.max.x);
    result.max.y = fmaxf(a.max.y, b.max.y);
    result.max.z = fmaxf(a.max.z, b.max.z);
    return result;
  }

  __host__ __device__ __inline__ AABB identity() const {
    AABB result;
    result.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    result.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    return result;
  }
};

struct CalculateCentroid {
  __host__ __device__ __inline__ float3
  operator()(const F3Triangle &triangle) const {
    float3 centroid;
    centroid.x = (triangle.v0.x + triangle.v1.x + triangle.v2.x) / 3.0f;
    centroid.y = (triangle.v0.y + triangle.v1.y + triangle.v2.y) / 3.0f;
    centroid.z = (triangle.v0.z + triangle.v1.z + triangle.v2.z) / 3.0f;
    return centroid;
  }
};

__device__ __inline__ __host__ inline float getComponent(const float3 &vec,
                                                         int axis) {
  switch (axis) {
  case 0:
    return vec.x;
  case 1:
    return vec.y;
  case 2:
    return vec.z;
  default:
    return 0.0f; // or handle error
  }
}

// Recursive function to build the BVH
void buildBVH_AABB_Recursive(thrust::device_vector<F3Triangle> &triangles,
                             thrust::device_vector<AABB> &aabbs,
                             thrust::device_vector<float3> &centroids,
                             thrust::device_vector<BVHNodeAABB> &nodes,
                             int &nodeIndex, int start, int end) {
  BVHNodeAABB *raw_ptr = thrust::raw_pointer_cast(nodes.data());
  BVHNodeAABB &node = raw_ptr[nodeIndex];
  node.firstTriangle = start;
  node.triangleCount = end - start;
  node.bounds = thrust::reduce(thrust::device, aabbs.begin() + start,
                               aabbs.begin() + end, AABB(), MergeAABB());

  if (node.triangleCount <= 2) {
    // Leaf node
    node.leftChild = -1;
    node.rightChild = -1;
  } else {
    // Internal node
    int axis = 0;
    // float splitPos = (node.bounds.min[axis] + node.bounds.max[axis]) * 0.5f;
    float splitPos = 0.5f * (getComponent(node.bounds.min, axis) +
                             getComponent(node.bounds.max, axis));

    // Partition the triangles
    auto splitIter = thrust::partition(
        thrust::device,
        thrust::make_zip_iterator(
            thrust::make_tuple(triangles.begin() + start, aabbs.begin() + start,
                               centroids.begin() + start)),
        thrust::make_zip_iterator(thrust::make_tuple(triangles.begin() + end,
                                                     aabbs.begin() + end,
                                                     centroids.begin() + end)),
        [=] __device__(const thrust::tuple<F3Triangle, AABB, float3> &t) {
          return getComponent(thrust::get<2>(t), axis) < splitPos;
        });

    int mid = start + thrust::distance(
                          thrust::make_zip_iterator(thrust::make_tuple(
                              triangles.begin() + start, aabbs.begin() + start,
                              centroids.begin() + start)),
                          splitIter);

    // Check if the partition actually divided the triangles
    if (mid == start || mid == end) {
      // If the partition did not divide the triangles, force a division in the
      // middle
      mid = start + (end - start) / 2;
    }

    // std::cout<<"mid="<<mid<<"\n"; CTRL OK

    // Create the child nodes
    node.leftChild = ++nodeIndex;
    buildBVH_AABB_Recursive(triangles, aabbs, centroids, nodes, nodeIndex,
                            start, mid);
    node.rightChild = ++nodeIndex;
    buildBVH_AABB_Recursive(triangles, aabbs, centroids, nodes, nodeIndex, mid,
                            end);
  }
}

struct BVHBuildTask {
  int nodeIndex;
  int start;
  int end;
  BVHBuildTask(int ni, int s, int e) : nodeIndex(ni), start(s), end(e) {}
};

void buildBVH_AABB_Iterative(thrust::device_vector<F3Triangle> &triangles,
                             thrust::device_vector<AABB> &aabbs,
                             thrust::device_vector<float3> &centroids,
                             thrust::device_vector<BVHNodeAABB> &nodes,
                             int &nodeIndex, int start, int end) {

  std::stack<BVHBuildTask> taskStack;
  taskStack.push(BVHBuildTask(nodeIndex, start, end));

  while (!taskStack.empty()) {
    BVHBuildTask task = taskStack.top();
    taskStack.pop();

    int currentNodeIndex = task.nodeIndex;
    int currentStart = task.start;
    int currentEnd = task.end;

    BVHNodeAABB *raw_ptr = thrust::raw_pointer_cast(nodes.data());
    BVHNodeAABB &node = raw_ptr[currentNodeIndex];

    node.firstTriangle = currentStart;
    node.triangleCount = currentEnd - currentStart;

    // std::cout<<"current nodeIndex="<<currentNodeIndex<<" Start=
    // "<<currentStart<<" End="<<currentEnd <<"\n";
    // std::cout<<"node.triangleCount="<<node.triangleCount<<"\n";
    // getchar();

    // Calculates the AABB of the node
    node.bounds =
        thrust::reduce(thrust::device, aabbs.begin() + currentStart,
                       aabbs.begin() + currentEnd, AABB(), MergeAABB());

    if (node.triangleCount <= 2) {
      // Leaf node
      node.leftChild = -1;
      node.rightChild = -1;
    } else {
      // Internal node
      int axis =
          0; // // separation axis (can be optimized). To be seen later, if ...
      float splitPos = 0.5f * (getComponent(node.bounds.min, axis) +
                               getComponent(node.bounds.max, axis));
      // Partition the triangles
      auto splitIter = thrust::partition(
          thrust::device,
          thrust::make_zip_iterator(thrust::make_tuple(
              triangles.begin() + currentStart, aabbs.begin() + currentStart,
              centroids.begin() + currentStart)),
          thrust::make_zip_iterator(thrust::make_tuple(
              triangles.begin() + currentEnd, aabbs.begin() + currentEnd,
              centroids.begin() + currentEnd)),
          [=] __device__(const thrust::tuple<F3Triangle, AABB, float3> &t) {
            return getComponent(thrust::get<2>(t), axis) < splitPos;
          });

      int mid = currentStart +
                thrust::distance(thrust::make_zip_iterator(thrust::make_tuple(
                                     triangles.begin() + currentStart,
                                     aabbs.begin() + currentStart,
                                     centroids.begin() + currentStart)),
                                 splitIter);

      // Check if the partition actually divided the triangles
      if (mid == currentStart || mid == currentEnd) {
        // If the partition did not divide the triangles, force a division in
        // the middle
        mid = currentStart + (currentEnd - currentStart) / 2;
      }

      // std::cout<<"mid="<<mid<<"\n";

      node.leftChild = ++nodeIndex;
      node.rightChild = ++nodeIndex;

      taskStack.push(BVHBuildTask(node.rightChild, mid, currentEnd));
      taskStack.push(BVHBuildTask(node.leftChild, currentStart, mid));
    }
  }
}

void buildBVH_AABB(thrust::device_vector<F3Triangle> &triangles,
                   thrust::device_vector<BVHNodeAABB> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);
  // Calculate AABBs and centroids for all triangles
  thrust::device_vector<AABB> aabbs(numTriangles);
  thrust::device_vector<float3> centroids(numTriangles);

  thrust::transform(thrust::device, triangles.begin(), triangles.end(),
                    aabbs.begin(), CalculateAABB());
  thrust::transform(thrust::device, triangles.begin(), triangles.end(),
                    centroids.begin(), CalculateCentroid());

  // Build the BVH recursively or iteratively
  int rootNodeIndex = 0;
  buildBVH_AABB_Recursive(
      triangles, aabbs, centroids, nodes, rootNodeIndex, 0,
      numTriangles); // Nota: it is a bit faster than compared to the iterative
  // buildBVH_AABB_Iterative(triangles, aabbs, centroids, nodes, rootNodeIndex,
  // 0, numTriangles);
}

__global__ void
intersectBVH_AABB(const BVHNodeAABB *nodes, const F3Triangle *triangles,
                  const F3Ray *rays, int *hitResults, float *hitDistances,
                  float3 *intersectionPoint, int *hitId, int numRays)

{
  int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rayIdx >= numRays)
    return;

  F3Ray ray = rays[rayIdx];
  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  // float closestHit = ray.tMax;
  float closestHit = INFINITY;
  int closestTriangle = -1;
  int closesIntersectionId = -1;
  float3 closestIntersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  bool isView = false; // isView=true;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    const BVHNodeAABB &node = nodes[nodeIdx];

    if (intersectAABB(ray, node.bounds)) {
      // printf("nodeIdx=%i\n",nodeIdx);
      if (node.leftChild == -1 && node.rightChild == -1) {
        // Leaf node
        for (int i = 0; i < node.triangleCount; ++i) {
          const F3Triangle &tri = triangles[node.firstTriangle + i];
          float t;
          float3 intersectionPointT;

          if (intersectTriangleVersion2(ray, tri, t, intersectionPointT)) {
            if (isView)
              printf("[%i] %f \n", rayIdx, t);
            if (t < closestHit) {
              closestHit = t;
              closestTriangle = node.firstTriangle + i;
              closestIntersectionPoint = intersectionPointT;
              closesIntersectionId = triangles[closestTriangle].id;
            }
          }
        }
      } else {
        if (node.rightChild != -1)
          stack[stackPtr++] = node.rightChild;
        if (node.leftChild != -1)
          stack[stackPtr++] = node.leftChild;
      }
    }
  }

  hitResults[rayIdx] = closestTriangle;
  hitDistances[rayIdx] = closestHit;
  intersectionPoint[rayIdx] = closestIntersectionPoint;
  hitId[rayIdx] = closesIntersectionId;
}

__global__ void rayTracingImgKernel_AABB(unsigned char *image, int width,
                                         int height, Camera camera,
                                         BVHNodeAABB *nodes,
                                         F3Triangle *triangles, int *hitResults,
                                         float *hitDistances,
                                         float3 *intersectionPoint, int *hitId)

{
  int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rayIdx >= width * height)
    return;

  int x = rayIdx % width;
  int y = rayIdx / width;

  F3Ray ray;
  camera.getRay(x, y, width, height, &ray.origin, &ray.direction);
  ray.tMin = -INFINITY;
  ray.tMax = INFINITY;

  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  // float closestHit = ray.tMax;
  float closestHit = INFINITY;
  int closestTriangle = -1;
  int closesIntersectionId = -1;
  float3 closestIntersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  bool isView = false; // isView=true;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    const BVHNodeAABB &node = nodes[nodeIdx];

    if (intersectAABB(ray, node.bounds)) {
      // printf("nodeIdx=%i\n",nodeIdx);
      if (node.leftChild == -1 && node.rightChild == -1) {
        // Leaf node
        for (int i = 0; i < node.triangleCount; ++i) {
          const F3Triangle &tri = triangles[node.firstTriangle + i];
          float t;
          float3 intersectionPointT;

          if (intersectTriangleVersion2(ray, tri, t, intersectionPointT)) {
            if (isView)
              printf("[%i] %f \n", rayIdx, t);
            if (t < closestHit) {
              closestHit = t;
              closestTriangle = node.firstTriangle + i;
              closestIntersectionPoint = intersectionPointT;
              closesIntersectionId = triangles[closestTriangle].id;
            }
          }
        }
      } else {
        if (node.rightChild != -1)
          stack[stackPtr++] = node.rightChild;
        if (node.leftChild != -1)
          stack[stackPtr++] = node.leftChild;
      }
    }
  }

  hitResults[rayIdx] = closestTriangle;
  hitDistances[rayIdx] = closestHit;
  intersectionPoint[rayIdx] = closestIntersectionPoint;
  hitId[rayIdx] = closesIntersectionId;

  // if (closestTriangle!=-1) { printf("t=%f\n",closestT); } OK

  if (hitId[rayIdx] != -1) {
    int modeColor = 0;
    int value = 255;
    float d1 = length(camera.position - camera.target);
    float d2 = hitDistances[rayIdx];
    if (modeColor == 0) {
      value = 255.0f * (1.0f - d2 / (1.5f * d1));
      image[(y * width + x) * 3] = value;
      image[(y * width + x) * 3 + 1] = value;
      image[(y * width + x) * 3 + 2] = value;
    }

    if (modeColor == 1) {
      float R, G, B;
      float d = 255.0f * d2 / d1;
      d = (fmax(fmin(d, 255.0f), 0.0f)) / 255.0f;
      R = 1.0 - d;
      B = d;
      G = sqrt(1.0 - B * B - R * R);
      image[(y * width + x) * 3] = int(255.0f * R);
      image[(y * width + x) * 3 + 1] = int(255.0f * G);
      image[(y * width + x) * 3 + 2] = int(255.0f * B);
    }
  } else {
    image[(y * width + x) * 3] = 0;     // Red chanel
    image[(y * width + x) * 3 + 1] = 0; // Green chanel
    image[(y * width + x) * 3 + 2] = 0; // Blue chanel
  }
}

void buildPicturRayTracingPPM_AABB(thrust::device_vector<F3Triangle> &triangles,
                                   thrust::device_vector<BVHNodeAABB> &nodes,
                                   Camera camera, int width, int height,
                                   const std::string &filename, bool isSave) {
  // Before using this function, the BVH must already be calculated and the
  // triangles must be in the device. Ray Tracing
  const int threadsPerBlock = 512;
  const int numRays =
      width * height; // Total number of rays based on image dimensions
  int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

  //...
  thrust::device_vector<unsigned char> deviceImage(width * height * 3);
  thrust::device_vector<int> deviceHitResults(numRays);
  thrust::device_vector<float> deviceDistanceResults(numRays);
  thrust::device_vector<float3> deviceIntersectionPoint(numRays);
  thrust::device_vector<int> deviceIdResults(numRays);

  //...

  rayTracingImgKernel_AABB<<<blocksPerGrid, threadsPerBlock>>>(
      thrust::raw_pointer_cast(deviceImage.data()), width, height, camera,
      thrust::raw_pointer_cast(nodes.data()),
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(deviceHitResults.data()),
      thrust::raw_pointer_cast(deviceDistanceResults.data()),
      thrust::raw_pointer_cast(deviceIntersectionPoint.data()),
      thrust::raw_pointer_cast(deviceIdResults.data()));

  hipDeviceSynchronize();

  //...
  if (isSave) {
    thrust::host_vector<unsigned char> hostImage = deviceImage;
    savePPM(filename, hostImage.data(), width, height);
  }

  // Memory cleaning
  deviceHitResults.clear();
  deviceDistanceResults.clear();
  deviceIntersectionPoint.clear();
  deviceIdResults.clear();
  deviceImage.clear();
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
//#########################################################################################################################################################

//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
// SAH Method 1

struct SAHBinInfo {
  float3 bounds_min;
  float3 bounds_max;
  int count;
};

__device__ __inline__ float calculateNodeCost(const BVHNodeSAH &node) {
  float3 extent = node.bounds_max - node.bounds_min;
  return 2.0f *
         (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

__device__ __inline__ float calculateBinCost(const SAHBinInfo &bin) {
  float3 extent = bin.bounds_max - bin.bounds_min;
  return 2.0f *
         (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

__device__ __inline__ inline float3 elementwise_min(const float3 &a,
                                                    const float3 &b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __inline__ inline float3 elementwise_max(const float3 &a,
                                                    const float3 &b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __inline__ void updateNodeBounds(BVHNodeSAH &node,
                                            const F3Triangle &triangle) {
  node.bounds_min = elementwise_min(node.bounds_min, triangle.v0);
  node.bounds_min = elementwise_min(node.bounds_min, triangle.v1);
  node.bounds_min = elementwise_min(node.bounds_min, triangle.v2);
  node.bounds_max = elementwise_max(node.bounds_max, triangle.v0);
  node.bounds_max = elementwise_max(node.bounds_max, triangle.v1);
  node.bounds_max = elementwise_max(node.bounds_max, triangle.v2);
}

__device__ __inline__ void updateBinBounds(SAHBinInfo &bin,
                                           const F3Triangle &triangle) {
  bin.bounds_min = elementwise_min(bin.bounds_min, triangle.v0);
  bin.bounds_min = elementwise_min(bin.bounds_min, triangle.v1);
  bin.bounds_min = elementwise_min(bin.bounds_min, triangle.v2);
  bin.bounds_max = elementwise_max(bin.bounds_max, triangle.v0);
  bin.bounds_max = elementwise_max(bin.bounds_max, triangle.v1);
  bin.bounds_max = elementwise_max(bin.bounds_max, triangle.v2);
}

/*
__global__ void initializeLeavesSAH(F3Triangle* triangles, BVHNodeSAH* nodes,
int numTriangles) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx <
numTriangles) {
        //if (idx >= numTriangles) return;
                BVHNodeSAH& node = nodes[idx];
                const F3Triangle& tri = triangles[idx];
                node.bounds_min = elementwise_min(elementwise_min(tri.v0,
tri.v1), tri.v2); node.bounds_max = elementwise_max(elementwise_max(tri.v0,
tri.v1), tri.v2); node.triangleIndex = idx; node.left = node.right = -1;
                updateNodeBounds(node, triangles[idx]);
                //node.left = 0;
                //node.right = 0;
        }
}
*/

__global__ void initializeLeavesSAH(F3Triangle *triangles, BVHNodeSAH *nodes,
                                    int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    BVHNodeSAH &node = nodes[idx];
    const F3Triangle &tri = triangles[idx];

    // Calcul des bornes minimales et maximales directement à partir des sommets
    // du triangle
    node.bounds_min = elementwise_min(elementwise_min(tri.v0, tri.v1), tri.v2);
    node.bounds_max = elementwise_max(elementwise_max(tri.v0, tri.v1), tri.v2);

    // Initialisation de l'indice du triangle et des enfants
    node.triangleIndex = idx;
    node.left = node.right = -1;
  }
}

__device__ __inline__ void swapNode(BVHNodeSAH &a, BVHNodeSAH &b) {
  BVHNodeSAH temp = a;
  a = b;
  b = temp;
}

__device__ __inline__ void swapTriangle(F3Triangle &a, F3Triangle &b) {
  F3Triangle temp = a;
  a = b;
  b = temp;
}

__global__ void buildInternalNodesSAHOld(F3Triangle *triangles,
                                         BVHNodeSAH *nodes, int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTriangles - 1)
    return;

  int nodeIdx = numTriangles + idx;
  // int nodeIdx = idx;
  BVHNodeSAH &node = nodes[nodeIdx];

  if (nodeIdx >= numTriangles) {
    node.triangleIndex = -1;
  }

  float bestCost = FLT_MAX;
  int bestAxis = -1;
  int bestSplit = -1;

  const int numBins = 32;
  float3 global_centroid_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 global_centroid_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (int i = 0; i < numTriangles; ++i) {
    float3 centroid =
        (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) * (1.0f / 3.0f);
    global_centroid_min = elementwise_min(global_centroid_min, centroid);
    global_centroid_max = elementwise_max(global_centroid_max, centroid);
  }

  for (int axis = 0; axis < 3; ++axis) {
    SAHBinInfo bins[numBins];
    for (int i = 0; i < numBins; ++i) {
      bins[i].bounds_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
      bins[i].bounds_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      bins[i].count = 0;
    }

    float axisMin = (axis == 0) ? global_centroid_min.x
                                : ((axis == 1) ? global_centroid_min.y
                                               : global_centroid_min.z);
    float axisMax = (axis == 0) ? global_centroid_max.x
                                : ((axis == 1) ? global_centroid_max.y
                                               : global_centroid_max.z);

    bool allInOneBin = true;
    int nonEmptyBin = -1;

    for (int i = 0; i < numTriangles; ++i) {
      float3 centroid = (nodes[i].bounds_min + nodes[i].bounds_max) * 0.5f;
      float axisValue =
          (axis == 0) ? centroid.x : ((axis == 1) ? centroid.y : centroid.z);

      int binIdx = min(numBins - 1, int((axisValue - axisMin) /
                                        (axisMax - axisMin) * numBins));
      updateBinBounds(bins[binIdx], triangles[nodes[i].triangleIndex]);
      bins[binIdx].count++;

      if (nonEmptyBin == -1) {
        nonEmptyBin = binIdx;
      } else if (nonEmptyBin != binIdx) {
        allInOneBin = false;
      }
    }

    printf("axis=%i\n", axis);
    printf("allInOneBin=%i\n", allInOneBin);
    if (allInOneBin) {
      int midTriangle = numTriangles / 2;
      bestAxis = axis;
      bestSplit = nonEmptyBin;
      bestCost = 0;
      break;
    }

    SAHBinInfo leftAccum = bins[0];
    SAHBinInfo rightAccum = {make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
                             make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), 0};

    for (int i = 1; i < numBins; ++i) {
      rightAccum.bounds_min =
          elementwise_min(rightAccum.bounds_min, bins[i].bounds_min);
      rightAccum.bounds_max =
          elementwise_max(rightAccum.bounds_max, bins[i].bounds_max);
      rightAccum.count += bins[i].count;
    }

    for (int split = 1; split < numBins; ++split) {
      float cost = (calculateBinCost(leftAccum) * leftAccum.count +
                    calculateBinCost(rightAccum) * rightAccum.count) /
                   numTriangles;

      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestSplit = split;
      }

      leftAccum.bounds_min =
          elementwise_min(leftAccum.bounds_min, bins[split].bounds_min);
      leftAccum.bounds_max =
          elementwise_max(leftAccum.bounds_max, bins[split].bounds_max);
      leftAccum.count += bins[split].count;

      rightAccum.bounds_min =
          elementwise_max(rightAccum.bounds_min, bins[split].bounds_min);
      rightAccum.bounds_max =
          elementwise_min(rightAccum.bounds_max, bins[split].bounds_max);
      rightAccum.count -= bins[split].count;
    }
  }

  // Perform split
  if (bestAxis != -1) {
    float splitPos;

    if (bestCost == 0) {
      splitPos = (getComponent(global_centroid_min, bestAxis) +
                  getComponent(global_centroid_max, bestAxis)) *
                 0.5f;
    } else {
      splitPos = getComponent(global_centroid_min, bestAxis) +
                 (getComponent(global_centroid_max, bestAxis) -
                  getComponent(global_centroid_min, bestAxis)) *
                     (float)bestSplit / numBins;
    }

    // begin::avant
    int mid = 0;
    for (int i = 0; i < numTriangles; ++i) {
      float3 centroid = (nodes[i].bounds_min + nodes[i].bounds_max) * 0.5f;
      float centroidValue = getComponent(centroid, bestAxis);
      if (centroidValue < splitPos) {
        swapNode(nodes[mid], nodes[i]);
        swapTriangle(triangles[mid], triangles[i]);
        mid++;
      }
    }
    // printf("mid=%i\n",mid);
    //  end::avant

    if (mid == 0)
      mid = 1;
    if (mid == numTriangles)
      mid = numTriangles - 1;

    node.left = mid;
    node.right = numTriangles + idx + 1;

    printf("In idx[%i] mid after=%i  node.left=%i node.right=%i\n", idx, mid,
           node.left, node.right);

    node.bounds_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    node.bounds_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // begin::avant

    for (int i = 0; i < numTriangles; ++i) {
      node.bounds_min = elementwise_min(node.bounds_min, nodes[i].bounds_min);
      node.bounds_max = elementwise_max(node.bounds_max, nodes[i].bounds_max);
    }

    // end::avant
  }
}

__global__ void buildInternalNodesSAH(F3Triangle *triangles, BVHNodeSAH *nodes,
                                      int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTriangles - 1)
    return;

  int nodeIdx = numTriangles + idx;
  BVHNodeSAH &node = nodes[nodeIdx];

  // Initialisation de l'indice du triangle pour les nœuds internes
  if (nodeIdx >= numTriangles) {
    node.triangleIndex = -1;
  }

  float bestCost = FLT_MAX;
  int bestAxis = -1;
  int bestSplit = -1;

  const int numBins = 32;
  float3 global_centroid_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 global_centroid_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  // Calcul des bornes globales des centroids
  for (int i = 0; i < numTriangles; ++i) {
    float3 centroid =
        (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) * (1.0f / 3.0f);
    global_centroid_min = elementwise_min(global_centroid_min, centroid);
    global_centroid_max = elementwise_max(global_centroid_max, centroid);
  }

  for (int axis = 0; axis < 3; ++axis) {
    SAHBinInfo bins[numBins];

    // Initialisation des bins
    for (int i = 0; i < numBins; ++i) {
      bins[i].bounds_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
      bins[i].bounds_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      bins[i].count = 0;
    }

    float axisMin = (axis == 0) ? global_centroid_min.x
                                : ((axis == 1) ? global_centroid_min.y
                                               : global_centroid_min.z);
    float axisMax = (axis == 0) ? global_centroid_max.x
                                : ((axis == 1) ? global_centroid_max.y
                                               : global_centroid_max.z);

    bool allInOneBin = true;
    int nonEmptyBin = -1;

    // Remplissage des bins
    for (int i = 0; i < numTriangles; ++i) {
      float3 centroid = (nodes[i].bounds_min + nodes[i].bounds_max) * 0.5f;
      float axisValue =
          (axis == 0) ? centroid.x : ((axis == 1) ? centroid.y : centroid.z);

      int binIdx = min(numBins - 1, int((axisValue - axisMin) /
                                        (axisMax - axisMin) * numBins));
      updateBinBounds(bins[binIdx], triangles[nodes[i].triangleIndex]);
      bins[binIdx].count++;

      if (nonEmptyBin == -1) {
        nonEmptyBin = binIdx;
      } else if (nonEmptyBin != binIdx) {
        allInOneBin = false;
      }
    }

    // Si tous les centroids sont dans un seul bin
    if (allInOneBin) {
      int midTriangle = numTriangles / 2;
      bestAxis = axis;
      bestSplit = nonEmptyBin;
      bestCost = 0;
      break;
    }

    SAHBinInfo leftAccum = bins[0];
    SAHBinInfo rightAccum = {make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
                             make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), 0};

    // Accumulation des informations sur les bins
    for (int i = 1; i < numBins; ++i) {
      rightAccum.bounds_min =
          elementwise_min(rightAccum.bounds_min, bins[i].bounds_min);
      rightAccum.bounds_max =
          elementwise_max(rightAccum.bounds_max, bins[i].bounds_max);
      rightAccum.count += bins[i].count;
    }

    // Calcul du coût pour chaque split possible
    for (int split = 1; split < numBins; ++split) {
      float cost = (calculateBinCost(leftAccum) * leftAccum.count +
                    calculateBinCost(rightAccum) * rightAccum.count) /
                   numTriangles;

      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestSplit = split;
      }

      leftAccum.bounds_min =
          elementwise_min(leftAccum.bounds_min, bins[split].bounds_min);
      leftAccum.bounds_max =
          elementwise_max(leftAccum.bounds_max, bins[split].bounds_max);
      leftAccum.count += bins[split].count;

      rightAccum.bounds_min =
          elementwise_max(rightAccum.bounds_min, bins[split].bounds_min);
      rightAccum.bounds_max =
          elementwise_min(rightAccum.bounds_max, bins[split].bounds_max);
      rightAccum.count -= bins[split].count;
    }
  }

  // Effectuer le split si un meilleur axe a été trouvé
  if (bestAxis != -1) {
    float splitPos;

    if (bestCost == 0) {
      splitPos = (getComponent(global_centroid_min, bestAxis) +
                  getComponent(global_centroid_max, bestAxis)) *
                 0.5f;
    } else {
      splitPos = getComponent(global_centroid_min, bestAxis) +
                 (getComponent(global_centroid_max, bestAxis) -
                  getComponent(global_centroid_min, bestAxis)) *
                     static_cast<float>(bestSplit) / numBins;
    }

    // Réorganisation des nœuds et triangles selon le split
    int mid = 0;
    for (int i = 0; i < numTriangles; ++i) {
      float3 centroid = (nodes[i].bounds_min + nodes[i].bounds_max) * 0.5f;
      float centroidValue = getComponent(centroid, bestAxis);
      if (centroidValue < splitPos) {
        swapNode(nodes[mid], nodes[i]);
        swapTriangle(triangles[mid], triangles[i]);
        mid++;
      }
    }

    // Ajustement des indices gauche et droit du nœud
    if (mid == 0)
      mid++;
    if (mid == numTriangles)
      mid--;

    node.left = mid;
    node.right = numTriangles + idx + 1;

    // Mise à jour des bornes du nœud interne après le split
    node.bounds_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    node.bounds_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < numTriangles; ++i) {
      node.bounds_min = elementwise_min(node.bounds_min, nodes[i].bounds_min);
      node.bounds_max = elementwise_max(node.bounds_max, nodes[i].bounds_max);
    }
  }
}

__global__ void checkBVHConsistency(BVHNodeSAH *nodes, int numNodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numNodes)
    return;

  BVHNodeSAH &node = nodes[idx];

  printf("[%i] left=%i right=%i\n", idx, nodes[idx].left, nodes[idx].right);

  if (idx < numNodes / 2) {
    if (node.triangleIndex == -1) {
      printf("Error: Node feuille %d with triangleIndex invalide\n", idx);
    }
  } else {
    if (node.triangleIndex != -1) {
      printf("Error: Node interne %d with triangleIndex valide\n", idx);
    }
  }
}

void buildBVHWithTriangleVersion3SAH(
    thrust::device_vector<F3Triangle> &triangles,
    thrust::device_vector<BVHNodeSAH> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);

  int blockSize = 512;
  int numBlocks = (numTriangles + blockSize - 1) / blockSize;

  initializeLeavesSAH<<<numBlocks, blockSize>>>(
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(nodes.data()), numTriangles);

  hipDeviceSynchronize();

  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    printf("Error in initializeLeavesSAH: %s\n", hipGetErrorString(error));
    return;
  }

  buildInternalNodesSAH<<<numBlocks, blockSize>>>(
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(nodes.data()), numTriangles);
  hipDeviceSynchronize();

  error = hipGetLastError();
  if (error != hipSuccess) {
    printf("Error in buildInternalNodesSAH: %s\n", hipGetErrorString(error));
    return;
  }

  printf("[TEST if OK]\n");
  int numNodes = nodes.size();
  checkBVHConsistency<<<(numNodes + 255) / 256, 256>>>(
      thrust::raw_pointer_cast(nodes.data()), numNodes);
  hipDeviceSynchronize();
}

__global__ void rayTracingImgKernelVersion3SAH(
    unsigned char *image, const int width, const int height,
    const Camera camera, const BVHNodeSAH *nodes, const F3Triangle *triangles,
    int *hitResults, float *distance, float3 *intersectionPoint, int *hitId) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height)
    return;

  const int x = idx % width;
  const int y = idx / width;

  F3Ray ray;
  camera.getRay(x, y, width, height, &ray.origin, &ray.direction);

  constexpr int MAX_STACK_SIZE = 64;
  int stack[MAX_STACK_SIZE];
  int stackPtr = 0;

  float closestT = INFINITY;
  int closestTriangleIndex = -1;
  int closestIntersectionId = -1;
  float3 closestIntersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);

  const float3 invDir = make_float3(
      1.0f / (fabsf(ray.direction.x) > 1e-8f ? ray.direction.x : 1e-8f),
      1.0f / (fabsf(ray.direction.y) > 1e-8f ? ray.direction.y : 1e-8f),
      1.0f / (fabsf(ray.direction.z) > 1e-8f ? ray.direction.z : 1e-8f));

  const int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};

  // Start from the root
  stack[stackPtr++] = 0;

  while (stackPtr > 0) {
    const int nodeIdx = stack[--stackPtr];
    const BVHNodeSAH &node = nodes[nodeIdx];

    if (nodeIdx > 0)
      printf("idx [%i] nodeIdx[%i] stask [%i] %i %i\n", idx, nodeIdx, stackPtr,
             node.left, node.right);

    // Optimized ray-box intersection
    float tmin = (node.bounds_min.x - ray.origin.x) * invDir.x;
    float tmax = (node.bounds_max.x - ray.origin.x) * invDir.x;
    if (dirIsNeg[0])
      SWAP(float, tmin, tmax);

    float tymin = (node.bounds_min.y - ray.origin.y) * invDir.y;
    float tymax = (node.bounds_max.y - ray.origin.y) * invDir.y;
    if (dirIsNeg[1])
      SWAP(float, tymin, tymax);

    tmin = max(tmin, tymin);
    tmax = min(tmax, tymax);

    float tzmin = (node.bounds_min.z - ray.origin.z) * invDir.z;
    float tzmax = (node.bounds_max.z - ray.origin.z) * invDir.z;
    if (dirIsNeg[2])
      SWAP(float, tzmin, tzmax);

    tmin = max(tmin, tzmin);
    tmax = min(tmax, tzmax);

    // Culling
    if (tmax < 0 || tmin > tmax || tmin > closestT)
      continue;

    // Leaf node intersection
    if (node.triangleIndex != -1) {
      float t;
      float3 intersectionPointT;
      if (rayTriangleIntersectVersion2(ray, triangles[node.triangleIndex], t,
                                       intersectionPointT)) {
        if (t < closestT) {
          closestT = t;
          closestTriangleIndex = node.triangleIndex;
          closestIntersectionPoint = intersectionPointT;
          closestIntersectionId = triangles[node.triangleIndex].id;
        }
      }
      continue; // Skip to the next iteration after processing a leaf node
    }

    // Internal node traversal
    if (stackPtr < MAX_STACK_SIZE - 2) { // Ensure space for both children
      if (node.right != -1)
        stack[stackPtr++] =
            node.right; // Push right child first for backtracking
      if (node.left != -1)
        stack[stackPtr++] = node.left; // Push left child next
    }
  }

  // Store results
  hitResults[idx] = closestTriangleIndex;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closestIntersectionId;

  // Image coloring based on hit results
  int ypx = (y * width + x) * 3; // Calculate pixel index
  if (closestTriangleIndex != -1) {
    float d1 = length(camera.position - camera.target);
    float d2 = closestT;
    float intensity = min(1.0f, max(0.0f, 1.0f - d2 / (1.5f * d1)));

    unsigned char value = static_cast<unsigned char>(255.0f * intensity);

    image[ypx] = value;     // Red channel
    image[ypx + 1] = value; // Green channel
    image[ypx + 2] = value; // Blue channel
  } else {
    image[ypx] = 0;     // Red channel
    image[ypx + 1] = 0; // Green channel
    image[ypx + 2] = 0; // Blue channel
  }
}

void buildPicturRayTracingPPM3_SAH(thrust::device_vector<F3Triangle> &triangles,
                                   thrust::device_vector<BVHNodeSAH> &nodes,
                                   Camera camera, int width, int height,
                                   const std::string &filename, bool isSave) {
  // Before using this function, the BVH must already be calculated and the
  // triangles must be in the device. Ray Tracing
  // const int threadsPerBlock = 256;
  const int threadsPerBlock = 256;
  const int numRays =
      width * height; // Total number of rays based on image dimensions
  int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

  //...
  thrust::device_vector<unsigned char> deviceImage(width * height * 3);
  thrust::device_vector<int> deviceHitResults(numRays);
  thrust::device_vector<float> deviceDistanceResults(numRays);
  thrust::device_vector<float3> deviceIntersectionPoint(numRays);
  thrust::device_vector<int> deviceIdResults(numRays);

  //...

  rayTracingImgKernelVersion3SAH<<<blocksPerGrid, threadsPerBlock>>>(
      thrust::raw_pointer_cast(deviceImage.data()), width, height, camera,
      thrust::raw_pointer_cast(nodes.data()),
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(deviceHitResults.data()),
      thrust::raw_pointer_cast(deviceDistanceResults.data()),
      thrust::raw_pointer_cast(deviceIntersectionPoint.data()),
      thrust::raw_pointer_cast(deviceIdResults.data()));

  hipDeviceSynchronize();

  //...
  if (isSave) {
    thrust::host_vector<unsigned char> hostImage = deviceImage;
    savePPM(filename, hostImage.data(), width, height);
  }

  // Memory cleaning
  deviceHitResults.clear();
  deviceDistanceResults.clear();
  deviceIntersectionPoint.clear();
  deviceIdResults.clear();
  deviceImage.clear();
}

void writeBVHNodesSAH1(const thrust::device_vector<BVHNodeSAH> &nodes) {

  std::vector<BVHNodeSAH> hostNodes(nodes.size());
  thrust::copy(nodes.begin(), nodes.end(), hostNodes.begin());

  std::cout << "BVH Nodes SAH1:" << std::endl;
  for (size_t i = 0; i < hostNodes.size(); ++i) {
    const BVHNodeSAH &node = hostNodes[i];
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "  leftChild: " << node.left << "\n";
    std::cout << "  rigthChild: " << node.right << "\n";
    std::cout << "  triangleIndex: " << node.triangleIndex << "\n";
    std::cout << "  Bounds:\n";
    std::cout << "    Min: ";
    std::cout << "(" << node.bounds_min.x << ", " << node.bounds_min.y << ", "
              << node.bounds_min.z << ")"
              << "\n";
    std::cout << "    Max: ";
    std::cout << "(" << node.bounds_max.x << ", " << node.bounds_max.y << ", "
              << node.bounds_max.z << ")"
              << "\n";
    std::cout << std::endl;
  }
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
// SAH Method 2

struct SAHBucket {
  AABB_SAH2 bounds;
  int count;
  __host__ __device__ __inline__ SAHBucket()
      : count(0), bounds() {} // Initialisation de bounds
};

struct HitInfo {
  float t;
  float3 normal;
  int triangleIndex;

  __device__ __inline__ HitInfo()
      : t(FLT_MAX), normal(make_float3(0.0f, 0.0f, 0.0f)), triangleIndex(-1) {}
};

__device__ __inline__ float calculateNodeCost(const AABB_SAH2 &nodeBounds) {
  float3 extent = nodeBounds.max - nodeBounds.min;
  return 2.0f *
         (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

__device__ __inline__ float maxComponent(float3 v) {
  return fmaxf(v.x, fmaxf(v.y, v.z));
}

__device__ __inline__ float
evaluateSAH(const AABB_SAH2 &nodeBounds, const AABB_SAH2 &leftBounds,
            const AABB_SAH2 &rightBounds, int leftCount, int rightCount,
            float traversalCost, float intersectionCost) {
  float pLeft = calculateNodeCost(leftBounds) / calculateNodeCost(nodeBounds);
  float pRight = calculateNodeCost(rightBounds) / calculateNodeCost(nodeBounds);
  return traversalCost +
         intersectionCost * (pLeft * leftCount + pRight * rightCount);
}

__global__ void buildBVHKernelSAH2(F3Triangle *triangles, BVHNodeSAH2 *nodes,
                                   int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Correction de l'indexation pour traiter tous les triangles
  if (idx >= numTriangles)
    return;

  BVHNodeSAH2 &node = nodes[idx];

  // Si le nœud est une feuille ou ne contient qu'un seul triangle, on
  // initialise ses bornes
  if (node.isLeaf || node.count <= 1) {
    node.bounds = AABB_SAH2(); // Réinitialiser les bornes
    for (int i = node.left; i < node.left + node.count; ++i) {
      F3Triangle &tri = triangles[i];
      node.bounds.expand(tri.v0);
      node.bounds.expand(tri.v1);
      node.bounds.expand(tri.v2);
    }
    return;
  }

  const int numBuckets = 12;
  SAHBucket buckets[numBuckets]; // Allocation automatique

  // Initialisation des buckets
  for (int i = 0; i < numBuckets; ++i) {
    buckets[i] = SAHBucket();
  }

  AABB_SAH2 nodeBounds;

  // Correction de l'accès aux triangles en utilisant le bon index
  for (int i = node.left; i < node.left + node.count; ++i) {
    F3Triangle &tri = triangles[i];
    float3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    nodeBounds.expand(centroid);

    printf("node.left=%i <%f,%f,%f> <%f,%f,%f>\ ", i, nodeBounds.min.x,
           nodeBounds.min.y, nodeBounds.min.z, nodeBounds.max.x,
           nodeBounds.max.y, nodeBounds.max.z);

    int bucketIndex =
        numBuckets * maxComponent((centroid - nodeBounds.min) /
                                  (nodeBounds.max - nodeBounds.min));
    bucketIndex = min(numBuckets - 1, bucketIndex);

    buckets[bucketIndex].bounds.expand(tri.v0);
    buckets[bucketIndex].bounds.expand(tri.v1);
    buckets[bucketIndex].bounds.expand(tri.v2);

    printf("bucket <%f,%f,%f> <%f,%f,%f>\n", buckets[bucketIndex].bounds.min.x,
           buckets[bucketIndex].bounds.min.y, buckets[bucketIndex].bounds.min.z,
           buckets[bucketIndex].bounds.max.x, buckets[bucketIndex].bounds.max.y,
           buckets[bucketIndex].bounds.max.z);

    buckets[bucketIndex].count++;
  }

  node.bounds = nodeBounds;

  // Évaluation des splits
  float minCost = FLT_MAX;
  int minCostSplitBucket = -1;

  const float traversalCost = 0.125f;
  const float intersectionCost = 1.0f;

  for (int i = 1; i < numBuckets; ++i) {
    AABB_SAH2 leftBounds, rightBounds;
    int leftCount = 0, rightCount = 0;

    for (int j = 0; j < i; ++j) {
      leftBounds.expand(buckets[j].bounds);
      leftCount += buckets[j].count;

      printf("node left <%f,%f,%f> <%f,%f,%f> \n", leftBounds.min.x,
             leftBounds.min.y, leftBounds.min.z, leftBounds.max.x,
             leftBounds.max.y, leftBounds.max.z);
    }

    for (int j = i; j < numBuckets; ++j) {
      rightBounds.expand(buckets[j].bounds);
      rightCount += buckets[j].count;
    }

    float cost = evaluateSAH(nodeBounds, leftBounds, rightBounds, leftCount,
                             rightCount, traversalCost, intersectionCost);

    if (cost < minCost) {
      minCost = cost;
      minCostSplitBucket = i;
    }

    printf("%i cost=%f minCost=%f\n", minCostSplitBucket, cost, minCost);
  }

  // Exécution du split si nécessaire
  if (minCostSplitBucket != -1) {
    int splitIndex = node.left;

    for (int i = 0; i < numBuckets; ++i) {
      for (int j = 0; j < buckets[i].count; ++j) {
        if (i < minCostSplitBucket) {
          if (splitIndex != node.left + j) {
            swapTriangle(triangles[splitIndex], triangles[node.left + j]);
          }
          splitIndex++;
        }
      }
    }

    // Création des nœuds enfants
    int leftChildIndex = numTriangles + idx * 2;
    int rightChildIndex = leftChildIndex + 1;

    nodes[leftChildIndex] = BVHNodeSAH2(node.left, splitIndex - node.left);
    nodes[rightChildIndex] =
        BVHNodeSAH2(splitIndex, node.count - (splitIndex - node.left));

    node.left = leftChildIndex;
    node.count = 2; // Indique que ce n'est plus une feuille
    node.isLeaf = false;

    node.bounds = nodes[leftChildIndex].bounds;
    node.bounds.expand(nodes[rightChildIndex].bounds);

    printf("node END <%f,%f,%f> <%f,%f,%f> \n", node.bounds.min.x,
           node.bounds.min.y, node.bounds.min.z, node.bounds.max.x,
           node.bounds.max.y, node.bounds.max.z);
  }
}

void buildBVHWithTriangleVersion3SAH2(
    thrust::device_vector<F3Triangle> &triangles,
    thrust::device_vector<BVHNodeSAH2> &nodes) {
  int numTriangles = triangles.size();
  nodes.resize(2 * numTriangles - 1);

  nodes[0] = BVHNodeSAH2(0, numTriangles);
  int blockSize = 256;
  int numBlocks = (numTriangles - 1 + blockSize - 1) / blockSize;
  // Build BVH
  for (int level = 0; level < log2(numTriangles); ++level) {
    buildBVHKernelSAH2<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(triangles.data()),
        thrust::raw_pointer_cast(nodes.data()), numTriangles);
    hipDeviceSynchronize();
  }

  printf("%i\n", nodes.size());
}

/*
__global__ void buildBVHKernelSAH2(F3Triangle* triangles, BVHNodeSAH2* nodes,
int numTriangles, int* nodesToProcess, int numNodesToProcess) { int idx =
blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numNodesToProcess) return;

    int nodeIndex = nodesToProcess[idx];
    BVHNodeSAH2& node = nodes[nodeIndex];

    // Si le nœud est une feuille ou ne contient qu'un seul triangle, ne rien
faire if (node.isLeaf || node.count <= 1) return;

    const int numBuckets = 12;
    SAHBucket buckets[numBuckets];

    // Initialisation des buckets
    for (int i = 0; i < numBuckets; ++i) {
        buckets[i] = SAHBucket();
    }

    AABB_SAH2 nodeBounds;

    // Calcul des bornes du nœud et distribution dans les buckets
    for (int i = node.left; i < node.left + node.count; ++i) {
        F3Triangle& tri = triangles[i];
        float3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
        nodeBounds.expand(centroid);

        int bucketIndex = min(numBuckets - 1, int(numBuckets *
maxComponent((centroid - nodeBounds.min) / (nodeBounds.max - nodeBounds.min))));

        buckets[bucketIndex].bounds.expand(tri.v0);
        buckets[bucketIndex].bounds.expand(tri.v1);
        buckets[bucketIndex].bounds.expand(tri.v2);
        buckets[bucketIndex].count++;
    }

    // Mise à jour des bornes du nœud
    node.bounds = nodeBounds;

    // Évaluation des splits
    float minCost = FLT_MAX;
    int minCostSplitBucket = -1;

    const float traversalCost = 0.125f;
    const float intersectionCost = 1.0f;

    for (int i = 1; i < numBuckets; ++i) {
        AABB_SAH2 leftBounds, rightBounds;
        int leftCount = 0, rightCount = 0;

        for (int j = 0; j < i; ++j) {
            leftBounds.expand(buckets[j].bounds);
            leftCount += buckets[j].count;
        }

        for (int j = i; j < numBuckets; ++j) {
            rightBounds.expand(buckets[j].bounds);
            rightCount += buckets[j].count;
        }

        float cost = evaluateSAH(nodeBounds, leftBounds, rightBounds, leftCount,
rightCount, traversalCost, intersectionCost);

        if (cost < minCost) {
            minCost = cost;
            minCostSplitBucket = i;
        }
    }

    // Exécution du split si nécessaire
    if (minCostSplitBucket != -1) {
        int splitIndex = node.left;

        for (int i = 0; i < numBuckets; ++i) {
            for (int j = 0; j < buckets[i].count; ++j) {
                if (i < minCostSplitBucket) {
                    if (splitIndex != node.left + j) {
                        swapTriangle(triangles[splitIndex], triangles[node.left
+ j]);
                    }
                    splitIndex++;
                }
            }
        }

        // Création des nœuds enfants
        int leftChildIndex = numTriangles + nodeIndex * 2;
        int rightChildIndex = leftChildIndex + 1;

        nodes[leftChildIndex] = BVHNodeSAH2(node.left, splitIndex - node.left);
        nodes[rightChildIndex] = BVHNodeSAH2(splitIndex, node.count -
(splitIndex - node.left));

        node.left = leftChildIndex;
        node.count = 2; // Indique que ce n'est plus une feuille
        node.isLeaf = false;

        // Ajouter les nouveaux nœuds à traiter (de manière atomique)
        int newIndex = atomicAdd(nodesToProcess + numNodesToProcess, 2);
        nodesToProcess[newIndex] = leftChildIndex;
        nodesToProcess[newIndex + 1] = rightChildIndex;
    }
}


void buildBVHWithTriangleVersion3SAH2(thrust::device_vector<F3Triangle>&
triangles, thrust::device_vector<BVHNodeSAH2>& nodes) { int numTriangles =
triangles.size(); nodes.resize(2 * numTriangles - 1);

    // Initialize all nodes
    thrust::fill(nodes.begin(), nodes.end(), BVHNodeSAH2());
    nodes[0] = BVHNodeSAH2(0, numTriangles);

    // Allocate and initialize nodesToProcess on the device
    thrust::device_vector<int> d_nodesToProcess(2 * numTriangles - 1);
    d_nodesToProcess[0] = 0;  // Start with the root node

    int* d_numNodesToProcess;
    hipMalloc(&d_numNodesToProcess, sizeof(int));
    int h_numNodesToProcess = numTriangles; // Host variable to store the number
of nodes to process hipMemcpy(d_numNodesToProcess, &h_numNodesToProcess,
sizeof(int), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numTriangles - 1 + blockSize) / blockSize;

    while (true) {
        // Copy the number of nodes to process from device to host
        int temp;
        hipMemcpy(&temp, d_numNodesToProcess, sizeof(int),
hipMemcpyDeviceToHost); if (temp == 0) break;

        buildBVHKernelSAH2<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(triangles.data()),
            thrust::raw_pointer_cast(nodes.data()),
            numTriangles,
            thrust::raw_pointer_cast(d_nodesToProcess.data()),
            d_numNodesToProcess
        );
        hipDeviceSynchronize();

    }

    hipFree(d_numNodesToProcess);

    printf("Total number of nodes: %i\n", nodes.size());
}
*/

__device__ __inline__ bool intersectAABB_SAH2(const F3Ray &ray,
                                              const AABB_SAH2 &aabb) {
  float3 invDir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y,
                              1.0f / ray.direction.z);
  float3 t0 = make_float3((aabb.min.x - ray.origin.x) * invDir.x,
                          (aabb.min.y - ray.origin.y) * invDir.y,
                          (aabb.min.z - ray.origin.z) * invDir.z);
  float3 t1 = make_float3((aabb.max.x - ray.origin.x) * invDir.x,
                          (aabb.max.y - ray.origin.y) * invDir.y,
                          (aabb.max.z - ray.origin.z) * invDir.z);
  float tmin =
      fmaxf(fmaxf(fminf(t0.x, t1.x), fminf(t0.y, t1.y)), fminf(t0.z, t1.z));
  float tmax =
      fminf(fminf(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y)), fmaxf(t0.z, t1.z));
  return tmax >= tmin && tmin < ray.tMax && tmax > ray.tMin;
}

__device__ __inline__ bool intersectTriangle(const F3Ray &ray,
                                             const F3Triangle &triangle,
                                             HitInfo &hitInfo) {
  const float EPSILON = 0.0000001f;
  float3 edge1, edge2, h, s, q;
  float a, f, u, v;

  edge1 = triangle.v1 - triangle.v0;
  edge2 = triangle.v2 - triangle.v0;
  h = cross(ray.direction, edge2);
  a = dot(edge1, h);

  if (a > -EPSILON && a < EPSILON)
    return false;

  f = 1.0f / a;
  s = ray.origin - triangle.v0;
  u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  q = cross(s, edge1);
  v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  float t = f * dot(edge2, q);

  if (t > EPSILON && t < hitInfo.t) {
    hitInfo.t = t;
    hitInfo.normal = normalize(cross(edge1, edge2));
    return true;
  }
  return false;
}

__device__ __inline__ bool traverseBVH(const F3Ray &ray,
                                       const BVHNodeSAH2 *nodes,
                                       const F3Triangle *triangles,
                                       HitInfo &hitInfo) {
  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0; // Start with root node

  bool hit = false;
  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    const BVHNodeSAH2 &node = nodes[nodeIdx];

    if (!intersectAABB_SAH2(ray, node.bounds))
      continue;

    if (node.isLeaf) {
      printf("node.count=%i\n", node.count);
      for (int i = 0; i < node.count; ++i) {
        const F3Triangle &tri = triangles[node.left + i];
        if (intersectTriangle(ray, tri, hitInfo)) {
          hit = true;
        }
      }
    } else {
      stack[stackPtr++] = node.left + 1; // Right child
      stack[stackPtr++] = node.left;     // Left child
    }
  }

  return hit;
}

__global__ void
rayTracingKernelSAH2(unsigned char *image, const int width, const int height,
                     const Camera camera, const BVHNodeSAH2 *nodes,
                     const F3Triangle *triangles, int *hitResults,
                     float *distance, float3 *intersectionPoint, int *hitId)

{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  F3Ray ray;
  camera.getRay(x, y, width, height, &ray.origin, &ray.direction);
  HitInfo hitInfo;
  hitInfo.t = INFINITY;
  int ypx = (y * width + x) * 3;

  bool hit = traverseBVH(ray, nodes, triangles, hitInfo);

  if (hit) {
    int value = 255;
    image[ypx] = value;
    image[ypx + 1] = value;
    image[ypx + 2] = value;
  } else {
    image[ypx] = 0;
    image[ypx + 1] = 0;
    image[ypx + 2] = 0;
  }
}

void buildPicturRayTracingPPM3_SAH2(
    thrust::device_vector<F3Triangle> &triangles,
    thrust::device_vector<BVHNodeSAH2> &nodes, Camera camera, int width,
    int height, const std::string &filename, bool isSave) {
  const int numRays =
      width * height; // Total number of rays based on image dimensions
  //...
  thrust::device_vector<unsigned char> deviceImage(width * height * 3);
  thrust::device_vector<int> deviceHitResults(numRays);
  thrust::device_vector<float> deviceDistanceResults(numRays);
  thrust::device_vector<float3> deviceIntersectionPoint(numRays);
  thrust::device_vector<int> deviceIdResults(numRays);

  //...
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  rayTracingKernelSAH2<<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(deviceImage.data()), width, height, camera,
      thrust::raw_pointer_cast(nodes.data()),
      thrust::raw_pointer_cast(triangles.data()),
      thrust::raw_pointer_cast(deviceHitResults.data()),
      thrust::raw_pointer_cast(deviceDistanceResults.data()),
      thrust::raw_pointer_cast(deviceIntersectionPoint.data()),
      thrust::raw_pointer_cast(deviceIdResults.data()));
  hipDeviceSynchronize();

  //...
  if (isSave) {
    thrust::host_vector<unsigned char> hostImage = deviceImage;
    savePPM(filename, hostImage.data(), width, height);
  }

  // Memory cleaning
  deviceHitResults.clear();
  deviceDistanceResults.clear();
  deviceIntersectionPoint.clear();
  deviceIdResults.clear();
  deviceImage.clear();
}

void writeBVHNodesSAH2(const thrust::device_vector<BVHNodeSAH2> &nodes) {
  std::vector<BVHNodeSAH2> hostNodes(nodes.size());
  thrust::copy(nodes.begin(), nodes.end(), hostNodes.begin());

  std::cout << "BVH Nodes SAH2:" << std::endl;
  for (size_t i = 0; i < hostNodes.size(); ++i) {
    const BVHNodeSAH2 &node = hostNodes[i];
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "  count: " << node.count << "\n";
    std::cout << "  left: " << node.left << "\n";
    std::cout << "  isLeaf: " << node.isLeaf << "\n";
    std::cout << "  Bounds:\n";
    std::cout << "    Min: ";
    std::cout << "(" << node.bounds.min.x << ", " << node.bounds.min.y << ", "
              << node.bounds.min.z << ")"
              << "\n";
    std::cout << "    Max: ";
    std::cout << "(" << node.bounds.max.x << ", " << node.bounds.max.y << ", "
              << node.bounds.max.z << ")"
              << "\n";
    std::cout << std::endl;
  }
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
