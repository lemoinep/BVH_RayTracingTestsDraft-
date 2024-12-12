
#pragma clang diagnostic push

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <optional>
#include <random>
#include <cfloat>
#include <stdexcept>



 #include "bvhgpu.hpp"


 //Links Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>



bool loadOBJTriangle(const std::string& filename, std::vector<F3Triangle>& triangles,const int& id) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::vector<float3> vertices;
    std::string line;
    bool isView=false;
    while (std::getline(file, line)) {
        if (line[0] == 'v') {
            float x, y, z;
            sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
            vertices.push_back(make_float3(x, y, z));
            if (isView) std::cout <<"v=<" << x <<","<<y<<","<<z<< ">\n";
        } else if (line[0] == 'f') {
            unsigned int i1, i2, i3;
            sscanf(line.c_str(), "f %u %u %u", &i1, &i2, &i3);
            if (isView) std::cout <<"f=<" << i1 <<","<<i2<<","<<i3<< ">\n";
            triangles.push_back({vertices[i1-1], vertices[i2-1], vertices[i3-1]});
            triangles.back().id=id;
        }
    }
    return true;
}

 void testRayTracingPicture005(const std::string& filename,const std::string& filenamePPM)
{

    std::chrono::steady_clock::time_point t_begin_0,t_begin_1,t_end_1;
    std::chrono::steady_clock::time_point t_end_0;
    long int t_laps;
    
    const int width = 1024;
    const int height = 768;

    //const int width = 200;
    //const int height = 200;

    //const int width = 10;
    //const int height = 10;

    //const int width = 800;
    //const int height = 600;

    Camera camera;
    camera.position = {-5.0f, 5.0f, 5.0f};
    camera.target   = {0.75f, 0.75f, 0.75f};

    camera.position = {11.0f, 2.0f, 2.5f};
    camera.target   = {0.0f, 0.0f, 0.0f};

    camera.up       = {0.0f, 1.0f, 0.0f};
    camera.fov      = M_PI / 4;
    camera.aspect   = static_cast<float>(width) / static_cast<float>(height);

    int idObject=123321;
    std::vector<F3Triangle> hostTriangles;
    loadOBJTriangle(filename, hostTriangles,idObject);
    thrust::device_vector<F3Triangle> deviceTriangles = hostTriangles;

    
    //BVH
    t_begin_0 = std::chrono::steady_clock::now();
    thrust::device_vector<BVHNode> deviceNodes;
    buildBVHWithTriangleVersion1(deviceTriangles, deviceNodes);
    t_end_0 = std::chrono::steady_clock::now();

    //Ray Tracing
    const int threadsPerBlock = 256;
    const int numRays = width * height; // Total number of rays based on image dimensions
    int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;
    
    //...
    thrust::device_vector<unsigned char> deviceImage(width * height * 3);
    thrust::device_vector<int>           deviceHitResults(numRays);
    thrust::device_vector<float>         deviceDistanceResults(numRays);
    thrust::device_vector<float3>        deviceIntersectionPoint(numRays);
    thrust::device_vector<int>           deviceIdResults(numRays);

    t_begin_1 = std::chrono::steady_clock::now();
    //...
    rayTracingImgKernel<<<blocksPerGrid, threadsPerBlock>>>(
    //rayTracingImgKernelVersion2<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(deviceImage.data()),
        width,
        height,
        camera,
        thrust::raw_pointer_cast(deviceNodes.data()),
        thrust::raw_pointer_cast(deviceTriangles.data()),
        thrust::raw_pointer_cast(deviceHitResults.data()),
        thrust::raw_pointer_cast(deviceDistanceResults.data()),
        thrust::raw_pointer_cast(deviceIntersectionPoint.data()),
        thrust::raw_pointer_cast(deviceIdResults.data())
    );

    t_end_1 = std::chrono::steady_clock::now();    
    //...
    thrust::host_vector<unsigned char> hostImage = deviceImage;
    savePPM(filenamePPM, hostImage.data(), width, height);
    //convertPPMtoBMP("output_image.ppm","output_image.bmp");


    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0).count();
    std::cout << "[INFO]: Elapsed microseconds inside BVH : "<<t_laps<< " us\n";

    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1).count();
    std::cout << "[INFO]: Elapsed microseconds inside Ray : "<<t_laps<< " us\n";

    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_0).count();
    std::cout << "[INFO]: Elapsed microseconds inside Application: "<<t_laps<< " us\n";

    // Memory cleaning
    deviceNodes.clear();        
    deviceTriangles.clear();    
    deviceHitResults.clear(); 
    deviceDistanceResults.clear(); 
    deviceIntersectionPoint.clear(); 
    deviceIdResults.clear(); 
	deviceImage.clear(); 
    hostTriangles.clear(); 
}


void testRayTracingPicture006(const std::string& filename,const std::string& filenamePPM,int numVersion)
{
    
    std::chrono::steady_clock::time_point t_begin_0,t_begin_1;
    std::chrono::steady_clock::time_point t_end_0,t_end_1;
    long int t_laps;
    
    const int width = 1024;
    const int height = 768;

    Camera camera;
    camera.position = {-5.0f, 5.0f, 5.0f};
    camera.target   = {0.75f, 0.75f, 0.75f};

    camera.position = {-0.0f, 0.75f, 0.75f};
    camera.target   = { 1.5f, 0.75f, 0.75f};

    //camera.position = {-0.9f, 0.0f, 0.0f};
    //camera.position = {-10.0f, 0.0f, 0.0f};
    //camera.target   = { 1.5f, 0.0f, 0.0f};

    camera.position = {10.5f, 1.9f, 2.4f};
    camera.target   = {0.0f, 0.0f, 0.0f};

    camera.up       = {0.0f, 1.0f, 0.0f};
    camera.fov      = M_PI / 4;
    camera.aspect   = static_cast<float>(width) / static_cast<float>(height);

    

    int idObject=123321;
    std::vector<F3Triangle> hostTriangles;
    loadOBJTriangle(filename, hostTriangles,idObject);
    thrust::device_vector<F3Triangle> deviceTriangles = hostTriangles;


    //BVH
    t_begin_0 = std::chrono::steady_clock::now();  
    thrust::device_vector<BVHNode> deviceNodes;
    if (numVersion == 1) buildBVHWithTriangleVersion1(deviceTriangles, deviceNodes);
    if (numVersion == 2) buildBVHWithTriangleVersion2(deviceTriangles, deviceNodes);
    if (numVersion == 3) buildBVHWithTriangleVersion3(deviceTriangles, deviceNodes);
    if (numVersion == 4) buildBVHWithTriangleVersion4(deviceTriangles, deviceNodes);
    if (numVersion == 5) buildBVHWithTriangleVersion5(deviceTriangles, deviceNodes);
    thrust::device_vector<BVHNodeAABB> deviceNodes_AABB;
    if (numVersion == 6) buildBVH_AABB(deviceTriangles, deviceNodes_AABB);

    thrust::device_vector<BVHNodeSAH> deviceNodes_SAH;
    if (numVersion == 7) buildBVHWithTriangleVersion3SAH(deviceTriangles, deviceNodes_SAH); //ERROR

    thrust::device_vector<BVHNodeSAH2> deviceNodes_SAH2;
    if (numVersion == 8) buildBVHWithTriangleVersion3SAH2(deviceTriangles, deviceNodes_SAH2); //ERROR


    t_end_0 = std::chrono::steady_clock::now();   

    //if (numVersion == 3) writeBVHNodes(deviceNodes);

    //if (numVersion == 7) writeBVHNodesSAH1(deviceNodes_SAH);
    if (numVersion == 8) writeBVHNodesSAH2(deviceNodes_SAH2);

    bool isSave=true; //isSave=false;
    t_begin_1 = std::chrono::steady_clock::now();
    if (numVersion  < 6) buildPicturRayTracingPPM(deviceTriangles, deviceNodes, camera, width, height, filenamePPM,isSave);
    if (numVersion == 6) buildPicturRayTracingPPM_AABB(deviceTriangles, deviceNodes_AABB, camera, width, height, filenamePPM,isSave);

    // SAH
    if (numVersion == 7) buildPicturRayTracingPPM3_SAH(deviceTriangles, deviceNodes_SAH, camera, width, height, filenamePPM,isSave); //ERROR
    if (numVersion == 8) buildPicturRayTracingPPM3_SAH2(deviceTriangles, deviceNodes_SAH2, camera, width, height, filenamePPM,isSave); //ERROR
    t_end_1 = std::chrono::steady_clock::now();    


    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0).count();
    std::cout << "[INFO]: Elapsed microseconds inside BVH : "<<t_laps<< " us\n";

    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1).count();
    std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : "<<t_laps<< " us\n";
}





Eigen::Vector3d sphericalToCartesian(double r, double theta, double alpha) {
    Eigen::Vector3d position;
    position[0] = r * sin(theta) * cos(alpha);
    position[1] = r * sin(theta) * sin(alpha);
    position[2] = r * cos(theta);
    return position;
}

void buildExplorationLight()
{
    Eigen::Vector3d ray_origin={0.0f,0.0f,0.0f};
    double thetaStart = 0.0f;        
    double thetaEnd = M_PI;       
    double alphaStart = 0.0f;        
    double alphaEnd = 2.0f * M_PI;   
    double thetaStep = M_PI / 18.0f; 
    double alphaStep = M_PI / 18.0f; 

    for (double theta = thetaStart; theta <= thetaEnd; theta += thetaStep) {
        for (double alpha = alphaStart; alpha <= alphaEnd; alpha += alphaStep) {
            Eigen::Vector3d ray_direction = sphericalToCartesian(1.0f, theta, alpha);
            std::cout << "Origine: <" << ray_origin[0] << ", " << ray_origin[1] << ", " << ray_origin[2] << ">" <<  " ";
            std::cout << "Direction: <" << ray_direction[0] << ", " << ray_direction[1] << ", " << ray_direction[2] << ">" << std::endl;
        }
    }
}



int main(){

    std::string filename;
    filename = "Triangle2Cube.obj";
    filename = "Test.obj";
    testRayTracingPicture005(filename,"output_image.ppm");
    std::cout << "[INFO]: WELL DONE"<<"\n";

    std::cout << "[INFO]: Version 1"<<"\n";
    testRayTracingPicture006(filename,"output_image1.ppm",1);
    std::cout << "[INFO]: WELL DONE"<<"\n";

    std::cout << "[INFO]: Version 2"<<"\n";
    testRayTracingPicture006(filename,"output_image2.ppm",2);
    std::cout << "[INFO]: WELL DONE"<<"\n";

    std::cout << "[INFO]: Version 3"<<"\n";
    testRayTracingPicture006(filename,"output_image3.ppm",3);
    std::cout << "[INFO]: WELL DONE"<<"\n";
 
    std::cout << "[INFO]: Version 4"<<"\n";
    testRayTracingPicture006(filename,"output_image4.ppm",4);

    std::cout << "[INFO]: Version 5"<<"\n";
    testRayTracingPicture006(filename,"output_image5.ppm",5);

    std::cout << "[INFO]: Version 6"<<"\n";
    testRayTracingPicture006(filename,"output_image6.ppm",6);

/*
    std::cout << "\n";
    std::cout << "[INFO]: Version 7"<<"\n";
    testRayTracingPicture006(filename,"output_image7.ppm",7); //Error


    std::cout << "\n";
    std::cout << "[INFO]: Version 8"<<"\n";
    testRayTracingPicture006(filename,"output_image8.ppm",8); //Error
*/
    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";
    return 0;
}



#pragma clang diagnostic pop



