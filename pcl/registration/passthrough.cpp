
#define nil Boost_nil
#define Nil Boost_Nil
#ifdef check
#undef check
#endif

#include <iostream>

#include <pcl/common/common_headers.h>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#undef Nil
#undef nil

//To view files
//     /Applications/pcl_viewer.app/Contents/MacOS/pcl_viewer table_filtered2.pcd

//#include <boost/thread/thread.hpp>
//#include <pcl/features/normal_3d.h>


//#include <pcl/console/parse.h>


int main(){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered1 (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZ>);


  pcl::io::loadPCDFile<pcl::PointXYZ> ("table_scene_lms400.pcd", *cloud);

  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (-1.5, 100.0);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_filtered1);
  
  //pcl::copyPointCloud(*cloud, *cloud_filtered1 );
  pcl::io::savePCDFileASCII ("table_filtered1.pcd", *cloud_filtered1);

  pass.setFilterLimits (-10.0, -1.4);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_filtered2);
   pcl::io::savePCDFileASCII ("table_filtered2.pcd", *cloud_filtered2);



  }