#include <iostream>

#include <pcl/common/common_headers.h>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>



//#include <boost/thread/thread.hpp>
//#include <pcl/features/normal_3d.h>


//#include <pcl/console/parse.h>


int main(){
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud);

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();


while (!viewer->wasStopped ())
{
  viewer->spinOnce (100);
  boost::this_thread::sleep (boost::posix_time::microseconds (100000));
}
  }
