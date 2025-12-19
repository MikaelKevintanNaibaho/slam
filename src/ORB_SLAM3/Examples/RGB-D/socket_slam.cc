#include <System.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

bool recv_all(int sock, char * buffer, int size)
{
  int received = 0;
  while (received < size) {
    int r = recv(sock, buffer + received, size - received, 0);
    if (r <= 0) return false;
    received += r;
  }
  return true;
}

int main(int argc, char ** argv)
{
  if (argc != 3) {
    cerr << "Usage: ./socket_slam vocab settings" << endl;
    return 1;
  }

  ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);

  int server_fd, new_socket;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);

  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(8080);
  bind(server_fd, (struct sockaddr *)&address, sizeof(address));
  listen(server_fd, 3);

  cout << "=== READY: Waiting for Python Fusion Script ===" << endl;
  new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);

  int width, height;
  double timestamp;
  char header_buf[16];

  while (true) {
    if (!recv_all(new_socket, header_buf, 16)) break;
    memcpy(&width, header_buf, 4);
    memcpy(&height, header_buf + 4, 4);
    memcpy(&timestamp, header_buf + 8, 8);

    vector<char> rgb_data(width * height * 3);
    vector<char> depth_data(width * height * 2);
    if (!recv_all(new_socket, rgb_data.data(), rgb_data.size())) break;
    if (!recv_all(new_socket, depth_data.data(), depth_data.size())) break;

    cv::Mat imRGB(height, width, CV_8UC3, rgb_data.data());
    cv::Mat imD(height, width, CV_16UC1, depth_data.data());

    // 1. Track Frame
    Sophus::SE3f Tcw_sophus = SLAM.TrackRGBD(imRGB, imD, timestamp);
    Eigen::Matrix4f Tcw = Tcw_sophus.matrix();

    // 2. Extract "Skeleton" (Sparse True Depth)
    std::vector<ORB_SLAM3::MapPoint *> pMPs = SLAM.GetTrackedMapPoints();
    std::vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();

    std::vector<float> skeleton_data;
    int nPoints = 0;

    for (size_t i = 0; i < pMPs.size(); i++) {
      if (pMPs[i] && !pMPs[i]->isBad()) {
        Eigen::Vector3f Pw = pMPs[i]->GetWorldPos();
        // Transform point to camera frame to get depth (Z)
        Eigen::Vector3f Pc = Tcw.block<3, 3>(0, 0) * Pw + Tcw.block<3, 1>(0, 3);

        skeleton_data.push_back(vKeys[i].pt.x);  // u (pixel x)
        skeleton_data.push_back(vKeys[i].pt.y);  // v (pixel y)
        skeleton_data.push_back(Pc(2));          // z (meters)
        nPoints++;
      }
      if (nPoints >= 300) break;
    }

    // 3. Send Pose and Skeleton
    int status = (nPoints > 10) ? 1 : 0;
    float pose_arr[16];
    memcpy(pose_arr, Tcw.data(), 64);  // Column-major Eigen data

    send(new_socket, &status, 4, 0);
    send(new_socket, pose_arr, 64, 0);
    send(new_socket, &nPoints, 4, 0);
    if (nPoints > 0) send(new_socket, skeleton_data.data(), nPoints * 12, 0);
  }
  SLAM.Shutdown();
  return 0;
}
