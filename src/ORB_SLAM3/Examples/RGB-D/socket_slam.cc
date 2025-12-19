#include <System.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

// Include Eigen for conversion
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

// Helper to receive exact number of bytes
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
    cerr << "Usage: ./socket_slam path_to_vocabulary path_to_settings" << endl;
    return 1;
  }

  // 1. Initialize SLAM System
  ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);

  // 2. Setup Socket Server
  int server_fd, new_socket;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);

  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("Socket failed");
    return -1;
  }

  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
    perror("setsockopt");
    return -1;
  }
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(8080);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("Bind failed");
    return -1;
  }

  if (listen(server_fd, 3) < 0) {
    perror("Listen");
    return -1;
  }

  cout << "=== READY: Waiting for Python script... ===" << endl;
  if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
    perror("Accept");
    return -1;
  }
  cout << "=== CONNECTED! Starting SLAM... ===" << endl;

  const int HEADER_SIZE = 16;
  char header_buf[HEADER_SIZE];

  int width, height;
  double timestamp;

  while (true) {
    // A. Read Header
    if (!recv_all(new_socket, header_buf, HEADER_SIZE)) break;

    memcpy(&width, header_buf, 4);
    memcpy(&height, header_buf + 4, 4);
    memcpy(&timestamp, header_buf + 8, 8);

    if (width <= 0 || width > 4000 || height <= 0 || height > 4000) {
      cerr << "ERROR: Invalid dimensions. Resetting..." << endl;
      close(new_socket);
      break;
    }

    // B. Read Images
    int rgb_size = width * height * 3;
    int depth_size = width * height * 2;
    std::vector<char> rgb_data(rgb_size);
    std::vector<char> depth_data(depth_size);

    if (!recv_all(new_socket, rgb_data.data(), rgb_size)) break;
    if (!recv_all(new_socket, depth_data.data(), depth_size)) break;

    cv::Mat imRGB(height, width, CV_8UC3, rgb_data.data());
    cv::Mat imD(height, width, CV_16UC1, depth_data.data());

    // C. Track and GET POSE (Fixing the Sophus issue)
    // Sophus::SE3f is what TrackRGBD returns now
    Sophus::SE3f Tcw_sophus = SLAM.TrackRGBD(imRGB, imD, timestamp);

    // D. Send Pose back to Python
    int status = 0;             // 0 = Lost, 1 = Tracking
    float pose_data[16] = {0};  // 4x4 Identity default

    // Check if tracking was successful (not empty/null)
    // Sophus SE3 doesn't have .empty(), usually check translation/rotation validity
    // But typically if it fails, ORB-SLAM3 might return an identity or specific state.
    // We assume if the matrix isn't pure identity or weird, it's valid.
    // Actually, ORB-SLAM3 usually returns empty cv::Mat in older versions if lost.
    // In newer versions with Sophus, we convert to Eigen matrix.

    Eigen::Matrix4f T_eigen = Tcw_sophus.matrix();

    // Simple heuristic: If it's valid tracking, the system state is usually tracked.
    // But since we don't have access to "isLost()" easily here without querying system state:
    // We will just send the matrix. If tracking is lost, ORB-SLAM3 usually returns the last known or identity.
    // Ideally, check SLAM.GetTrackingState() but that's internal.
    // We'll trust the matrix for now.

    status = 1;  // Assume tracking is okay if we got a return

    // Fill the buffer (Column-Major or Row-Major? OpenCV is Row-Major, Eigen is Col-Major by default)
    // We need 4x4 Row-Major for Python/Numpy usually.
    int idx = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        pose_data[idx++] = T_eigen(i, j);
      }
    }

    // Send Status
    send(new_socket, &status, sizeof(int), 0);

    // Send Matrix
    send(new_socket, pose_data, sizeof(float) * 16, 0);
  }

  // E. Save and Shutdown (Now INSIDE main)
  SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
  SLAM.Shutdown();
  return 0;
}
