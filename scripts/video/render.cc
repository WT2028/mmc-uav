// Render recorded link poses with the Gazebo Harmonic rendering engine.
// This program performs no dynamics integration or controller execution.
#include <gz/rendering.hh>
#include <gz/rendering/MeshDescriptor.hh>
#include <gz/common/MeshManager.hh>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>

using namespace gz;

int main(int argc, char **argv) {
  if (argc != 4) { std::cerr << "Usage: render scene.txt poses.bin output_fd\n"; return 2; }
  std::ifstream config(argv[1]);
  unsigned width, height, count, frames;
  double floor;
  config >> width >> height >> count >> frames >> floor;
  if (!config || count != 7 || !frames) return 2;
  auto engine = rendering::engine("ogre2", {{"headless", "true"}});
  if (!engine) return 3;
  auto scene = engine->CreateScene("mmc_recorded_motion");
  scene->SetAmbientLight(.55, .55, .55);
  scene->SetBackgroundColor(.94, .95, .97);
  auto root = scene->RootVisual();
  auto light = scene->CreateDirectionalLight();
  light->SetDirection(-.4, .3, -1.);
  light->SetDiffuseColor(.85, .85, .85);
  light->SetSpecularColor(.2, .2, .2);
  root->AddChild(light);
  auto material = [&](double r, double g, double b) {
    auto m = scene->CreateMaterial();
    m->SetAmbient(r, g, b); m->SetDiffuse(r, g, b);
    m->SetSpecular(.12, .12, .12); m->SetShininess(30.);
    m->SetCastShadows(false);
    return m;
  };
  std::vector<rendering::VisualPtr> links;
  for (unsigned i = 0; i < count; ++i) {
    std::string path;
    double r, g, b;
    config >> path >> r >> g >> b;
    auto mesh = common::MeshManager::Instance()->Load(path);
    if (!mesh) { std::cerr << "Missing mesh: " << path << '\n'; return 4; }
    auto visual = scene->CreateVisual();
    visual->AddGeometry(scene->CreateMesh(rendering::MeshDescriptor(mesh)));
    visual->SetMaterial(material(r, g, b));
    root->AddChild(visual); links.push_back(visual);
  }
  auto floorMat = material(.90, .91, .93);
  auto ground = scene->CreateVisual();
  ground->AddGeometry(scene->CreateBox());
  ground->SetLocalScale(200., 200., .02);
  ground->SetLocalPosition(0, 0, floor - .015);
  ground->SetMaterial(floorMat); root->AddChild(ground);
  auto gridMat = material(.77, .79, .82);
  for (int i = -30; i <= 30; ++i) {
    for (int axis = 0; axis < 2; ++axis) {
      auto grid = scene->CreateVisual(); grid->AddGeometry(scene->CreateBox());
      grid->SetLocalScale(axis ? .003 : 30., axis ? 30. : .003, .001);
      grid->SetLocalPosition(axis ? i * .5 : 0., axis ? 0. : i * .5, floor);
      grid->SetMaterial(gridMat); root->AddChild(grid);
    }
  }
  auto camera = scene->CreateCamera();
  camera->SetImageWidth(width); camera->SetImageHeight(height);
  camera->SetImageFormat(rendering::PF_R8G8B8);
  camera->SetHFOV(1.0); camera->SetAspectRatio(double(width) / height);
  camera->SetAntiAliasing(4); camera->SetNearClipPlane(.02); camera->SetFarClipPlane(250.);
  root->AddChild(camera);
  auto image = camera->CreateImage();
  std::ifstream input(argv[2], std::ios::binary);
  int fd = std::stoi(argv[3]);
  // Each frame stores camera pose followed by seven visual poses (xyz, wxyz).
  std::vector<double> pose((count + 1) * 7);
  auto makePose = [](const double *p) {
    return math::Pose3d(math::Vector3d(p[0],p[1],p[2]), math::Quaterniond(p[3],p[4],p[5],p[6]));
  };
  for (unsigned f = 0; f < frames; ++f) {
    input.read(reinterpret_cast<char*>(pose.data()), pose.size() * sizeof(double));
    if (!input) return 5;
    camera->SetLocalPose(makePose(pose.data()));
    for (unsigned i = 0; i < count; ++i) links[i]->SetLocalPose(makePose(pose.data() + (i+1)*7));
    camera->Capture(image);
    const char *data = reinterpret_cast<const char*>(image.Data());
    size_t remaining = image.MemorySize();
    while (remaining) {
      auto n = write(fd, data, remaining);
      if (n <= 0) return 6;
      data += n; remaining -= n;
    }
    if (f % 200 == 0) std::cerr << "frame " << f << "/" << frames << '\n';
  }
  close(fd);
  engine->Destroy();
  return 0;
}
