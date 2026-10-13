// Test-only initializer and prescribed body-origin wrench for isolated multibody tests.
// No production controller/model changes. Velocities/joint positions are set
// once to establish an initial condition; every later step is physical.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/JointPositionReset.hh>
#include <gz/sim/components/JointVelocityReset.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace mmc::analysis {
class JointIsolationWrench final : public gz::sim::System,
  public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemPostUpdate {
 public:
  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &) override {
    model = gz::sim::Model(_entity);
    base = gz::sim::Link(model.LinkByName(_ecm, "base_link"));
    base.EnableVelocityChecks(_ecm);
    base.EnableAccelerationChecks(_ecm);
    thrust0 = _sdf->Get<double>("thrust0");
    rise = _sdf->Get<double>("thrust_relative_rise");
    duration = _sdf->Get<double>("transition_time");
    omega0 = _sdf->Get<gz::math::Vector3d>("initial_omega_body");
    drag = _sdf->Get<bool>("enable_nominal_aero");
    auto copy = _sdf->Clone();
    if (copy->HasElement("initial_joint")) {
      auto element = copy->GetElement("initial_joint");
      while (element) {
        jointPositions.emplace_back(element->Get<std::string>("name"),
                                    element->Get<double>("position"));
        element = element->GetNextElement("initial_joint");
      }
    }
    if (copy->HasElement("initial_link")) {
      auto element = copy->GetElement("initial_link");
      while (element) {
        initialLinks.emplace_back(element->Get<std::string>("name"),
                                  element->Get<gz::math::Vector3d>("position"));
        element = element->GetNextElement("initial_link");
      }
    }
    publisher = node.Advertise<gz::msgs::Double_V>(_sdf->Get<std::string>("topic"));
    bodyPublisher = node.Advertise<gz::msgs::Double_V>(_sdf->Get<std::string>("topic")+"/body_state");
    for (const auto &[name, position] : initialLinks) {
      gz::sim::Link link(model.LinkByName(_ecm,name));
      link.EnableVelocityChecks(_ecm);
      link.EnableAccelerationChecks(_ecm);
      linkPublishers[name] = node.Advertise<gz::msgs::Double_V>(
          _sdf->Get<std::string>("topic")+"/link/"+name);
    }
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override {
    if (_info.paused) return;
    const double now = std::chrono::duration<double>(_info.simTime).count();
    if (!initialized) {
      for (const auto &[name, position] : jointPositions) {
        const auto entity = model.JointByName(_ecm, name);
        _ecm.CreateComponent(entity, gz::sim::components::JointPositionReset({position}));
        _ecm.CreateComponent(entity, gz::sim::components::JointVelocityReset({0.0}));
        // Establish initial servo feedback before its first PreUpdate. This
        // only accompanies the marked physical joint-position reset.
        _ecm.SetComponentData<gz::sim::components::JointPosition>(entity, {position});
        _ecm.SetComponentData<gz::sim::components::JointVelocity>(entity, {0.0});
      }
      // Gazebo commands the common free group's root-link velocity, even
      // when the command is attached to a child. Set that group only once;
      // setting all seven links would successively overwrite its velocity.
      base.SetAngularVelocity(_ecm, omega0);
      base.SetLinearVelocity(_ecm, gz::math::Vector3d::Zero);
      initialized = true;
      start = now + std::chrono::duration<double>(_info.dt).count();
      // This marker explicitly identifies the one initial-condition reset.
      gz::msgs::Double_V message;
      for (double value : {now, -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.})
        message.add_data(value);
      publisher.Publish(message);
      return;
    }
    if (!velocityReleased) {
      // Link velocity commands remain ECS components and otherwise suppress
      // subsequent physical wrenches. Release every one after initialization.
      for (const auto &[name, position] : initialLinks) {
        const auto entity = model.LinkByName(_ecm, name);
        _ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(entity);
        _ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(entity);
      }
      velocityReleased = true;
    }
    auto pose = base.WorldPose(_ecm);
    if (!pose) return;
    const double t = std::max(0., now-start);
    const double x = std::clamp(t/duration, 0., 1.);
    const double q5 = x*x*x*(10.+x*(-15.+6.*x));
    const double thrust = thrust0*(1.+rise*q5);
    gz::math::Vector3d forceBody(0., 0., thrust), momentBody(0., 0., 0.);
    if (drag) {
      auto linear = base.WorldLinearVelocity(_ecm);
      auto angular = base.WorldAngularVelocity(_ecm);
      if (linear && angular) {
        auto velocity = pose->Rot().RotateVectorReverse(*linear);
        auto omega = pose->Rot().RotateVectorReverse(*angular);
        for (int axis = 0; axis < 3; ++axis) {
          const double area = axis == 2 ? .16 : .072;
          const double cd = axis == 2 ? 1.3 : 1.2;
          forceBody[axis] -= .5*1.225*cd*area*std::abs(velocity[axis])*velocity[axis];
          momentBody[axis] -= (axis == 2 ? .02 : .04)*omega[axis];
        }
      }
    }
    const auto forceWorld = pose->Rot().RotateVector(forceBody);
    const auto momentWorld = pose->Rot().RotateVector(momentBody);
    // Link.hh specifies AddWorldWrench at the link origin. Thrust and drag
    // share the same declared body reference used by the analytical model.
    base.AddWorldWrench(_ecm, forceWorld, momentWorld);
    gz::msgs::Double_V message;
    for (double value : {now, t, thrust, forceBody.X(), forceBody.Y(), forceBody.Z(),
                         momentBody.X(), momentBody.Y(), momentBody.Z(), forceWorld.X(),
                         forceWorld.Y(), forceWorld.Z(), momentWorld.X(), momentWorld.Y(), momentWorld.Z()})
      message.add_data(value);
    publisher.Publish(message);
  }
  void PostUpdate(const gz::sim::UpdateInfo &_info,
                  const gz::sim::EntityComponentManager &_ecm) override {
    if (_info.paused || !initialized) return;
    PublishState(_info,_ecm,base,bodyPublisher);
    for (auto &[name,pub] : linkPublishers)
      PublishState(_info,_ecm,gz::sim::Link(model.LinkByName(_ecm,name)),pub);
  }
  void PublishState(const gz::sim::UpdateInfo &_info,
                    const gz::sim::EntityComponentManager &_ecm,
                    const gz::sim::Link &link,
                    gz::transport::Node::Publisher &pub) {
    const auto pose = link.WorldPose(_ecm);
    const auto linear = link.WorldLinearVelocity(_ecm);
    const auto angular = link.WorldAngularVelocity(_ecm);
    const auto acceleration = link.WorldLinearAcceleration(_ecm);
    const auto alpha = link.WorldAngularAcceleration(_ecm);
    if (!pose || !linear || !angular || !acceleration || !alpha) return;
    const auto omega = pose->Rot().RotateVectorReverse(*angular);
    const auto alphaBody = pose->Rot().RotateVectorReverse(*alpha);
    const double now = std::chrono::duration<double>(_info.simTime).count();
    gz::msgs::Double_V message;
    for (double value : {now, pose->Pos().X(), pose->Pos().Y(), pose->Pos().Z(),
                         pose->Rot().X(), pose->Rot().Y(), pose->Rot().Z(), pose->Rot().W(),
                         linear->X(), linear->Y(), linear->Z(), omega.X(), omega.Y(), omega.Z(),
                         acceleration->X(), acceleration->Y(), acceleration->Z(),
                         alphaBody.X(), alphaBody.Y(), alphaBody.Z()})
      message.add_data(value);
    pub.Publish(message);
  }
 private:
  gz::sim::Model model;
  gz::sim::Link base;
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  gz::transport::Node::Publisher bodyPublisher;
  std::map<std::string,gz::transport::Node::Publisher> linkPublishers;
  std::vector<std::pair<std::string, double>> jointPositions;
  std::vector<std::pair<std::string, gz::math::Vector3d>> initialLinks;
  gz::math::Vector3d omega0;
  double thrust0{}, rise{}, duration{}, start{};
  bool initialized{false}, velocityReleased{false}, drag{false};
};
}
GZ_ADD_PLUGIN(mmc::analysis::JointIsolationWrench, gz::sim::System,
              gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate,
              gz::sim::ISystemPostUpdate)
