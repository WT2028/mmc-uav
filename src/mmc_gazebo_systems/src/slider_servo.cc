// Force-actuated second-order moving-mass servo for Gazebo Harmonic.
// Only JointForceCmd is actuated: the physics engine retains the equal and
// opposite parent reaction, real inertia, and mechanical stops. No state reset
// or velocity command is used. All feedback is the actual joint state.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <gz/common/Console.hh>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/transport/Node.hh>

namespace mmc
{
class SliderServo final : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPreUpdate
{
  public: void Configure(const gz::sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &ecm, gz::sim::EventManager &) override
  {
    try
    {
      this->joint = gz::sim::Model(entity).JointByName(ecm,
          sdf->Get<std::string>("joint_name"));
      if (this->joint == gz::sim::kNullEntity)
        throw std::runtime_error("joint not found");
      const auto read = [&](const std::string &key, double fallback)
      {
        const double value = sdf->Get<double>(key, fallback).first;
        if (!std::isfinite(value) || value <= 0)
          throw std::runtime_error("invalid " + key);
        return value;
      };
      this->wn = read("wn", 20.0);
      this->zeta = read("zeta", 1.2);
      this->mass = read("moving_mass", 0.05);
      this->velocityLimit = read("velocity_limit", 0.25);
      this->commandLimit = read("command_limit", 0.135);
      this->mechanicalLimit = read("mechanical_limit", 0.15);
      this->effortLimit = read("effort_limit", 2.0);
      this->observerTau = read("load_observer_tau", 0.02);
      if (this->commandLimit >= this->mechanicalLimit)
        throw std::runtime_error("command limit must be inside mechanical stops");
      this->topic = sdf->Get<std::string>("topic");
      this->node.Subscribe(this->topic, &SliderServo::OnCommand, this);
      this->telemetry = this->node.Advertise<gz::msgs::Double_V>(
          this->topic + "/servo_state");
      if (!ecm.Component<gz::sim::components::JointPosition>(this->joint))
        ecm.CreateComponent(this->joint, gz::sim::components::JointPosition());
      if (!ecm.Component<gz::sim::components::JointVelocity>(this->joint))
        ecm.CreateComponent(this->joint, gz::sim::components::JointVelocity());
      this->configured = true;
    }
    catch (const std::exception &err)
    {
      gzerr << "SliderServo configuration failed: " << err.what() << '\n';
    }
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &info,
      gz::sim::EntityComponentManager &ecm) override
  {
    if (!this->configured || info.paused) return;
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0)
    {
      this->previousValid = false;
      this->loadAcceleration = 0;
      this->request.store(0);
      return;
    }
    const auto *position = ecm.Component<gz::sim::components::JointPosition>(this->joint);
    const auto *velocity = ecm.Component<gz::sim::components::JointVelocity>(this->joint);
    if (!position || !velocity || position->Data().empty() || velocity->Data().empty()) return;
    const double q = position->Data()[0], v = velocity->Data()[0];
    if (!std::isfinite(q) || !std::isfinite(v)) return;
    // Estimate the load/recoil acceleration left by the force applied in the
    // previous physics step. This measured-state observer compensates gravity
    // and free-body recoil without removing their physical reaction. Tau is
    // fixed in SI time, independent of simulation real-time factor.
    if (this->previousValid)
    {
      const double residual = (v - this->previousVelocity) / dt -
          this->previousForce / this->mass;
      const double alpha = dt / (this->observerTau + dt);
      this->loadAcceleration += alpha * (residual - this->loadAcceleration);
      this->loadAcceleration = std::clamp(this->loadAcceleration,
          -this->effortLimit / this->mass, this->effortLimit / this->mass);
    }
    const double requested = this->request.load();
    const double target = std::clamp(requested, -this->commandLimit, this->commandLimit);
    double acceleration = this->wn * this->wn * (target - q) -
        2 * this->zeta * this->wn * v;
    // Limit the physical next-step speed; a setpoint step is not a rail speed.
    // Braking is still delivered as a bounded joint force, never by overwriting
    // the velocity. Account for one-step stopping distance near rail limits.
    const double maxAcceleration = this->effortLimit / this->mass;
    const double positiveSpeed = std::min(this->velocityLimit, std::sqrt(
        2 * maxAcceleration * std::max(0.0, this->commandLimit - q)));
    const double negativeSpeed = std::min(this->velocityLimit, std::sqrt(
        2 * maxAcceleration * std::max(0.0, this->commandLimit + q)));
    const double nextVelocity = std::clamp(v + acceleration * dt,
        -negativeSpeed, positiveSpeed);
    acceleration = (nextVelocity - v) / dt;
    const double force = std::clamp(this->mass *
        (acceleration - this->loadAcceleration), -this->effortLimit, this->effortLimit);
    ecm.SetComponentData<gz::sim::components::JointForceCmd>(this->joint, {force});
    this->previousVelocity = v;
    this->previousForce = force;
    this->previousValid = true;
    gz::msgs::Double_V message;
    const double simTime = std::chrono::duration<double>(info.simTime).count();
    for (double value : {simTime, q, v, requested, target, force, acceleration, this->loadAcceleration})
      message.add_data(value);
    this->telemetry.Publish(message);
  }

  private: void OnCommand(const gz::msgs::Double &msg)
  {
    if (std::isfinite(msg.data())) this->request.store(msg.data());
  }
  private: gz::sim::Entity joint{gz::sim::kNullEntity};
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher telemetry;
  private: std::atomic<double> request{0};
  private: std::string topic;
  private: bool configured{false}, previousValid{false};
  private: double wn{20}, zeta{1.2}, mass{0.05}, velocityLimit{0.25};
  private: double commandLimit{0.135}, mechanicalLimit{0.15}, effortLimit{2};
  private: double observerTau{0.02}, loadAcceleration{0}, previousVelocity{0}, previousForce{0};
};
}
GZ_ADD_PLUGIN(mmc::SliderServo, gz::sim::System,
    mmc::SliderServo::ISystemConfigure, mmc::SliderServo::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(mmc::SliderServo, "mmc::SliderServo")
