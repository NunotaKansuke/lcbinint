#include "event.hpp"
#include <stdexcept>

namespace lcbinint::obs {

Event::Event(std::string name, double ra, double dec, double t_ref)
    : name_(std::move(name)), ra_(ra), dec_(dec), t_ref_(t_ref)
{}

void Event::add(std::shared_ptr<LightCurveData> data)
{
    if (!data)
        throw std::invalid_argument("data must not be null");
    datasets_.push_back(std::move(data));
}

} // namespace lcbinint::obs
