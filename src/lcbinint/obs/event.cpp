#include "event.hpp"
#include <stdexcept>

namespace lcbinint::obs {

Event::Event(std::string name, std::shared_ptr<SkyCoord> sky_coord)
    : name_(std::move(name)), sky_coord_(std::move(sky_coord))
{}

void Event::add(std::shared_ptr<LightCurveData> data)
{
    if (!data)
        throw std::invalid_argument("data must not be null");
    datasets_.push_back(std::move(data));
}

} // namespace lcbinint::obs
