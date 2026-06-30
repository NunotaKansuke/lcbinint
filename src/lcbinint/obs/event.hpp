#pragma once
#include "coordinates.hpp"
#include "light_curve_data.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lcbinint::obs {

class Event {
public:
    Event(
        std::string               name      = {},
        std::shared_ptr<SkyCoord> sky_coord = nullptr
    );

    void add(std::shared_ptr<LightCurveData> data);

    std::size_t           size()            const noexcept { return datasets_.size(); }
    const LightCurveData& at(std::size_t i) const { return *datasets_.at(i); }

    const std::string&               name()      const noexcept { return name_; }
    const std::shared_ptr<SkyCoord>& sky_coord() const noexcept { return sky_coord_; }
    // Convenience: return 0 if no sky_coord
    double ra()  const noexcept { return sky_coord_ ? sky_coord_->ra_deg()  : 0.0; }
    double dec() const noexcept { return sky_coord_ ? sky_coord_->dec_deg() : 0.0; }

    // Iteration support (dereferences shared_ptr transparently)
    class const_iterator {
    public:
        using It = std::vector<std::shared_ptr<LightCurveData>>::const_iterator;
        explicit const_iterator(It it) : it_(it) {}
        const LightCurveData& operator*()  const { return **it_; }
        const LightCurveData* operator->() const { return it_->get(); }
        const_iterator& operator++()       { ++it_; return *this; }
        const_iterator  operator++(int)    { auto t = *this; ++it_; return t; }
        bool operator==(const const_iterator& o) const { return it_ == o.it_; }
        bool operator!=(const const_iterator& o) const { return it_ != o.it_; }
    private:
        It it_;
    };

    const_iterator begin() const { return const_iterator(datasets_.begin()); }
    const_iterator end()   const { return const_iterator(datasets_.end()); }

private:
    std::string               name_;
    std::shared_ptr<SkyCoord> sky_coord_;
    std::vector<std::shared_ptr<LightCurveData>> datasets_;
};

} // namespace lcbinint::obs
