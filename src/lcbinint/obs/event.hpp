#pragma once
#include "light_curve_data.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lcbinint::obs {

class Event {
public:
    Event(
        std::string name  = {},
        double      ra    = 0.0,
        double      dec   = 0.0,
        double      t_ref = 0.0
    );

    void add(std::shared_ptr<LightCurveData> data);

    std::size_t           size()            const noexcept { return datasets_.size(); }
    const LightCurveData& at(std::size_t i) const { return *datasets_.at(i); }

    const std::string& name()  const noexcept { return name_; }
    double             ra()    const noexcept { return ra_; }
    double             dec()   const noexcept { return dec_; }
    double             t_ref() const noexcept { return t_ref_; }

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
    std::string name_;
    double ra_;
    double dec_;
    double t_ref_;
    std::vector<std::shared_ptr<LightCurveData>> datasets_;
};

} // namespace lcbinint::obs
