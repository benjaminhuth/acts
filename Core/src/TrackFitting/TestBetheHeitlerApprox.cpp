#include <iostream>
#include <ranges>
#include <algorithm>
#include <array>
#include <random>

#include <matplotlibcpp.h>
#include <fmt/format.h>

#include "Acts/TrackFitting/BetheHeitlerApprox.hpp"

// #include "common.hpp"
// #include "minimize_annealing.hpp"
// #include "minimize_fruewirth.hpp"

namespace plt = matplotlibcpp;

using namespace Acts;
using namespace Acts::Experimental;

void test_flat_cache() {
    detail::FlatCache<1> cache(0.1);

    cache.insert({0.0, {{detail::GaussianComponent{0, 0, 0}}}});
    cache.insert({1.0, {{detail::GaussianComponent{1, 1, 1}}}});
    cache.insert({2.0, {{detail::GaussianComponent{2, 2, 2}}}});
    cache.insert({3.0, {{detail::GaussianComponent{3, 3, 3}}}});

    throw_assert(not cache.findApprox(1.5).has_value(), "bla");
    throw_assert(not cache.findApprox(4.0).has_value(), "bla");
    throw_assert(not cache.findApprox(-1.0).has_value(), "bla");
    throw_assert(cache.findApprox(1.0).has_value() && cache.findApprox(1.0)->at(0).weight == 1.0, "bla");
    throw_assert(cache.findApprox(2.05).has_value() && cache.findApprox(2.05)->at(0).weight == 2.0, "bla");
    throw_assert(cache.findApprox(2.97).has_value() && cache.findApprox(2.97)->at(0).weight == 3.0, "bla");
}


int main() {
#if 1
    const auto atlasBetheHeitlerApprox = AtlasBetheHeitlerApprox<6, 5>::loadFromFile(
        "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par",
        "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par"
    );
#else
    const auto atlasBetheHeitlerApprox = AtlasBetheHeitlerApprox<6, 5>(bh_cdf_cmps6_order5_data, bh_cdf_cmps6_order5_data, true, true);
#endif

    const double start = 0.0;
    const double step = 0.0001;
    const double stop = 0.9999;
    std::vector<double> x(static_cast<std::size_t>((stop-start)/step));
    std::ranges::generate(x, [&, n = 1]() mutable { return start + n++ * step; });

    constexpr static int NComponents = 12;

    auto startValue = [&]() {
        std::array<detail::GaussianComponent, NComponents> ret{};

        double m = 0.;
        for(auto i=0ul; i<ret.size(); ++i) {
            ret[i].weight = 1./(ret.size() -i);

            m += std::pow(0.5,i+1);
            ret[i].mean = m;
            ret[i].var = 0.0001 / (10*i + 1);
        }

        auto d = 1-ret.back().mean;

        for(auto &cmp : ret) {
            cmp.mean += 0.99*d;
        }

        detail::normalizeWeights(ret,
                                [](auto &a) -> double & { return a.weight; });

        return ret;
    }();

    // my handcrafted
    startValue = std::array<detail::GaussianComponent, NComponents>{{
        {1, 0.5, 1.e-5},
        {2, 0.99, 1.e-5},
        {2, 0.99, 1.e-5},
        {2, 0.99, 1.e-5},
        {2,.99, 1.e-5},
        {2,.99, 1.e-5},
        {2, 0.99, 1.e-5},
        {2,.99, 1.e-5},
        {2,.99, 1.e-5},
        {2, 0.99, 1.e-5},
        {2,.99, 1.e-5},
        {2,.99, 1.e-5},
    }};



        detail::normalizeWeights(startValue,
                                [](auto &a) -> double & { return a.weight; });

    using Kwargs = std::map<std::string, std::string>;
    auto mix_plot = [&](auto this_mixture, double thickness, Kwargs kwargs = Kwargs{}) {
        auto mixture_function = detail::GaussianMixtureModelPDF<this_mixture.size()>{this_mixture};
        auto bh_distribution = detail::BetheHeitlerPDF{thickness};

        std::vector<double> view_bh, view_mx;
        std::transform(x.begin(), x.end(), std::back_inserter(view_bh), bh_distribution);
        std::transform(x.begin(), x.end(), std::back_inserter(view_mx), mixture_function);
        // auto view_bh = x | std::ranges::views::transform(bh_distribution);
        // auto view_mx = x | std::ranges::views::transform(mixture_function);

        std::vector<double> y_bh(view_bh.begin(), view_bh.end());
        std::vector<double> y_mx(view_mx.begin(), view_mx.end());


        plt::plot(x, y_bh, Kwargs{{"label", "bethe_heitler"}, {"color","black"}});
        plt::plot(x, y_mx, kwargs);
    };

    auto gen = std::make_shared<std::mt19937>(std::random_device{}());

    // Minimize mixture
    const auto iterations = 3000;
    std::vector<double> temperatures;
    for(int i=0; i<iterations; ++i) {
        temperatures.push_back(1*std::exp(-0.0001*i));
    }

    auto next = [](std::array<double, 3*NComponents> ps, std::mt19937 &generator) {
      auto val_dist = std::uniform_real_distribution{-0.5, 0.5};
#if 1
      auto idx = std::uniform_int_distribution(0ul, ps.size())(generator);
      ps[idx] += ps[idx]*val_dist(generator);
#else
      for (auto &p : ps) {
        p += p*val_dist(generator);
      }
#endif
      return ps;
    };

    BetheHeitlerSimulatedAnnealingMinimizer bh(temperatures, startValue, gen, next);

    const std::vector<double> thicknesses = {0.02, 0.1, 0.2};
    for(auto i=0ul; i<thicknesses.size(); ++i) {

        plt::subplot(3, thicknesses.size(), i+1);
        mix_plot(startValue, thicknesses[i]);
        plt::title(fmt::format("Start distribution at x/x0 = {}", thicknesses[i]));

        std::vector<double> cdf_history;
        const auto cdf_mixture = bh.mixture(thicknesses[i], &cdf_history);

        // Make atlas mixture for comparison
        const auto atlas_mixture = atlasBetheHeitlerApprox.mixture(thicknesses[i]);
        const auto atlas_dist = boost::math::quadrature::trapezoidal(
            detail::CDFIntegrant<atlas_mixture.size()>{
                detail::GaussianMixtureModelCDF<atlas_mixture.size()>{atlas_mixture},
                detail::BetheHeitlerCDF{thicknesses[i]}},
            -0.5, 1.5);

        plt::subplot(3, thicknesses.size(), i+4);
        plt::plot(cdf_history, Kwargs{{"label", "cdf"}});
        plt::title(fmt::format("History for x/x0 = {}", thicknesses[i]));
        plt::legend();

        plt::subplot(3, thicknesses.size(), i+7);
        mix_plot(atlas_mixture, thicknesses[i], Kwargs{{"label", fmt::format("atlas, Nc={}, dist={:.5f}", atlas_mixture.size(), atlas_dist)}});
        mix_plot(cdf_mixture, thicknesses[i], Kwargs{{"label", fmt::format("cdf, Nc={}, dist={:.5f}", NComponents, *std::ranges::min_element(cdf_history))}});
        plt::title(fmt::format("Mixtures for x/x0 = {}", thicknesses[i]));
        plt::legend();
    }
    plt::show();
}

