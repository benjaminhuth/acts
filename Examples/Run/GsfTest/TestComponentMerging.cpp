#define BOOST_TEST_MODULE GSFTESTS

#include <boost/test/unit_test.hpp>

#include "KLMixtureReduction.hpp"

using DummyComponent = Acts::detail::GsfComponentCache<int>;

BOOST_AUTO_TEST_CASE(test_component_merging) {
  DummyComponent a;
  a.predictedPars = Acts::BoundVector::Random();
  a.predictedCov = Acts::BoundSymMatrix::Random().cwiseAbs();
  *a.predictedCov *= a.predictedCov->transpose();
  a.weight = 0.5;

  DummyComponent b;
  b.predictedPars = Acts::BoundVector::Random();
  b.predictedCov = Acts::BoundSymMatrix::Random().cwiseAbs();
  *b.predictedCov *= b.predictedCov->transpose();
  b.weight = 0.5;

  DummyComponent c = Acts::detail::mergeComponents(a, a);
  BOOST_CHECK(c.predictedPars == a.predictedPars);
  BOOST_CHECK(*c.predictedCov == *a.predictedCov);
  BOOST_CHECK(c.weight == 1.0);

  DummyComponent d = Acts::detail::mergeComponents(a, b);
  BOOST_CHECK(d.predictedPars == 0.5 * (a.predictedPars + b.predictedPars));
  BOOST_CHECK(d.weight == 1.0);

  std::vector<DummyComponent> cmps = {a, b, d};

  const auto [pars, cov] = Acts::detail::combineComponentRange(
      cmps.begin(), cmps.end(), [](auto &a) {
        return std::tie(a.weight, a.predictedPars, a.predictedCov);
      });

  BOOST_CHECK(pars == (a.predictedPars * a.weight + b.predictedPars * b.weight +
                       d.predictedPars * d.weight));
}

BOOST_AUTO_TEST_CASE(test_component_reduction) {
  const std::size_t NCompsBefore = 10;
  const std::size_t NCompsAfter = 5;

  // Create start state
  std::vector<DummyComponent> cmps;

  for (auto i = 0ul; i < NCompsBefore; ++i) {
    DummyComponent a;
    a.predictedPars = Acts::BoundVector::Random();
    a.predictedCov = Acts::BoundSymMatrix::Random().cwiseAbs();
    *a.predictedCov *= a.predictedCov->transpose();
    a.weight = 1.0 / NCompsBefore;
    cmps.push_back(a);
  }

  // Determine mean
  const auto meanBefore = std::accumulate(
      cmps.begin(), cmps.end(), Acts::BoundVector::Zero().eval(),
      [](auto sum, const auto &cmp) -> Acts::BoundVector {
        return sum + cmp.weight * cmp.predictedPars;
      });

  const double weightSumBefore = std::accumulate(
      cmps.begin(), cmps.end(), 0.0,
      [](auto sum, const auto &cmp) { return sum + cmp.weight; });

  BOOST_CHECK_CLOSE(weightSumBefore, 1.0, 0.0001);

  // Combine
  Acts::detail::reduceWithKLDistance(cmps, NCompsAfter);

  const auto meanAfter = std::accumulate(
      cmps.begin(), cmps.end(), Acts::BoundVector::Zero().eval(),
      [](auto sum, const auto &cmp) -> Acts::BoundVector {
        return sum + cmp.weight * cmp.predictedPars;
      });

  const double weightSumAfter = std::accumulate(
      cmps.begin(), cmps.end(), 0.0,
      [](auto sum, const auto &cmp) { return sum + cmp.weight; });

  // Check
  BOOST_CHECK_CLOSE(weightSumAfter, weightSumBefore, 0.0001);
  BOOST_CHECK(cmps.size() == NCompsAfter);
  std::cout << "mean before: " << meanBefore.transpose() << "\n";
  std::cout << "mean after: " << meanAfter.transpose() << "\n";
  BOOST_CHECK((meanBefore - meanAfter).cwiseAbs().all() < 1.e-4);
}
