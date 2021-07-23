#define BOOST_TEST_MODULE GSFTESTS

#include <boost/test/unit_test.hpp>

#include "KLMixtureReduction.hpp"

using DummyComponent = Acts::detail::GsfComponentCache<int>;

BOOST_AUTO_TEST_CASE(test_merge_two_equal_components) {
  DummyComponent a;
  a.predictedPars = Acts::BoundVector::Random();
  a.predictedCov = Acts::BoundSymMatrix::Random().cwiseAbs();
  *a.predictedCov *= a.predictedCov->transpose();
  a.weight = 0.5;

  DummyComponent c = Acts::detail::mergeComponents(a, a);
  BOOST_CHECK(c.predictedPars == a.predictedPars);
  BOOST_CHECK(*c.predictedCov == *a.predictedCov);
  BOOST_CHECK(c.weight == 1.0);
}

BOOST_AUTO_TEST_CASE(test_merge_two_different_components) {
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

  DummyComponent c = Acts::detail::mergeComponents(a, b);
  BOOST_CHECK(c.predictedPars == 0.5 * (a.predictedPars + b.predictedPars));
  BOOST_CHECK(c.weight == 1.0);
}

BOOST_AUTO_TEST_CASE(test_component_reduction) {
  const std::size_t NCompsBefore = 10;

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
  while (cmps.size() >= 2) {
    auto merge_iter_a = cmps.begin();
    auto merge_iter_b = std::next(cmps.begin());

    *merge_iter_a = Acts::detail::mergeComponents(*merge_iter_a, *merge_iter_b);
    cmps.erase(merge_iter_b);

    const auto mean = std::accumulate(
        cmps.begin(), cmps.end(), Acts::BoundVector::Zero().eval(),
        [](auto sum, const auto &cmp) -> Acts::BoundVector {
          return sum + cmp.weight * cmp.predictedPars;
        });

    const double weightSum = std::accumulate(
        cmps.begin(), cmps.end(), 0.0,
        [](auto sum, const auto &cmp) { return sum + cmp.weight; });

    BOOST_CHECK((mean - meanBefore).cwiseAbs().all() < 1.e-4);
    BOOST_CHECK_CLOSE(weightSum, 1.0, 0.0001);
  }
}

BOOST_AUTO_TEST_CASE(test_kl_mixture_reduction)
{
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

  Acts::detail::reduceWithKLDistance(cmps, NCompsAfter);

  const auto meanAfter = std::accumulate(
      cmps.begin(), cmps.end(), Acts::BoundVector::Zero().eval(),
      [](auto sum, const auto &cmp) -> Acts::BoundVector {
        return sum + cmp.weight * cmp.predictedPars;
      });

  const double weightSumAfter = std::accumulate(
      cmps.begin(), cmps.end(), 0.0,
      [](auto sum, const auto &cmp) { return sum + cmp.weight; });

  BOOST_CHECK_CLOSE(weightSumAfter, 1.0, 0.0001);
  BOOST_CHECK((meanAfter - meanBefore).cwiseAbs().all() < 1.e-4);
  BOOST_CHECK(cmps.size() == NCompsAfter);
}
